import genjaxmix.model.dsl as dsl
from abc import ABC, abstractmethod, ABCMeta
import jax
import jax.numpy as jnp
import genjaxmix.analytical.logpdf as logpdf
from genjaxmix.model.utils import topological_sort, count_unique
from dataclasses import dataclass
from typing import List, Dict
from genjaxmix.analytical.posterior import (
    get_segmented_posterior_sampler,
    get_posterior_sampler,
)


class PostInitCaller(ABCMeta):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class Model(ABC):
    def __init__(self):
        self._nodes = dict()

    def __post_init__(self):
        self._discover_nodes()

    def __setattr__(self, name, value):
        if isinstance(value, dsl.Node):
            self._nodes[name] = value
        super().__setattr__(name, value)

    def __getitem__(self, key):
        return self.node_to_id[self._nodes[key]]

    def __len__(self):
        return len(self.nodes)

    def initalize_parameters(self, key):
        environment = dict()

        keys = jax.random.split(key, len(self.nodes))
        for id in range(len(self.nodes)):
            environment[id] = self.nodes[id].initialize(keys[id])
        self.environment = environment

    def observe(self, observations):
        if set(observations.keys()) != set(self.observations()):
            missing_keys = set(self._nodes.keys()) - set(observations.keys())
            extraneous_keys = set(observations.keys()) - set(self._nodes.keys())
            raise ValueError(
                f"Observation keys do not match model keys. Missing keys: {missing_keys}. Extraneous keys: {extraneous_keys}"
            )

        for key, value in observations.items():
            id = self.node_to_id[self._nodes[key]]
            self.environment[id] = value

    @abstractmethod
    def observations(self):
        return self.observables

    def compile(self, observables=None):
        self._discover_nodes()

        if observables is None:
            observables = self.observations()

        for observable in observables:
            if observable not in self._nodes:
                raise ValueError(
                    f"Observation variable {observable} not found in model"
                )

        observables = [self._nodes[observable] for observable in observables]

        return self._codegen(observables)

    def _discover_nodes(self):
        node_to_id = dict()
        nodes = []

        edges = dict()
        backedges = dict()

        types = []
        environment = dict()

        queue = list(self._nodes.values())
        if len(queue) == 0:
            raise ValueError("no nodes found in model")

        id = 0

        for node in queue:
            edges[id] = []
            backedges[id] = []
            types.append(type(node))
            nodes.append(node)
            node_to_id[node] = id
            id += 1

        while len(queue) > 0:
            node = queue.pop(0)
            child_id = node_to_id[node]

            for parent in node.parents():
                if parent not in node_to_id:
                    queue.append(parent)
                    edges[id] = []
                    backedges[id] = []
                    types.append(type(parent))
                    nodes.append(parent)
                    node_to_id[parent] = id
                    id += 1

                parent_id = node_to_id[parent]

                if parent_id not in edges[child_id]:
                    edges[child_id].append(parent_id)

                if child_id not in backedges[parent_id]:
                    backedges[parent_id].append(child_id)

        self.edges = edges
        self.backedges = backedges
        self.types = types
        self.ordering = topological_sort(self.backedges)
        self.environment = environment
        self.nodes = nodes
        self.node_to_id = node_to_id

    def _codegen(self, observables):
        self.build_proportions_proposal()
        self.build_parameter_proposal(observables)
        self.build_assignment_proposal(observables)

        # combine all proposals in one program

        def proposal(key, environment, pi, assignments):
            environment = environment.copy()
            subkeys = jax.random.split(key, 4)
            pi = self.pi_proposal(subkeys[0], assignments, pi)
            environment = self.parameter_proposal(subkeys[1], environment, assignments)
            assignments = self.assignment_proposal(
                subkeys[2], environment, pi, assignments
            )
            return environment, assignments, pi

        self.infer = proposal
        return proposal

    def build_proportions_proposal(self):
        self.pi_proposal = gibbs_pi

    def build_parameter_proposal(self, observables):
        proposals = dict()
        for id in range(len(self.nodes)):
            blanket = MarkovBlanket.from_model(self, id, observables)
            proposal = build_parameter_proposal(blanket)
            if proposal:
                proposals[id] = proposal
        self.parameter_proposals = proposals

        # combine parameter proposals
        def parameter_proposal(key, environment, assignments):
            environment = environment.copy()
            for id in self.parameter_proposals.keys():
                environment = self.parameter_proposals[id](
                    key, environment, assignments
                )
            return environment

        self.parameter_proposal = parameter_proposal

        return self.parameter_proposal

    def build_assignment_proposal(self, observables):
        likelihoods = dict()
        observed_likelihoods = dict()
        latent_likelihoods = dict()
        for id in range(len(self.nodes)):
            if self.types[id] == dsl.Constant:
                continue
            blanket = MarkovBlanket.from_model(self, id, observables)
            logpdf, is_vectorized = build_loglikelihood_at_node(blanket)
            if logpdf:
                likelihoods[id] = logpdf
                if is_vectorized:
                    observed_likelihoods[id] = logpdf
                else:
                    latent_likelihoods[id] = logpdf

        self.likelihood_fns = likelihoods

        def assignment_proposal(key, environment, pi, assignments):
            K = count_unique(assignments)
            K_max = pi.shape[0]
            log_p = jnp.zeros(pi.shape[0])
            for id in latent_likelihoods.keys():
                log_p += latent_likelihoods[id](environment)

            for id in observed_likelihoods.keys():
                log_p += observed_likelihoods[id](environment)

            log_p += jnp.log(pi)
            log_p = jnp.where(jnp.arange(K_max) < K, log_p, -jnp.inf)
            z = jax.random.categorical(key, log_p)
            return z

        self.assignment_proposal = assignment_proposal
        return self.assignment_proposal


@dataclass
class MarkovBlanket:
    id: int
    parents: List[int]
    children: List[int]
    cousins: Dict
    types: Dict
    observed: Dict

    @classmethod
    def from_model(cls, model: Model, id: int, observations):
        edges = model.edges
        backedges = model.backedges
        _types = model.types

        types = dict()
        observed = dict()

        id = id
        types[id] = _types[id]
        observed[id] = model.nodes[id] in observations

        parents = edges[id]
        for parent in parents:
            types[parent] = _types[parent]
            observed[parent] = model.nodes[parent] in observations

        children = backedges[id]
        for child in children:
            types[child] = _types[child]
            observed[child] = model.nodes[child] in observations

        coparent = dict()
        for child in children:
            coparent[child] = edges[child]
            for ii in coparent[child]:
                types[ii] = _types[ii]
                observed[ii] = model.nodes[ii] in observations

        return cls(id, parents, children, coparent, types, observed)


def gibbs_pi(key, assignments, pi):
    alpha = 1.0
    K = count_unique(assignments)
    K_max = pi.shape[0]
    counts = jnp.bincount(assignments, length=K_max)
    alpha_new = jnp.where(jnp.arange(K_max) < K, counts, alpha)
    alpha_new = jnp.where(jnp.arange(K_max) < K + 1, alpha_new, 0.0)
    pi_new = jax.random.dirichlet(key, alpha_new)
    return pi_new


def build_parameter_proposal(blanket: Model):
    id = blanket.id
    if blanket.observed[id] or blanket.types[id] == dsl.Constant:
        return None

    if has_conjugate_rule(blanket):
        return build_gibbs_proposal(blanket)
    else:  # default to MH
        return build_mh_proposal(blanket)


def build_loglikelihood_at_node(blanket: MarkovBlanket):
    id = blanket.id
    observed = blanket.observed
    is_vectorized = observed[id] or any(
        [observed[parent] for parent in blanket.parents]
    )

    logpdf_lambda = logpdf.get_logpdf(blanket.types[id])
    if is_vectorized:
        inner_axes = (None,) + tuple(
            None if blanket.observed[ii] else 0 for ii in blanket.parents
        )
        outer_axes = (0,) + tuple(
            0 if blanket.observed[ii] else None for ii in blanket.parents
        )

        def loglikelihood(environment):
            return jax.vmap(
                jax.vmap(logpdf_lambda, in_axes=inner_axes), in_axes=outer_axes
            )(environment[id], *[environment[parent] for parent in blanket.parents])
    else:
        axes = (0,) + tuple(0 for ii in blanket.parents)

        def loglikelihood(environment):
            return jax.vmap(logpdf_lambda, in_axes=axes)(
                environment[id], *[environment[parent] for parent in blanket.parents]
            )

    return loglikelihood, is_vectorized


#############
# Conjugacy #
#############
def has_conjugate_rule(blanket: MarkovBlanket):
    if len(blanket.children) != 1:
        return False

    prior = blanket.types[blanket.id]
    likelihood = blanket.types[blanket.children[0]]
    return (prior, likelihood) in CONJUGACY_RULES


def get_conjugate_rule(blanket: MarkovBlanket):
    prior = blanket.types[blanket.id]
    likelihood = blanket.types[blanket.children[0]]
    return CONJUGACY_RULES[(prior, likelihood)]


def jax_normal_normal_sigma_known(blanket: MarkovBlanket):
    mu_0_id, sig_0_id = blanket.parents
    sig_id = blanket.cousins[blanket.children[0]][1]
    observations = blanket.children[0]
    return (mu_0_id, sig_0_id, sig_id, observations)


def jax_gamma_normal_mu_known(blanket: MarkovBlanket):
    alpha_id, beta_id = blanket["parents"]
    mu_id = blanket["cousins"][0]
    return (alpha_id, beta_id, mu_id)


CONJUGACY_RULES = {
    (dsl.Normal, dsl.Normal): jax_normal_normal_sigma_known,
    (dsl.Gamma, dsl.Normal): jax_gamma_normal_mu_known,
}


def build_gibbs_proposal(blanket: MarkovBlanket):
    arg_rule = get_conjugate_rule(blanket)
    if arg_rule is None:
        raise ValueError("No conjugate rule found???")

    id = blanket.id
    proposal_signature = (blanket.types[id], blanket.types[blanket.children[0]])
    posterior_args = arg_rule(blanket)

    # check if likelihood is an observable

    likelihood_observed = blanket.observed[blanket.children[0]]

    if not likelihood_observed:
        parameter_proposal = get_posterior_sampler(*proposal_signature)

        def gibbs_sweep(key, environment, assignments):
            environment = environment.copy()
            conditionals = [environment[ii] for ii in posterior_args]
            environment[id] = parameter_proposal(key, conditionals)
            return environment

        return gibbs_sweep
    else:
        parameter_proposal = get_segmented_posterior_sampler(*proposal_signature)

        def gibbs_sweep(key, environment, assignments):
            environment = environment.copy()
            conditionals = [environment[ii] for ii in posterior_args]
            environment[id] = parameter_proposal(key, conditionals, assignments)

            return environment

        return gibbs_sweep


###############
# MH Proposal #
###############


def _build_obs_likelihood_at_node(blanket: MarkovBlanket, substitute_id):
    # TODO: Both cases seem to be the same. See if combining works.
    id = blanket.id
    observed = blanket.observed
    is_vectorized = observed[id] or any(
        [observed[parent] for parent in blanket.parents]
    )

    logpdf_lambda = logpdf.get_logpdf(blanket.types[id])
    if is_vectorized:
        axes = (0,) + tuple(0 for ii in blanket.parents)

        def loglikelihood(substituted_value, assignments, environment):
            def promote_shape(ii, arr, assignments):
                if blanket.observed[ii]:
                    return arr
                else:
                    return arr[assignments]

            def swap(id):
                if id == substitute_id:
                    return substituted_value
                else:
                    return environment[id]

            x = swap(id)

            return jax.vmap(logpdf_lambda, in_axes=axes)(
                promote_shape(id, x, assignments),
                *[
                    promote_shape(parent, swap(parent), assignments)
                    for parent in blanket.parents
                ],
            )
    else:
        axes = (0,) + tuple(0 for ii in blanket.parents)

        def loglikelihood(substituted_value, assignments, environment):
            def swap(id):
                if id == substitute_id:
                    return substituted_value
                else:
                    return environment[id]

            return jax.vmap(logpdf_lambda, in_axes=axes)(
                swap(id), *[swap(parent) for parent in blanket.parents]
            )

    return loglikelihood, is_vectorized


def build_mh_proposal(blanket: MarkovBlanket):
    likelihood_fns = {"observed": dict(), "unobserved": dict()}
    id = blanket.id
    likelihood, is_vectorized = _build_obs_likelihood_at_node(blanket, id)
    if is_vectorized:
        likelihood_fns["observed"][id] = likelihood
    else:
        likelihood_fns["unobserved"][id] = likelihood

    for child in blanket.children:
        fake_blanket = MarkovBlanket(
            child, blanket.cousins[child], [], [], blanket.types, blanket.observed
        )
        likelihood, is_vectorized = _build_obs_likelihood_at_node(fake_blanket, id)
        if is_vectorized:
            likelihood_fns["observed"][child] = likelihood
        else:
            likelihood_fns["unobserved"][child] = likelihood

    def mh_move(key, environment, assignments):
        environment = environment.copy()

        # TODO: random walk must use the correct sample space
        x_old = environment[id]
        x_new = x_old + 0.1 * jax.random.normal(key, shape=x_old.shape)

        ratio = 0.0
        for ii, likelihood_fn in likelihood_fns["unobserved"].items():
            ratio += likelihood_fn(x_new, assignments, environment)
            ratio -= likelihood_fn(x_old, assignments, environment)

        for ii, likelihood_fn in likelihood_fns["observed"].items():
            increment = likelihood_fn(x_new, assignments, environment) - likelihood_fn(
                x_old, assignments, environment
            )
            increment = jax.ops.segment_sum(
                increment, assignments, num_segments=x_old.shape[0]
            )
            ratio += increment

        logprob = jnp.minimum(0.0, ratio)

        u = jax.random.uniform(key, shape=ratio.shape)
        accept = u < jnp.exp(logprob)
        x = jnp.where(accept[:, None], x_new, x_old)
        environment[id] = x
        return environment

    return mh_move
