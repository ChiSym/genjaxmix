import genjaxmix.model.dsl as dsl
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import genjaxmix.dpmm.dpmm as dpmm
import genjaxmix.analytical.logpdf as logpdf
from genjaxmix.model.utils import topological_sort, count_unique
from dataclasses import dataclass
from typing import List, Dict
from genjaxmix.analytical.posterior import get_segmented_posterior_sampler, get_posterior_sampler


class Model(ABC):
    def __init__(self):
        self._nodes = dict()

    def __setattr__(self, name, value):
        if isinstance(value, dsl.Node):
            self._nodes[name] = value
        super().__setattr__(name, value)

    def initalize_parameters(self, key):
        self._discover_nodes()

        keys = jax.random.split(key, len(self.nodes))
        for id in range(len(self.nodes)):
            self.environment[id] = self.nodes[id].initialize(keys[id])

    @abstractmethod
    def observations(self):
        return self.observables

    def compile(self, observables = None):
        self._discover_nodes()
        
        if observables is None:
            observables = self.observations()

        for observable in observables:
            if observable not in self._nodes:
                raise ValueError(f"Observation variable {observable} not found in model")
        
        observables = [self._nodes[observable] for observable in observables]

        self._discover_nodes()
        self._codegen(observables)

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
        self.build_assignment_proposal()

        # combine
        def proposal(key, environment, observations, assignments):
            return environment

        return proposal

    def build_proportions_proposal(self):
        self.proposals["pi"] = gibbs_pi

    def build_parameter_proposal(self, observables):
        proposals = dict()
        for id in range(len(self.nodes)):
            blanket = MarkovBlanket(self, id, observables)
            proposal = build_parameter_proposal(blanket)
            if proposal:
                proposals[id] = proposal
        self.proposals = proposals

        # combine all proposals in one program
        def parameter_proposal(key, environment, observations, assignments):
            # jax function should be pure?
            environment = environment.copy()

            for id, proposal in proposals.items():
                environment[id] = proposal(key, environment, observations, assignments)
            return environment

        return parameter_proposal

    def build_assignment_proposal(self):
        pass
        # def assignment_proposal(key, environment, observations):
        #     return dpmm.dpmm(key, environment, observations, self.proposals["pi"], self.proposals)

        # self.proposals["assignments"] = assignment_proposal

@dataclass
class MarkovBlanket:
    id: int
    parents: List[int]
    children: List[int]
    cousins: Dict
    types: Dict
    observed: Dict

    def __init__(self, model: Model, id, observations):
        edges = model.edges
        backedges = model.backedges
        types = model.types

        self.types = dict()
        self.observed = dict()

        self.id = id
        self.types[id] = types[id]
        self.observed[id] = model.nodes[id] in observations

        self.parents = edges[id]
        for parent in self.parents:
            self.types[parent] = types[parent]
            self.observed[parent] = model.nodes[parent] in observations

        self.children = backedges[id]
        for child in self.children:
            self.types[child] = types[child]
            self.observed[child] = model.nodes[child] in observations

        self.cousins = dict()
        for child in self.children:
            self.cousins[child] = edges[child]
            for cousin in self.cousins[child]:
                self.types[cousin] = types[cousin]
                self.observed[cousin] = model.nodes[cousin] in observations


def build_proportions_proposal():
    return gibbs_pi

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
    return (mu_0_id, sig_0_id, sig_id)


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
        def gibbs_sweep(key, environment, observations, assignments):
            environment = environment.copy()
            conditionals = [environment[ii] for ii in posterior_args]
            environment[id] = parameter_proposal(key, conditionals)
            return environment
        
        raise NotImplementedError("Incorrect logic")
        return gibbs_sweep
    else:
        parameter_proposal = get_segmented_posterior_sampler(*proposal_signature)

        def gibbs_sweep(key, environment, observations, assignments):
            environment = environment.copy()
            conditionals = [environment[ii] for ii in posterior_args]
            environment[id] = parameter_proposal(
                key, conditionals, observations, assignments
            )

            return environment

        return gibbs_sweep


###############
# MH Proposal #
###############


def build_mh_proposal(blanket: MarkovBlanket):
    # is_observation_node = []
    # for child in blanket["children"]:
    #     is_observation_node.append(child in self.observed)

    # likelihood_fns = [logpdf.logpdf(self.nodes[child]) for child in blanket["children"]]

    def mh_move(key, environment, observations, assignments):
        environment = environment.copy()
        x = environment[blanket.id]
        x_new = x + 0.1 * jax.random.normal(key, shape=x.shape)

        prior_conditionals = [environment[ii] for ii in blanket.parents]
        ratio = jnp.zeros(x.shape[0])

    #     for child in blanket["children"]:
    #         input_nodes = self.edges[child]
    #         assert id in input_nodes
    #         idx = input_nodes.index(id)
    #         # TODO: perform shape promotion to ensure parameters have same shape as observations
    #         is_observation_node = True
    #         if is_observation_node:
    #             likelihood_conditionals = [
    #                 environment[jj][assignments] for jj in input_nodes
    #             ]
    #             log_p_old = likelihood_fns[child](observations, likelihood_conditionals)
    #             likelihood_conditionals[idx] = x_new[assignments]
    #             log_p_new = likelihood_fns[child](observations, likelihood_conditionals)
    #             increment = log_p_new - log_p_old
    #             increment = jax.ops.segment_sum(
    #                 increment, assignments, num_segments=x.shape[0]
    #             )
    #             ratio += increment
    #         else:
    #             likelihood_conditionals = [
    #                 environment[jj][assignments] for jj in input_nodes
    #             ]
    #             log_p_old = likelihood_fns[child](likelihood_conditionals)
    #             likelihood_conditionals[idx] = x_new[assignments]
    #             log_p_new = likelihood_fns[child](likelihood_conditionals)
    #             ratio += log_p_new - log_p_old

    #     ratio += prior_fn(x_new, prior_conditionals) - prior_fn(x, prior_conditionals)
    #     logprob = jnp.minimum(0.0, ratio)

        u = jax.random.uniform(key, shape=ratio.shape)
        # accept = u < jnp.exp(logprob)
        # x = jnp.where(accept[:, None], x_new, x)
        # return x

    return mh_move