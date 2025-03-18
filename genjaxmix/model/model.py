import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod, ABCMeta
from genjaxmix.model.utils import count_unique, topological_sort
from genjaxmix.model.compile import (
    MarkovBlanket,
    build_loglikelihood_at_node,
    build_parameter_proposal,
    gibbs_pi,
)
import genjaxmix.model.dsl as dsl


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
            blanket = blanket_from_model(self, id, observables)
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
            blanket = blanket_from_model(self, id, observables)
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

    def sample(self, key):
        environment = dict()
        for id in self.ordering:
            node = self.nodes[id]
            args = [environment[parent] for parent in self.edges[id]]
            environment[id] = node.sample(key, *args)

        return environment


def blanket_from_model(model: Model, id: int, observations):
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

    return MarkovBlanket(id, parents, children, coparent, types, observed)
