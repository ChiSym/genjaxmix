import genjaxmix.model.dsl as dsl
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import genjaxmix.dpmm.dpmm as dpmm
import genjaxmix.analytical.logpdf as logpdf
from genjaxmix.model.utils import topological_sort
from plum import dispatch


class Program:
    edges: dict
    backedges: dict
    types: list
    nodes: list
    node_to_id: dict
    environment: dict
    proposals: dict

    def __init__(self, model):
        edges = dict()
        backedges = dict()
        types = []
        nodes = []
        node_to_id = dict()
        environment = dict()

        edges[0] = []
        backedges[0] = []
        types.append(type(model))
        nodes.append(model)
        node_to_id[model] = 0
        id = 1

        queue = [model]
        while len(queue) > 0:
            node = queue.pop(0)
            child_id = node_to_id[node]
            # environment[child_id] = node.value()
            for parent in node.children():
                if parent not in node_to_id:
                    queue.append(parent)
                    types.append(type(parent))
                    nodes.append(parent)
                    edges[id] = []
                    backedges[id] = []
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
        self.environment = environment
        self.nodes = nodes
        self.node_to_id = node_to_id
        self.ordering = topological_sort(self.backedges)

    def initalize_parameters(self, key):
        keys = jax.random.split(key, len(self.nodes))
        for id in range(len(self.nodes)):
            self.environment[id] = self.nodes[id].initialize(keys[id])

    def markov_blanket(self, id: int):
        parents = self.edges[id]
        children = self.backedges[id]
        cousins = []
        for child in children:
            cousins += filter(lambda i: i != id, self.edges[child])
        return {"parents": parents, "children": children, "cousins": cousins}

    @dispatch
    def build_parameter_proposal(self, id: int):
        if isinstance(self.types[id], dsl.Constant):
            return None

        blanket = self.markov_blanket(id)

        if len(blanket["children"]) == 1:  # shortcut to conjugate
            child_id = blanket["children"][0]

            prior_type = self.types[id]
            likelihood_type = self.types[child_id]

            arg_rule = conjugate_rule(prior_type, likelihood_type)
            if arg_rule:
                posterior_pair = (self.nodes[id], self.nodes[child_id])
                posterior_args = arg_rule(blanket)
                parameter_proposal = dpmm.gibbs_parameters_proposal(*posterior_pair)

                def gibbs_sweep(key, environment, observations, assignments):
                    conditionals = [environment[ii] for ii in posterior_args]
                    environment[id] = parameter_proposal(
                        key, conditionals, observations, assignments
                    )

                    return environment

                return gibbs_sweep

            prior_fn = logpdf.logpdf(self.nodes[id])
            # TODO: is an observation node
            is_observation_node = []
            for child in blanket["children"]:
                is_observation_node.append(True)

            likelihood_fns = [
                logpdf.logpdf(self.nodes[child]) for child in blanket["children"]
            ]

            def mh_move(key, environment, observations, assignments):
                x = environment[id]
                x_new = x + 0.1 * jax.random.normal(key, shape=x.shape)

                prior_conditionals = [environment[ii] for ii in blanket["parents"]]
                ratio = 0.0

                for child in blanket["children"]:
                    input_nodes = self.edges[child]
                    assert id in input_nodes
                    idx = input_nodes.index(id)
                    # TODO: perform shape promotion to ensure parameters have same shape as observations
                    is_observation_node = True
                    if is_observation_node:
                        likelihood_conditionals = [
                            environment[jj][assignments] for jj in input_nodes
                        ]
                        log_p_old = likelihood_fns[child](
                            observations, likelihood_conditionals
                        )
                        likelihood_conditionals[idx] = x_new[assignments]
                        log_p_new = likelihood_fns[child](
                            observations, likelihood_conditionals
                        )
                        increment = log_p_new - log_p_old
                        increment = jax.ops.segment_sum(
                            increment, assignments, num_segments=x.shape[0]
                        )
                        ratio += increment
                    else:
                        likelihood_conditionals = [
                            environment[jj][assignments] for jj in input_nodes
                        ]
                        log_p_old = likelihood_fns[child](likelihood_conditionals)
                        likelihood_conditionals[idx] = x_new[assignments]
                        log_p_new = likelihood_fns[child](likelihood_conditionals)
                        ratio += log_p_new - log_p_old

                ratio += prior_fn(x_new, prior_conditionals) - prior_fn(
                    x, prior_conditionals
                )
                logprob = jnp.minimum(0.0, ratio)

                u = jax.random.uniform(key, shape=ratio.shape)
                accept = u < jnp.exp(logprob)
                x = jnp.where(accept[:, None], x_new, x)
                return x

            return mh_move

    def build_single_proposal(self, id: int):
        if self.types[id] == dsl.Constant:
            return None
        blanket = self.markov_blanket(id)
        if len(blanket["children"]) == 0:
            assert len(blanket["cousins"]) == 0

            parent_types = tuple([self.types[i] for i in blanket["parents"]])
            signature = (self.types[id], parent_types)

            rule = conjugate_rule(signature)
            if rule:
                posterior_args, posterior_type = rule(self.edges, self.nodes, id)
                logpdf_pair = self.nodes[id]
                logpdf_args = tuple(self.edges[id])
                parameter_proposal = dpmm.gibbs_parameters_proposal(*posterior_type)
                z_proposal = dpmm.gibbs_z_proposal(logpdf_pair)

                def gibbs_sweep(key, pi, environment, observations, assignments):
                    subkeys = jax.random.split(key, 2)
                    conditionals = [environment[ii] for ii in posterior_args]
                    environment[id] = parameter_proposal(
                        subkeys[0], conditionals, observations, assignments
                    )

                    conditionals = [environment[ii] for ii in logpdf_args]
                    assignments = z_proposal(
                        subkeys[1], conditionals, observations, pi, 2
                    )
                    return environment, assignments

                return gibbs_sweep

    @dispatch
    def build_parameter_proposal(self):
        proposals = dict()
        for id in range(len(self.nodes)):
            proposal = self.build_parameter_proposal(id)
            if proposal:
                proposals[id] = proposal

        # combine all proposals in one program
        def parameter_proposal(key, environment, observations, assignments):
            # jax function should be pure?
            environment = environment.copy()

            for id, proposal in proposals.items():
                environment[id] = proposal(key, environment, observations, assignments)
            return environment

        return parameter_proposal


############
# ANALYSIS #
############


def _arg_normal_normal(blanket):
    mu_0_id, sig_0_id = blanket["parents"]
    sig_id = blanket["cousins"][0]
    return (mu_0_id, sig_0_id, sig_id)


def _arg_gamma_normal(blanket):
    alpha_id, beta_id = blanket["parents"]
    mu_id = blanket["cousins"][0]
    return (alpha_id, beta_id, mu_id)


# prior-likelihood
CONJUGACY_RULES = {
    (dsl.Normal, dsl.Normal): _arg_normal_normal,
    (dsl.Gamma, dsl.Normal): _arg_gamma_normal,
    (dsl.InverseGamma, dsl.Normal): _arg_gamma_normal,
}


def conjugate_rule(prior, likelihood):
    return CONJUGACY_RULES.get((prior, likelihood), None)
