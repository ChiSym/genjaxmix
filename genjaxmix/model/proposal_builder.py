import genjaxmix.model.dsl as dsl
import jax
import genjaxmix.dpmm.dpmm as dpmm
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

    def initalize_parameters(self, key):
        keys = jax.random.split(key, len(self.nodes))
        for id in range(len(self.nodes)):
            self.environment[id] = self.nodes[id].initialize(keys[id])

    def markov_blanket(self, id: int):
        parents = self.edges[id]
        children = self.backedges[id]
        cousins = []
        for child in children:
            cousins += self.edges[child]
        return {"parents": parents, "children": children, "cousins": cousins}

    @dispatch
    def build_parameter_proposal(self, id: int):
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

    def build_proportion_proposal(self):
        pass

    @dispatch
    def build_parameter_proposal(self):
        proposals = dict()
        for id in range(len(self.nodes)):
            proposal = self.build_parameter_proposal(id)
            if proposal:
                proposals[id] = proposal
        print(proposals)

    def build_assignment_proposal(self):
        pass


############
# ANALYSIS #
############


# CONJUGACY_RULES = {
# }


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
}


def conjugate_rule(prior, likelihood):
    return CONJUGACY_RULES.get((prior, likelihood), None)
