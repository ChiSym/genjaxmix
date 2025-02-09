from genjaxmix.model.dsl import Node, DType
from plum import dispatch
import genjaxmix.core as core
import jax

class Environment:
    vars: dict
    def __init__(self):
        self.vars = dict()
    def __setitem__(self, id, value):
        if id in self.vars:
            raise Exception("Variable already defined")
        self.vars[id] = value
    def __getitem__(self, id):
        return self.vars[id]

class Program:
    edges: dict
    backedges: dict
    types: list
    nodes: list
    node_to_id: dict
    environment: Environment
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
        types.append(model.type())
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
                    types.append(parent.type())
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

    @dispatch
    def markov_blanket(self, id: int):
        parents = self.edges[id]
        children = self.backedges[id]
        cousins = []
        for child in children:
            cousins += self.edges[child]
        return {"parents": parents, "children": children, "cousins": cousins}

    @dispatch
    def markov_blanket(self, node: Node):
        self.markov_blanket(self.node_to_id[node])

    def build_proposal(self, id: int):
        blanket = self.markov_blanket(id)
        if len(blanket["children"]) == 0:
            assert len(blanket["cousins"]) == 0

            likelihood = self.types[id]
            priors = tuple([self.types[parent] for parent in blanket["parents"]])

            # mu = Normal(mu_0, sigma_0)
            # sigma = Constant
            mu_id = blanket["parents"][0]
            mu_0_id = self.edges[mu_id][0]
            sig_0_id = self.edges[mu_id][1]
            sig_id = blanket["parents"][1]

            proposal = {
                "var": mu_id,
                "posterior_args": (mu_0_id, sig_0_id, sig_id),
                "posterior": (core.Normal(), core.Normal()),
                "logpdf_args": (mu_id, sig_id),
                "logpdf": (core.Normal(), core.Normal())
            }
            return proposal
            


############
# ANALYSIS #
############




CONJUGACY_RULES = {
    DType.NORMAL: {
        (DType.NORMAL, DType.CONSTANT): True,
        (DType.CONSTANT, DType.GAMMA): True,
        (DType.CONSTANT, DType.INVERSE_GAMMA): True,
        (DType.NORMAL_INVERSE_GAMMA,): True,
    },
}


def conjugacy_rule(parent, child):
    if parent not in CONJUGACY_RULES:
        return None
    return CONJUGACY_RULES[parent].get(child, None)
