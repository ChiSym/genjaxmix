from typing import Union
from jax import Array
from plum import dispatch
from abc import abstractmethod, ABC
from enum import Enum


class DType(Enum):
    CONSTANT = 0
    NORMAL = 1
    GAMMA = 2
    INVERSE_GAMMA = 3
    NORMAL_INVERSE_GAMMA = 4


class Node(ABC):
    shape: list

    @abstractmethod
    def children(self):
        pass

    @abstractmethod
    def type(self):
        pass


class Constant(Node):
    def __init__(self, value):
        self.value = value
        self.shape = value.shape

    def children(self):
        return []

    def type(self):
        return DType.CONSTANT

    def __repr__(self):
        return f"Constant({self.value})"


class Normal(Node):
    fused: bool

    @dispatch
    def __init__(self, mu, sigma):
        if len(mu.shape) != 2:
            raise ValueError("mu must be a 2D array")
        if len(sigma.shape) != 2:
            raise ValueError("sigma must be a 2D array")
        if mu.shape != sigma.shape:
            raise ValueError("mu and sigma must have the same shape")

        if is_constant(mu):
            mu = Constant(mu)

        if is_constant(sigma):
            sigma = Constant(sigma)

        self.mu = mu
        self.sigma = sigma
        self.shape = mu.shape
        self.fused = False

    @dispatch
    def __init__(self, mu_and_sigma: Node):
        if is_constant(mu_and_sigma):
            raise NotImplementedError("mu_and_sigma must be a 3D array")
        self.mu = mu_and_sigma
        self.sigma = mu_and_sigma
        self.fused = True

    def children(self):
        return [self.mu, self.sigma]

    def type(self):
        return DType.NORMAL


class Gamma(Node):
    @dispatch
    def __init__(self, alpha, beta):
        if len(alpha.shape) != 2:
            raise ValueError("alpha must be a 2D array")
        if len(beta.shape) != 2:
            raise ValueError("beta must be a 2D array")
        if alpha.shape != beta.shape:
            raise ValueError("alpha and beta must have the same shape")

        if is_constant(alpha):
            alpha = Constant(alpha)
        if is_constant(beta):
            beta = Constant(beta)

        self.alpha = alpha
        self.beta = beta
        self.shape = alpha.shape

    @dispatch
    def __init__(self, alpha_and_beta: Node):
        self.alpha = alpha_and_beta
        self.beta = alpha_and_beta

    def children(self):
        return [self.alpha, self.beta]

    def type(self):
        return DType.GAMMA


def is_constant(obj):
    return isinstance(obj, Array)


def build_normal_normal_proposal(normal, mu, sigma):
    hyperparameters = (mu.mu.value, mu.sigma.value, sigma.value)
    print(hyperparameters)


CONJUGACY_RULES = {
    DType.NORMAL: {
        (DType.NORMAL, DType.CONSTANT): build_normal_normal_proposal,
        (DType.CONSTANT, DType.GAMMA): True,
        (DType.CONSTANT, DType.INVERSE_GAMMA): True,
        (DType.NORMAL_INVERSE_GAMMA,): True,
    },
}


def get_conjugacy_rule(parent, child):
    if parent not in CONJUGACY_RULES:
        return None
    return CONJUGACY_RULES[parent].get(child, None)


class IR:
    def __init__(self, edges, backedges, environment, node_to_id):
        self.edges = edges
        self.backedges = backedges
        self.environment = environment
        self.node_to_id = node_to_id

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
            likelihood = self.environment[id][0]
            priors = tuple(
                [self.environment[parent][0] for parent in blanket["parents"]]
            )
            var = self.environment[id][1]
            args = tuple([self.environment[parent][1] for parent in blanket["parents"]])
            print("*****")
            print(var)
            print(args)
            return get_conjugacy_rule(likelihood, priors)(var, *args)


def trace(model: Node):
    environment = dict()
    edges = dict()
    backedges = dict()
    node_to_id = dict()

    environment = [(model.type(), model)]
    edges[0] = []
    backedges[0] = []
    node_to_id[model] = 0
    id = 1

    queue = [model]
    while len(queue) > 0:
        node = queue.pop(0)
        child_id = node_to_id[node]
        for parent in node.children():
            if parent not in node_to_id:
                queue.append(parent)
                environment.append((parent.type(), parent))
                edges[id] = []
                backedges[id] = []
                node_to_id[parent] = id
                id += 1

            parent_id = node_to_id[parent]

            if parent_id not in edges[child_id]:
                edges[child_id].append(parent_id)

            if child_id not in backedges[parent_id]:
                backedges[parent_id].append(child_id)
    return IR(edges, backedges, environment, node_to_id)
