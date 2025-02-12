from typing import Union
from jax import Array
import jax
from plum import dispatch
from abc import abstractmethod, ABC
from enum import Enum


class Node(ABC):
    shape: list

    @abstractmethod
    def children(self):
        pass

    @abstractmethod
    def initialize(self, key):
        pass


class Constant(Node):
    def __init__(self, value):
        self.value = value
        self.shape = value.shape

    def children(self):
        return []

    def initialize(self, key):
        return self.value

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

    def initialize(self, key):
        if not self.fused:
            return jax.random.normal(key, self.mu.shape)

    def __repr__(self):
        return f"Normal({self.mu}, {self.sigma})"


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

    def initialize(self, key):
        if isinstance(self.alpha, Constant):
            return jax.random.gamma(key, self.alpha.value) * self.beta.value
        else:
            raise NotImplementedError("WIP: Need to process in topological order")


class InverseGamma(Node):
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

    def children(self):
        return [self.alpha, self.beta]

    def initialize(self, key):
        if isinstance(self.alpha, Constant):
            return jax.random.gamma(key, self.alpha.value) * self.beta.value
        else:
            raise NotImplementedError("WIP: Need to process in topological order")

class Exponential(Node):
    def __init__(self, rate):
        if len(rate.shape) != 2:
            raise ValueError("rate must be a 2D array")

        if is_constant(rate):
            rate = Constant(rate)

        self.rate = rate
        self.shape = rate.shape

    def children(self):
        return [self.rate]
    
    def initialize(self, key):
        if isinstance(self.rate, Constant):
            return jax.random.exponential(key, shape=self.shape)/self.rate.value

class Poisson(Node):
    pass


class Categorical(Node):
    pass


class Bernoulli(Node):
    pass


class Dirichlet(Node):
    pass


class NormalInverseGamma(Node):
    pass


class Beta(Node):
    pass


def is_constant(obj):
    return isinstance(obj, Array)
