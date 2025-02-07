import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Array


@jax.tree_util.register_dataclass
@dataclass
class Parameter:
    data: Array


# @jax.tree_util.register_dataclass
# @dataclass
# class Hyperparameters:
#     data: tuple
#     K: int


#################
# Distributions #
#################


class Distribution(ABC):
    pass


class NormalInverseGamma(Distribution):
    pass


class Normal(Distribution):
    pass


class Dirichlet(Distribution):
    pass


class Categorical(Distribution):
    pass


class Beta(Distribution):
    pass


class Bernoulli(Distribution):
    pass


class Gamma(Distribution):
    pass


class Poisson(Distribution):
    pass


class InverseGamma(Distribution):
    pass
