import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float, Array, Int


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
    @abstractmethod
    def sample(self, key, *args):
        pass


class NormalInverseGamma(Distribution):
    def _sampler(self, key, *args):
        mu_0, v_0, a_0, b_0 = args
        shape = mu_0.shape
        return (mu_0, v_0)


class Normal(Distribution):
    def sample(self, key, *args):
        mu = args[0]
        sigma_sq = args[1]
        shape = mu.shape
        noise = jax.random.normal(key, shape=shape)
        return (noise * jnp.sqrt(sigma_sq) + mu,)

    def logpdf(self, *args):
        pass


class Dirichlet(Distribution):
    def sample(self, key, *args):
        alphas = args[0]


class Categorical(Distribution):
    def _sample(self, key):
        pass


class Beta(Distribution):
    pass


class Bernoulli(Distribution):
    pass


class Gamma(Distribution):
    def sample(self, key, *args):
        return


class Poisson(Distribution):
    pass


class InverseGamma(Distribution):
    pass
