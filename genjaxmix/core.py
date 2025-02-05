import jax
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float, Array, Int


@jax.tree_util.register_dataclass
@dataclass
class Latents:
    pi: Float[Array, ""]  # noqa: F722
    latent: Float[Array, ""]  # noqa: F722
    assignments: Int[Array, ""]  # noqa: F722
    K: int


@jax.tree_util.register_dataclass
@dataclass
class Hyperparameters:
    alpha: Float
    hyperparameters: tuple
    K: int


#################
# Distributions #
#################


class Distribution(ABC):
    @abstractmethod
    def sampler(self):
        return self._sampler


class NormalInverseGamma(Distribution):
    def _sampler(self, key, parameters):
        mu_0, v_0, a_0, b_0 = parameters


class Normal(Distribution):
    def _sampler(self, key, parameters, shape):
        mu, sigma = parameters
        return jax.random.normal(key, mu, sigma, shape=shape)


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
