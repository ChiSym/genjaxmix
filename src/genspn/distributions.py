import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import Array, Float, Int
from plum import dispatch
from typing import Optional

class NormalInverseGamma(eqx.Module):
    m: Float[Array, "*batch n_dim"]
    l: Float[Array, "*batch n_dim"]
    a: Float[Array, "*batch n_dim"]
    b: Float[Array, "*batch n_dim"]

class Dirichlet(eqx.Module):
    alpha: Float[Array, "*batch n_dim k"]

class Normal(eqx.Module):
    mu: Float[Array, "*batch n_dim"]
    std: Float[Array, "*batch n_dim"]

class Categorical(eqx.Module):
    # assumed normalized, padded
    logprobs: Float[Array, "*batch n_dim k"]

class Mixed(eqx.Module):
    normal: Normal
    categorical: Categorical

class MixedConjugate(eqx.Module):
    nig: NormalInverseGamma
    dirichlet: Dirichlet

@dispatch
def sample(key: Array, dist: Dirichlet) -> Categorical:
    probs = jax.random.dirichlet(key, dist.alpha)
    return Categorical(jnp.log(probs))

@dispatch 
def sample(key: Array, dist: NormalInverseGamma) -> Normal:
    """ See Kevin Murphy's Conjugate Bayesian analysis of the Gaussian distribution:
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf """
    keys = jax.random.split(key)

    log_lambda = jax.random.loggamma(key, dist.a) - jnp.log(dist.b)
    log_sigma = -jnp.log(dist.l) - log_lambda
    # log_sigma = (jnp.log(dist.b) - jax.random.loggamma(key, dist.a)) / 2
    std = jnp.exp(log_sigma/ 2)
    mu = dist.m + jax.random.normal(keys[1], shape=dist.m.shape) * std

    return Normal(mu=mu, std=jnp.exp(-log_lambda/2))

@dispatch
def sample(key: Array, dist: MixedConjugate) -> Mixed:
    keys = jax.random.split(key)
    normal = sample(keys[0], dist.nig)
    categorical = sample(keys[1], dist.dirichlet)

    return Mixed(normal=normal, categorical=categorical)

@dispatch
def posterior(dist: MixedConjugate, x: tuple[Float[Array, "batch n_normal_dim"], Int[Array, "batch n_categorical_dim"]]) -> MixedConjugate:
    nig = posterior(dist.nig, x[0])
    dirichlet = posterior(dist.dirichlet, x[1])

    return MixedConjugate(nig=nig, dirichlet=dirichlet)

@dispatch
def posterior(dist: MixedConjugate, x: tuple[Float[Array, "batch n_normal_dim"], Int[Array, "batch n_categorical_dim"]], c: Int[Array, "batch"], max_clusters:Optional[int]=None) -> MixedConjugate:
    nig = posterior(dist.nig, x[0], c, max_clusters)
    dirichlet = posterior(dist.dirichlet, x[1], c, max_clusters)

    return MixedConjugate(nig=nig, dirichlet=dirichlet)

@dispatch
def posterior(dist: NormalInverseGamma, x: Float[Array, "batch n_dim"], c: Int[Array, "batch"], max_clusters:Optional[int]=None) -> NormalInverseGamma:
    N = jax.ops.segment_sum(jnp.ones(x.shape[0], dtype=jnp.int32), c, num_segments=max_clusters)
    sum_x = jax.ops.segment_sum(x, c, num_segments=max_clusters)
    sum_x_sq = jax.ops.segment_sum(x ** 2, c, num_segments=max_clusters)

    return jax.vmap(posterior, in_axes=(None, 0, 0, 0))(dist, N, sum_x, sum_x_sq)

@dispatch
def posterior(dist: NormalInverseGamma, x: Float[Array, "batch n_dim"]) -> NormalInverseGamma:
    N = x.shape[0]
    sum_x = jnp.sum(x, axis=0)
    sum_x_sq = jnp.sum(x ** 2, axis=0)

    return posterior(dist, N, sum_x, sum_x_sq)

@dispatch
def posterior(dist: NormalInverseGamma, N: Int[Array, ""], sum_x: Float[Array, "n_dim"], sum_x_sq: Float[Array, "n_dim"]) -> NormalInverseGamma:
    l = dist.l + N
    m = (dist.l * dist.m + sum_x) / l
    a = dist.a + N / 2
    b = dist.b + 0.5 * (sum_x_sq + dist.l * dist.m ** 2 - l * m ** 2)

    return NormalInverseGamma(m=m, l=l, a=a, b=b)

@dispatch
def posterior(dist: Dirichlet, x: Int[Array, "batch n_dim"], c: Int[Array, "batch"], max_clusters:Optional[int]=None) -> Dirichlet:
    one_hot_x = jax.nn.one_hot(x, num_classes=dist.alpha.shape[-1], dtype=jnp.int32)
    counts = jax.ops.segment_sum(one_hot_x, c, num_segments=max_clusters)
    return jax.vmap(posterior, in_axes=(None, 0))(dist, counts)

@dispatch
def posterior(dist: Dirichlet, counts: Int[Array, "n_dim k"]) -> Dirichlet:
    return Dirichlet(alpha=dist.alpha + counts)

@dispatch
def logpdf(dist: Normal, x: Float[Array, "n_dim"]) -> Float[Array, ""]:
    return jnp.sum(-0.5 * jnp.log(2 * jnp.pi) - jnp.log(dist.std) - 0.5 * ((x - dist.mu) / dist.std) ** 2)

@dispatch
def logpdf(dist: Categorical, x: Int[Array, "n_dim"]) -> Float[Array, ""]:
    return jnp.sum(dist.logprobs[jnp.arange(x.shape[-1]), x])

@dispatch
def logpdf(dist: Mixed, x: tuple[Float[Array, "n_normal_dim"], Int[Array, "n_categorical_dim"]]) -> Float[Array, ""]:
    return logpdf(dist.normal, x[0]) + logpdf(dist.categorical, x[1])