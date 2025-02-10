from plum import dispatch

# import genjaxmix.core as core
import genjaxmix.model.dsl as core
from jax.scipy.stats import norm
from jax.scipy.stats import poisson
import jax.numpy as jnp
import jax


@dispatch
def logpdf(dist: core.Normal):  # noqa: F811
    return _logpdf_normal


@dispatch
def logpdf(dist: core.Gamma):
    return _logpdf_gamma


@dispatch
def logpdf(likelihood: core.Poisson):
    return _logpdf_poisson


@dispatch
def logpdf(likelihood: core.Categorical):
    return _logpdf_categorical


@dispatch
def logpdf(likelihood: core.Bernoulli):
    return _logpdf_bernoulli


def _logpdf_normal(parameters, x, K):
    mu, sigma_sq = parameters
    K_max = mu.shape[0]
    log_p = jax.vmap(norm.logpdf, in_axes=(None, 0, 0))(x, mu, jnp.sqrt(sigma_sq))
    log_p = jnp.sum(log_p, axis=1)
    log_p = jnp.where(jnp.arange(K_max) < K, log_p, -jnp.inf)
    return log_p


def _logpdf_gamma(parameters, x, K):
    raise NotImplementedError()


def _logpdf_poisson(parameters, x, K):
    (mu,) = parameters
    K_max = mu.shape[0]
    log_p = jax.vmap(poisson.logpmf, in_axes=(None, 0))(x, mu)
    log_p = jnp.sum(log_p, axis=1)
    log_p = jnp.where(jnp.arange(K_max) < K, log_p, -jnp.inf)
    return log_p


def _logpdf_bernoulli(parameters, x, K):
    raise NotImplementedError()


def _logpdf_categorical(parameters, x, K):
    log_p, K_arr = parameters
    K_max = K_arr.shape[0]
    log_pdf = jnp.sum(log_p * x[:, None], axis=1)
    log_pdf = jnp.where(jnp.arange(K_max) < K, log_pdf, -jnp.inf)
    return log_pdf
