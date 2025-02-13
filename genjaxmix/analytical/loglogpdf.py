from plum import dispatch

# import genjaxmix.core as core
import genjaxmix.model.dsl as core
from jax.scipy.stats import norm
from jax.scipy.stats import poisson
import jax.scipy.stats as stats
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


@dispatch
def logpdf(likelihood: core.Exponential):
    return _logpdf_exponential


def _logpdf_normal(x, parameters):
    mu, sigma = parameters
    log_p = jnp.sum(norm.logpdf(x, mu, sigma), axis=1)
    return log_p


def _logpdf_gamma(x, parameters):
    shape, scale = parameters
    log_p = jnp.sum(stats.gamma.logpdf(x, shape, 0, scale), axis=1)
    return log_p


def _logpdf_poisson(x, parameters):
    (mu,) = parameters
    log_p = jnp.sum(poisson.logpmf(x, mu), axis=1)
    return log_p


def _logpdf_bernoulli(x, parameters):
    (p,) = parameters
    log_p = jnp.sum(stats.bernoulli.logpmf(x, p), axis=1)
    return log_p


def _logpdf_categorical(x, parameters):
    (log_p,) = parameters
    log_pdf = jnp.sum(log_p * x[:, None], axis=1)
    return log_pdf


def _logpdf_exponential(x, parameters):
    (rate,) = parameters
    log_p = jnp.sum(stats.expon.logpdf(x, 0.0, rate), axis=1)
    return log_p
