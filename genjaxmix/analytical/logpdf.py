from plum import dispatch

import genjaxmix.model.dsl as dsl
import jax.scipy.stats as stats
import jax.numpy as jnp


def get_logpdf(distribution):
    logpdfs = {
        dsl.Bernoulli: _logpdf_bernoulli,
        dsl.Beta: _logpdf_beta,
        dsl.Categorical: _logpdf_categorical,
        dsl.Exponential: _logpdf_exponential,
        dsl.Gamma: _logpdf_gamma,
        dsl.InverseGamma: _logpdf_inverse_gamma,
        dsl.Normal: _logpdf_normal,
        dsl.NormalInverseGamma: _logpdf_nig,
        dsl.Pareto: _logpdf_pareto,
        dsl.Poisson: _logpdf_poisson,
        dsl.Uniform: _logpdf_uniform,
        dsl.Weibull: _logpdf_weibull,
    }
    return logpdfs[distribution]


@dispatch
def logpdf(likelihood: dsl.Bernoulli):
    return _logpdf_bernoulli


@dispatch
def logpdf(likelihood: dsl.Beta):  # noqa: F811
    return _logpdf_beta


@dispatch
def logpdf(likelihood: dsl.Categorical):  # noqa: F811
    return _logpdf_categorical


@dispatch
def logpdf(likelihood: dsl.Exponential):  # noqa: F811
    return _logpdf_exponential


@dispatch
def logpdf(dist: dsl.Gamma):  # noqa: F811
    return _logpdf_gamma


@dispatch
def logpdf(dist: dsl.InverseGamma):  # noqa: F811
    return _logpdf_inverse_gamma


@dispatch
def logpdf(dist: dsl.Normal):  # noqa: F811
    return _logpdf_normal


@dispatch
def logpdf(dist: dsl.NormalInverseGamma):  # noqa: F811
    return _logpdf_nig


@dispatch
def logpdf(dist: dsl.Pareto):  # noqa: F811
    return _logpdf_pareto


@dispatch
def logpdf(likelihood: dsl.Poisson):  # noqa: F811
    return _logpdf_poisson


@dispatch
def logpdf(likelihood: dsl.Uniform):  # noqa: F811
    return _logpdf_uniform


@dispatch
def logpdf(likelihood: dsl.Weibull):  # noqa: F811
    return _logpdf_weibull


def _logpdf_beta(x, alpha, beta):
    log_p = jnp.sum(stats.beta.logpdf(x, alpha, beta), axis=1)
    return log_p


def _logpdf_bernoulli(x, p):
    log_p = jnp.sum(stats.bernoulli.logpmf(x, p), axis=1)
    return log_p


def _logpdf_categorical(x, log_p):
    log_pdf = jnp.sum(log_p * x[:, None], axis=1)
    return log_pdf


def _logpdf_exponential(x, rate):
    log_p = jnp.sum(stats.expon.logpdf(x, 0.0, rate))
    return log_p


def _logpdf_gamma(x, shape, scale):
    log_p = jnp.sum(stats.gamma.logpdf(x, shape, 0, scale), axis=1)
    return log_p


def _logpdf_inverse_gamma(x, shape, scale):
    log_p = jnp.sum(stats.gamma.logpdf(1 / x, shape, 0, scale) - 2 * jnp.log(x), axis=1)
    return log_p


def _logpdf_normal(x, mu, sigma):
    log_p = jnp.sum(stats.norm.logpdf(x, mu, sigma))
    return log_p


def _logpdf_nig(x1, x2, alpha, beta, mu, tau):
    log_p = stats.norm.logpdf(x1, mu, tau) + stats.gamma.logpdf(x2, alpha, 0, beta)
    log_p = jnp.sum(log_p, axis=1)
    return log_p


def _logpdf_pareto(x, shape, scale):
    log_p = jnp.sum(stats.pareto.logpdf(x, shape, 0, scale), axis=1)
    return log_p


def _logpdf_poisson(x, mu):
    log_p = jnp.sum(stats.poisson.logpmf(x, mu), axis=1)
    return log_p


def _logpdf_uniform(x, a, b):
    log_p = jnp.sum(stats.uniform.logpdf(x, a, b - a), axis=1)
    return log_p


def _logpdf_weibull(x, shape, scale):
    log_p = (
        (shape - 1) * jnp.log(x / scale) - (x / scale) ** shape + jnp.log(shape / scale)
    )
    log_p = jnp.sum(log_p, axis=1)
    return log_p
