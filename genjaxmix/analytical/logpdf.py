from plum import dispatch
import genjaxmix.core as core
from jax.scipy.stats import norm
from jax.scipy.stats import poisson
import jax.numpy as jnp
import jax


@dispatch
def logpdf(prior: core.Normal, likelihood: core.Normal):  # noqa: F811
    return _logpdf_normal_normal


@dispatch
def logpdf(prior: core.Gamma, likelihood: core.Normal):  # noqa: F811
    return _logpdf_gamma_normal


@dispatch
def logpdf(prior: core.NormalInverseGamma, likelihood: core.Normal):
    return _logpdf_nig_normal


@dispatch
def logpdf(prior: core.InverseGamma, likelihood: core.Normal):
    return _logpdf_inverse_gamma_normal


@dispatch
def logpdf(prior: core.Gamma, likelihood: core.Poisson):
    return _logpdf_gamma_poisson


@dispatch
def logpdf(prior: core.Dirichlet, likelihood: core.Categorical):
    return _logpdf_dirichlet_categorical


@dispatch
def logpdf(prior: core.Beta, likelihood: core.Bernoulli):
    return _logpdf_beta_bernoulli


def _logpdf_normal_normal(hyperparameters, parameters, x, K):
    _, _, sigma_sq = hyperparameters
    (mu,) = parameters
    K_max = mu.shape[0]
    log_p = jax.vmap(norm.logpdf, in_axes=(None, 0, 0))(x, mu, jnp.sqrt(sigma_sq))
    log_p = jnp.sum(log_p, axis=1)
    log_p = jnp.where(jnp.arange(K_max) < K, log_p, -jnp.inf)
    return log_p


def _logpdf_gamma_normal(hyperparameters, parameters, x, K):
    _, _, mu = hyperparameters
    (sigma_sq,) = parameters
    K_max = mu.shape[0]
    log_p = jax.vmap(norm.logpdf, in_axes=(None, 0, 0))(x, mu, jnp.sqrt(sigma_sq))
    log_p = jnp.sum(log_p, axis=1)
    log_p = jnp.where(jnp.arange(K_max) < K, log_p, -jnp.inf)
    return log_p


def _logpdf_nig_normal(hyperparameters, parameters, x, K):
    mu, sig_sq = parameters
    K_max = mu.shape[0]
    log_p = jax.vmap(norm.logpdf, in_axes=(None, 0, 0))(x, mu, jnp.sqrt(sig_sq))
    log_p = jnp.sum(log_p, axis=1)
    log_p = jnp.where(jnp.arange(K_max) < K, log_p, -jnp.inf)
    return log_p


def _logpdf_inverse_gamma_normal(hyperparameters, parameters, x, K):
    raise NotImplementedError()


def _logpdf_gamma_poisson(hyperparameters, parameters, x, K):
    (mu,) = parameters
    K_max = mu.shape[0]
    log_p = jax.vmap(poisson.logpmf, in_axes=(None, 0))(x, mu)
    log_p = jnp.sum(log_p, axis=1)
    log_p = jnp.where(jnp.arange(K_max) < K, log_p, -jnp.inf)
    return log_p


def _logpdf_beta_bernoulli(hyperparameters, parameters, x, K):
    raise NotImplementedError()


def _logpdf_dirichlet_categorical(hyperparameters, parameters, x, K):
    _, _, K_arr, _ = hyperparameters
    (log_p,) = parameters
    K_max = K_arr.shape[0]
    log_pdf = jnp.sum(log_p * x[:, None], axis=1)
    log_pdf = jnp.where(jnp.arange(K_max) < K, log_pdf, -jnp.inf)
    return log_pdf
