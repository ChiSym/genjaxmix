from plum import dispatch
import genjaxmix.core as core
from jax.scipy.stats import norm
import jax.numpy as jnp
import jax


@dispatch
def logpdf(prior: core.Normal, likelihood: core.Normal):  # noqa: F811
    return _logpdf_normal_normal


@dispatch
def logpdf(prior: core.Gamma, likelihood: core.Normal):  # noqa: F811
    return _log_gamma_normal


def _logpdf_normal_normal(hyperparameters, parameters, x, K):
    _, _, sigma_sq = hyperparameters
    (mu,) = parameters
    K_max = mu.shape[0]
    log_p = jax.vmap(norm.logpdf, in_axes=(None, 0, 0))(x, mu, jnp.sqrt(sigma_sq))
    log_p = jnp.sum(log_p, axis=1)
    log_p = jnp.where(jnp.arange(K_max) < K, log_p, -jnp.inf)
    return log_p


def _log_gamma_normal(hyperparameters, parameters, x, K):
    _, _, mu = hyperparameters
    (sigma_sq,) = parameters
    K_max = mu.shape[0]
    log_p = jax.vmap(norm.logpdf, in_axes=(None, 0, 0))(x, mu, jnp.sqrt(sigma_sq))
    log_p = jnp.sum(log_p, axis=1)
    log_p = jnp.where(jnp.arange(K_max) < K, log_p, -jnp.inf)
    return log_p
