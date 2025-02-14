import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, Float, Int
from genjaxmix.analytical import logpdf
from genjaxmix.analytical.posterior import segmented_posterior_sampler


def gibbs_pi(
    key: Key[Array, ""],  # noqa: F722
    alpha: Float[Array, ""],  # noqa: F722
    assignments: Float[Array, " n"],  # noqa: F722
    pi: Float[Array, " k"],  # noqa: F722
    K: Int[Array, ""],  # noqa: F722
):
    K_max = pi.shape[0]
    counts = jnp.bincount(assignments, length=K_max)
    alpha_new = jnp.where(jnp.arange(K_max) < K, counts, alpha)
    alpha_new = jnp.where(jnp.arange(K_max) < K + 1, alpha_new, 0.0)
    pi_new = jax.random.dirichlet(key, alpha_new)
    return pi_new


def gibbs_z_proposal(likelihood):
    """
    Returns a function that samples the assignments given the hyperparameters, parameters, observations, and pi.

    Args:
        prior: The prior distribution.
        likelihood: The likelihood distribution.
    """
    # logpdf_lambda = vlogpdf.logpdf(likelihood)
    logpdf_lambda = logpdf.logpdf(likelihood)

    def _gibbs_z(key, parameters, observations, pi, K):
        log_pdfs = jax.vmap(
            jax.vmap(logpdf_lambda, in_axes=(None, 0)), in_axes=(0, None)
        )(observations, parameters)
        log_pdfs = log_pdfs + jnp.log(pi)

        K_max = parameters[0].shape
        log_pdfs = jnp.where(jnp.arange(K_max) < K, log_pdfs, -jnp.inf)
        z = jax.random.categorical(key, log_pdfs)
        return z

    return _gibbs_z


def gibbs_parameters_proposal(prior, likelihood):
    """
    Returns a function that samples the parameters from the posterior given the hyperparameters, observations, and assignments.

    Args:
        prior: The prior distribution.
        likelihood: The likelihood distribution.

    Returns:
        A closure that samples the parameters given the hyperparameters, observations, and assignments.
    """
    sampler = segmented_posterior_sampler(prior, likelihood)

    def proposal(key, hyperparameters, observations, assignments):
        return sampler(key, hyperparameters, observations, assignments)

    return proposal
