import jax
from plum import dispatch
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key
from genjaxmix.core import Parameter, Distribution
from genjaxmix.analytical.marginal_likelihood import segmented_marginal_likelihood
from genjaxmix.analytical import sufficient_statistics, marginal_likelihood
from genjaxmix.analytical.posterior import segmented_posterior_sampler


def gibbs_pi(
    key: Key[Array, ""],  # noqa: F722
    hyperparameters: Parameter,  # noqa: F722
    latents: Parameter,  # noqa: F722
):
    alpha = hyperparameters.alpha
    pi = latents.pi
    assignments = latents.assignments

    K_max = pi.pi.shape[0]
    K = pi.K
    counts = jnp.bincount(assignments, length=K_max)
    alpha_new = jnp.where(jnp.arange(K_max) < K, counts, alpha)
    alpha_new = jnp.where(jnp.arange(K_max) < K + 1, alpha_new, 0.0)
    pi_new = jax.random.dirichlet(key, alpha_new)
    return pi_new


def gibbs_z_proposal(prior: Distribution, likelihood: Distribution):
    sml = segmented_marginal_likelihood(prior, likelihood)

    def _gibbs_z(key, hyperparameters, latents, observations):
        pi = hyperparameters.pi
        assignments = latents.assignments
        weights = sml(hyperparameters, observations, assignments)
        weights = weights + jnp.log(pi)
        labels = jax.random.categorical(key, weights)
        return labels

    return _gibbs_z


def gibbs_parameters_proposal(prior: Distribution, likelihood):
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
