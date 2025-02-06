from plum import dispatch
import genjaxmix.core as core
import jax.numpy as jnp
import jax


@dispatch
def segmented_posterior_sampler(prior: core.Normal, likelihood: core.Normal):
    return _sps_normal_normal


@dispatch
def segmented_posterior_sampler(prior: core.Gamma, likelihood: core.Normal):
    pass


@dispatch
def segmented_posterior_sampler(prior: core.Dirichlet, likelihood: core.Categorical):
    pass


@dispatch
def segmented_posterior_sampler(
    prior: core.NormalInverseGamma, likelihood: core.Normal
):
    pass


@dispatch
def segmented_posterior_sampler(prior: core.Beta, likelihood: core.Bernoulli):
    pass


@dispatch
def segmented_posterior_sampler(prior: core.Gamma, likelihood: core.Poisson):
    pass


@dispatch
def segmented_posterior_sampler(prior: core.InverseGamma, likelihood: core.Normal):
    pass


#######
def _sps_normal_normal(
    key,
    hyperparameters: core.Parameter,  # noqa: F722
    x: core.Parameter,  # noqa: F722
    assignments: core.Parameter,  # noqa: F722
):
    mu_0, sig_sq_0, sig_sq = hyperparameters

    counts = jnp.bincount(assignments, length=mu_0.shape[0])
    x_sum = jax.ops.segment_sum(x, assignments, mu_0.shape[0])
    sig_sq_post = 1 / (1 / sig_sq_0 + counts[:, None] / sig_sq)
    mu_post = sig_sq_post * (mu_0 / sig_sq_0 + x_sum / sig_sq)

    noise = jax.random.normal(key, shape=mu_0.shape)
    return noise * jnp.sqrt(sig_sq_post) + mu_post


def _sps_gamma_normal(
    key,
    hyperparameters: core.Parameter,  # noqa: F722
    x: core.Parameter,  # noqa: F722
    assignments: core.Parameter,  # noqa: F722
):
    alpha_0, beta_0, mu_0, sig_sq_0, sig_sq = hyperparameters

    counts = jnp.bincount(assignments, length=alpha_0.shape[0])
    x_sum = jax.ops.segment_sum(x, assignments, alpha_0.shape[0])
    sig_sq_post = 1 / (1 / sig_sq_0 + counts[:, None] / sig_sq)
    mu_post = sig_sq_post * (mu_0 / sig_sq_0 + x_sum / sig_sq)

    noise = jax.random.normal(key, shape=mu_0.shape)
    return noise * jnp.sqrt(sig_sq_post) + mu_post


def _sps_dirichlet_categorical(
    key,
    hyperparameters: core.Parameter,  # noqa: F722
    x: core.Parameter,  # noqa: F722
    assignments: core.Parameter,  # noqa: F722
):
    alpha_0, mu_0 = hyperparameters

    counts = jnp.bincount(assignments, length=alpha_0.shape[0])
    x_sum = jax.ops.segment_sum(x, assignments, alpha_0.shape[0])
    mu_post = (mu_0 + x_sum) / (alpha_0 + counts[:, None])

    noise = jax.random.dirichlet(key, alpha_0)
    return noise + mu_post
