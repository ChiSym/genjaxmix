from plum import dispatch
import genjaxmix.model.dsl as core
import jax.numpy as jnp
import jax


@dispatch
def segmented_posterior_sampler(prior: core.Normal, likelihood: core.Normal):  # noqa: F811
    return _sps_normal_normal


@dispatch
def segmented_posterior_sampler(prior: core.Gamma, likelihood: core.Normal):  # noqa: F811
    return _sps_gamma_normal


@dispatch
def segmented_posterior_sampler(prior: core.Dirichlet, likelihood: core.Categorical):  # noqa: F811
    return _sps_dirichlet_categorical


@dispatch
def segmented_posterior_sampler(  # noqa: F811
    prior: core.NormalInverseGamma, likelihood: core.Normal
):
    return _sps_nig_normal


@dispatch
def segmented_posterior_sampler(prior: core.Beta, likelihood: core.Bernoulli):  # noqa: F811
    return _sps_beta_bernoulli


@dispatch
def segmented_posterior_sampler(prior: core.Gamma, likelihood: core.Poisson):  # noqa: F811
    pass


@dispatch
def segmented_posterior_sampler(prior: core.InverseGamma, likelihood: core.Normal):  # noqa: F811
    return _sps_inverse_gamma_normal


@dispatch
def segmented_posterior_sampler(prior: core.Gamma, likelihood: core.Poisson): # noqa: F811
    return _sps_gamma_poisson


#######
def _sps_normal_normal(
    key,
    hyperparameters,  # noqa: F722
    x,  # noqa: F722
    assignments,  # noqa: F722
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
    hyperparameters,  # noqa: F722
    x,  # noqa: F722
    assignments,  # noqa: F722
):
    alpha_0, beta_0, mu = hyperparameters
    counts = jnp.bincount(assignments, length=alpha_0.shape[0])
    sum_squared_diff = jax.ops.segment_sum(
        (x - mu[assignments]) ** 2, assignments, alpha_0.shape[0]
    )

    alpha_new = alpha_0 + 0.5 * counts[:, None]
    beta_new = beta_0 + 0.5 * sum_squared_diff
    return jax.random.gamma(key, alpha_new) * beta_new


def _sps_inverse_gamma_normal(
    key,
    hyperparameters,  # noqa: F722
    x,  # noqa: F722
    assignments,  # noqa: F722
):
    alpha_0, beta_0, mu = hyperparameters
    counts = jnp.bincount(assignments, length=alpha_0.shape[0])
    sum_squared_diff = jax.ops.segment_sum(
        (x - mu[assignments]) ** 2, assignments, alpha_0.shape[0]
    )

    alpha_new = alpha_0 + 0.5 * counts[:, None]
    beta_new = beta_0 + 0.5 * sum_squared_diff
    return 1 / (jax.random.gamma(key, alpha_new) * beta_new)


def _sps_nig_normal(
    key,
    hyperparameters,  # noqa: F722
    x,  # noqa: F722
    assignments,  # noqa: F722
):
    alpha_0, beta_0, mu_0, v_0 = hyperparameters
    K_max = alpha_0.shape[0]

    counts = jnp.bincount(assignments, length=K_max)
    x_sum = jax.ops.segment_sum(x, assignments, K_max)
    x_sum_sq = jax.ops.segment_sum(x**2, assignments, K_max)

    v_post = 1 / (1 / v_0 + counts)
    mu_post = 1 / (v_0) * mu_0 + x_sum

    alpha_post = alpha_0 + counts / 2
    beta_post = beta_0 + 0.5 * (x_sum_sq + mu_0**2 / v_0 - mu_post**2 / v_post)

    sigma_new = jax.random.gamma(key, alpha_post, 1 / beta_post)
    mu_new = jax.random.normal(key, mu_0.shape) * jnp.sqrt(sigma_new) + mu_post
    return mu_new, sigma_new


def _sps_dirichlet_categorical(
    key,
    hyperparameters,  # noqa: F722
    x,  # noqa: F722
    assignments,  # noqa: F722
):
    # See https://github.com/chi-collective/minijaxmix/blob/main/minijaxmix/query.py#L29
    alpha, segment_ids, K_arr, F_arr = hyperparameters
    K = K_arr.shape[0]
    F = F_arr.shape[0]

    counts = jax.ops.segment_sum(x.astype(jnp.int32), assignments, K)
    alpha_new = alpha + counts

    y = jax.random.loggamma(key, alpha_new)
    c = jnp.max(y, axis=1)
    y_exp = jnp.exp(y - c[:, None])
    y_sum = jax.ops.segment_sum(
        y_exp.T, segment_ids, num_segments=F, indices_are_sorted=True
    )
    y_sum = jnp.log(y_sum)
    y_sum = y_sum.T
    y_sum += c[:, None]
    y_sum_full = y_sum.take(segment_ids, axis=1)
    return y - y_sum_full


def _sps_beta_bernoulli(
    key,
    hyperparameters,  # noqa: F722
    x,  # noqa: F722
    assignments,  # noqa: F722
):
    alpha_0, beta_0 = hyperparameters

    counts = jnp.bincount(assignments, length=alpha_0.shape[0])
    x_sum = jax.ops.segment_sum(x, assignments, alpha_0.shape[0])
    alpha_post = alpha_0 + x_sum
    beta_post = beta_0 + counts - x_sum

    return jax.random.beta(key, alpha_post, beta_post)


def _sps_gamma_poisson(
    key,
    hyperparameters,  # noqa: F722
    x,  # noqa: F722
    assignments,  # noqa: F722
):
    shape, scale = hyperparameters
    K = shape.shape[0]
    x_sum = jax.ops.segment_sum(x, assignments, K)
    counts = jnp.bincount(assignments, length=K)
    shape_new = shape + x_sum
    scale_new = scale / (counts[:, None] * scale + 1)
    mu_new = jax.random.gamma(key, shape_new)
    mu_new *= scale_new
    return mu_new
