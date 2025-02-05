from plum import dispatch
import jax.numpy as jnp
import jax
import genjaxmix.core as dist
from jaxtyping import Array, Float, Int


@dispatch
def segment_sufficient_statistic(prior: dist.Beta, likelihood: dist.Bernoulli):
    return _sss_beta_bernoulli


@dispatch
def segment_sufficient_statistic(prior: dist.Gamma, likelihood: dist.Poisson):  # noqa: F811
    return _sss_gamma_poisson


@dispatch
def segment_sufficient_statistic(prior: dist.Dirichlet, likelihood: dist.Categorical):  # noqa: F811
    return _sss_dirichlet_categorical


@dispatch
def segment_sufficient_statistic(  # noqa: F811
    prior: dist.NormalInverseGamma, likelihood: dist.Normal
):
    return _sss_nig_normal


@dispatch
def segment_sufficient_statistic(prior: dist.Normal, likelihood: dist.Normal):  # noqa: F811
    """
    Normal with known variance
    """
    return _sss_normal_normal


@dispatch
def segment_sufficient_statistic(prior: dist.InverseGamma, likelihood: dist.Normal):  # noqa: F811
    """
    Normal with known mean
    """
    return _sss_inverse_gamma_normal


@dispatch
def segment_sufficient_statistic(prior: dist.Gamma, likelihood: dist.Normal):  # noqa: F811
    return _sss_gamma_normal


############
def _sss_gamma_normal(
    parameters: Float[Array, "kc"],  # noqa: F821
    x: Float[Array, "nl"],  # noqa: F821
    assignments: Int[Array, "n"],  # noqa: F821
):
    alpha, beta, mu_likelihood = parameters
    counts = jnp.bincount(assignments, length=alpha.shape[0])
    sum_squared_diff = jax.ops.segment_sum(
        (x - mu_likelihood[assignments]) ** 2, assignments, alpha.shape[0]
    )

    alpha_new = alpha + 0.5 * counts
    beta_new = beta + 0.5 * sum_squared_diff

    return (alpha_new, beta_new)


def _sss_normal_normal(
    parameters: Float[Array, "kc"],  # noqa: F821
    x: Float[Array, "nl"],  # noqa: F821
    assignments: Int[Array, "n"],  # noqa: F821
):
    mu_0, sig_0, sig_sq_likelihood = parameters
    K = mu_0.shape[0]
    counts = jnp.bincount(assignments, length=K)
    sum_x = jax.ops.segment_sum(x, assignments, K)

    precision_new = 1 / sig_0 + counts / sig_sq_likelihood
    sig_new = 1 / precision_new

    mu_new = precision_new * (mu_0 / sig_0 + sum_x / sig_sq_likelihood)
    return (mu_new, sig_new)


def _sss_inverse_gamma_normal(
    parameters: Float[Array, "kc"],  # noqa: F821
    x: Float[Array, "nl"],  # noqa: F821
    assignments: Float[Array, "n"],  # noqa: F821
):
    alpha, beta, mu_likelihood = parameters
    counts = jnp.bincount(assignments, length=alpha.shape[0])
    sum_squared_diff = jax.ops.segment_sum(
        (x - mu_likelihood[assignments]) ** 2, assignments, alpha.shape[0]
    )

    alpha_new = alpha + 0.5 * counts
    beta_new = beta + 0.5 * sum_squared_diff

    return (alpha_new, beta_new)


def _sss_nig_normal(
    parameters: Float[Array, "kc"],  # noqa: F821
    x: Float[Array, "nl"],  # noqa: F821
    assignments: Float[Array, "n"],  # noqa: F821
):
    v_0, mu_0, a_0, b_0 = parameters
    K = v_0.shape[0]
    counts = jnp.bincount(assignments, length=K)
    sum_x = jax.ops.segment_sum(x, assignments, K)
    sum_x_sq = jax.ops.segment_sum(x**2, assignments, K)

    v_n_inv = 1 / v_0 + counts
    m = (1 / v_0 * mu_0 + sum_x) / v_n_inv
    a = a_0 + counts / 2
    b = b_0 + 0.5 * (sum_x_sq + 1 / v_0 * mu_0**2 - v_n_inv * m**2)
    return (m, 1 / v_n_inv, a, b)


def _sss_dirichlet_categorical(
    parameters: Float[Array, "kc"],  # noqa: F821
    x: Float[Array, "nl"],  # noqa: F821
    assignments: Float[Array, "n"],  # noqa: F821
):
    (alpha,) = parameters
    K = alpha.shape[0]
    L = alpha.shape[1]
    one_hot_c = jax.nn.one_hot(assignments, K)
    one_hot_y = jax.nn.one_hot(x, L)
    frequency_matrix = one_hot_c.T @ one_hot_y
    return (frequency_matrix,)


def _sss_beta_bernoulli(
    parameters: Float[Array, "kc"],  # noqa: F821
    x: Float[Array, "nl"],  # noqa: F821
    assignments,  # noqa: F821
):
    alpha, beta = parameters
    K = alpha.shape[0]
    x_sum = jax.ops.segment_sum(x, assignments, K)
    alpha_new = alpha + x_sum
    beta_new = beta + jnp.bincount(assignments, length=K) - x_sum
    return (alpha_new, beta_new)


def _sss_gamma_poisson(
    parameters: Float[Array, "kc"],  # noqa: F821
    x: Float[Array, "nl"],  # noqa: F821
    assignments: Int[Array, "n"],  # noqa: F821
):
    shape, scale = parameters
    K = shape.shape[0]
    x_sum = jax.ops.segment_sum(x, assignments, K)
    shape_new = shape + x_sum
    scale_new = scale / (jnp.bincount(assignments, length=K) * scale + 1)
    return (shape_new, scale_new)
