from plum import dispatch
import jax
import genjaxmix.core as dist
import jax.numpy as jnp
from jaxtyping import Float, Array, Int
from jax.scipy.special import gammaln


@dispatch
def segmented_marginal_likelihood(prior: dist.Normal, likelihood: dist.Normal):  # noqa: F811
    return _sml_normal_normal


@dispatch
def segmented_marginal_likelihood(prior: dist.Gamma, likelihood: dist.Normal):  # noqa: F811
    return _sml_gamma_normal


@dispatch
def segmented_marginal_likelihood(prior: dist.Dirichlet, likelihood: dist.Categorical):  # noqa: F811
    return _sml_dirichlet_categorical


@dispatch
def segmented_marginal_likelihood(  # noqa: F811
    prior: dist.NormalInverseGamma, likelihood: dist.Normal
):
    return _sml_nig_normal


######


def _sml_normal_normal(
    hyperparameters: Float[Array, " kc"],  # noqa: F722
    parameters: Float[Array, " kd"],  # noqa: F722
    x: Float[Array, " nd"],  # noqa: F722
    assignments: Int[Array, " n"],  # noqa: F722
):
    """
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    """
    mu_0, tau_sq = hyperparameters
    (sig_sq,) = parameters
    K = mu_0.shape[0]
    counts = jnp.bincount(assignments, length=K)

    factor = (
        0.5 * jnp.log(sig_sq)
        - counts * 0.5 * jnp.log(2 * jnp.pi * tau_sq)
        - 0.5 * jnp.log(counts * tau_sq + sig_sq)
    )

    sum_x = jax.ops.segment_sum(x, assignments, K)
    sum_x_sq = jax.ops.segment_sum(x**2, assignments, K)

    A = -sum_x_sq / 2 / sig_sq - (mu_0**2) / 2 / tau_sq

    B = (
        0.5
        * 1
        / (counts * tau_sq + sig_sq)
        * (2 * sum_x * mu_0 + sig_sq * mu_0**2 / tau_sq + tau_sq * (sum_x**2) / sig_sq)
    )
    return factor + A + B


def _sml_gamma_normal(prior: dist.Gamma, likelihood: dist.Normal):
    raise NotImplementedError()


def _sml_dirichlet_categorical(
    hyperparameters: Float[Array, " kl"],  # noqa: F722
    observations: Int[Array, " nl"],  # noqa: F722
    assignments: Int[Array, " n"],  # noqa: F722
    cardinalities,
):
    # jax.nn.one_hot(assignments, num_classes=cardinalities[0])
    return jax.vmap(jax.nn.one_hot, in_axes=(None, 0))(assignments, cardinalities)


def _sml_nig_normal(hyperparameters, x, assignments):
    a_0, b_0, _, v_0 = hyperparameters
    count = jnp.bincount(assignments, length=a_0.shape[0])
    v_n = 1 / v_0 + count
    m_n = (1 / v_0 * a_0 + jax.ops.segment_sum(x, assignments, a_0.shape[0])) / v_n
    a_n = a_0 + count / 2
    b_n = b_0 + 0.5 * (
        jax.ops.segment_sum(x**2, assignments, a_0.shape[0])
        + 1 / v_0 * a_0**2
        - v_n * m_n**2
    )

    numerator = (
        0.5 * jnp.log(v_n)
        + a_0 * jnp.log(b_0)
        + gammaln(a_n)
        + (a_0 - a_n) * jnp.log(2)
    )
    denominator = (
        0.5 * jnp.log(v_0)
        + a_n * jnp.log(b_n)
        + gammaln(a_0)
        - count(0.5 * jnp.log(jnp.pi) - jnp.log(2))
    )

    return numerator - denominator
