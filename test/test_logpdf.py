import jax.numpy as jnp
import pytest
from jax.scipy.special import gammaln
import genjaxmix.analytical.logpdf as logpdf


def test_logpdf_bernoulli():
    p = jnp.array([[0.5]])
    x = jnp.array([[1]])

    output = logpdf._logpdf_bernoulli(x, p)
    expected = jnp.log(p)

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_beta():
    alpha, beta = jnp.array([[1.0]]), jnp.array([[2.0]])
    x = jnp.array([[0.5]])

    output = logpdf._logpdf_beta(x, alpha, beta)
    expected = (
        gammaln(alpha + beta)
        - gammaln(alpha)
        - gammaln(beta)
        + (alpha - 1) * jnp.log(x)
        + (beta - 1) * jnp.log(1 - x)
    )

    assert jnp.all(jnp.isclose(output, expected))


@pytest.mark.skip(reason="no way of currently testing this")
def test_logpdf_categorical():
    raise NotImplementedError()


@pytest.mark.skip(reason="no way of currently testing this")
def test_logpdf_dirichlet():
    raise NotImplementedError()


def test_logpdf_exponential():
    lamb = jnp.array([[2.0]])
    x = jnp.array([[0.0]])

    output = logpdf._logpdf_exponential(x, lamb)
    expected = jnp.log(1 / lamb)

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_gamma():
    shape, scale = jnp.array([[1.0]]), jnp.array([[2.0]])
    x = jnp.array([[1.0]])

    output = logpdf._logpdf_gamma(x, shape, scale)
    expected = (
        -shape * jnp.log(scale) - gammaln(shape) + (shape - 1) * jnp.log(x) - x / scale
    )

    assert jnp.all(jnp.isclose(output, expected))


@pytest.mark.skip(reason="incorrect")
def test_logpdf_inverse_gamma():
    shape, scale = jnp.array([[2.0]]), jnp.array([[3.0]])
    x = jnp.array([[4.0]])

    output = logpdf._logpdf_inverse_gamma(x, shape, scale)
    expected = (
        -shape * jnp.log(1 / scale)
        - gammaln(shape)
        - (shape + 1) * jnp.log(x)
        - (1 / scale) / x
    )
    print("expected")
    print(expected)
    print("output")
    print(output)

    assert jnp.all(jnp.isclose(output, expected))


@pytest.mark.skip(reason="no way of currently testing this")
def test_logpdf_nig():
    alpha = jnp.array([[1.0]])
    beta = jnp.array([[2.0]])
    mu = jnp.array([[0.0]])
    tau = jnp.array([[1.0]])

    x = jnp.array([[0.0]])

    output = logpdf._logpdf_nig(x, x, alpha, beta, mu, tau)
    expected = (
        -0.5 * jnp.log(2 * jnp.pi * tau)
        - 0.5 * ((x - mu) / tau) ** 2
        + alpha * jnp.log(beta)
        + gammaln(alpha)
        - (alpha + 1) * jnp.log(x)
        - beta / x
    )

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_normal():
    mu, sigma = jnp.array([[0.0]]), jnp.array([[2.0]])
    x = jnp.array([[0.0]])

    output = logpdf._logpdf_normal(x, mu, sigma)
    expected = jnp.log(
        1
        / (2 * jnp.pi * sigma * sigma) ** 0.5
        * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)
    )

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_pareto():
    shape = jnp.array([[1.0]])
    scale = jnp.array([[2.0]])
    x = jnp.array([[2.0]])

    output = logpdf._logpdf_pareto(x, shape, scale)
    expected = jnp.log(shape * scale**shape / x ** (shape + 1))

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_poison():
    lamb = jnp.array([[2.0]])
    x = jnp.array([[1]], dtype=jnp.int32)

    output = logpdf._logpdf_poisson(x, lamb)
    expected = x * jnp.log(lamb) - lamb - gammaln(x + 1)

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_uniform():
    a, b = jnp.array([[0.0]]), jnp.array([[2.0]])
    x = jnp.array([[1.0]])

    output = logpdf._logpdf_uniform(x, a, b)
    expected = jnp.log(1 / (b - a))

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_weibull():
    shape = jnp.array([[2.0]])
    concentration = jnp.array([[3.0]])

    x = jnp.array([[4.0]])

    output = logpdf._logpdf_weibull(x, shape, concentration)
    expected = jnp.log(2.0 / 3.0 * (4.0 / 3.0)) - (4.0 / 3.0) ** 2

    assert jnp.all(jnp.isclose(output, expected))
