import jax.numpy as jnp
from jax.scipy.special import gammaln
import genjaxmix.analytical.logpdf as logpdf


def test_logpdf_bernoulli():
    p = jnp.array([[0.5]])
    x = jnp.array([[1]])

    output = logpdf._logpdf_bernoulli(x, (p,))
    expected = jnp.log(p)

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_beta():
    alpha, beta = jnp.array([[1.0]]), jnp.array([[2.0]])
    x = jnp.array([[0.5]])

    output = logpdf._logpdf_beta(x, (alpha, beta))
    expected = (
        gammaln(alpha + beta)
        - gammaln(alpha)
        - gammaln(beta)
        + (alpha - 1) * jnp.log(x)
        + (beta - 1) * jnp.log(1 - x)
    )

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_categorical():
    raise NotImplementedError()


def test_logpdf_dirichlet():
    raise NotImplementedError()


def test_logpdf_exponential():
    lamb = jnp.array([[2.0]])
    x = jnp.array([[0.0]])

    output = logpdf._logpdf_exponential(x, (lamb,))
    expected = jnp.log(1 / lamb)

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_gamma():
    shape, scale = jnp.array([[1.0]]), jnp.array([[2.0]])
    x = jnp.array([[1.0]])

    output = logpdf._logpdf_gamma(x, (shape, scale))
    expected = (
        -shape * jnp.log(scale) - gammaln(shape) + (shape - 1) * jnp.log(x) - x / scale
    )

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_inverse_gamma():
    raise NotImplementedError()


def test_logpdf_nig():
    a_0, b_0, mu_0, tau_sq = (
        jnp.array([[1.0]]),
        jnp.array([[2.0]]),
        jnp.array([[0.0]]),
        jnp.array([[2.0]]),
    )
    x = jnp.array([[0.0, 1.0]])
    raise NotImplementedError()


def test_logpdf_normal():
    mu, sigma = jnp.array([[0.0]]), jnp.array([[2.0]])
    x = jnp.array([[0.0]])

    output = logpdf._logpdf_normal(x, (mu, sigma))
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

    output = logpdf._logpdf_pareto(x, (shape, scale))
    expected = jnp.log(shape * scale**shape / x ** (shape + 1))

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_poison():
    lamb = jnp.array([[2.0]])
    x = jnp.array([[1]], dtype=jnp.int32)

    output = logpdf._logpdf_poisson(x, (lamb,))
    expected = x * jnp.log(lamb) - lamb - gammaln(x + 1)

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_uniform():
    a, b = jnp.array([[0.0]]), jnp.array([[2.0]])
    x = jnp.array([[1.0]])

    output = logpdf._logpdf_uniform(x, (a, b))
    expected = jnp.log(1 / (b - a))

    assert jnp.all(jnp.isclose(output, expected))


def test_logpdf_weibull():
    raise NotImplementedError()
