import genjaxmix.model.dsl as dsl
import jax.numpy as jnp
import pytest


def test_is_constant():
    assert dsl.is_constant(1)
    assert dsl.is_constant(2.0)
    assert dsl.is_constant(jnp.array([1.0, 1.0]))
    assert not dsl.is_constant("a")
    assert not dsl.is_constant(dsl.Bernoulli(jnp.array([[1.0, 1.0]])))


def test_bernoulli():
    p = jnp.ones((3, 2))
    dist = dsl.Bernoulli(p)

    assert len(dist.children()) == 1
    assert jnp.array_equal(dist.children()[0].value, p)


def test_beta():
    alpha = jnp.ones((3, 2))
    beta = 2 * jnp.ones((3, 2))
    dist = dsl.Beta(alpha, beta)

    assert len(dist.children()) == 2
    assert jnp.array_equal(dist.children()[0].value, alpha)
    assert jnp.array_equal(dist.children()[1].value, beta)


def test_constant():
    dist = dsl.Constant(jnp.ones((3, 2)))
    assert dist.children() == []


def test_exponential():
    rate = jnp.ones((3, 2))
    dist = dsl.Exponential(rate)

    assert len(dist.children()) == 1
    assert jnp.array_equal(dist.children()[0].value, rate)


def test_gamma():
    alpha = jnp.ones((3, 2))
    beta = 2 * jnp.ones((3, 2))
    dist = dsl.Gamma(alpha, beta)

    assert len(dist.children()) == 2
    assert jnp.array_equal(dist.children()[0].value, alpha)
    assert jnp.array_equal(dist.children()[1].value, beta)


def test_inverse_gamma():
    alpha = jnp.ones((3, 2))
    beta = 2 * jnp.ones((3, 2))
    dist = dsl.InverseGamma(alpha, beta)

    assert len(dist.children()) == 2
    assert jnp.array_equal(dist.children()[0].value, alpha)
    assert jnp.array_equal(dist.children()[1].value, beta)


def test_normal():
    mu = jnp.zeros((3, 2))
    sigma = jnp.ones((3, 2))
    dist = dsl.Normal(mu, sigma)

    assert len(dist.children()) == 2
    assert jnp.array_equal(dist.children()[0].value, mu)
    assert jnp.array_equal(dist.children()[1].value, sigma)

    mu_and_sigma = jnp.ones((3, 3))
    # TODO: Consider fused args


def test_normal_inverse_gamma():
    alpha = jnp.ones((3, 2))
    beta = 2 * jnp.ones((3, 2))
    mu = jnp.zeros((3, 2))
    sigma = 3 * jnp.ones((3, 2))

    dist = dsl.NormalInverseGamma(alpha, beta, mu, sigma)

    assert len(dist.children()) == 4
    assert jnp.array_equal(dist.children()[0].value, alpha)
    assert jnp.array_equal(dist.children()[1].value, beta)
    assert jnp.array_equal(dist.children()[2].value, mu)
    assert jnp.array_equal(dist.children()[3].value, sigma)


def test_pareto():
    concentration = jnp.ones((3, 2))
    scale = 2 * jnp.ones((3, 2))
    dist = dsl.Pareto(concentration, scale)

    assert len(dist.children()) == 2
    assert jnp.array_equal(dist.children()[0].value, concentration)
    assert jnp.array_equal(dist.children()[1].value, scale)


def test_uniform():
    lower = jnp.ones((3, 2))
    upper = 2 * jnp.ones((3, 2))
    dist = dsl.Uniform(lower, upper)

    assert len(dist.children()) == 2
    assert jnp.array_equal(dist.children()[0].value, lower)
    assert jnp.array_equal(dist.children()[1].value, upper)


def test_weibull():
    shape = jnp.ones((3, 2))
    scale = 2 * jnp.ones((3, 2))
    dist = dsl.Weibull(shape, scale)

    assert len(dist.children()) == 2
    assert jnp.array_equal(dist.children()[0].value, shape)
    assert jnp.array_equal(dist.children()[1].value, scale)


def test_incorrect_dimensions():
    with pytest.raises(ValueError):
        dsl.Beta(jnp.zeros((3, 2)), jnp.ones((3, 3)))

    with pytest.raises(ValueError):
        dsl.Gamma(jnp.zeros((3, 2)), jnp.ones((3, 3)))

    with pytest.raises(ValueError):
        dsl.InverseGamma(jnp.zeros((3, 2)), jnp.ones((3, 3)))

    with pytest.raises(ValueError):
        dsl.Normal(jnp.zeros((3, 2)), jnp.ones((3, 3)))

    with pytest.raises(ValueError):
        dsl.NormalInverseGamma(
            jnp.zeros((3, 2)), jnp.ones((3, 3)), jnp.ones((3, 2)), jnp.ones((3, 2))
        )

    with pytest.raises(ValueError):
        dsl.Pareto(jnp.zeros((3, 2)), jnp.ones((3, 3)))

    with pytest.raises(ValueError):
        dsl.Uniform(jnp.zeros((3, 2)), jnp.ones((3, 3)))

    with pytest.raises(ValueError):
        dsl.Weibull(jnp.zeros((3, 2)), jnp.ones((3, 3)))


def test_incorrect_dimensions_2():
    gamma = dsl.Gamma(jnp.ones((3, 2)), jnp.ones((3, 2)))
    with pytest.raises(ValueError):
        dsl.Normal(jnp.ones((3, 3)), gamma)
