import jax
import jax.numpy as jnp
import genjaxmix.analytical.sufficient_statistics as ss


def test_sss_nig_normal():
    a_0 = jnp.ones(2)
    b_0 = jnp.ones(2)
    mu_0 = jnp.zeros(2)
    v_0 = jnp.ones(2)

    x = jnp.array([0.0, 1.0, 1.0])
    assignments = jnp.array([0, 0, 1], dtype=jax.numpy.int32)
    output = ss._sss_nig_normal((a_0, b_0, mu_0, v_0), x, assignments)
    expected = (
        jnp.array([1 / 3, 1 / 2]),
        jnp.array([1 / 3, 1 / 2]),
        jnp.array([2, 3 / 2]),
        jnp.array([4 / 3, 5 / 4]),
    )

    assert len(output) == 4
    assert jnp.all(jnp.isclose(output[0], expected[0]))
    assert jnp.all(jnp.isclose(output[1], expected[1]))
    assert jnp.all(jnp.isclose(output[2], expected[2]))
    assert jnp.all(jnp.isclose(output[3], expected[3]))


def test_sss_dirichlet_categorical():
    alpha = (jnp.ones((2, 2)),)
    x = jnp.array([0, 1, 1, 0, 1], dtype=jax.numpy.int32)
    assignments = jnp.array([0, 0, 1, 1, 1], dtype=jax.numpy.int32)
    output = ss._sss_dirichlet_categorical(alpha, x, assignments)
    expected = jnp.array([[1, 1], [1, 2]])

    assert len(output) == 1
    assert jnp.all(jnp.isclose(output[0], expected))


def test_sss_beta_bernoulli():
    parameters = (jnp.ones(2), jnp.ones(2))
    x = jnp.array([0, 1, 1, 0, 1], dtype=jax.numpy.int32)
    assignments = jnp.array([0, 0, 1, 1, 1], dtype=jax.numpy.int32)
    output = ss._sss_beta_bernoulli(parameters, x, assignments)
    expected = (jnp.array([2, 3]), jnp.array([2, 2]))

    assert len(output) == 2
    assert jnp.all(jnp.isclose(output[0], expected[0]))
    assert jnp.all(jnp.isclose(output[1], expected[1]))


def test_sss_gamma_poisson():
    parameters = (jnp.ones(2), jnp.ones(2))
    x = jnp.array([0, 1, 1, 0, 1], dtype=jax.numpy.int32)
    assignments = jnp.array([0, 0, 1, 1, 1], dtype=jax.numpy.int32)
    output = ss._sss_gamma_poisson(parameters, x, assignments)
    expected = (jnp.array([2, 3]), jnp.array([1 / 3, 1 / 4]))

    assert len(output) == 2
    assert jnp.all(jnp.isclose(output[0], expected[0]))
    assert jnp.all(jnp.isclose(output[1], expected[1]))


def test_sss_normal_normal():
    parameters = (jnp.zeros(2), jnp.ones(2), 2 * jnp.ones(2))
    x = jnp.array([0.0, 1.0, 1.0])
    assignments = jnp.array([0, 0, 1], dtype=jax.numpy.int32)
    output = ss._sss_normal_normal(parameters, x, assignments)
    expected = (jnp.array([1.0, 3 / 4]), jnp.array([1 / 2, 2 / 3]))

    assert len(output) == 2
    assert jnp.all(jnp.isclose(output[0], expected[0]))
    assert jnp.all(jnp.isclose(output[1], expected[1]))


def test_sss_gamma_normal():
    parameters = (jnp.ones(2), jnp.ones(2), jnp.zeros(2))
    x = jnp.array([0.0, 1.0, 1.0])
    assignments = jnp.array([0, 0, 1], dtype=jax.numpy.int32)
    output = ss._sss_gamma_normal(parameters, x, assignments)
    expected = (jnp.array([2, 3 / 2]), jnp.array([3 / 2, 3 / 2]))

    assert len(output) == 2
    assert jnp.all(jnp.isclose(output[0], expected[0]))
    assert jnp.all(jnp.isclose(output[1], expected[1]))


def test_sss_inverse_gamma_normal():
    parameters = (jnp.ones(2), jnp.ones(2), jnp.zeros(2))
    x = jnp.array([0.0, 1.0, 1.0])
    assignments = jnp.array([0, 0, 1], dtype=jax.numpy.int32)
    output = ss._sss_inverse_gamma_normal(parameters, x, assignments)
    expected = (jnp.array([2, 3 / 2]), jnp.array([3 / 2, 3 / 2]))

    assert len(output) == 2
    assert jnp.all(jnp.isclose(output[0], expected[0]))
    assert jnp.all(jnp.isclose(output[1], expected[1]))
