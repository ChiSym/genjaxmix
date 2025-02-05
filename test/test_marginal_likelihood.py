import jax.numpy as jnp
import genjaxmix.analytical.marginal_likelihood as ml


def test_sml_nig_normal():
    # hyperparameters = Hyperparameters(
    #     jnp.zeros(2), jnp.zeros(2), jnp.ones(1.0), jnp.ones(1.0)
    # )
    # parameters = jnp.zeros(2)

    # x = jnp.array([1.0])
    # ml._sml_normal_normal(hyperparameters, parameters, x, assignments)
    pass


def test_sml_normal_normal():
    hyperparameters = (jnp.array([0.0, 0.0, 1.0]), jnp.ones(3))
    parameters = (jnp.ones(3),)

    x = jnp.array([0.0, 1.0, -1.0, 1.0])
    assignments = jnp.array([0, 0, 1, 2])

    output = ml._sml_normal_normal(hyperparameters, parameters, x, assignments)

    expected = jnp.array(
        [
            jnp.log(1 / (2 * jnp.pi * jnp.sqrt(3))) - 1 / 3,
            jnp.log(1 / jnp.sqrt(2 * jnp.pi * 2)) - 1 / 4,
            jnp.log(1 / jnp.sqrt(2 * jnp.pi * 2)),
        ]
    )

    assert len(output) == 3
    assert jnp.allclose(output, expected)


def test_sml_dirichlet_categorical():
    pass


def test_sml_gamma_normal():
    pass
