import jax
import jax.numpy as jnp
import genjaxmix.model.dsl as dsl
import genjaxmix.model as mm


def test_normal_normal_known_mu():
    # TODO: Swap this out with a Markov blanket and call build_parameter_proposal directly
    class Test(mm.Model):
        def __init__(self):
            super().__init__()
            self.mu = dsl.Normal(jnp.zeros((1, 1)), jnp.ones((1, 1)))
            self.x = dsl.Normal(self.mu, jnp.ones((1, 1)))

        def observations(self):
            return []

    model = Test()
    model.compile()
    proposal = model.parameter_proposals[0]

    outputs = []
    key = jax.random.key(0)
    for i in range(10):
        key, subkey = jax.random.split(key)
        assignments = jnp.array([])
        environment = {
            0: jnp.ones((1, 1)),
            1: jnp.ones((1, 1)),
            2: jnp.zeros((1, 1)),
            3: jnp.ones((1, 1)),
            4: jnp.ones((1, 1)),
        }
        output = proposal(subkey, environment, assignments)[0]
        outputs.append(output)

    output = jnp.concat(outputs)

    empirical_mean = jnp.mean(output)
    empirical_std = jnp.std(output)

    expected_mean = jnp.array(0.27979577)
    expected_std = jnp.array(0.74080914)

    assert jnp.isclose(expected_mean, empirical_mean)
    assert jnp.isclose(expected_std, empirical_std)
