import genjaxmix.model.dsl as dsl
import genjaxmix.model.compile as pb
import jax.numpy as jnp

def test_basic():
    gamma = dsl.Gamma(jnp.ones((3,2)), jnp.ones((3,2)))
    normal = dsl.Normal(jnp.zeros((3,2)), gamma)

    program = pb.Program(normal)
