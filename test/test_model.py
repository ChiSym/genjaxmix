import genjaxmix.model.dsl as dsl
import genjaxmix.model as mm
import jax.numpy as jnp


def test_basic():
    class Test(mm.Model):
        def __init__(self):
            super().__init__()
            self.gamma = dsl.Gamma(jnp.ones((3, 2)), jnp.ones((3, 2)))
            self.normal = dsl.Normal(jnp.zeros((3, 2)), self.gamma)

        def observations(self):
            return ["normal"]

    model = Test()
    model._discover_nodes()

    edges = model.edges
    backedges = model.backedges
    types = model.types
    ordering = model.ordering

    expected_edges = {0: [2, 3], 1: [4, 0], 2: [], 3: [], 4: []}
    assert expected_edges == edges, f"Expected {expected_edges}, but got {edges}"

    expected_backedges = {0: [1], 1: [], 2: [0], 3: [0], 4: [1]}
    assert expected_backedges == backedges, (
        f"Expected {expected_backedges}, but got {backedges}"
    )

    expected_types = [
        dsl.Gamma,
        dsl.Normal,
        dsl.Constant,
        dsl.Constant,
        dsl.Constant,
    ]

    assert expected_types == types, f"Expected {expected_types}, but got {types}"
    expected_ordering = [4, 3, 2, 0, 1]

    assert expected_ordering == ordering, (
        f"Expected {expected_ordering}, but got {ordering}"
    )
