import genjaxmix.model.dsl as dsl
import genjaxmix.model.compile as pb
import jax.numpy as jnp


def test_basic():
    gamma = dsl.Gamma(jnp.ones((3, 2)), jnp.ones((3, 2)))
    normal = dsl.Normal(jnp.zeros((3, 2)), gamma)

    program = pb.Program(normal)
    expected_edges = {0: [1, 2], 1: [], 2: [3, 4], 3: [], 4: []}
    assert expected_edges == program.edges, (
        f"Expected {expected_edges}, but got {program.edges}"
    )

    expected_backedges = {0: [], 1: [0], 2: [0], 3: [2], 4: [2]}
    assert expected_backedges == program.backedges, (
        f"Expected {expected_backedges}, but got {program.backedges}"
    )

    expected_types = [
        dsl.Normal,
        dsl.Constant,
        dsl.Gamma,
        dsl.Constant,
        dsl.Constant,
    ]

    assert expected_types == program.types, (
        f"Expected {expected_types}, but got {program.types}"
    )
    expected_ordering = [4, 3, 2, 1, 0]

    assert expected_ordering == program.ordering, (
        f"Expected {expected_ordering}, but got {program.ordering}"
    )

def test_blanket_1():
    gamma = dsl.Gamma(jnp.ones((3, 2)), jnp.ones((3, 2)))
    normal = dsl.Normal(jnp.zeros((3, 2)), gamma)
    program = pb.Program(normal)
    blanket_0 = program.markov_blanket(0)
    blanket_1 = program.markov_blanket(1)
    blanket_2 = program.markov_blanket(2)
    blanket_3 = program.markov_blanket(3)
    blanket_4 = program.markov_blanket(4)

    expected_blanket_0 = {
        "parents": [1,2],
        "children": [],
        "cousins": []
    }

    expected_blanket_1 = {
        "parents": [],
        "children": [0],
        "cousins": [2]
    }

    expected_blanket_2 = {
        "parents": [3, 4],
        "children": [0],
        "cousins": [1]
    }

    expected_blanket_3 = {
        "parents": [],
        "children": [2],
        "cousins": [4]
    }

    expected_blanket_4 = {
        "parents": [],
        "children": [2],
        "cousins": [3]
    }

    assert blanket_0 == expected_blanket_0, (
        f"Expected {expected_blanket_1}, but got {blanket_1}"
    )

    assert blanket_1 == expected_blanket_1, (
        f"Expected {expected_blanket_1}, but got {blanket_1}"
    )

    assert blanket_2 == expected_blanket_2, (
        f"Expected {expected_blanket_2}, but got {blanket_2}"
    )

    assert blanket_3 == expected_blanket_3, (
        f"Expected {expected_blanket_3}, but got {blanket_3}"
    )

    assert blanket_4 == expected_blanket_4, (
        f"Expected {expected_blanket_4}, but got {blanket_4}"
    )

