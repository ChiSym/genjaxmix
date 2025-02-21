import jax.numpy as jnp


def topological_sort(graph):
    visited = set()
    ordering = []

    for node in graph:
        if node in visited:
            continue

        visited.add(node)
        sublist = [node]
        queue = [node]
        while len(queue) > 0:
            u = queue.pop(0)
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    sublist.append(v)
                    queue.append(v)

        ordering = sublist + ordering
    return ordering


def count_unique(array):
    # Sort the array first
    sorted_arr = jnp.sort(array)
    # Compare adjacent elements to find unique transitions
    # Add 1 to account for the first element
    return jnp.sum(jnp.concatenate([jnp.array([1]), sorted_arr[1:] != sorted_arr[:-1]]))
