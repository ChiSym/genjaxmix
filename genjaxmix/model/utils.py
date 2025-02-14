
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

