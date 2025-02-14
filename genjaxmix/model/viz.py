from genjaxmix.model.compile import Program
import genjaxmix.model.dsl as dsl
from graphviz import Digraph


def visualize(program: Program):
    dot = Digraph()

    def pretty_print(node):
        if isinstance(node, dsl.Constant):
            return str(node.value)

        if isinstance(node, dsl.Normal):
            return "Normal"

        if isinstance(node, dsl.Exponential):
            return "Exponential"

        return str(node)

    for i, node in enumerate(program.nodes):
        dot.node(str(i), pretty_print(node))

    for i, adj in program.backedges.items():
        for j in adj:
            dot.edge(str(i), str(j))

    return dot
