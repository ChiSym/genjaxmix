import genjaxmix.model as mm
import genjaxmix.model.dsl as dsl
from graphviz import Digraph


def visualize(model: mm.Model):
    dot = Digraph()

    def pretty_print(i):
        node = model.nodes[i]
        out = ""
        if isinstance(node, dsl.Constant):
            out = str(node.value)
        elif isinstance(node, dsl.Normal):
            out = "Normal"
        elif isinstance(node, dsl.Exponential):
            out = "Exponential"
        else:
            out = str(node)
        return f"{i}: {out}"

    for i in range(len(model.nodes)):
        dot.node(str(i), pretty_print(i))

    for i, adj in model.backedges.items():
        for j in adj:
            dot.edge(str(i), str(j))

    return dot
