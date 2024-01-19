import pylab
import networkx as nx


def draw_graph(G, colors=None, pos=None):
    if colors is None:
        colors = ["r" for i in range(len(G))]
    if pos is None:
        pos = nx.spring_layout(G)

    default_axes = pylab.axes(frameon=True)
    nx.draw_networkx(
        G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
