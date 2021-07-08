import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_euclidean_graph(
    G,
    draw_edge=False, 
    node_size=300, 
    node_color='red',
    with_label=False,
    width=1.0
    ):
    """Function to draw euclidean graph

    Args:
        G (networkx.classes.graph.Graph): NetworkX Graph
        draw_edge (bool, optional): Whether or not to draw edges. Defaults to False.
        node_size (int, optional): Size of nodes. Defaults to 300.
        node_color (str, optional): [description]. Defaults to 'red'.
        with_label (bool, optional): Set to True to draw labels on the nodes.
        width (float, optional): Line width of edges. Defaults to 1.0.
    """

    # Draw network
    plt.figure()

    # Get positions
    pos = nx.get_node_attributes(G, 'pos')

    # If you want to add edgelabels
    for edge in G.edges():
        # Create a new attribute in the edge
        G[edge[0]][edge[1]]['cost_label'] = np.round(G[edge[0]][edge[1]]['cost'],2)

    cost_labels = nx.get_edge_attributes(G, 'cost_label')

    # Draw the edge labels on the graph
    if draw_edge:
        nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels=cost_labels, 
            font_size=8,
            font_color=node_color
        )

    # Draw the graph
    nx.draw_networkx(
        G, 
        pos,
        node_color=node_color,
        node_size=node_size,
        with_labels=with_label,
        width=width
    )