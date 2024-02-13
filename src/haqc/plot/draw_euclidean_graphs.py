from src.haqc.classical.greedy_tsp import greedy_tsp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_euclidean_graph(
    G,
    draw_edge=False,
    node_size=300,
    node_color=False,
    with_label=False,
    width=1.0,
    draw_sol=False,
):
    """Function to draw euclidean graph

    Args:
        G (networkx.classes.graph.Graph): NetworkX Graph
        draw_edge (bool, optional): Whether or not to draw edges. Defaults to False.
        node_size (int, optional): Size of nodes. Defaults to 300.
        node_color (bool, optional): Whether or not to color node 0. Defaults to False.
        with_label (bool, optional): Set to True to draw labels on the nodes.
        width (float, optional): Line width of edges. Defaults to 1.0.
    """

    # Draw network
    plt.figure()

    # Get positions
    pos = nx.get_node_attributes(G, "pos")

    # If you want to add edgelabels
    for edge in G.edges():
        # Create a new attribute in the edge
        G[edge[0]][edge[1]]["cost_label"] = np.round(G[edge[0]][edge[1]]["cost"], 2)

    cost_labels = nx.get_edge_attributes(G, "cost_label")

    if node_color:
        color_map = []
        for node in G:
            if node == 0:
                color_map.append("yellow")
            elif "tag" in G.nodes()[node]:
                if G.nodes()[node]["tag"] == "outlier":
                    color_map.append("green")
                else:
                    color_map.append("skyblue")
            else:
                color_map.append("skyblue")
    else:
        color_map = "skyblue"

    # Draw the edge labels on the graph
    if draw_edge and draw_sol:
        path = greedy_tsp(G)
        for e in G.edges():
            G[e[0]][e[1]]["color"] = "grey"
            G[e[0]][e[1]]["width"] = width
        # Set color of edges of the shortest path to green
        for i in range(len(path) - 1):
            G[int(path[i])][int(path[i + 1])]["color"] = "red"
            G[int(path[i])][int(path[i + 1])]["width"] = 2 * width
        # Store in a list to use for drawing
        edge_color_list = [G[e[0]][e[1]]["color"] for e in G.edges()]
        edge_width_list = [G[e[0]][e[1]]["width"] for e in G.edges()]

        # Draw into the netowrkx
        nx.draw_networkx(
            G,
            pos,
            node_color=color_map,
            node_size=node_size,
            with_labels=with_label,
            edge_color=edge_color_list,
            width=edge_width_list,
        )

    elif draw_edge == True:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=cost_labels, font_size=8)

        # Draw the graph
        nx.draw_networkx(
            G,
            pos,
            node_color=color_map,
            node_size=node_size,
            with_labels=with_label,
            width=width,
        )
    else:
        # Draw the graph
        nx.draw_networkx(
            G,
            pos,
            node_color=color_map,
            node_size=node_size,
            with_labels=with_label,
            width=width,
        )
