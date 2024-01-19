import networkx as nx
from networkx.algorithms.structuralholes import constraint
import numpy as np


def build_json_graph(data):
    """A function that builds a graph from JSON data provided by the webapp.

    Args:
        data (dict): A dictionary representation of the json response
    """
    G = nx.DiGraph()
    for i in range(len(data["nodes"])):
        G.add_node(data["nodes"][i]["id"])
        # Extract depot info from json
        if data["nodes"][i]["tag"] == "Depot":
            depot_info = data["nodes"][i]

    for i in range(len(data["edges"])):
        # Enrich edge information from json and the costs
        G.add_edge(
            data["edges"][i]["source"],
            data["edges"][i]["target"],
            cost=data["edges"][i]["cost"],
            id=data["edges"][i]["id"],
        )

    G = nx.DiGraph.to_undirected(G)

    return G, depot_info
