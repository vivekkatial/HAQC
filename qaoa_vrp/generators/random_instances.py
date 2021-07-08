import networkx as nx
import numpy as np
import random
import uuid

from qaoa_vrp.utils import distance


def generate_random_instance(num_nodes, num_vehicles, instance_type):
    """
    This function creates random graphs with each node having dimension of 2

    Args:
        num_nodes (int): The number of nodes in the graph
    Returns:
        G (object): Networkx graph object
    """

    verify = False

    while verify == False:

        assert num_nodes >= 3, "num_nodes cannot be {}, must be >= 3".format(num_nodes)
        if instance_type not in [
            "watts_strogatz",
            "erdos_renyi",
            "complete",
            "newman_watts_strogatz",
            "euclidean_tsp"
        ]:
            raise ValueError("Incorrect Instance Type Requested")

        if instance_type == "watts_strogatz":
            G = generate_watts_strogatz_graph(num_nodes, num_vehicles)
        elif instance_type == "erdos_renyi":
            G = generate_erdos_renyi(num_nodes, num_vehicles)
        elif instance_type == "complete":
            G = complete_graph(num_nodes, num_vehicles)
        elif instance_type == "newman_watts_strogatz":
            G = generate_newman_watts_strogatz_graph(num_nodes, num_vehicles)
        elif instance_type == "euclidean_tsp":
            G = generate_euclidean_graph(num_nodes)
        elif instance_type == "euclidean_tsp_outlier":
            # G = generate_euclidean_graph(num_nodes, outlier_info)
            raise ValueError("This has not yet been implemented")

        for (u, v) in G.edges():
            if "euclidean" not in instance_type:
                G.edges[u, v]["cost"] = round(np.random.random(), 2)
            G.edges[u, v]["id"] = uuid.uuid4().hex

        # Randomly select depot
        depot_node_id = 0
        for node in G.nodes:
            if node == depot_node_id:
                G.nodes[depot_node_id]["tag"] = "Depot"
            else:
                G.nodes[node]["tag"] = ""

        verify = verify_graph(G, num_vehicles)

    return G


def verify_graph(G, num_vehicles):
    """
    This function verifies that the graph generated is appropriate for the LMRO problem

    Args:
        G (object): the graph as a networkx graph object

    Returns:
        is_feasible (bool): whether the graph is feasible for the project
    """

    is_feasible = False
    depot_check = False
    nodes_check = False

    for node in G.nodes:
        if G.nodes[node]["tag"] == "Depot":
            if len(G.edges(node)) >= 2 * num_vehicles:
                depot_check = True
        if len(G.edges(node)) >= 2:
            nodes_check = True
    if depot_check == True and nodes_check == True:
        is_feasible = True

    # Check for Depot
    depot_exists = False
    for node in G.nodes:
        tag = G.nodes[node]["tag"]
        if tag == "Depot":
            depot_exists = True

    if depot_exists == False:
        is_feasible = False

    return is_feasible


def generate_watts_strogatz_graph(num_nodes, num_vehicles, k=4, p=0.5):
    """ Build Watts Strogatz Graph """
    G = nx.connected_watts_strogatz_graph(num_nodes, k, p, num_vehicles)
    return G


def generate_erdos_renyi(num_nodes, num_vehicles, p=0.5):
    """ Build Erdors-Renyi Graph"""
    G = nx.erdos_renyi_graph(num_nodes, p, num_vehicles)
    return G


def generate_newman_watts_strogatz_graph(num_nodes, num_vehicles, k=2, p=0.5):
    """ Build Newman Wattz Strogatz Graph"""
    G = nx.newman_watts_strogatz_graph(num_nodes, k, p)
    return G


def complete_graph(num_nodes, num_vehicles):
    """ Build Complete Graph """
    G = nx.complete_graph(num_nodes)
    return G

def generate_euclidean_graph(num_nodes):
    """A function to generate a euclidean graph 'G' based on:
    2. Initialise an empty graph
    3. Randomly generate positions on a 2D plane and allocate these points as nodes
    4. Create a complete graph by connecting all edges together and 
    make the cost the euclidean distance between the two points

    Args:
        num_nodes (int): Number of nodes
    """

    # Init range for vertices
    V=range(num_nodes)

    # Initialise empty graph
    G = nx.Graph()

    # Build nodes
    nodes = [(i,{'pos':tuple(np.random.random(2))}) for i in V]
    G.add_nodes_from(nodes)

    # Get positions
    pos = nx.get_node_attributes(G, 'pos')

    # Add edges to the graph
    for i in V:
        for j in V:
            if i != j:
                G.add_edge(i, j, cost=distance(pos[i],pos[j]))

    return G