import networkx as nx
import numpy as np
import uuid

from src.utils import distance, get_direction


def generate_random_instance(
    num_nodes: int,
    num_vehicles: int,
    instance_type: str,
    num_outliers: int = 1,
    gamma: int = 2,
    quasi: bool = False,
    noise: float = 0.1,
) -> nx.classes.graph.Graph:
    """This function creates random graphs with each node having dimension of 2

    Args:
        num_nodes (int): [description]
        num_vehicles (int): [description]
        instance_type (str): Must be one of "watts_strogatz",
            "erdos_renyi",
            "complete",
            "newman_watts_strogatz",
            "euclidean_tsp",
            "euclidean_tsp_outlier",
            "asymmetric_tsp",
            "quasi_asymmetric_tsp"
        num_outliers (int, optional): [description]. Defaults to 1.
        gamma (int, optional): [description]. Defaults to 2.
        quasi (bool, optional): [description]. Defaults to False.
        noise (float, optional): [description]. Defaults to 0.1.

    Raises:
        ValueError: [description]

    Returns:
        nx.classes.graph.Graph: [description]
    """

    verify = False

    while verify == False:

        assert num_nodes >= 3, "num_nodes cannot be {}, must be >= 3".format(num_nodes)
        if instance_type not in [
            "watts_strogatz",
            "erdos_renyi",
            "complete",
            "newman_watts_strogatz",
            "euclidean_tsp",
            "euclidean_tsp_outlier",
            "asymmetric_tsp",
            "quasi_asymmetric_tsp",
        ]:
            raise ValueError("Incorrect Instance Type Requested")

        if instance_type == "watts_strogatz":
            G = generate_watts_strogatz_graph(num_nodes, num_vehicles)
        elif instance_type == "erdos_renyi":
            G = generate_erdos_renyi(num_nodes, num_vehicles)
        elif instance_type == "complete":
            G = complete_graph(num_nodes)
        elif instance_type == "newman_watts_strogatz":
            G = generate_newman_watts_strogatz_graph(num_nodes)
        elif instance_type == "euclidean_tsp":
            G = generate_euclidean_graph(num_nodes)
        elif instance_type == "euclidean_tsp_outlier":
            G = generate_euclidean_graph_with_outliers(
                num_nodes=num_nodes, num_outliers=num_outliers, gamma=gamma
            )
        elif instance_type == "asymmetric_tsp":
            G = generate_asymmetric_euclidean_graph(num_nodes, quasi, noise)
        elif instance_type == "quasi_asymmetric_tsp":
            G = generate_asymmetric_euclidean_graph(num_nodes, quasi=True, noise=noise)

        for (u, v) in G.edges():
            if "tsp" not in instance_type:
                G.edges[u, v]["cost"] = round(np.random.random(), 2)
            G.edges[u, v]["id"] = uuid.uuid4().hex

        # Randomly select depot
        depot_node_id = 0
        for node in G.nodes:
            if node == depot_node_id:
                G.nodes[depot_node_id]["tag"] = "Depot"
            elif G.nodes[node].get("tag") is None:
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
    """Build Watts Strogatz Graph"""
    G = nx.connected_watts_strogatz_graph(num_nodes, k, p, num_vehicles)
    return G


def generate_erdos_renyi(num_nodes, num_vehicles, p=0.5):
    """Build Erdors-Renyi Graph"""
    G = nx.erdos_renyi_graph(num_nodes, p, num_vehicles)
    return G


def generate_newman_watts_strogatz_graph(num_nodes, k=4, p=0.5):
    """Build Newman Wattz Strogatz Graph"""
    G = nx.newman_watts_strogatz_graph(num_nodes, k, p)
    return G


def complete_graph(num_nodes):
    """Build Complete Graph"""
    G = nx.complete_graph(num_nodes)
    return G


def generate_euclidean_graph(num_nodes: int) -> nx.classes.graph.Graph:
    """A function to generate a euclidean graph 'G' based on:
    2. Initialise an empty graph
    3. Randomly generate positions on a 2D plane and allocate these points as nodes
    4. Create a complete graph by connecting all edges together and
    make the cost the euclidean distance between the two points

    Args:
        num_nodes (int): Number of nodes
    """

    # Init range for vertices
    V = range(num_nodes)

    # Initialise empty graph
    G = nx.Graph()

    # Build nodes
    nodes = [(i, {"pos": tuple(np.random.random(2))}) for i in V]
    G.add_nodes_from(nodes)

    # Get positions
    pos = nx.get_node_attributes(G, "pos")

    # Add edges to the graph
    for i in V:
        for j in V:
            if i != j:
                G.add_edge(i, j, cost=distance(pos[i], pos[j]))

    return G


def generate_euclidean_graph_with_outliers(
    num_nodes: int, num_outliers: int, gamma: float
) -> nx.classes.graph.Graph:
    """A function to generate a euclidean graph with outlier structure

    Args:
        num_nodes (int): Number of nodes
        num_outliers (int): Number of outliers (must be less than number of nodes)
        gamma (float): A parameter to decide how far away the nodes are from each other based on $\sqrt{2}$

    Raises:
        ValueError: Number of nodes must be greater than number of outliers

    Returns:
        nx.classes.graph.Graph: A network.X graph object
    """
    G = generate_euclidean_graph(num_nodes)

    # Randomly select k nodes from the network (check k < N)
    if num_outliers > G.number_of_nodes():
        raise ValueError(
            "k=%s cannot be higher than the number of nodes N=%s"
            % (num_nodes, G.number_of_nodes())
        )
    else:
        # Ensure we get k distinct nodes being selected
        random_nodes = np.random.choice(
            range(G.number_of_nodes()), num_outliers, replace=False
        )

    # Update the node locations
    for node in random_nodes:
        # Move node
        x_move_direction = get_direction()
        y_move_direction = get_direction()
        x_new = G.nodes()[node]["pos"][0] + x_move_direction * gamma * np.sqrt(2)
        y_new = G.nodes()[node]["pos"][1] + y_move_direction * gamma * np.sqrt(2)
        G.nodes()[node]["pos"] = (x_new, y_new)
        G.nodes()[node]["tag"] = "outlier"

    # Get new position data
    pos = nx.get_node_attributes(G, "pos")

    V = range(num_nodes)
    # Recalculate edge distances
    for i in V:
        for j in V:
            if i != j:
                G.add_edge(i, j, cost=distance(pos[i], pos[j]))

    return G


def generate_asymmetric_euclidean_graph(
    num_nodes: int, quasi: bool = False, noise: float = 0.1
) -> nx.classes.graph.Graph:
    """A function to generate asymetric euclidean graph

    Args:
        num_nodes (int): Number of nodes
        quasi (bool, optional): If graph is a quasi graph. Defaults to False.
        noise (float, optional): Noise parameter. Defaults to 0.1.

    Returns:
        nx.classes.graph.Graph: networkX Graph
    """

    # Generate random euclidean graph
    G = generate_euclidean_graph(num_nodes)
    adj = nx.adjacency_matrix(G, weight="cost")

    # Randomly generate an adjacency matrix with random costs for each edge
    rand = np.random.rand(len(G), len(G))
    np.fill_diagonal(rand, 0)

    # An asymmetric graph adjacency can be represented by:
    # A_{\text{asym}} = A - L(A) + A_{\text{rand}}
    if quasi:
        asymmetric_adj = adj + rand * noise
    else:
        asymmetric_adj = adj.toarray() - np.tril(adj.toarray()) + rand

    dt = [("cost", float)]
    asymmetric_adj = np.array(asymmetric_adj, dtype=dt)

    # Convert this adjacency matrix into a graph
    G_asym = nx.from_numpy_array(asymmetric_adj, create_using=nx.DiGraph)

    # Update information regarding tags and position into the new graph
    pos = nx.get_node_attributes(G, "pos")
    for node in G_asym.nodes():
        G_asym.nodes()[node]["pos"] = pos[node]

    return G_asym
