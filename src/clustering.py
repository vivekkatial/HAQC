import networkx as nx
from sklearn import cluster
from itertools import groupby


def create_clusters(G, num_vehicles, algorithm, cost_mat, *args, **kwargs):
    """This function generates and builds the clusters based on the graph, num vehicles and the algorithm provided

    Args:
        G (object): NetworkX graph Object
        num_vehicles (int): The number of vehicles
        algorithm (str): The algorithm used for clustering

    Returns:
        tuple: A tuple containing the graph object in [0] and a cluster mapping in [1]
    """

    # TODO: Error and input validation for the function
    if algorithm not in ["k-means", "spectral-clustering"]:
        print("error")
    else:
        if algorithm == "k-means":
            G, cluster_mapping = build_clusters_k_means(G, cost_mat, num_vehicles)
        if algorithm == "spectral-clustering":
            G, cluster_mapping = build_clusters_spectral_clustering(
                G, cost_mat, num_vehicles
            )

    return G, cluster_mapping


def build_clusters_k_means(G, cost_mat, num_vehicles):
    """Creates clusters based on k means clustering

    Args:
        G (object): NetworkX graph Object
        cost_mat (array): `np.array` object which contains the cost matrix
        num_vehicles (int): The number of vehicles

    Returns:
        tuple: A tuple containing the `networkX` graph object in [0] and a cluster mapping in [1]
    """
    raise ValueError("Currently NOT Functional")

    return G, clusters


def build_clusters_spectral_clustering(G, cost_mat, num_vehicles):
    """Creates clusters based on SpectralClustering

    Args:
        G (object): NetworkX graph Object
        cost_mat (array): `np.array` object which contains the cost matrix
        num_vehicles (int): The number of vehicles

    Returns:
        tuple: A tuple containing the `networkX` graph object in [0] and a cluster mapping in [1]
    """

    # TODO: Add error checking and input validation

    # Conduct Clustering using Spectral Clustering
    sc = cluster.SpectralClustering(
        num_vehicles, affinity="precomputed", n_init=1000, assign_labels="discretize"
    )
    sc.fit_predict(cost_mat)
    cluster_mapping = sc.labels_
    labelled_nodes = {
        node: {"cluster": cluster + 1}
        for (node, cluster) in zip(list(G.nodes()), cluster_mapping)
    }
    nx.set_node_attributes(G, labelled_nodes)

    return G, cluster_mapping


def _get_depot_edges_for_cluster(subgraph, depot_edges):
    """For a given subgraph (that is a cluster) find the edges connected to the depot

    Args:
        subgraph (object): A networkX graph object
        depot_edges (list): list of networkX edges

    Returns:
        [type]: Edge list
    """
    edge_list = []
    for node in subgraph:
        for edge in depot_edges:
            if node == edge[0] or node == edge[1]:
                edge_list.append(edge)
    return edge_list


def build_sub_graphs(G, depot_node, depot_edges):
    """Build Subgraphs from the clustered Graph Object

    Args:
        G (object): A networkX graph object
        depot_node (str): node id for the Depot
        depot_edges (list): List of edges

    Returns:
        [list]: List of Subgraphs
    """
    # Sort the nodes by their cluster (this is required for the groupby to work)
    sorted_by_cluster = sorted(
        G.nodes(data=True), key=lambda node_data: node_data[1]["cluster"]
    )
    # Group objects with same cluster
    grouped_cluster = groupby(
        sorted_by_cluster, key=lambda node_data: node_data[1]["cluster"]
    )
    grouped_cluster

    # Initialise dictionary for subgraphs
    subgraphs = dict()
    for key, group in grouped_cluster:
        nodes_in_group, _ = zip(*list(group))
        subgraphs[key] = G.subgraph(nodes_in_group)
        subgraphs[key] = nx.Graph(subgraphs[key])
        cluster_edges = _get_depot_edges_for_cluster(subgraphs[key], depot_edges)
        subgraphs[key].add_nodes_from([(depot_node, {"tag": "DEPOT", "cluster": 0})])
        subgraphs[key].add_edges_from(cluster_edges)

    return subgraphs


def draw_sub_graph(subgraph):
    """A function to draw a subgraph (that is clustered)

    Args:
        subgraph (object): A networkX graph object
    """
    clusters = [node[1]["cluster"] for node in subgraph.nodes(data=True)]
    print(clusters)
    nx.draw(subgraph, with_labels=True)
