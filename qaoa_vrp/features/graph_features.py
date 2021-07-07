import networkx as nx
import numpy as np


def get_graph_features(G):
    """
    Generates a list of features for the given graph

    Args:
        G (object): networkx graph object

    Returns:
        features (dict): a dictionary of the features in the given graph
    """

    features = {}

    L = nx.laplacian_matrix(G, weight="cost")
    # L  doesn't work for what we're triyng to do here (so e will not either)
    e = np.linalg.eigvals(L.A)

    features["acyclic"] = nx.is_directed_acyclic_graph(G)
    features[
        "algebraic_connectivity"
    ] = nx.linalg.algebraicconnectivity.algebraic_connectivity(G)
    features["average_distance"] = nx.average_shortest_path_length(G)
    features["bipartite"] = nx.is_bipartite(G)

    # features['Chromatic Index'] =
    # features['Chromatic Number'] =
    # features['Circumference'] =
    features["clique_number"] = nx.graph_clique_number(G)
    features["connected"] = nx.algorithms.components.is_connected(G)
    features["density"] = nx.classes.function.density(G)
    features["diameter"] = nx.algorithms.distance_measures.diameter(G)
    features[
        "edge_connectivity"
    ] = nx.algorithms.connectivity.connectivity.edge_connectivity(G)
    features["eulerian"] = nx.algorithms.euler.is_eulerian(G)
    # features['Genus'] =
    # features['Girth'] =
    # features['Hamiltonian'] =
    # features['independence_number'] = nx.algorithms.mis.maximal_independent_set(G)
    # features['Index'] =
    features["laplacian_largest_eigenvalue"] = max(e)
    # features['Longest Induced Cycle'] =
    # features['Longest Induced Path'] =
    # features['Matching Number'] =

    features["maximum_degree"] = max([G.degree[i] for i in G.nodes])
    features["minimum_degree"] = min([G.degree[i] for i in G.nodes])
    features["minimum_dominating_set"] = len(nx.algorithms.dominating.dominating_set(G))
    features[
        "number_of_components"
    ] = nx.algorithms.components.number_connected_components(G)
    features["number_of_edges"] = G.number_of_edges()
    # features['number_of_triangles'] = nx.algorithms.cluster.triangles(G)
    features["number_of_vertices"] = G.number_of_nodes()
    features["planar"] = nx.algorithms.planarity.check_planarity(G)[0]
    features["radius"] = nx.algorithms.distance_measures.radius(G)
    features["regular"] = nx.algorithms.regular.is_regular(G)
    features["laplacian_second_largest_eigenvalue"] = sorted(e)[1]
    features["ratio_of_two_largest_laplacian_eigenvaleus"] = max(e)/sorted(e)[1]
    features["smallest_eigenvalue"] = min(e)
    features[
        "vertex_connectivity"
    ] = nx.algorithms.connectivity.connectivity.node_connectivity(G)

    # Incude features that are typically used in a VRP/TSP (cost matrix, etc.)

    return features
