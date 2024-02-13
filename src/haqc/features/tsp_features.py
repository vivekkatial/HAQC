import networkx as nx
import numpy as np
import scipy


def is_symmetric(A, tol=1e-8):
    return scipy.sparse.linalg.norm(A - A.T, scipy.Inf) < tol


def get_tsp_features(G):
    """
    Generates a list of TSP based features for the given graph

    Args:
        G (object): networkx graph object

    Returns:
        features (dict): a dictionary of the features in the given graph
    """

    features = {}

    adj = nx.adjacency_matrix(G, weight="cost")
    shortest1 = nx.shortest_path_length(G, weight="cost")
    shortest2 = dict(shortest1)
    ecc = nx.eccentricity(G, sp=shortest2)

    # Find Nearest Neighbours
    nearest_neighbours = np.asarray(
        [
            min([edge[2]["cost"] for edge in G.edges(node, data=True)])
            for node in G.nodes
        ]
    )
    normalised_nearest_neighbours = nearest_neighbours / np.sqrt(
        np.sum(nearest_neighbours**2)
    )
    normalised_nearest_neighbours

    # Fraction of distinct distances
    cost_one_dp = [np.round(edge[2]["cost"], 1) for edge in G.edges(data=True)]

    features["tsp_nnd_var"] = np.var(normalised_nearest_neighbours)
    features["tsp_nnd_coefficient_var"] = 100 * (
        np.std(normalised_nearest_neighbours) / np.mean(normalised_nearest_neighbours)
    )
    features["tsp_radius"] = nx.algorithms.distance_measures.radius(G, e=ecc)
    features["tsp_mean"] = np.mean(adj)
    features["tsp_std"] = np.std(nx.to_numpy_matrix(G, weight="cost"))
    features["tsp_frac_distinct_dist_one_dp"] = len(set(cost_one_dp)) / len(cost_one_dp)
    features["tsp_clustering_coeff_variance"] = np.var(
        [item[1] for item in nx.clustering(G).items()]
    )
    features["tsp_symmetric"] = is_symmetric(adj)

    # Asymmetry features
    diff = abs(adj - adj.T)
    diff = diff.toarray()
    features["tsp_asym_diff_matrix_sd"] = np.std(
        diff[np.triu_indices(diff.shape[0], k=1)]
    )
    features["tsp_asym_diff_matrix_mean"] = np.mean(
        diff[np.triu_indices(diff.shape[0], k=1)]
    )

    return features
