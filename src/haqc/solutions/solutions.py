import numpy as np
import networkx as nx
from itertools import combinations


def compute_max_cut_brute_force(G):
    nodes = G.nodes()
    n = len(nodes)
    max_cut_value = 0
    max_cut_partition = None

    # Iterate over all possible ways to split the nodes into two sets
    for size in range(1, n // 2 + 1):
        for subset in combinations(nodes, size):
            cut_value = sum(
                (G.has_edge(i, j) for i in subset for j in G.nodes() if j not in subset)
            )
            if cut_value > max_cut_value:
                max_cut_value = cut_value
                max_cut_partition = subset

    return max_cut_partition, max_cut_value


def compute_distance(
    n_layers, beta_values, beta_optimised_values, gamma_values, gamma_optimised_values
):
    """
    Compute the distance based on the given parameters.

    Parameters:
    n_layers (int): The number of layers to sum over.
    beta_values (list): List of beta values.
    beta_optimised_values (list): List of optimised beta values.
    gamma_values (list): List of gamma values.
    gamma_optimised_values (list): List of optimised gamma values.

    Returns:
    float: Computed distance.
    """
    distance = 0
    for i in range(n_layers):
        distance += abs(beta_values[i] - beta_optimised_values[i]) + abs(
            gamma_values[i] - gamma_optimised_values[i]
        )
    return distance


# Define functions to compute the optimal parameter values based on parameter concentration
def get_analytical_parameters(n, n_layers):
    """A function to compute the optimal parameter values based on parameter concentration paper.
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.L010401

    Args:
        n (int): System size of the MAXCUT instance
        n_layers (int): Number of layers in the QAOA circuit
    """

    # Check that argements are valid
    if n < 1:
        raise ValueError("n must be greater than 0")
    if n_layers < 1:
        raise ValueError("n_layers must be greater than 0")

    # Compute based on number of layers

    # 1 layer (beta = pi/(n+4) and gamma = pi - 2*beta)
    if n_layers == 1:
        beta = np.pi - np.pi / (n + 4)
        gamma = 2 * np.pi - np.pi - 2 * beta
        return [beta, gamma]

    # 2 Layers (based on eq 15-18 in paper)
    if n_layers == 2:
        beta_1 = np.pi / n
        beta_2 = np.pi / (n + 4)
        gamma_1 = np.pi
        gamma_2 = np.pi * ((n + 2) / (n + 4))
        return [beta_1, beta_2, gamma_1, gamma_2]

    # For more than 2 layers, use the lookup formula in eq 19
    if n_layers > 2 and n_layers < 6:
        betas = []
        gammas = []
        for i in range(n_layers):
            beta, gamma = get_analytical_parameters_lookup(n, i + 1)
            betas.append(beta)
            gammas.append(gamma)

        # Concatenate beta and gamma into a single list
        return np.concatenate((betas, gammas))


def get_analytical_parameters_lookup(n, n_layers):
    # Define lookup dictionary
    data = {
        1: {'a1': 1.04, 'a2': 0.92, 'b1': 1.06, 'b2': 2.07},
        2: {'a1': 0.98, 'a2': 1.23, 'b1': 1.05, 'b2': 2.04},
        3: {'a1': 0.94, 'a2': 1.58, 'b1': 1.05, 'b2': 1.96},
        4: {'a1': 0.88, 'a2': 2.32, 'b1': 1.03, 'b2': 1.83},
        5: {'a1': 1.09, 'a2': 5.25, 'b1': 1.00, 'b2': 2.00},
    }

    # Check that argements are valid
    if n < 1:
        raise ValueError("n must be greater than 0")
    if n_layers < 1:
        raise ValueError("n_layers must be greater than 0")

    # Extract a1, a2, b1, b2 from dictionary based on n_layers
    a1 = data[n_layers]['a1']
    a2 = data[n_layers]['a2']
    b1 = data[n_layers]['b1']
    b2 = data[n_layers]['b2']

    # Calculate beta
    beta = np.pi / (a1 * n + a2)

    # Calculate gamma
    gamma = b1 * np.pi - b2 * beta

    return beta, gamma
