import numpy as np
import cvxpy as cvx
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Tuple
from haqc.solutions.solutions import compute_max_cut_brute_force

def goemans_williamson(graph: nx.Graph) -> Tuple[np.ndarray, float, float]:
    """
    Implements the Goemans-Williamson algorithm for the Max-Cut problem.
    
    Based on the Github code available at:  https://github.com/rigetti/quantumflow-qaoa/blob/master/gw.py
    Reference:
        Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
        algorithms for maximum cut and satisfiability problems using
        semidefinite programming. Journal of the ACM (JACM), 42(6), pp.1115-1145.

    Returns:
        partitions (np.ndarray): Partitioning of the graph nodes (+1 or -1).
        score (float): The GW score for the calculated cut.
        bound (float): The GW bound from the SDP relaxation.
    """
    # Calculate the Laplacian matrix and scale it
    laplacian = 0.25 * nx.laplacian_matrix(graph).todense()

    # Define the semidefinite program (SDP)
    psd_matrix = cvx.Variable(laplacian.shape, PSD=True)
    objective = cvx.Maximize(cvx.trace(laplacian @ psd_matrix))
    constraints = [cvx.diag(psd_matrix) == 1]  # Constraint for unit norm
    problem = cvx.Problem(objective, constraints)

    # Solve the SDP
    problem.solve(solver=cvx.SCS)

    # Extract the solution and compute the eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(psd_matrix.value)
    sdp_vectors = eigenvectors[:, eigenvalues > 1e-6]

    # Compute the bound from the SDP relaxation
    bound = np.trace(laplacian @ psd_matrix.value)

    # Generate a random hyperplane to partition the graph
    random_vector = np.random.randn(sdp_vectors.shape[1])
    random_vector /= np.linalg.norm(random_vector)  # Normalize
    partitions = np.sign(sdp_vectors @ random_vector)  # Determine partitions

    # Calculate the score of the cut
    score = partitions @ laplacian @ partitions
    # Convert score to a float
    score = score.item()
    return partitions, score, bound


if __name__ == '__main__':
    # Initialize lists to store results
    optimal_scores = []
    gw_bounds = []
    gw_scores_means = []
    approx_ratios = []

    for _ in range(2000):
        # Generate a new instance of the graph
        G = nx.erdos_renyi_graph(10, 0.5)
        laplacian = np.array(0.25 * nx.laplacian_matrix(G).todense())
        
        # Compute GW bound and optimal score for the current graph
        bound = goemans_williamson(G)[2]
        scores = [goemans_williamson(G)[1] for _ in range(10)]  # Run GW 10 times per instance for variability
        _, optimal_score = compute_max_cut_brute_force(G)
        
        # Store results for analysis
        optimal_scores.append(optimal_score)
        gw_bounds.append(bound)
        gw_scores_means.append(np.mean(scores))
        approx_ratios.extend(np.array(scores) / optimal_score)

    # After running all instances, print aggregated results
    print(f"Optimal score mean: {np.mean(optimal_scores)}")
    print(f"GW bound approximation mean: {np.mean(gw_bounds)/np.mean(optimal_scores)}")
    print(f"GW score mean approximation: {np.mean(gw_scores_means)/np.mean(optimal_scores)}")

    # Set the aesthetic style of the plots
    sns.set(style="whitegrid")

    # Plot the approximation ratio of the GW scores for all instances using Seaborn
    sns.histplot(approx_ratios, bins=50, color='orange', alpha=0.7, edgecolor='white', linewidth=1.5)
    plt.axvline(np.mean(approx_ratios), color='black', linestyle='dashed', linewidth=1)
    plt.axvline(0.878, color='red', linestyle='dashed', linewidth=1)  # GW performance guarantee
    plt.legend(["Mean", "GW performance guarantee"])
    plt.xlabel("GW algorithm approximation ratio")
    plt.ylabel("Frequency")
    plt.show()
    
