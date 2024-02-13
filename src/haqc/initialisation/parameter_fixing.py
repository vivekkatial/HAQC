import numpy as np
import networkx as nx
import logging

from haqc.algorithms.eval_qaoa import eval_qaoa
from qiskit_optimization.applications import Maxcut

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_optimal_parameters_from_parameter_fixing(n_layers, graph_instance, n=10):
    """
    Perform the Parameters Fixing algorithm for QAOA.

    Args:
    n_layers (int): The circuit depth.
    qaoa_circuit: The QAOA circuit to evaluate.
    n (int): The number of trials.

    Returns:
    tuple: The best parameters and the best score.
    """
    
    adjacency_matrix = nx.adjacency_matrix(graph_instance)
    max_cut = Maxcut(adjacency_matrix)
    qubitOp, offset = max_cut.to_quadratic_program().to_ising()

    # Initialize the best parameters and the best score
    best_params = ()
    best_expectation = -np.inf
    total_fevals = 0

    # Iterate over the depth of the circuit
    for q in range(1, n_layers + 1):
        logging.info(f"Starting layer {q} optimization with {n} trials.")
        
        # Initialize a list to store the scores for each trial
        scores = []

        # Perform n trials
        for k in range(n):
            # Generate random parameters beta and gamma
            beta = np.random.uniform(0, np.pi)
            gamma = np.random.uniform(0, 2 * np.pi)

            # For the first step, just use the generated parameters
            if q == 1:
                current_params = (beta, gamma)
            else:
                # Append the new parameters to the best parameters from the previous iteration
                current_params = best_params + (beta, gamma)

            # Evaluate the QAOA with the current parameters
            logging.getLogger().setLevel(logging.WARNING)
            result = eval_qaoa(current_params, qubitOp)
            logging.getLogger().setLevel(logging.INFO)


            score = result.optimal_value
            fevals = result.cost_function_evals
            total_fevals += fevals

            logging.info(f"Trial {k+1}/{n} for layer {q}: Score = {score}, Function Evaluations = {fevals}")

            scores.append(score)

        # Find the parameters that gave the maximum score in this iteration
        max_score_index = np.argmax(scores)
        best_expectation = scores[max_score_index]
        best_params = current_params[:2 * (q - 1)] + (current_params[2 * (q - 1)], current_params[2 * q - 1])
        logging.info(f"Best expectation value for layer {q}: {best_expectation}")

    logging.info(f"Total function evaluations: {total_fevals}")
    assert len(best_params) == 2 * n_layers

    # Initialise parameters for p+1 layer as a random final point
    beta = np.random.uniform(0, np.pi)
    gamma = np.random.uniform(0, 2 * np.pi)
    betas = best_params[n_layers:] + (beta,)
    gammas = best_params[:n_layers] + (gamma,)
    initial_point = np.concatenate((gammas, betas))
    # Return the best parameters and the best score
    initial_point_info = dict(best_expectation=best_expectation, total_fevals=total_fevals)
    return initial_point, initial_point_info


if __name__ == "__main__":
    G = nx.connected_watts_strogatz_graph(10, k=2, p=0.5)
    N_LAYERS = 1  # Set the circuit depth
    n = 2  # Set the number of trials
    inital_point, initial_point_info = get_optimal_parameters_from_parameter_fixing(N_LAYERS-1, G, n)
    logging.info(f"Total number of function evaluations to achieve the best parameters: {initial_point_info['total_fevals']}")
    logging.info(f"Initial point for layer {N_LAYERS}: {inital_point}")
