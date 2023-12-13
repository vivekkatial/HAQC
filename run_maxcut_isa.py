import os
import logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info('Script started')

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import time
import mlflow
import json

from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms.optimizers import COBYLA, ADAM
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.utils import QuantumInstance
from qiskit_optimization.applications import Maxcut
from qiskit.circuit import Parameter
from itertools import combinations

# Custom imports
from qaoa_vrp.generators.graph_instance import create_graphs_from_all_sources
from qaoa_vrp.exp_utils import str2bool, to_snake_case, make_temp_directory, check_boto3_credentials
from qaoa_vrp.features.graph_features import get_graph_features
from qaoa_vrp.parallel.landscape_parallel import parallel_computation


def run_qaoa_script(track_mlflow, graph_type, node_size, quant_alg, n_layers=1):

    if track_mlflow:
        # Configure MLFlow Stuff
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        experiment_name = "QAOA-Parameter-layers-vanilla"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    # Generate all graph sources
    G_instances = create_graphs_from_all_sources(instance_size=node_size, sources="ALL")

    G_instances = [g for g in G_instances if g.graph_type == graph_type]
    graph_instance = G_instances[0]
    G = graph_instance.G

    logging.info(
        f"\n{'-'*10} This run is for a {graph_instance.graph_type} graph with {len(G.nodes())} nodes  {'-'*10}\n"
    )

    # Show instance features
    graph_features = get_graph_features(graph_instance.G)
    instance_class = to_snake_case(graph_instance.graph_type)

    graph_features = {str(key): val for key, val in graph_features.items()}
    logging.info(f"Graph Features {json.dumps(graph_features, indent=2)}")

    if track_mlflow:
        mlflow.log_param("instance_class", instance_class)
        mlflow.log_param("instance_size", node_size)
        mlflow.log_param("quantum_algorithm", quant_alg)
        mlflow.log_param("n_layers", n_layers)
        mlflow.log_params(graph_features)

    # Generate the adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G)
    max_cut = Maxcut(adjacency_matrix)
    qubitOp, offset = max_cut.to_quadratic_program().to_ising()

    # Landscape Analysis of Instance (at p=1)

    p = 1
    qaoa = QAOA(optimizer=COBYLA(), reps=1)
    # Use constrained search space
    beta = np.linspace(-np.pi / 4, np.pi / 4, 100)
    gamma = np.linspace(-np.pi / 2, np.pi / 2, 100)

    # Example usage
    obj_vals = parallel_computation(beta,gamma, qubitOp, qaoa)

    # ### Plotting the Parameter Landscape
    # The heatmap below represents the landscape of the objective function across 
    # different values of \$\gamma\$ and \$\beta\$.
    # The color intensity indicates the expectation value of the Hamiltonian, 
    # helping identify the regions where optimal parameters may lie.

    with make_temp_directory() as tmp_dir:
        Beta, Gamma = np.meshgrid(beta, gamma)

        # Plotting
        plt.figure(figsize=(10, 8))
        cp = plt.contourf(Beta, Gamma, obj_vals.T, cmap='viridis')  # Transpose obj_vals if necessary
        plt.colorbar(cp)
        plt.title(f'QAOA Objective Function Landscape (p=1) for  {graph_type}')
        plt.xlabel('Beta')
        plt.ylabel('Gamma')

        # Adjust the x and y limits to show the new range
        plt.xlim(-np.pi / 2, np.pi / 2)
        plt.ylim(-np.pi / 4, np.pi / 4)

        # Adjust the x and y labels to show the new pi values
        plt.xticks(np.linspace(-np.pi / 2, np.pi / 2, 5), 
                ['-π/2', '-π/4', '0', 'π/4', 'π/2'])
        plt.yticks(np.linspace(-np.pi / 4, np.pi / 4, 5), 
                ['-π/4', '-π/8', '0', 'π/8', 'π/4'])

        plt.savefig(os.path.join(tmp_dir, 'landscape_plot.png'))

        if track_mlflow:
            mlflow.log_artifact(os.path.join(tmp_dir, 'landscape_plot.png'))
        # Clear plots
        plt.clf()


    # ### Brute Force Solution for the Max-Cut Problem
    # A brute-force solution to the Max-Cut problem involves evaluating every possible partition of the graph's nodes into two sets.
    # We calculate the 'cut' for each partition, which is the number of edges between the two sets. The goal is to maximize this cut.
    # NOTE: This method is computationally intensive and not practical for large graphs,
    # but it gives an exact solution for smaller ones.

    def compute_max_cut_brute_force(G):
        nodes = G.nodes()
        n = len(nodes)
        max_cut_value = 0
        max_cut_partition = None

        # Iterate over all possible ways to split the nodes into two sets
        for size in range(1, n // 2 + 1):
            for subset in combinations(nodes, size):
                cut_value = sum(
                    (
                        G.has_edge(i, j)
                        for i in subset
                        for j in G.nodes()
                        if j not in subset
                    )
                )
                if cut_value > max_cut_value:
                    max_cut_value = cut_value
                    max_cut_partition = subset

        return max_cut_partition, max_cut_value

    # Apply the brute force solution to our graph
    max_cut_partition, max_cut_value = compute_max_cut_brute_force(G)

    # ### Visualizing the Brute Force Solution
    with make_temp_directory() as tmp_dir:
        # Define the colors for each node based on the brute force solution partition
        node_colors = [
            'pink' if node in max_cut_partition else 'lightblue' for node in G.nodes()
        ]

        # Draw the graph with nodes colored based on the solution
        nx.draw(
            G,
            with_labels=True,
            node_color=node_colors,
            edge_color='gray',
            node_size=700,
            font_size=10,
        )
        plt.savefig(os.path.join(tmp_dir, 'maxcut_solution_plot.png'))
        if track_mlflow:
            mlflow.log_artifact(os.path.join(tmp_dir, 'maxcut_solution_plot.png'))
        # Clear plots
        plt.clf()

    logging.info(f"\n{'-'*10} Solving for Exact Ground State {'-'*10}\n")

    exact_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(
        operator=qubitOp
    )

    logging.info(f"Minimum Energy is {exact_result}")
    logging.info(f"\n{'-'*10} Simulating Instance on Quantum using VQE {'-'*10}\n")

    MAX_ITERATIONS = 1000
    N_RESTARTS = 1
    N_LAYERS = n_layers

    # Run optimisation code
    optimizer = ADAM()



    backend = Aer.get_backend("aer_simulator_statevector")
    quantum_instance = QuantumInstance(backend)
    logging.info(f"Testing Optimizer {type(optimizer).__name__}")
    
    # Callback function to store intermediate values
    intermediate_values = []

    def store_intermediate_result(eval_count, parameters, mean, std):
        if eval_count % 100 == 0:
            logging.info(f"{type(optimizer).__name__} iteration {eval_count} \t cost function {mean}")
        betas = parameters[:N_LAYERS]   # Extracting beta values
        gammas = parameters[N_LAYERS:]  # Extracting gamma values
        intermediate_values.append({
            'eval_count': eval_count,
            'parameters': {'gammas': gammas, 'betas': betas},
            'mean': mean,
            'std': std
        })

    for restart in range(N_RESTARTS):
        logging.info(f"Running Optimization at n_restart={restart}")
        # Initialize the initial gamma and beta parameters
        initial_beta = np.random.uniform(-np.pi / 4, np.pi / 4, N_LAYERS)
        initial_gamma = np.random.uniform(-np.pi / 2, np.pi / 2, N_LAYERS)
        initial_point = np.concatenate([initial_beta, initial_gamma])

        # QAOA definition
        qaoa = QAOA(
            optimizer=optimizer,
            reps=N_LAYERS,
            initial_point=initial_point, # initial parameter values for beta and gamma
            callback=store_intermediate_result, 
            quantum_instance=quantum_instance,
        )

        qaoa_result = qaoa.compute_minimum_eigenvalue(qubitOp)

    logging.info(f"\n{'-'*10} Optimization Complete {'-'*10}\n")

    # Plot the convergence
    with make_temp_directory() as tmp_dir:
        # Extract values for plotting
        total_counts = [info['eval_count'] for info in intermediate_values]
        values = [info['mean'] for info in intermediate_values]
        gamma_values = [info['parameters']['gammas'] for info in intermediate_values]
        beta_values = [info['parameters']['betas'] for info in intermediate_values]

        # Plot the convergence
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.plot(total_counts, values, label=type(optimizer).__name__)
        plt.xlabel("Eval count")
        plt.ylabel("Energy")
        plt.title("Energy convergence for various optimizers")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(tmp_dir, 'energy_convergence_optimisation_plot.png'))
        if track_mlflow:
            mlflow.log_artifact(
                os.path.join(tmp_dir, 'energy_convergence_optimisation_plot.png')
            )
        plt.clf()

        # Optionally, plot gamma and beta values over iterations
        plt.figure(figsize=(12, 6))
        for i in range(N_LAYERS):
            plt.subplot(1, 2, 1)
            plt.plot(total_counts, [gamma[i] for gamma in gamma_values], label=f'Gamma {i+1}')
            plt.xlabel('Eval count')
            plt.ylabel('Gamma value')
            plt.title('Gamma Convergence Over Iterations')

            plt.subplot(1, 2, 2)
            plt.plot(total_counts, [beta[i] for beta in beta_values], label=f'Beta {i+1}')
            plt.xlabel('Eval count')
            plt.ylabel('Beta value')
            plt.title('Beta Convergence Over Iterations')

        plt.legend()
        plt.savefig(os.path.join(tmp_dir, 'gamma_beta_convergence_plot.png'))
        if track_mlflow:
            mlflow.log_artifact(
                os.path.join(tmp_dir, 'gamma_beta_convergence_plot.png')
            )
        plt.clf()


    # QAOA Result Analysis for MaxCut Problem

    # Extract the most likely solution from QAOA results
    most_likely_solution = max_cut.sample_most_likely(qaoa_result.eigenstate)

    # Calculate the energy gap
    energy_gap = exact_result.eigenvalue.real - qaoa_result.eigenvalue.real

    # Convert exact result eigenstate to matrix form and get QAOA state
    exact_result_vector = exact_result.eigenstate.to_matrix()
    qaoa_state_vector = qaoa_result.eigenstate

    # Compute inner product between exact result and QAOA state
    inner_product = np.dot(exact_result_vector.conj(), qaoa_state_vector)

    # Calculate the probability of success (adjusting for MAXCUT symmetry)
    success_probability = (np.abs(inner_product) ** 2) * 2

    # Calculate the approximation ratio
    approximation_ratio = qaoa_result.eigenvalue.real / exact_result.eigenvalue.real

    # Extract optimal parameters
    optimal_params = qaoa_result.optimal_parameters
    optimal_gammas = [optimal_params[param] for param in optimal_params.keys() if 'γ' in str(param)]
    optimal_betas = [optimal_params[param] for param in optimal_params.keys() if 'β' in str(param)]


    # Output performance metrics
    logging.info(f"\n{'-'*10} MAXCUT Performance Metrics {'-'*10}\n")
    logging.info(f"Final energy <C>: {qaoa_result.eigenvalue.real}")
    logging.info(f"Energy gap: {energy_gap}")
    logging.info(
        f"Probability of being in the ground state P(C_max): {success_probability}"
    )
    logging.info(f"Approximation Ratio: {approximation_ratio}")
    logging.info(f"Optimal Parameters: {optimal_params}")
    
    if track_mlflow:
        mlflow.log_metric("final_energy", qaoa_result.eigenvalue.real)
        mlflow.log_metric("energy_gap", energy_gap)
        mlflow.log_metric("p_success", success_probability)
        mlflow.log_metric("approximation_ratio", approximation_ratio)
        # Log each optimal parameter in MLFlow
        for i in range(N_LAYERS):
            mlflow.log_metric(f'optimal_gamma_{i+1}', optimal_gammas[i])
            mlflow.log_metric(f'optimal_beta_{i+1}', optimal_betas[i])
        

    # Output other additional information
    logging.info(f"\n{'-'*10} Other Performance Information {'-'*10}\n")

    logging.info(f"Optimization time: {qaoa_result.optimizer_time}")
    logging.info(f"Max-cut objective: {qaoa_result.eigenvalue.real + offset}")
    logging.info(f"QAOA most likely solution: {most_likely_solution}")
    logging.info(f"Actual solution: {max_cut.sample_most_likely(exact_result_vector)}")
    logging.info(
        f"Solution objective: {max_cut.to_quadratic_program().objective.evaluate(most_likely_solution)}"
    )

    # Only do optimisation heatmap if n_layers=1
    if n_layers == 1:
        # Your existing code for plotting the heatmap
        with make_temp_directory() as tmp_dir:

            Beta, Gamma = np.meshgrid(beta, gamma)

            # Plotting
            plt.figure(figsize=(10, 8))
            cp = plt.contourf(Beta, Gamma, obj_vals.T, cmap='viridis')  # Transpose obj_vals if necessary
            plt.colorbar(cp)
            plt.title(f'QAOA Objective Function Landscape (p=1) for  {graph_type}')
            plt.xlabel('Beta')
            plt.ylabel('Gamma')

            # Adjust the x and y limits to show the new range
            plt.xlim(-np.pi / 2, np.pi / 2)
            plt.ylim(-np.pi / 4, np.pi / 4)

            # Adjust the x and y labels to show the new pi values
            plt.xticks(np.linspace(-np.pi / 2, np.pi / 2, 5), 
                    ['-π/2', '-π/4', '0', 'π/4', 'π/2'])
            plt.yticks(np.linspace(-np.pi / 4, np.pi / 4, 5), 
                    ['-π/4', '-π/8', '0', 'π/8', 'π/4'])
            # Plot the optimization path
            if beta_values and gamma_values:
                plt.plot(beta_values, gamma_values, '-', color='cyan', label='Optimization Path', linewidth=1, zorder=1)
                # Highlight the start and end points
                plt.scatter(beta_values[0], gamma_values[0], color='red', s=20, label='Start', zorder=2)
                plt.scatter( beta_values[-1], gamma_values[-1], color='magenta', s=20, label='End', zorder=2)

            plt.legend()
            plt.savefig(os.path.join(tmp_dir, 'landscape_optimisation_plot.png'))
            if track_mlflow:
                mlflow.log_artifact(
                    os.path.join(tmp_dir, 'landscape_optimisation_plot.png')
                )
            plt.clf()
    
    # Compute the performance metrics for using analytically found beta and gamma parameters found
    # This is based on this research: https://ar5iv.labs.arxiv.org/html/2103.11976#S4.E19



    logging.info('Script finished')


if __name__ == "__main__":
    check_boto3_credentials()

    parser = argparse.ArgumentParser(
        description="Run QAOA script with custom parameters."
    )

    parser.add_argument(
        "-T",
        "--track_mlflow",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Activate MlFlow Tracking.",
    )
    parser.add_argument(
        "-G",
        "--graph_type",
        type=str,
        default="3-Regular Graph",
        help="Type of Graph to test (based on qaoa_vrp/generators/graph_instance.py)",
    )
    parser.add_argument("-n", "--node_size", type=int, default=6, help="Size of Graph")
    parser.add_argument(
        "-q",
        "--quantum_algorithm",
        type=str,
        default="QAOA",
        help="Quantum Algorithm to test",
    )
    parser.add_argument(
        "-l", "--n_layers", type=int, default=1, help="Number of layers for QAOA"
    )

    args = parser.parse_args()
    print(vars(args))

    start_time = time.time()
    run_qaoa_script(
        track_mlflow=args.track_mlflow,
        graph_type=args.graph_type,
        node_size=args.node_size,
        quant_alg=args.quantum_algorithm,
        n_layers=args.n_layers,
    )
    end_time = time.time()

    print(f"Result found in: {end_time - start_time:.3f} seconds")
