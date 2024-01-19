import os
import logging

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

# Adjust Qiskit's logger to only display errors or critical messages
qiskit_logger = logging.getLogger('qiskit')
qiskit_logger.setLevel(logging.ERROR)  # or use logging.CRITICAL


logging.info('Script started')

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import time
import mlflow
import json
import pandas as pd

from qiskit import Aer
from qiskit.algorithms.optimizers import ADAM
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.utils import QuantumInstance
from qiskit_optimization.applications import Maxcut

# Custom imports
from src.haqc.generators.graph_instance import create_graphs_from_all_sources
from src.haqc.exp_utils import (
    str2bool,
    to_snake_case,
    make_temp_directory,
    check_boto3_credentials,
)
from src.haqc.features.graph_features import get_graph_features
from src.haqc.generators.parameter import get_optimal_parameters
from src.haqc.solutions.solutions import compute_max_cut_brute_force, compute_distance
from src.haqc.parallel.landscape_parallel import parallel_computation
from src.haqc.initialisation.initialisation import Initialisation
from src.haqc.plot.utils import *

# Theme plots to be seaborn style
plt.style.use('seaborn')

# Check that optimal parameters csv file exists
if not os.path.exists('data/optimal-parameters.csv'):
    raise FileNotFoundError('Optimal parameters csv file not found.')

# Load the optimal parameters DataFrame from the csv file
df = pd.read_csv('data/optimal-parameters.csv')


def run_qaoa_script(track_mlflow, graph_type, node_size, quant_alg, n_layers=1):

    if track_mlflow:
        # Configure MLFlow Stuff
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        experiment_name = "QAOA-Instance-Based-Parameter-Optimization"
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

    # ### Brute Force Solution for the Max-Cut Problem
    # A brute-force solution to the Max-Cut problem involves evaluating every possible partition of the graph's nodes into two sets.
    # We calculate the 'cut' for each partition, which is the number of edges between the two sets. The goal is to maximize this cut.
    # NOTE: This method is computationally intensive and not practical for large graphs,
    # but it gives an exact solution for smaller ones.
    logging.info(f"\n{'-'*10} Solving for Exact Ground State {'-'*10}\n")
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

    if track_mlflow:
        mlflow.log_metric("ground_state_energy", exact_result.eigenvalue.real)

    logging.info(f"\n{'-'*10} Simulating Instance on Quantum{'-'*10}\n")
    N_LAYERS = n_layers

    # Initialize the optimizer and backend for the Quantum Algorithm
    optimizer = ADAM()
    backend = Aer.get_backend("aer_simulator_statevector")
    quantum_instance = QuantumInstance(backend)

    logging.info(f"Using Classical Optimizer {type(optimizer).__name__}")

    # Initialise Quantum Algorithm list
    algos_initializations = []

    #### QAOA with Instance Based Initialization

    # Get instance optimised paramters
    optimal_params = get_optimal_parameters(instance_class, n_layers, df)
    # Check if optimal parameters were found
    if isinstance(optimal_params, str):
        logging.warning(optimal_params)
    else:
        optimal_beta = np.array(optimal_params['beta'])
        optimal_gamma = np.array(optimal_params['gamma'])
        initial_point_optimal = np.concatenate([optimal_beta, optimal_gamma])
        # Add QAOA with optimal initialization
        algos_initializations.append(
            ('QAOA', initial_point_optimal, 'instance_class_optimsed')
        )

    # Add QAOA with random initialization as well
    initial_point_random = np.concatenate(
        [
            np.random.uniform(-np.pi / 4, np.pi / 4, N_LAYERS),
            np.random.uniform(-np.pi / 2, np.pi / 2, N_LAYERS),
        ]
    )

    #### QAOA with 3-regular graph initialization

    # Add QAOA with 3-regular graph initialization
    optimal_params = get_optimal_parameters('three_regular_graph', n_layers, df)
    # Check if optimal parameters were found
    if isinstance(optimal_params, str):
        logging.warning(optimal_params)
    else:
        optimal_beta = np.array(optimal_params['beta'])
        optimal_gamma = np.array(optimal_params['gamma'])
        initial_point_optimal = np.concatenate([optimal_beta, optimal_gamma])
        # Add QAOA with optimal initialization
        algos_initializations.append(
            ('QAOA', initial_point_optimal, 'three_regular_graph_optimised')
        )

    #### QAOA with TQA (Trotterised Quantum Annealing) initialization

    # Add QAOA with TQA initialization
    initial_point_tqa = Initialisation().trotterized_quantum_annealing(n_layers)
    algos_initializations.append(('QAOA', initial_point_tqa, 'tqa_initialisation'))

    #### QAOA with random initialization

    # Add QAOA with random initialization as well
    algos_initializations.append(
        ('QAOA', initial_point_random, 'random_initialisation')
    )

    # Initialise empty dataframe to store results for each algorithm and init type (for evolution at each time step)
    results_df = pd.DataFrame(
        columns=['algo', 'init_type', 'eval_count', 'parameters', 'energy', 'std']
    )

    # Loop through each algorithm and initialization
    for algo_name, initial_point, init_type in algos_initializations:
        logging.info(f"Running {algo_name} with {init_type} Initialization")

        # Print initial values from algorithm
        logging.info(f"Initial point ({algo_name} - {init_type}): {initial_point}")

        # Callback function to store intermediate values
        intermediate_values = []

        def store_intermediate_result(eval_count, parameters, mean, std):
            if eval_count % 100 == 0:
                logging.info(
                    f"{type(optimizer).__name__} iteration {eval_count} \t cost function {mean}"
                )
            betas = parameters[:N_LAYERS]  # Extracting beta values
            gammas = parameters[N_LAYERS:]  # Extracting gamma values
            intermediate_values.append(
                {
                    'eval_count': eval_count,
                    'parameters': {'gammas': gammas, 'betas': betas},
                    'mean': mean,
                    'std': std,
                }
            )

        # Initialize the algorithm based on its name (e.g., QAOA)
        if algo_name == 'QAOA':
            qaoa = QAOA(
                optimizer=optimizer,
                reps=N_LAYERS,
                initial_point=initial_point,
                callback=store_intermediate_result,
                quantum_instance=quantum_instance,
                include_custom=True,
            )
            algo_result = qaoa.compute_minimum_eigenvalue(qubitOp)
        else:
            # Add initialization for other algorithms here (could start off with VQE here too)
            pass

        # Compute performance metrics
        eval_counts = [
            intermediate_result['eval_count']
            for intermediate_result in intermediate_values
        ]
        most_likely_solution = max_cut.sample_most_likely(algo_result.eigenstate)
        # Calculate the energy gap
        energy_gap = exact_result.eigenvalue.real - algo_result.eigenvalue.real

        # Convert exact result eigenstate to matrix form and get QAOA state
        exact_result_vector = exact_result.eigenstate.to_matrix()
        qaoa_state_vector = algo_result.eigenstate

        # Compute inner product between exact result and QAOA state
        inner_product = np.dot(exact_result_vector.conj(), qaoa_state_vector)

        # Calculate the probability of success (adjusting for MAXCUT symmetry)
        success_probability = (np.abs(inner_product) ** 2) * 2

        # Calculate the approximation ratio
        approximation_ratio = algo_result.eigenvalue.real / exact_result.eigenvalue.real

        # Calculate Distance
        distance = compute_distance(
            N_LAYERS,
            initial_point[:N_LAYERS],
            algo_result.optimal_point[:N_LAYERS],
            initial_point[N_LAYERS:],
            algo_result.optimal_point[N_LAYERS:],
        )
        logging.info(f"Distance between initial point and optimal point: {distance}")

        # Compile results into a dataframe from intermediate values
        results_df = results_df.append(
            pd.DataFrame(
                {
                    'algo': [algo_name] * len(intermediate_values),
                    'init_type': [init_type] * len(intermediate_values),
                    'eval_count': [
                        intermediate_result['eval_count']
                        for intermediate_result in intermediate_values
                    ],
                    'parameters': [
                        intermediate_result['parameters']
                        for intermediate_result in intermediate_values
                    ],
                    'energy': [
                        intermediate_result['mean']
                        for intermediate_result in intermediate_values
                    ],
                    'std': [
                        intermediate_result['std']
                        for intermediate_result in intermediate_values
                    ],
                }
            )
        )

        # Log results to MLFlow
        if track_mlflow:
            mlflow.log_param(f"{algo_name}_{init_type}_initial_point", initial_point)
            mlflow.log_metric(
                f"{algo_name}_{init_type}_final_energy", algo_result.eigenvalue.real
            )
            # Convert array to string for logging
            most_likely_solution = np.array2string(most_likely_solution)
            mlflow.log_param(
                f"{algo_name}_{init_type}_most_likely_solution", most_likely_solution
            )
            mlflow.log_metric(
                f"{algo_name}_{init_type}_success_probability", success_probability
            )
            mlflow.log_metric(
                f"{algo_name}_{init_type}_approximation_ratio", approximation_ratio
            )
            mlflow.log_metric(f"{algo_name}_{init_type}_energy_gap", energy_gap)
            # log number of iterations
            mlflow.log_metric(
                f"{algo_name}_{init_type}_num_iterations", len(eval_counts)
            )
            # Log distance between initial point and optimal point
            mlflow.log_metric(f"{algo_name}_{init_type}_distance", distance)
            # Log each optimal parameter in MLFlow
            for i, (beta, gamma) in enumerate(
                zip(
                    algo_result.optimal_point[:N_LAYERS],
                    algo_result.optimal_point[N_LAYERS:],
                )
            ):
                mlflow.log_metric(f"{algo_name}_{init_type}_optimal_beta_{i}", beta)
                mlflow.log_metric(f"{algo_name}_{init_type}_optimal_gamma_{i}", gamma)

        logging.info(f"Results with {algo_name} {init_type} Initialization:")
        logging.info(
            f"Final energy for ({algo_name} {init_type})<C>: {algo_result.eigenvalue.real}"
        )
        logging.info(
            f"Most likely solution ({algo_name} {init_type}): {most_likely_solution}"
        )
        logging.info(
            f"Probability of success ({algo_name} {init_type}): {success_probability}"
        )
        logging.info(
            f"Approximation ratio ({algo_name} {init_type}): {approximation_ratio}"
        )
        logging.info(f"Energy gap ({algo_name} {init_type}): {energy_gap}")
        logging.info(
            f"Number of iterations ({algo_name} {init_type}): {len(eval_counts)}"
        )
        logging.info(
            f"Distance between initial point and optimal point ({algo_name} {init_type}): {distance}"
        )

    # Add column for approximation ratio
    results_df['approximation_ratio'] = (
        results_df['energy'] / exact_result.eigenvalue.real
    )

    # Save results dataframe to csv and log to mlflow (via tempdir)
    with make_temp_directory() as tmp_dir:
        results_df.to_csv(os.path.join(tmp_dir, 'results.csv'))
        if track_mlflow:
            mlflow.log_artifact(os.path.join(tmp_dir, 'results.csv'))

        # Plot energy vs iterations for each algorithm and initialization on a single chart
        plt.figure(figsize=(12, 8))
        for algo_name, initial_point, init_type in algos_initializations:
            # Filter results for specific algorithm and initialization
            filtered_results_df = results_df[
                (results_df['algo'] == algo_name)
                & (results_df['init_type'] == init_type)
            ]
            # Plot energy vs iterations
            plt.plot(
                filtered_results_df['eval_count'],
                filtered_results_df['energy'],
                label=f"{algo_name} {init_type}",
            )
        # Add dashed line for exact ground state energy
        plt.axhline(
            y=exact_result.eigenvalue.real,
            color='r',
            linestyle='--',
            label='Exact Ground State Energy',
        )
        plt.xlabel('Iterations')
        plt.ylabel('Energy')
        plt.legend()
        plt.savefig(os.path.join(tmp_dir, 'energy_vs_iterations.png'))
        if track_mlflow:
            mlflow.log_artifact(os.path.join(tmp_dir, 'energy_vs_iterations.png'))
        # Clear plots
        plt.clf()

        # Plot approximation ratio vs iterations for each algorithm and initialization on a single chart
        plt.figure(figsize=(12, 8))
        for algo_name, initial_point, init_type in algos_initializations:
            # Filter results for specific algorithm and initialization
            filtered_results_df = results_df[
                (results_df['algo'] == algo_name)
                & (results_df['init_type'] == init_type)
            ]
            # Plot approximation ratio vs iterations
            plt.plot(
                filtered_results_df['eval_count'],
                filtered_results_df['approximation_ratio'],
                label=f"{algo_name} {init_type}",
            )
        # Add dashed line for approximation ratio of 1
        plt.axhline(y=1, color='r', linestyle='--', label='Approximation Ratio of 1')
        plt.xlabel('Iterations')
        plt.ylabel('Approximation Ratio')
        plt.legend()
        plt.savefig(os.path.join(tmp_dir, 'approximation_ratio_vs_iterations.png'))
        if track_mlflow:
            mlflow.log_artifact(
                os.path.join(tmp_dir, 'approximation_ratio_vs_iterations.png')
            )

    # Logging the Parameter Landscape
    logging.info(f"\n{'-'*10} Logging the Parameter Landscape {'-'*10}\n")
    # Note this can take some time
    logging.info(f"Logging the Parameter Landscape for {graph_type}")

    # Landscape Analysis of Instance (at p=1)
    qaoa = QAOA(optimizer=ADAM(), reps=1)
    # Use constrained search space
    beta = np.linspace(-np.pi / 2, np.pi / 2, 40)
    gamma = np.linspace(-np.pi / 2, np.pi / 2, 40)
    # Example usage
    obj_vals = parallel_computation(beta, gamma, qubitOp, qaoa)
    # ### Plotting the Parameter Landscape https://ar5iv.labs.arxiv.org/html/2209.01159#A5
    # The heatmap below represents the landscape of the objective function across
    # different values of \$\gamma\$ and \$\beta\$.
    # The color intensity indicates the expectation value of the Hamiltonian,
    # helping identify the regions where optimal parameters may lie.
    with make_temp_directory() as tmp_dir:
        Beta, Gamma = np.meshgrid(beta, gamma)
        # Plotting
        plt.figure(figsize=(10, 8))
        cp = plt.contourf(
            Beta, Gamma, obj_vals.T, cmap='viridis'
        )  # Transpose obj_vals if necessary
        plt.colorbar(cp)
        plt.title(f'QAOA Objective Function Landscape (p=1) for  {graph_type}')
        plt.xlabel('Beta')
        plt.ylabel('Gamma')
        # Adjust the x and y limits to show the new range
        plt.xlim(-np.pi / 4, np.pi / 4)
        plt.ylim(-np.pi / 2, np.pi / 2)
        # Adjust the x and y labels to show the new pi values
        plt.xticks([-np.pi / 2, 0, np.pi / 2], [r'$-\pi/2$', r'$0$', r'$\pi/2$'])
        plt.yticks([-np.pi / 2, 0, np.pi / 2], [r'$-\pi/2$', r'$0$', r'$\pi/2$'])

        # Add markers for the optimal parameters from each initialization point
        for index, (algo_name, initial_point, init_type) in enumerate(
            algos_initializations
        ):
            # Get initialized parameters from the initial point
            beta_init = initial_point[:N_LAYERS][0]  # Only take first value for beta
            gamma_init = initial_point[N_LAYERS:][0]  # Only take first value for gamma

            # Use a different marker and color for each initialization
            plt.plot(
                beta_init,
                gamma_init,
                color=colors[index],
                marker=markers[index],
                markersize=10,
                label=f"{algo_name} {init_type}",
            )

        # Adjust legend position
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(algos_initializations),
            fancybox=True,
            shadow=True,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the plot to fit the legend
        plt.savefig(os.path.join(tmp_dir, 'landscape_plot.png'))
        if track_mlflow:
            mlflow.log_artifact(os.path.join(tmp_dir, 'landscape_plot.png'))
        # Clear plots
        plt.clf()


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
