import os
import logging
import warnings

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

# Adjust Qiskit's logger to only display errors or critical messages
qiskit_logger = logging.getLogger('qiskit')
qiskit_logger.setLevel(logging.ERROR)  # or use logging.CRITICAL


logging.info('Script started')
# Ignore divide by zero and invalid value encountered in det warnings
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, message='divide by zero encountered in det'
)
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, message='invalid value encountered in det'
)


class OptimizationTermination(Exception):
    """Exception raised for terminating the optimization process."""

    def __init__(self, message="Optimization terminated early."):
        self.message = message
        super().__init__(self.message)


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import time
import mlflow
import json
import pandas as pd

from qiskit import Aer
from qiskit.algorithms.optimizers import NELDER_MEAD
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
from src.haqc.solutions.solutions import compute_max_cut_brute_force, compute_distance
from src.haqc.initialisation.initialisation import Initialisation
from src.haqc.plot.utils import *
from src.haqc.plot.approximation_ratio import plot_approx_ratio_vs_iterations_for_layers

# Theme plots to be seaborn style
plt.style.use('seaborn-white')


def run_qaoa_script(
    track_mlflow, graph_type, node_size, quant_alg, max_layers, max_feval
):

    logging.info(
        f"{'-'*10} Running QAOA Script for investigating number of layers {'-'*10}"
    )

    if track_mlflow:
        # Configure MLFlow Stuff
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        experiment_name = "QAOA-Number-of-Layers"
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
        mlflow.log_param("max_layers", max_layers)
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

    logging.info(f"Minimum Energy is {exact_result.eigenvalue.real}")

    if track_mlflow:
        mlflow.log_metric("ground_state_energy", exact_result.eigenvalue.real)

    # ### Quantum Approximate Optimization Algorithm (QAOA)
    backend = Aer.get_backend("aer_simulator_statevector")
    quantum_instance = QuantumInstance(backend)
    results_df = pd.DataFrame(
        columns=['algo', 'init_type', 'eval_count', 'parameters', 'energy', 'std']
    )
    results_df.head()

    for layer in range(1, max_layers + 1):
        logging.info(f"{'-'*15} Solving for layer: {layer} {'-'*15}")
        initial_point = Initialisation().random_initialisation(layer)

        # Callback function to store intermediate values
        intermediate_values = []
        total_feval = 0
        n_restart = 0

        while total_feval < max_feval:
            logging.info(f"{' '*5 + '-'*10} Solving at restart: {n_restart} {'-'*15}")

            def store_intermediate_result(eval_count, parameters, mean, std):
                if eval_count % 100 == 0:
                    logging.info(
                        f"{type(NELDER_MEAD()).__name__} iteration {eval_count} \t cost function {mean}"
                    )
                betas = parameters[:layer]  # Extracting beta values
                gammas = parameters[layer:]  # Extracting gamma values
                intermediate_values.append(
                    {
                        'eval_count': eval_count,
                        'parameters': {'gammas': gammas, 'betas': betas},
                        'mean': mean,
                        'std': std,
                    }
                )

            qaoa = QAOA(
                # Optimize only from the remaining  budget
                optimizer=NELDER_MEAD(maxfev=max_feval - total_feval),
                reps=layer,
                initial_point=initial_point,
                callback=store_intermediate_result,
                quantum_instance=quantum_instance,
                include_custom=True,
            )

            algo_result = qaoa.compute_minimum_eigenvalue(qubitOp)

            # Compute performance metrics
            eval_counts = [
                intermediate_result['eval_count']
                for intermediate_result in intermediate_values
            ]

            total_feval += eval_counts[-1]

            initial_point = Initialisation().random_initialisation(layer)

            n_restart += 1

            # Compile results into a dataframe from intermediate values
        results_df = results_df.append(
            pd.DataFrame(
                {
                    'algo': [layer] * len(intermediate_values),
                    'optimizer_name': [type(NELDER_MEAD()).__name__]
                    * len(intermediate_values),
                    'eval_count': [i + 1 for i in range(total_feval)],
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

        most_likely_solution = max_cut.sample_most_likely(algo_result.eigenstate)

        # Find minimum energy from results_df as the minimum energy for that layer
        min_energy = results_df[results_df['algo'] == layer]['energy'].min()
        # Calculate the energy gap
        energy_gap = exact_result.eigenvalue.real - min_energy

        # Convert exact result eigenstate to matrix form and get QAOA state
        exact_result_vector = exact_result.eigenstate.to_matrix()
        qaoa_state_vector = algo_result.eigenstate

        # Compute inner product between exact result and QAOA state
        inner_product = np.dot(exact_result_vector.conj(), qaoa_state_vector)

        # Calculate the probability of success (adjusting for MAXCUT symmetry)
        success_probability = (np.abs(inner_product) ** 2) * 2

        # Calculate the approximation ratio
        approximation_ratio = min_energy / exact_result.eigenvalue.real

        # Calculate Distance
        distance = compute_distance(
            layer,
            initial_point[:layer],
            algo_result.optimal_point[:layer],
            initial_point[layer:],
            algo_result.optimal_point[layer:],
        )
        logging.info(f"Distance between initial point and optimal point: {distance}")

        # Log results to MLFlow
        if track_mlflow:
            algo_name = f"QAOA_{layer}_layers"
            mlflow.log_param(f"{algo_name}_initial_point", initial_point)
            mlflow.log_metric(f"{algo_name}_final_energy", algo_result.eigenvalue.real)
            # Convert array to string for logging
            most_likely_solution = np.array2string(most_likely_solution)
            mlflow.log_param(f"{algo_name}_most_likely_solution", most_likely_solution)
            mlflow.log_metric(f"{algo_name}_success_probability", success_probability)
            mlflow.log_metric(f"{algo_name}_approximation_ratio", approximation_ratio)
            mlflow.log_metric(f"{algo_name}_energy_gap", energy_gap)
            # log number of iterations
            mlflow.log_metric(f"{algo_name}_num_iterations", len(eval_counts))
            # Log distance between initial point and optimal point
            mlflow.log_metric(f"{algo_name}_distance", distance)
            # Log each optimal parameter in MLFlow
            for i, (beta, gamma) in enumerate(
                zip(
                    algo_result.optimal_point[:layer],
                    algo_result.optimal_point[layer:],
                )
            ):
                mlflow.log_metric(f"{algo_name}_optimal_beta_{i}", beta)
                mlflow.log_metric(f"{algo_name}_optimal_gamma_{i}", gamma)

            logging.info(f"Results with {algo_name}:")
            logging.info(
                f"Final energy for ({algo_name})<C>: {algo_result.eigenvalue.real}"
            )
            logging.info(f"Most likely solution ({algo_name}): {most_likely_solution}")
            logging.info(f"Probability of success ({algo_name}): {success_probability}")
            logging.info(f"Approximation ratio ({algo_name}): {approximation_ratio}")
            logging.info(f"Energy gap ({algo_name}): {energy_gap}")
            logging.info(f"Number of iterations ({algo_name}): {len(eval_counts)}")
            logging.info(
                f"Distance between initial point and optimal point ({algo_name}): {distance}"
            )

    # Add column for the approximation ratio
    results_df['approx_ratio'] = results_df['energy'] / exact_result.eigenvalue.real

    # Find the highest value of the approximation ratio
    max_approx_ratio = results_df['approx_ratio'].max()

    # Compute an acceptable approximation ratio as 0.95 * this maximum
    acceptable_approx_ratio = 0.95 * max_approx_ratio

    # Compute performance
    performance_dict = (
        {}
    )  # Initialize a dictionary to hold the performance of each algorithm with prefixed keys

    for algo in results_df['algo'].unique():
        # Filter DataFrame for the current algorithm
        algo_df = results_df[results_df['algo'] == algo]

        # Find the minimum eval_count where the approx_ratio is greater than or equal to the acceptable approximation ratio
        sufficient_evals = algo_df[algo_df['approx_ratio'] >= acceptable_approx_ratio][
            'eval_count'
        ]

        algo_key = f"algo_{algo}_perf"  # Prefix the algo identifier with "algo_" and suffix with "_perf"
        if sufficient_evals.empty:
            # If no such evaluation count exists, assign penalty score
            performance_dict[algo_key] = 100000
        else:
            # Otherwise, assign the minimum eval_count that meets or exceeds the acceptable approximation ratio
            performance_dict[algo_key] = sufficient_evals.min()

    # Plot the approximation ratio vs. iterations for each algorithm
    plot_approx_ratio_vs_iterations_for_layers(
        results_df, max_layers, f"n_layers_convergence.png"
    )
    # Track to MLFlow
    if track_mlflow:
        mlflow.log_metrics(performance_dict)
        # Make a plot of the approximation ratio vs. iterations and log it to MLFlow
        with make_temp_directory() as tmp_dir:
            results_df.to_csv(os.path.join(tmp_dir, "results_df.csv"), index=False)
            mlflow.log_artifact(os.path.join(tmp_dir, "results_df.csv"))

            plot_approx_ratio_vs_iterations_for_layers(
                results_df, max_layers, f"{tmp_dir}/approx_ratio_vs_iterations.png"
            )
            mlflow.log_artifact(os.path.join(tmp_dir, "approx_ratio_vs_iterations.png"))


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
        "-l", "--max_layers", type=int, default=1, help="Number of max layers for QAOA"
    )

    parser.add_argument(
        "-f",
        "--max_feval",
        type=int,
        default=1e3,
        help="Maximum number of function evaluations for the optimizer. Default is 1000.",
    )

    args = parser.parse_args()
    print(vars(args))

    start_time = time.time()
    run_qaoa_script(
        track_mlflow=args.track_mlflow,
        graph_type=args.graph_type,
        node_size=args.node_size,
        quant_alg=args.quantum_algorithm,
        max_layers=args.max_layers,
        max_feval=args.max_feval,
    )
    end_time = time.time()

    print(f"Result found in: {end_time - start_time:.3f} seconds")
