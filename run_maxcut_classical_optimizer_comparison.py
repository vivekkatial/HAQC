import os
import logging

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

# Adjust Qiskit's logger to only display errors or critical messages
qiskit_logger = logging.getLogger('qiskit')
qiskit_logger.setLevel(logging.ERROR)  # or use logging.CRITICAL
import warnings

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
from qiskit.algorithms.optimizers import (
    ADAM,
    AQGD,
    CG,
    COBYLA,
    GradientDescent,
    L_BFGS_B,
    NELDER_MEAD,
    NFT,
    POWELL,
    SLSQP,
    SPSA,
    TNC,
)
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.utils import QuantumInstance
from qiskit_optimization.applications import Maxcut

# Custom imports
from src.haqc.generators.graph_instance import create_graphs_from_all_sources, GraphInstance
from src.haqc.exp_utils import (
    str2bool,
    to_snake_case,
    make_temp_directory,
    check_boto3_credentials,
    find_instance_class
)

from src.haqc.features.graph_features import get_graph_features
from src.haqc.solutions.solutions import compute_max_cut_brute_force, compute_distance
from src.haqc.initialisation.initialisation import Initialisation
from src.haqc.plot.utils import *
from src.haqc.plot.approximation_ratio import (
    plot_approx_ratio_vs_iterations_for_optimizers,
)

# Theme plots to be seaborn style
plt.style.use('seaborn-white')
total_fevals = None


def run_qaoa_script(
    track_mlflow, graph_type, node_size, quant_alg, n_layers, max_feval
):
    global total_fevals

    

    if track_mlflow:
        # Configure MLFlow Stuff
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        experiment_name = "QAOA-Classical-Optimization"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    # if graph type is a `.pkl` file, load the graph from the file
    if graph_type.endswith(".pkl"):
        G = nx.read_gpickle(graph_type)
        # Extract the graph type from the file name
        G.graph_type = find_instance_class(graph_type)
        logging.info(
            f"\n{'-'*10} This run is for a custom graph with {len(G.nodes())} nodes of source {G.graph_type}  {'-'*10}\n"
        )
        graph_instance = GraphInstance(G, G.graph_type)
        if track_mlflow:
            mlflow.log_param("custom_graph", True)
    else:
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
        mlflow.log_param("max_feval", max_feval)
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

    # Initialize the optimizers and backend for the Quantum Algorithm
    backend = Aer.get_backend("aer_simulator_statevector")
    quantum_instance = QuantumInstance(backend)
    # Initialise Quantum Algorithm list
    algos_optimizers = [
        ('QAOA', 'COBYLA', COBYLA(tol=0.0001, maxiter=max_feval)),
        ('QAOA', 'ADAM', ADAM(tol=0.0001, lr=0.01, maxiter=max_feval)),
        ('QAOA', 'NELDER_MEAD', NELDER_MEAD(tol=0.0001, maxfev=max_feval)),
        (
            'QAOA',
            'SPSA',
            SPSA(
                maxiter=max_feval,
                blocking=True,
                learning_rate=0.01,
                perturbation=0.001,
                second_order=True,
            ),
        ),
        ('QAOA', 'L_BFGS_B', L_BFGS_B(maxfun=max_feval)),
        ('QAOA', 'GradientDescent', GradientDescent(maxiter=max_feval, tol=0.0001)),
        ('QAOA', 'CG', CG(maxiter=max_feval)),
        # Adding missing optimizers with basic configurations
        (
            'QAOA',
            'AQGD',
            AQGD(
                maxiter=max_feval,
                eta=0.5,
                tol=1e-5,
                momentum=0.25,
                param_tol=1e-5,
                averaging=50,
            ),
        ),
        # ('QAOA', 'GSLS', GSLS(max_eval=max_feval, max_failed_rejection_sampling=100)),
        (
            'QAOA',
            'NFT',
            NFT(
                maxfev=max_feval,
            ),
        ),
        ('QAOA', 'POWELL', POWELL(maxfev=max_feval, xtol=0.00001)),
        ('QAOA', 'SLSQP', SLSQP(maxiter=max_feval)),
        ('QAOA', 'TNC', TNC(maxiter=max_feval)),
    ]

    # Loop through each algorithm and initialization
    # Initialise empty dataframe to store results for each algorithm and init type (for evolution at each time step)
    results_df = pd.DataFrame(
        columns=['layers', 'optimizer', 'eval_count', 'parameters', 'energy', 'std']
    )

    for algo_name, optimizer_name, optimizer in algos_optimizers:
        print(f"{'-'*15} Solving for Optimizer: {optimizer_name} {'-'*15}")
        initial_point = Initialisation().random_initialisation(n_layers)

        # Callback function to store intermediate values
        intermediate_values = []
        total_feval = 0
        n_restart = 0

        while total_feval < max_feval:
            print(f"{' '*5 + '-'*10} Solving at restart: {n_restart} {'-'*15}")

            def store_intermediate_result(eval_count, parameters, mean, std):
                global total_fevals  # Refer to the global variable
                if eval_count >= max_feval:
                    total_fevals = eval_count
                    return

                if eval_count % 100 == 0:
                    logging.info(
                        f"{optimizer_name} iteration {eval_count} \t cost function {mean}"
                    )
                betas = parameters[:n_layers]  # Extracting beta values
                gammas = parameters[n_layers:]  # Extracting gamma values
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
                optimizer=optimizer,
                reps=n_layers,
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

            initial_point = Initialisation().random_initialisation(n_layers)

            n_restart += 1

            # If the algorithm has had more iterations than the maximum, continue to the next algorithm
            if total_feval >= max_feval:
                logging.info(
                    f"Maximum number of iterations reached. Continuing to next algorithm."
                )
                continue

        # Logging that optimization has finished
        logging.info(f"Optimization finished for {optimizer_name}")
        logging.info(f"Total number of evaluations: {total_feval}")
        logging.info(f"Number of restarts: {n_restart}")

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
                    'optimizer': [optimizer_name] * len(intermediate_values),
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
            mlflow.log_param(
                f"{algo_name}_{optimizer_name}_initial_point", initial_point
            )
            mlflow.log_metric(
                f"{algo_name}_{optimizer_name}_final_energy",
                algo_result.eigenvalue.real,
            )
            # Convert array to string for logging
            most_likely_solution = np.array2string(most_likely_solution)
            mlflow.log_param(
                f"{algo_name}_{optimizer_name}_most_likely_solution",
                most_likely_solution,
            )
            mlflow.log_metric(
                f"{algo_name}_{optimizer_name}_success_probability", success_probability
            )
            mlflow.log_metric(
                f"{algo_name}_{optimizer_name}_approximation_ratio", approximation_ratio
            )
            mlflow.log_metric(f"{algo_name}_{optimizer_name}_energy_gap", energy_gap)
            # log number of iterations
            mlflow.log_metric(
                f"{algo_name}_{optimizer_name}_num_iterations", len(eval_counts)
            )
            # Log distance between initial point and optimal point
            mlflow.log_metric(f"{algo_name}_{optimizer_name}_distance", distance)
            # Log each optimal parameter in MLFlow
            for i, (beta, gamma) in enumerate(
                zip(
                    algo_result.optimal_point[:n_layers],
                    algo_result.optimal_point[n_layers:],
                )
            ):
                mlflow.log_metric(
                    f"{algo_name}_{optimizer_name}_optimal_beta_{i}", beta
                )
                mlflow.log_metric(
                    f"{algo_name}_{optimizer_name}_optimal_gamma_{i}", gamma
                )

            logging.info(f"Results with {algo_name} {optimizer_name}:")
            logging.info(
                f"Final energy for ({algo_name} {optimizer_name})<C>: {algo_result.eigenvalue.real}"
            )
            logging.info(
                f"Most likely solution ({algo_name} {optimizer_name}): {most_likely_solution}"
            )
            logging.info(
                f"Probability of success ({algo_name} {optimizer_name}): {success_probability}"
            )
            logging.info(
                f"Approximation ratio ({algo_name} {optimizer_name}): {approximation_ratio}"
            )
            logging.info(f"Energy gap ({algo_name} {optimizer_name}): {energy_gap}")
            logging.info(
                f"Number of iterations ({algo_name} {optimizer_name}): {len(eval_counts)}"
            )
            logging.info(
                f"Distance between initial point and optimal point ({algo_name} {optimizer_name}): {distance}"
            )

    # Add column for approximation ratio
    results_df['approximation_ratio'] = (
        results_df['energy'] / exact_result.eigenvalue.real
    )

    results_df['layers'] = N_LAYERS
    results_df['total_count'] = results_df.groupby('optimizer').cumcount() + 1

    # Filter out the results where total eval count is greater than max_feval
    results_df = results_df[results_df['total_count'] <= max_feval]
    max_approx_ratio = results_df['approximation_ratio'].max()
    acceptable_approx_ratio = 0.95 * max_approx_ratio

    performance_dict = {}  # This will store the performance for each optimizer

    for optimizer in results_df['optimizer'].unique():
        # Filter the DataFrame for the current optimizer
        optimizer_df = results_df[results_df['optimizer'] == optimizer]

        # Find the minimum eval_count where approximation_ratio >= acceptable_approx_ratio
        acceptable_df = optimizer_df[
            optimizer_df['approximation_ratio'] >= acceptable_approx_ratio
        ]

        if not acceptable_df.empty:
            min_eval_count = acceptable_df['total_count'].min()
            performance_dict[f'algo_{optimizer}'] = min_eval_count
        else:
            # Assign penalty score if the acceptable level is not reached
            performance_dict[f'algo_{optimizer}'] = 100000

    if track_mlflow:
        mlflow.log_metrics(performance_dict)

    # Save results dataframe to csv and log to mlflow (via tempdir)
    with make_temp_directory() as tmp_dir:
        results_df.to_csv(os.path.join(tmp_dir, 'results.csv'))
        if track_mlflow:
            mlflow.log_artifact(os.path.join(tmp_dir, 'results.csv'))

        # Set up the plot
        plot_approx_ratio_vs_iterations_for_optimizers(
            results_df,
            acceptable_approx_ratio,
            #'convergence_plot.png'
            os.path.join(tmp_dir, 'convergence_plot.png'),
        )

        if track_mlflow:
            mlflow.log_artifact(os.path.join(tmp_dir, 'convergence_plot.png'))


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

    parser.add_argument(
        "-f",
        "--max_feval",
        type=int,
        default=1000,
        help="Maximum number of function evaluations for the optimizer. Default is 1000.",
    )

    # Parse the arguments
    args = parser.parse_args()
    print(vars(args))

    # Start the timer
    start_time = time.time()

    # Run the QAOA script
    run_qaoa_script(
        track_mlflow=args.track_mlflow,
        graph_type=args.graph_type,
        node_size=args.node_size,
        quant_alg=args.quantum_algorithm,
        n_layers=args.n_layers,
        max_feval=args.max_feval,
    )

    # End the timer
    end_time = time.time()

    print(f"Result found in: {end_time - start_time:.3f} seconds")
