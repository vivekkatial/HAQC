# Hide warnings
from logging import FATAL
import warnings

warnings.filterwarnings("ignore")

# Standard Libraries
import argparse
import json
import time
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
import mlflow
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Custom Libraries
import src.build_graph
import src.features.graph_features
import src.features.tsp_features
import src.build_circuit
import src.clustering
import src.utils
from src.exp_utils import str2bool, make_temp_directory
from src.quantum_burden import compute_quantum_burden
from src.classical.greedy_tsp import greedy_tsp
from src.plot.draw_euclidean_graphs import draw_euclidean_graph
from src.plot.feasibility_graph import plot_feasibility
from src.features.graph_features import get_graph_features
from src.features.tsp_features import get_tsp_features

# QISKIT stuff
from qiskit import Aer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, NELDER_MEAD, SPSA
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import MinimumEigenOptimizer
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.optimization.applications.ising import tsp

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_instance(filename, budget: int, p_max=10, mlflow_tracking=False):
    instance_path = "data/{}".format(filename)
    with open(instance_path) as f:
        data = json.load(f)
    G, depot_info = src.build_graph.build_json_graph(data["graph"])
    num_vehicles = int(data["numVehicles"])
    threshold = float(data["threshold"])
    n_max = int(data["n_max"])
    # Read in parameter file
    with open("config/mlflow_config.json") as file:
        params = json.load(file)

    if mlflow_tracking:
        # Configure MLFlow Stuff
        mlflow.set_tracking_uri(params["experiment"]["tracking-uri"])
        mlflow.set_experiment(params["experiment"]["name"])

        # Build Graph Feature Vector
        feature_vector = get_graph_features(G)
        # Build TSP Feature Vector
        tsp_feature_vector = get_tsp_features(G)
        # Add num vehicles
        feature_vector["num_vehicles"] = num_vehicles

        # Log Params
        mlflow.log_params(feature_vector)
        mlflow.log_params(tsp_feature_vector)
        mlflow.log_param("source", data["instance_type"])
        mlflow.log_param("instance_uuid", filename)
        mlflow.log_param("p_max", p_max)
        # Log instance
        mlflow.log_artifact(instance_path)

    edge_mat = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    cost_mat = np.array(nx.attr_matrix(G, edge_attr="cost", rc_order=list(G.nodes())))
    for edge in G.edges():
        G[edge[0]][edge[1]]["cost"] = 0

    edge_mat = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    cost_mat = np.array(nx.attr_matrix(G, edge_attr="cost", rc_order=list(G.nodes())))

    G, cluster_mapping = src.clustering.create_clusters(
        G, num_vehicles, "spectral-clustering", edge_mat
    )

    depot_edges = list(G.edges(depot_info["id"], data=True))
    depot_node = depot_info["id"]

    subgraphs = src.clustering.build_sub_graphs(G, depot_node, depot_edges)

    # big_offset = sum(sum(cost_mat))/2 + 1
    big_offset = 30
    qubos = src.build_circuit.build_qubos(subgraphs, depot_info, A=big_offset)

    cluster_mapping = [i + 1 for i in cluster_mapping]
    cluster_mapping.insert(0, 0)

    qubo = qubos[0]

    print("Running single tsp for Qubo")
    print(qubo)

    single_qubo_solution_data = {}
    single_qubo_solution_data["qubo_id"] = 0
    single_qubo_solution_data["cluster"] = [
        index for index, node in enumerate(cluster_mapping) if node == 1 or node == 0
    ]

    op, offset = qubo.to_ising()

    print("Offset:", offset)
    print("Ising Hamiltonian:")
    print(op.print_details())

    qp = QuadraticProgram()
    qp.from_ising(op, offset, linear=True)
    qp.to_docplex().prettyprint()

    exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    exact_result = exact.solve(qp)
    print(exact_result)

    ee = NumPyMinimumEigensolver(op)
    exact_result = ee.run()

    print("energy:", exact_result.eigenvalue.real)
    if mlflow_tracking:
        mlflow.log_metric("ground_state_energy", exact_result.eigenvalue.real)
    print("tsp objective:", exact_result.eigenvalue.real + offset)
    x = sample_most_likely(exact_result.eigenstate)
    print("feasible:", tsp.tsp_feasible(x))
    z = tsp.get_tsp_solution(x)
    print("solution:", z)
    print("solution objective:", tsp.tsp_value(z, cost_mat))

    if mlflow_tracking:
        mlflow.log_metric("ground_state_energy", exact_result.eigenvalue.real)
        mlflow.log_param("solution_objective", z)

    print("Quantum Optimisation Starting")
    # Quantum solution
    p = 1
    while p < p_max:

        print(f"Quantum Optimisation Starting for p={p}")

        # Initialise each budget parameter
        if p > 5:
            budget = 2000

        optimizers = [
            # SLSQP(maxiter=budget, disp=True, eps=0.001),
            COBYLA(maxiter=budget, disp=True, rhobeg=0.1),
            # NELDER_MEAD(maxfev=budget, disp=True, adaptive=True),
            # L_BFGS_B(maxfun=budget, factr=10, epsilon=0.001, iprint=100),
        ]

        # Make convergence counts and values
        converge_cnts = np.empty([len(optimizers)], dtype=object)
        converge_vals = np.empty([len(optimizers)], dtype=object)
        min_energy_vals = np.empty([len(optimizers)], dtype=object)
        min_energy_states = np.empty([len(optimizers)], dtype=object)
        backend = Aer.get_backend("aer_simulator_matrix_product_state")
        # mps_algo = "mps_apply_measure"
        # mps_algo = "mps_probabilities"
        print(f"Setting MPS Sample Measure Algorithim to be: {mps_algo}")
        backend.set_option("mps_sample_measure_algorithm", mps_algo)

        for i, optimizer in enumerate(optimizers):
            print("\rOptimizer: {}        ".format(type(optimizer).__name__))
            counts = []
            values = []
            # Run energy and results
            run_energy = []
            run_results = []
            global global_count
            global_count = 0
            n_restart = 0

            def store_intermediate_result(eval_count, parameters, mean, std):
                global global_count
                global_count += 1
                counts.append(eval_count)
                values.append(mean)

            while global_count < budget:

                # Increment n_restarts
                n_restart += 1
                # Initiate a random point uniformly from [0,1]
                initial_point = [np.random.uniform(0, 1) for i in range(2 * p)]
                # Set random seed
                aqua_globals.random_seed = np.random.default_rng(123)
                seed = 10598
                # Initate quantum instance
                quantum_instance = QuantumInstance(
                    backend,
                    seed_simulator=seed,
                    seed_transpiler=seed,
                )
                # Initate QAOA
                qaoa = QAOA(
                    operator=op,
                    optimizer=optimizer,
                    callback=store_intermediate_result,
                    p=p,
                    initial_point=initial_point,
                    quantum_instance=quantum_instance,
                )

                # Compute the QAOA result
                result = qaoa.compute_minimum_eigenvalue(operator=op)

                # Store in temp run info
                run_energy.append(result.eigenvalue.real)
                run_results.append(result.eigenstate)

                # Append the minimum value from temp run info into doc
                min_energy_vals[i] = min(run_energy)
                if mlflow_tracking:
                    mlflow.log_metric(
                        key=f"{type(optimizer).__name__}_p_{p}", value=min(run_energy)
                    )

                min_energy_ind = run_energy.index(min(run_energy))
                min_energy_states[i] = run_results[min_energy_ind]
                converge_cnts[i] = np.asarray(counts)
                converge_vals[i] = np.asarray(values)

        # Create a dictionary for results
        results_dict = {"optimizer": None, "n_eval": None, "value": None}

        # Dictionary for different optimisers we're exploring
        optimizer_dict = {
            # 0: "SLSQP",
            0: "COBYLA",
            # 0: "NELDER_MEAD",
            # 4: "L_BFGS_B"
        }

        d_results = []

        for i, (evals, values) in enumerate(zip(converge_cnts, converge_vals)):
            for cnt, val in zip(evals, values):
                results_dict_temp = results_dict.copy()
                results_dict_temp["n_eval"] = cnt
                results_dict_temp["value"] = val
                results_dict_temp["optimizer"] = optimizer_dict[i]
                d_results.append(results_dict_temp)

        d_results = pd.DataFrame.from_records(d_results)

        # Add counter for num_evals
        d_results["total_evals"] = d_results.groupby("optimizer").cumcount()

        with make_temp_directory() as temp_dir:
            results_layer_p_fn = f"results_large_offset_p_{p}.csv"
            results_layer_p_fn = os.path.join(temp_dir, results_layer_p_fn)

            # Create plots
            g = sns.relplot(
                data=d_results,
                x="total_evals",
                y="value",
                col="optimizer",
                hue="optimizer",
                kind="line",
            )

            (
                g.map(
                    plt.axhline, y=-180, color=".7", dashes=(2, 1), zorder=0
                ).tight_layout(w_pad=0)
            )

            optimization_plot_p_fn = f"optimization_plot_p_{p}.png"
            optimization_plot_p_fn = os.path.join(temp_dir, optimization_plot_p_fn)
            g.savefig(optimization_plot_p_fn)

            plt.clf()
            for ind in optimizer_dict.keys():
                # Make feasibility graph
                feasibility_p = plot_feasibility(min_energy_states[ind], exact_result)
                feasibility_p_fn = (
                    f"feasibility_plot_opt_{optimizer_dict[ind]}_p_{p}.png"
                )
                feasibility_p_fn = os.path.join(temp_dir, feasibility_p_fn)
                feasibility_p.figure.savefig(
                    feasibility_p_fn, dpi=300, bbox_inches="tight"
                )

            # Write results
            with open(results_layer_p_fn, "w") as file:
                d_results.to_csv(file)

            if mlflow_tracking:
                mlflow.log_artifact(results_layer_p_fn)
                mlflow.log_artifact(feasibility_p_fn)
                mlflow.log_artifact(optimization_plot_p_fn)

        # Increment p
        p += 1

    print("\rOptimization complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=True,
        help="Filename of the instance graph to be run (file must be a .json)",
    )

    parser.add_argument(
        "-b",
        "--budget",
        type=int,
        default=100,
        help="The budget for the optimziations (max function evals)",
    )

    parser.add_argument(
        "-p",
        "--p_max",
        type=int,
        default=2,
        help="The number of layers,p, to compute for",
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
    args = vars(parser.parse_args())
    t1 = time.time()
    print(f"Run starting at: {time.ctime()}")
    run_instance(
        args["filename"],
        budget=args["budget"],
        p_max=args["p_max"],
        mlflow_tracking=args["track_mlflow"],
    )
    t2 = time.time()
    print("Run complete in: {} seconds".format(round(t2 - t1, 3)))
