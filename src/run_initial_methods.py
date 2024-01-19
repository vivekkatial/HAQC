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
import concurrent.futures
import mlflow
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathos.pools as pp

plt.style.use("seaborn")

# Custom Libraries
import src.build_graph
import src.features.graph_features
import src.features.tsp_features
import src.build_circuit
import src.clustering
import src.utils
from src.exp_utils import str2bool, make_temp_directory
from src.plot.feasibility_graph import (
    plot_feasibility_results,
    generate_feasibility_results,
)
from src.initialisation.initialisation import Initialisation
from src.features.graph_features import get_graph_features
from src.features.tsp_features import get_tsp_features
from src.parallel.optimize_qaoa import run_qaoa_parallel_control_max_restarts

# Import Qiskit Dependencies
from qiskit import Aer
from qiskit.aqua import aqua_globals
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import NELDER_MEAD, COBYLA
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import MinimumEigenOptimizer
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.optimization.applications.ising import tsp

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_initialisation_methods_instance(
    filename: str,
    max_restarts: int,
    evolution_time: float,
    p_max: int,
    mlflow_tracking: bool,
):
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

    # Adding all methods
    methods = [
        "trotterized_quantum_annealing",
        "random_initialisation",
        "perturb_from_previous_layer",
        "ramped_up_initialisation",
        "fourier_transform",
    ]
    optimizers = [
        # NELDER_MEAD(disp=True, adaptive=True, tol=0.1, maxfev=10000),
        COBYLA(maxiter=10000, disp=True, rhobeg=1),
    ]
    results = []

    for opt in optimizers:
        for method in methods:
            print(f"Running optimisation with {method}")
            # Initiate optimizers for a parallel run
            p = 1
            # Randomly initialise layer alpha, beta at layer 1
            initial_point = Initialisation(
                evolution_time=evolution_time
            ).random_initialisation(p=p)
            while p <= p_max:
                run_args = [
                    opt,
                    max_restarts,
                    op,
                    p,
                    mlflow_tracking,
                    Initialisation(p=p, initial_point=initial_point, method=method),
                ]
                # Re-assign initial points
                result = run_qaoa_parallel_control_max_restarts(args=run_args)
                initial_point = result["min_energy_point"]
                p += 1
                results.append(result)

    # Clean up results
    d_results = []
    for res in results:
        d_res = pd.DataFrame(
            list(zip(res["converge_cnts"], res["converge_vals"])),
            columns=["n_eval", "value"],
        )
        d_res["optimizer"] = res["optimizer"]
        d_res["init_method"] = res["initialisation_method"]
        d_res["layer"] = res["layers"]
        d_results.append(d_res)
    d_results = pd.concat(d_results)

    # Add counter for num_evals (+1 so it matches up with n_eval)
    d_results["total_evals"] = (
        d_results.groupby(["init_method", "layer"]).cumcount() + 1
    )
    # Clean up method
    d_results["method"] = d_results["init_method"].apply(
        lambda x: x.replace("_", " ").title()
    )
    d_results.reset_index(inplace=True)
    with make_temp_directory() as temp_dir:
        results_layer_p_fn = f"results_large_offset.csv"
        results_layer_p_fn = os.path.join(temp_dir, results_layer_p_fn)
        # Write results
        with open(results_layer_p_fn, "w") as file:
            d_results.to_csv(file, index=False)

        # Create plots
        g = sns.relplot(
            data=d_results,
            x="total_evals",
            y="value",
            row="method",
            col="layer",
            hue="optimizer",
            kind="line",
        )

        axes = g.axes.flatten()
        for ax in axes:
            ax.axhline(-180, ls="--", linewidth=3, color="grey")
            ax.set_xlabel("Function Evals")

        optimization_plot_fn = f"optimization_plot.png"
        optimization_plot_fn = os.path.join(temp_dir, optimization_plot_fn)
        g.savefig(optimization_plot_fn)
        plt.clf()

        # Create plot for CIs
        g_ave = sns.relplot(
            data=d_results,
            x="n_eval",
            y="value",
            row="method",
            col="layer",
            hue="optimizer",
            kind="line",
        )

        axes = g_ave.axes.flatten()
        for ax in axes:
            ax.axhline(-180, ls="--", linewidth=3, color="grey")
            ax.set_xlabel("Function Evals")

        optimization_plot_ave_fn = f"optimization_plot_average.png"
        optimization_plot_ave_fn = os.path.join(temp_dir, optimization_plot_ave_fn)
        g_ave.savefig(optimization_plot_ave_fn)
        plt.clf()

        for res in results:
            # Make feasibility graph
            feasibility_p_res = generate_feasibility_results(
                res["min_energy_state"], exact_result
            )
            feasibility_p = plot_feasibility_results(feasibility_p_res)
            feasibility_p_fn = f"feasibility_plot_opt_{res['optimizer']}_p_{res['layers']}_meth_{res['initialisation_method']}.png"
            feasibility_p_fn = os.path.join(temp_dir, feasibility_p_fn)
            feasibility_p.figure.savefig(feasibility_p_fn, dpi=300, bbox_inches="tight")

            # Track on MLFlow for each optimizer
            if mlflow_tracking:
                mlflow.log_artifact(feasibility_p_fn)

        if mlflow_tracking:
            mlflow.log_artifact(results_layer_p_fn)
            mlflow.log_artifact(optimization_plot_fn)
            mlflow.log_artifact(optimization_plot_ave_fn)

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
        "-m",
        "--max_restarts",
        type=int,
        default=100,
        help="The max_restarts for the optimziations",
    )

    parser.add_argument(
        "-p",
        "--p_max",
        type=int,
        default=2,
        help="The number of layers,p, to compute for",
    )

    parser.add_argument(
        "-E",
        "--evolution_time",
        type=float,
        default=5.0,
        help="Evolution Time for when using Trotterized Quantum Annealing",
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
    run_initialisation_methods_instance(
        args["filename"],
        max_restarts=args["max_restarts"],
        p_max=args["p_max"],
        evolution_time=args["evolution_time"],
        mlflow_tracking=args["track_mlflow"],
    )
    t2 = time.time()
    print("Run complete in: {} seconds".format(round(t2 - t1, 3)))
