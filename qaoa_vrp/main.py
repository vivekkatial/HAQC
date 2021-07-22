"""
VRP simulation script

Author: Floyd creevey
"""
# Standard Libraries
import argparse
import json
import time
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
import mlflow
import os


# Custom Libraries
import qaoa_vrp.build_graph
import qaoa_vrp.features.graph_features
import qaoa_vrp.features.tsp_features
import qaoa_vrp.build_circuit
import qaoa_vrp.clustering
import qaoa_vrp.utils
from qaoa_vrp.exp_utils import str2bool, make_temp_directory
from qaoa_vrp.quantum_burden import compute_quantum_burden


def solve_qaoa(
    i, qubo, p_max, threshold, n_max, mlflow_tracking, filename, raw_build=True
):
    """
    This function is used as input to the parellelisation in run_vrp_instance
    It obtains the exact and QAOA results and prints them, in parellel for each subtour

    Args:
        i (int): the index of the parellel run
        qubo (object): qiskit qubo object
        p (int): the number of layers in the QAOA (p value)
        mlflow_tracking (bool): Boolean to enable MlFlow tracking
    """
    evolution_data = []
    n = 0
    n_run_probabilities = []

    while n < n_max:
        # Set p ticker
        p = 1
        probability = [0, 0]

        points = list(2 * np.pi * np.random.random(2 * p))

        prev_optimal_value = 100000
        optimal_value = 1000

        while p <= p_max:
            t_start = time.time()
            if n != 1:
                prev_optimal_value = optimal_value

            qaoa, circuit, params_expr = qaoa_vrp.build_circuit.qubo_to_qaoa(qubo, p)

            qaoa_result, exact_result, offset = qaoa_vrp.build_circuit.solve_qubo_qaoa(
                qubo, p, points
            )

            parameters = [
                qaoa_result["optimal_parameters"][parameter]
                for parameter in qaoa_result["optimal_parameters"]
            ]
            gammas = parameters[0::2]
            betas = parameters[1::2]

            (
                num_nodes,
                linear_terms,
                quadratic_terms,
            ) = qaoa_vrp.build_circuit.to_hamiltonian_dicts(qubo)

            optimal_value = qaoa_result["optimal_value"] + offset

            points = qaoa_vrp.build_circuit.interp_point(qaoa_result["optimal_point"])

            optimal_value = qaoa_result["optimal_value"] + offset

            print("Cluster QUBO {}:".format(i))
            print("Optimal value: {}".format(optimal_value))
            print("Exact result (p={}): {}".format(p, exact_result.samples))
            probability, solution_data = qaoa_vrp.build_circuit.print_result(
                qubo, qaoa_result, circuit.num_qubits, exact_result.samples[0][1]
            )

            if num_nodes > 1:
                p_success = probability[0] + probability[1]
            else:
                p_success = probability[0]

            if raw_build:
                # Evolution p_step
                evolution_p_data = {
                    "p": p,
                    "state": solution_data,
                    "probability_success": p_success,
                }
                # Attach evolution step
                evolution_data.append(evolution_p_data)

            p += 1
            t_end = time.time()
            print("Computed in {} seconds\n".format(round(t_end - t_start, 2)))

        n += 1

    print("--Result found for QUBO {}---\n".format(i))
    p = 1
    return evolution_data


def run_vrp_instance(filename, mlflow_tracking, raw_build=True):
    """
    This function runs a VRP instance from end to end
    e.g loads in graph json file, converts to networkx
    graph, creates clusters, creates Hamiltonians for
    each cluster, builds QAOA circuits for each
    Hamiltonian, solves the QAOA circuits.

    Args:
        filename (str): the filename of the graph json
        num_vehicles (int): the number of vehicles
        p (int): Number of layers for QAOA
        threshold (int): the percentage rise allowed for the optimal value in QAOA
        n_max (int): the number of restarts allowed for the QAOA,
        mlflow_tracking (bool): Boolean to enable MlFlow tracking
    """

    # Initiate empty dictionary
    if raw_build:
        instance_data_complete = {
            "filename": None,
            "instance": {},
            "solution_data": {"qubos": []},
        }

    # Load in json data
    instance_path = "data/{}".format(filename)
    with open(instance_path) as f:
        data = json.load(f)

    # Define global variables
    G, depot_info = qaoa_vrp.build_graph.build_json_graph(data["graph"])
    num_vehicles = int(data["numVehicles"])
    threshold = float(data["threshold"])
    n_max = int(data["n_max"])
    instance_type = data["instance_type"]
    p_max = data["p_max"]

    # Create QAOA parameter dictionary
    qaoa_dict = qaoa_vrp.utils.create_qaoa_params(threshold, n_max, p_max)

    # Read in parameter file
    with open("config/mlflow_config.json") as file:
        params = json.load(file)

    # MlFlow Configuration
    if mlflow_tracking:

        # Configure MLFlow Stuff
        mlflow.set_tracking_uri(params["experiment"]["tracking-uri"])
        mlflow.set_experiment(params["experiment"]["name"])

        # Build Graph Feature Vector
        feature_vector = qaoa_vrp.features.graph_features.get_graph_features(G)
        # Build TSP Feature Vector
        tsp_feature_vector = qaoa_vrp.features.tsp_features.get_tsp_features(G)
        # Add num vehicles
        feature_vector["num_vehicles"] = num_vehicles

        # Log Params
        mlflow.log_params(feature_vector)
        mlflow.log_params(tsp_feature_vector)
        mlflow.log_params(qaoa_dict)
        mlflow.log_param("source", instance_type)
        mlflow.log_param("instance_uuid", filename)
        mlflow.log_param("p_max", p_max)

    if raw_build:
        instance_data_complete["filename"] = filename
        instance_data_complete["instance"]["source"] = instance_type
        instance_data_complete["instance"]["raw_graph"] = data
        # MLFlow tracking
        if mlflow_tracking:
            instance_data_complete["instance"]["features"] = feature_vector
            instance_data_complete["instance"]["qaoa_params"] = qaoa_dict

    depot_edges = list(G.edges(depot_info["id"], data=True))
    depot_node = depot_info["id"]

    G.remove_node(depot_info["id"])

    edge_mat = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    cost_mat = np.array(nx.attr_matrix(G, edge_attr="cost", rc_order=list(G.nodes())))

    G, cluster_mapping = qaoa_vrp.clustering.create_clusters(
        G, num_vehicles, "spectral-clustering", edge_mat
    )

    subgraphs = qaoa_vrp.clustering.build_sub_graphs(G, depot_node, depot_edges)

    qubos = qaoa_vrp.build_circuit.build_qubos(subgraphs, depot_info)

    qubos_solution_data = []

    cluster_mapping = [i + 1 for i in cluster_mapping]
    cluster_mapping.insert(0, 0)

    for i, qubo in enumerate(qubos):
        # Build solution data
        single_qubo_solution_data = {}
        single_qubo_solution_data["qubo_id"] = i
        single_qubo_solution_data["cluster"] = [
            index
            for index, node in enumerate(cluster_mapping)
            if node == i + 1 or node == 0
        ]
        single_qubo_solution_data["evolution"] = solve_qaoa(
            i, qubo, p_max, threshold, n_max, mlflow_tracking, filename
        )

        # Solution data for QUBO stuff
        qubos_solution_data.append(single_qubo_solution_data)

    # Compute quantum burden
    quantum_burden = compute_quantum_burden(qubos_solution_data)

    if mlflow_tracking:
        mlflow.log_metrics(quantum_burden)

    # Log Results File
    if raw_build:
        instance_data_complete["solution_data"] = qubos_solution_data

        with make_temp_directory() as temp_dir:
            instance_data_complete_filename = filename.replace(".json", "")
            instance_data_complete_filename = filename + "_solution.json"
            instance_data_complete_filename = os.path.join(
                temp_dir, instance_data_complete_filename
            )

            # Write JSON
            with open(instance_data_complete_filename, "w") as file:
                json.dump(
                    instance_data_complete,
                    file,
                    indent=2,
                    default=qaoa_vrp.utils.np_encoder,
                )

            if mlflow_tracking:
                mlflow.log_artifact(instance_data_complete_filename)

    print("Run Successful! Well DONE :D")


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
    run_vrp_instance(args["filename"], args["track_mlflow"])
    t2 = time.time()
    print("Result found in: {} seconds".format(round(t2 - t1, 3)))
