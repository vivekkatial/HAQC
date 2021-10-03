# Hide warnings
import warnings
warnings.filterwarnings('ignore')

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
import pylab

# Custom Libraries
import qaoa_vrp.build_graph
import qaoa_vrp.features.graph_features
import qaoa_vrp.features.tsp_features
import qaoa_vrp.build_circuit
import qaoa_vrp.clustering
import qaoa_vrp.utils
from qaoa_vrp.exp_utils import str2bool, make_temp_directory
from qaoa_vrp.quantum_burden import compute_quantum_burden
from qaoa_vrp.classical.greedy_tsp import greedy_tsp
from qaoa_vrp.plot.draw_euclidean_graphs import draw_euclidean_graph

# QISKIT stuff
from qiskit import Aer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, NELDER_MEAD, SPSA
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import MinimumEigenOptimizer
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.optimization.applications.ising import tsp

filename="instanceType_euclidean_tsp_numNodes_4_numVehicles_1_87a170c748e240d0b71d5fb7fe7de707.json"



def run_instance(filename, budget: int, p_max=10):
    instance_path = "data/{}".format(filename)
    with open(instance_path) as f:
            data = json.load(f)
    G, depot_info = qaoa_vrp.build_graph.build_json_graph(data["graph"])
    num_vehicles = int(data["numVehicles"])
    threshold = float(data["threshold"])
    n_max = int(data["n_max"])



    edge_mat = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    cost_mat = np.array(nx.attr_matrix(G, edge_attr="cost", rc_order=list(G.nodes())))
    for edge in G.edges():
        G[edge[0]][edge[1]]['cost'] = 0
        
    edge_mat = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
    cost_mat = np.array(nx.attr_matrix(G, edge_attr="cost", rc_order=list(G.nodes())))

    G, cluster_mapping = qaoa_vrp.clustering.create_clusters(
        G, num_vehicles, "spectral-clustering", edge_mat
    )

    depot_edges = list(G.edges(depot_info["id"], data=True))
    depot_node = depot_info["id"]

    subgraphs = qaoa_vrp.clustering.build_sub_graphs(G, depot_node, depot_edges)

    # big_offset = sum(sum(cost_mat))/2 + 1
    big_offset=30
    qubos = qaoa_vrp.build_circuit.build_qubos(subgraphs, depot_info,A=big_offset)

    cluster_mapping = [i + 1 for i in cluster_mapping]
    cluster_mapping.insert(0, 0)

    qubo = qubos[0]

    print("Running single tsp for Qubo")
    print(qubo)

    single_qubo_solution_data = {}
    single_qubo_solution_data["qubo_id"] = 0
    single_qubo_solution_data["cluster"] = [
        index
        for index, node in enumerate(cluster_mapping)
        if node == 1 or node == 0
    ]

    op, offset = qubo.to_ising()

    print('Offset:', offset)
    print('Ising Hamiltonian:')
    print(op.print_details())

    qp = QuadraticProgram()
    qp.from_ising(op, offset, linear=True)
    qp.to_docplex().prettyprint()

    exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    exact_result = exact.solve(qp)
    print(exact_result)

    ee = NumPyMinimumEigensolver(op)
    exact_result = ee.run()

    print('energy:', exact_result.eigenvalue.real)
    print('tsp objective:', exact_result.eigenvalue.real + offset)
    x = sample_most_likely(exact_result.eigenstate)
    print('feasible:', tsp.tsp_feasible(x))
    z = tsp.get_tsp_solution(x)
    print('solution:', z)
    print('solution objective:', tsp.tsp_value(z, cost_mat))

    # Quantum solution
    p = 1
    while p < p_max:
        
        # Initialise each budget parameter

        if p > 5:
            budget = 2000

        optimizers = [
            SLSQP(maxiter=budget, disp=True, eps=0.001),
            COBYLA(maxiter=budget, disp=True, rhobeg=0.1), 
            NELDER_MEAD(maxfev=budget,disp=True,adaptive=True),
            SPSA(maxiter=budget,learning_rate=0.01,perturbation=0.01),
            L_BFGS_B(maxfun=budget,factr=10, epsilon=0.001,iprint=100)
        ]

        # Make convergence counts and values
        converge_cnts = np.empty([len(optimizers)], dtype=object)
        converge_vals = np.empty([len(optimizers)], dtype=object)
        backend = Aer.get_backend('aer_simulator_matrix_product_state')

        for i, optimizer in enumerate(optimizers):
            print('\rOptimizer: {}        '.format(type(optimizer).__name__))
            counts = []
            values = []
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
                initial_point = [np.random.uniform(0,1) for i in range(2*p)]         
                # Set random seed
                aqua_globals.random_seed = np.random.default_rng(123)
                seed = 10598
                # Initate quantum instance
                quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
                # Initate QAOA
                qaoa = QAOA(
                    operator=op,
                    optimizer=optimizer,
                    callback=store_intermediate_result,
                    p=p,
                    initial_point = initial_point,
                    quantum_instance=quantum_instance
                )
                result = qaoa.compute_minimum_eigenvalue(operator=op)
                converge_cnts[i] = np.asarray(counts)
                converge_vals[i] = np.asarray(values)
        
            # Create a dictionary for results
            results_dict = {
                "optimizer":None,
                "n_eval": None,
                "value": None
            }

            # Dictionary for different optimisers we're exploring
            optimizer_dict = {
                0: "NELDER_MEAD",
                1: "COBYLA",
            }

            d_results = []

            for i,(evals, values) in enumerate(zip(converge_cnts, converge_vals)):
                for cnt, val in zip(evals, values):
                    results_dict_temp = results_dict.copy()
                    results_dict_temp["n_eval"] = cnt
                    results_dict_temp["value"] = val
                    results_dict_temp["optimizer"] = optimizer_dict[i]
                    d_results.append(results_dict_temp)

            d_results = pd.DataFrame.from_records(d_results)

            # Add counter for num_evals
            d_results['total_evals']=d_results.groupby('optimizer').cumcount()
            d_results.to_csv(f"../data/results_large_offset_{p}.csv")

            # Create plots
            g = sns.relplot(
                data=d_results, x="total_evals", y="value",
                col="optimizer", hue="optimizer",
                kind="line"
            )

            (g.map(plt.axhline, y=-180, color=".7", dashes=(2, 1), zorder=0)
            .tight_layout(w_pad=0))

            g.savefig('plot.png')


        # Increment p        
        p += 1  
            
    print('\rOptimization complete')


if __name__ == "__main__":
    run_instance(filename, budget=10, p_max=2)