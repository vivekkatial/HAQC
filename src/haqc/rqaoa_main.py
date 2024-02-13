#!/usr/bin/env python
# coding: utf-8

# Hide warnings
from logging import FATAL
import warnings

warnings.filterwarnings("ignore")

# General dependencies
import json
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple

# Qiskit global
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit import Aer, QuantumCircuit

# Qiskit Algorithms
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
)
from qiskit_optimization import QuadraticProgram

from qiskit.quantum_info import PauliTable
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.quantum_info.operators.symplectic.sparse_pauli_op import SparsePauliOp

# Qiskit Visuals
from qiskit.visualization import plot_histogram

# Custom libraries
import src.build_graph
import src.features.graph_features
import src.features.tsp_features
import src.build_circuit
import src.clustering
import src.utils
from src.plot.feasibility_graph import (
    plot_feasibility_results,
    generate_feasibility_results,
)

from src.initialisation.initialisation import Initialisation
from src.features.graph_features import get_graph_features
from src.features.tsp_features import get_tsp_features
from src.parallel.optimize_qaoa import run_qaoa_parallel_control_max_restarts
from src.solutions.solutions import FEASIBLE_SOLUTIONS

import argparse
import json


parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--num_layers",
    type=int,
    default=2,
    help="The number of layers,p, to compute for",
)

args = vars(parser.parse_args())


########## DEFINE GLOBAL VARIABLES
# SET GLOBAL VARIABLES
NUM_LAYERS = args["num_layers"]
INITIAL_POINT = Initialisation().trotterized_quantum_annealing(p=NUM_LAYERS)

RQAOA_MIN_NUM_VARS = 3
RQAOA_N_RESTARTS = 15
# RQAOA_MIN_NUM_VARS = [1, 3, 5]

plt.style.use("seaborn")
warnings.filterwarnings("ignore", category=DeprecationWarning)

filename = "instanceType_asymmetric_tsp_numNodes_4_numVehicles_1_0083a1d22a6447f69091ac552ceb8ee2.json"
instance_path = "data/{}".format(filename)

with open(instance_path) as f:
    data = json.load(f)
    G, depot_info = src.build_graph.build_json_graph(data["graph"])
    num_vehicles = int(data["numVehicles"])
    threshold = float(data["threshold"])
    n_max = int(data["n_max"])

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

op_tsp, offset = qubo.to_ising()
print("offset: {}".format(offset))
print("operator: {}".format(op_tsp))


list_of_paulis = []
list_of_coeffs = []
for i in list(op_tsp):
    list_of_paulis.append(str(i.primitive))
    list_of_coeffs.append(i.coeff)

for i, pauli in enumerate(list_of_paulis):
    if i == 0:
        pauli_table = PauliTable(list_of_paulis[0])
    else:
        pauli_table += PauliTable(list_of_paulis[i])

op_tsp = PauliSumOp(SparsePauliOp(pauli_table, coeffs=list_of_coeffs))

qp = QuadraticProgram()
qp.from_ising(op_tsp, offset, linear=True)
print(qp.export_as_lp_string())

algorithm_globals.random_seed = 10598
quantum_instance = QuantumInstance(
    Aer.get_backend("aer_simulator_matrix_product_state"),
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
)
qaoa_mes = QAOA(quantum_instance=quantum_instance, reps=NUM_LAYERS)

exact_mes = NumPyMinimumEigensolver()
qaoa = MinimumEigenOptimizer(qaoa_mes)  # using QAOA
exact = MinimumEigenOptimizer(
    exact_mes
)  # using the exact classical numpy minimum eigen solver

ising_qubo = qubos[0].to_ising()

list_of_paulis = []
list_of_coeffs = []
for i in list(ising_qubo[0]):
    list_of_paulis.append(str(i.primitive))
    list_of_coeffs.append(i.coeff)


for i, pauli in enumerate(list_of_paulis):
    if i == 0:
        pauli_table = PauliTable(list_of_paulis[0])
    else:
        pauli_table += PauliTable(list_of_paulis[i])

op_tsp = PauliSumOp(SparsePauliOp(pauli_table, coeffs=list_of_coeffs))


qp = QuadraticProgram()
qp.from_ising(op_tsp, offset, linear=True)
print(qp.export_as_lp_string())

qaoa = MinimumEigenOptimizer(qaoa_mes)  # using QAOA
exact = MinimumEigenOptimizer(
    exact_mes
)  # using the exact classical numpy minimum eigen solver

exact_result = exact.solve(qp)
print(exact_result)

print(f"{'='*20} STARTING QAOA {'='*20}")

qaoa_result = qaoa.solve(qp)
print(qaoa_result)

# Recursive QAOA
# Find results and probabilities where valid
qaoa_feas_probs = []
for i in range(len(qaoa_result.raw_samples)):
    for j in FEASIBLE_SOLUTIONS:
        if np.array_equal(qaoa_result.raw_samples[i].x, j):
            qaoa_feas_probs.append(qaoa_result.raw_samples[i].probability)

print(f"{'='*20} STARTING Recursive QAOA {'='*20}")

rqaoa_sols = []
for restart in range(RQAOA_N_RESTARTS):
    rqaoa = RecursiveMinimumEigenOptimizer(
        qaoa, min_num_vars=RQAOA_MIN_NUM_VARS, min_num_vars_optimizer=exact
    )
    rqaoa_result = rqaoa.solve(qp)
    print(rqaoa_result)
    rqaoa_sols.append(rqaoa_result.x)

rqaoa_res = []
for rqaoa_sol in rqaoa_sols:
    check = any(
        all(np.array_equal(x, y) for x, y in zip(sol, rqaoa_sol))
        for sol in FEASIBLE_SOLUTIONS
    )
    rqaoa_res.append(check)

rqaoa_prob = sum(rqaoa_res) / len(rqaoa_res)
print(f"{'='*20} ALL RUNS FINISHED {'='*20}")

print(
    f"Run with following properties: \n NUM_LAYERS \t{NUM_LAYERS}\n RQAOA_N_RESTARTS \t{RQAOA_N_RESTARTS} \n Initialisaton \t 'TQA'"
)
print(f"QAOA Probability of Success is {np.sum(qaoa_feas_probs)}")
print(f"Recursive QAOA Probability of Success is {rqaoa_prob}")
