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
import qaoa_vrp.build_graph
import qaoa_vrp.features.graph_features
import qaoa_vrp.features.tsp_features
import qaoa_vrp.build_circuit
import qaoa_vrp.clustering
import qaoa_vrp.utils
from qaoa_vrp.plot.feasibility_graph import (
    plot_feasibility_results,
    generate_feasibility_results,
)
from qaoa_vrp.initialisation.initialisation import Initialisation
from qaoa_vrp.features.graph_features import get_graph_features
from qaoa_vrp.features.tsp_features import get_tsp_features
from qaoa_vrp.parallel.optimize_qaoa import run_qaoa_parallel_control_max_restarts

plt.style.use("seaborn")
warnings.filterwarnings("ignore", category=DeprecationWarning)

filename = "instanceType_asymmetric_tsp_numNodes_4_numVehicles_1_0083a1d22a6447f69091ac552ceb8ee2.json"
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
    G[edge[0]][edge[1]]["cost"] = 0

edge_mat = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
cost_mat = np.array(nx.attr_matrix(G, edge_attr="cost", rc_order=list(G.nodes())))

G, cluster_mapping = qaoa_vrp.clustering.create_clusters(
    G, num_vehicles, "spectral-clustering", edge_mat
)

depot_edges = list(G.edges(depot_info["id"], data=True))
depot_node = depot_info["id"]

subgraphs = qaoa_vrp.clustering.build_sub_graphs(G, depot_node, depot_edges)

# big_offset = sum(sum(cost_mat))/2 + 1
big_offset = 30
qubos = qaoa_vrp.build_circuit.build_qubos(subgraphs, depot_info, A=big_offset)

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
qaoa_mes = QAOA(quantum_instance=quantum_instance, reps=3)

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

qaoa_result = qaoa.solve(qp)
print(qaoa_result)

rqaoa = RecursiveMinimumEigenOptimizer(
    qaoa, min_num_vars=3, min_num_vars_optimizer=exact
)
rqaoa_result = rqaoa.solve(qp)
print(rqaoa_result)
