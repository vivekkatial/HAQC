"""
Solving different instance classes with QAOA and VQE in `qiskit`

Consider an $n$-node undirected graph $G = (V, E)$ where $|V| = n$ with edge weights $w_{ij} \geq 0$, $w_{ij} = w_{ji}$ for $(i, j) \in E$. A cut is defined as a partition of the original set $V$ into two subsets. The cost function to be optimized is in this case the sum of weights of edges connecting points in the two different subsets, crossing the cut.

Author: Vivek Katial
"""

import pylab
import numpy as np
import json
import networkx as nx
import random
import copy
import os

# useful additional packages
import numpy as np
import networkx as nx
import seaborn as sns
import time
import argparse
import mlflow

# Qiskit Imports
from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA, NELDER_MEAD

# Custom Imports
from qaoa_vrp.features.graph_features import *
from qaoa_vrp.exp_utils import str2bool, make_temp_directory, to_snake_case


sns.set_theme()

import qiskit

print(qiskit.__version__)


def draw_graph(G, colors, pos):
    default_axes = pylab.axes(frameon=True)
    nx.draw_networkx(
        G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)


class GraphInstance:
    def __init__(self, G, graph_type):
        self.G = G
        self.graph_type = graph_type
        self.weight_matrix = None
        self.brute_force_sol = None

    def __repr__(self):
        return f"This is a {self.graph_type} {self.G} graph instance"

    def allocate_random_weights(self):
        # Allocate random costs to the edges for now
        for (u, v) in self.G.edges():
            self.G.edges[u, v]["weight"] = random.randint(0, 10)

    def compute_weight_matrix(self):
        G = self.G
        n = len(G.nodes())
        w = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                temp = G.get_edge_data(i, j, default=0)
                if temp != 0:
                    w[i, j] = temp["weight"]
        self.weight_matrix = w

    def show_weight_matrix(self):
        print(self.weight_matrix)

    def build_qubo():
        pass

    def solve_qaoa():
        pass

    def solve_vqe():
        pass


def main(track_mlflow=False):
    if track_mlflow:
        # Configure MLFlow Stuff
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        experiment_name = "vqe-maxcut"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        
    # Number of nodes
    N = 10
    # MAX iterations
    MAX_ITERATIONS = 1000

    # Generating a graph of 10 nodes
    G_unif = GraphInstance(nx.gnm_random_graph(N, 30), "Uniform Random")
    G_pl_tree = GraphInstance(
        nx.random_powerlaw_tree(N, gamma=3, seed=None, tries=1000), "Power Law Tree"
    )
    G_wattz = GraphInstance(
        nx.connected_watts_strogatz_graph(N, k=4, p=0.5), "Watts-Strogatz small world"
    )
    G_geom = GraphInstance(nx.random_geometric_graph(N, radius=4), "Geometric")
    G_nc_bipart = GraphInstance(
        nx.complete_bipartite_graph(int(N / 2), int(N / 2)), "Nearly Complete BiPartite"
    )

    G_instances = [G_unif, G_pl_tree, G_wattz, G_geom, G_nc_bipart]

    for i, graph_instance in enumerate(G_instances):
        print(
            f"\n{'-'*50}\nRunning Experiment for {graph_instance.graph_type}\n{'-'*50}\n"
        )
        print(f"Instance Features for {graph_instance.graph_type}\n{'-'*50}\n")
        # Show instance features
        graph_features = get_graph_features(graph_instance.G)
        instance_type_logging = to_snake_case(graph_instance.graph_type)
        graph_features = {instance_type_logging + "_" + str(key): val for key, val in graph_features.items()}
        print(json.dumps(graph_features, indent=4))
        if track_mlflow:
            mlflow.log_params(graph_features)
        print(f"Solving Brute Force for {graph_instance.graph_type}\n{'-'*50}\n")
        G = graph_instance

        print(G)
        G.allocate_random_weights()
        G.compute_weight_matrix()
        print(G.weight_matrix)
        G = graph_instance.G
        w = graph_instance.weight_matrix
        n = len(G.nodes())

        best_cost_brute = 0
        for b in range(2**n):
            x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
            cost = 0
            for i in range(n):
                for j in range(n):
                    cost = cost + w[i, j] * x[i] * (1 - x[j])
            if best_cost_brute < cost:
                best_cost_brute = cost
                xbest_brute = x

        colors = ["r" if xbest_brute[i] == 0 else "c" for i in range(n)]
        pos = nx.spring_layout(G)
        print(
            "\nBest solution = " + str(xbest_brute) + " cost = " + str(best_cost_brute)
        )
        draw_graph(G, colors, pos)

        max_cut = Maxcut(graph_instance.weight_matrix)
        qp = max_cut.to_quadratic_program()
        print(qp.prettyprint())

        qubitOp, offset = qp.to_ising()
        print("Offset:", offset)
        print("Ising Hamiltonian:")
        print(str(qubitOp))

        # solving Quadratic Program using exact classical eigensolver
        exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        result = exact.solve(qp)
        print(result.prettyprint())
        print(
            "\nBest solution BRUTE FORCE = "
            + str(xbest_brute)
            + " cost = "
            + str(best_cost_brute)
        )

        # Check that the Hamiltonian gives the right cost
        print(
            f"\n{'-'*10} Checking that current Hamiltonian gives right cost {'-'*10}\n"
        )

        ee = NumPyMinimumEigensolver()

        # Calculate the min eigenvalue
        optimal_result = ee.compute_minimum_eigenvalue(qubitOp)

        ground_state = max_cut.sample_most_likely(optimal_result.eigenstate)
        print("ground state energy:", optimal_result.eigenvalue.real)
        print("optimal max-cut objective:", optimal_result.eigenvalue.real + offset)
        print("ground state solution:", ground_state)
        print("ground state objective:", qp.objective.evaluate(x))

        colors = ["r" if x[i] == 0 else "c" for i in range(n)]
        # draw_graph(G, colors, pos)

        ################################
        # Quantum Run -- VQE
        ################################

        print(f"\n{'-'*10} Simulating Instance on Quantum using VQE {'-'*10}\n")

        # Run optimisation code
        optimizer = COBYLA(maxiter=MAX_ITERATIONS)
        converge_cnts = np.empty([], dtype=object)
        converge_vals = np.empty([], dtype=object)
        num_qubits = qubitOp.num_qubits

        init_state = np.random.rand(num_qubits) * 2 * np.pi
        print(f"The initial state is {init_state}")
        n_restarts = 1
        optimizer_results = []

        result = {"algo": None, "result": None}

        ## Setting parameters for a run (Simulator Backend etc)
        algorithm_globals.random_seed = 12321
        seed = 10598
        backend = Aer.get_backend("aer_simulator_statevector")
        quantum_instance = QuantumInstance(
            backend, seed_simulator=seed, seed_transpiler=seed
        )

        print(f"The initial state is {init_state}")

        n_restarts = 3

        print(f"Testing Optimizer {i+1}: {type(optimizer).__name__}")

        counts = []
        values = []

        # Callback definition
        def store_intermediate_result(eval_count, parameters, mean, std):
            mlflow.log_metric(f"energy_{instance_type_logging}", mean)
            if eval_count % 100 == 0:
                print(
                    f"{type(optimizer).__name__} iteration {eval_count} \t cost function {mean}"
                )
            counts.append(eval_count)
            values.append(mean)

        for restart in range(n_restarts):
            print(f"Running Optimization at n_restart={restart}")
            init_state = np.random.rand(4) * 2 * np.pi

            # Define the systems of rotation for x and y
            ry = TwoLocal(num_qubits, "ry", "cz", reps=2, entanglement="linear")

            # VQE definition
            vqe = VQE(
                ry,
                optimizer=optimizer,
                quantum_instance=quantum_instance,
                callback=store_intermediate_result,
            )
            algo_result = vqe.compute_minimum_eigenvalue(qubitOp)

        # Convergence array
        total_counts = np.arange(0, len(counts))
        values = np.asarray(values)

        print(f"\n{'-'*10} Optimization Complete {'-'*10}\n")

        pylab.rcParams["figure.figsize"] = (12, 8)
        pylab.plot(total_counts, values, label=type(optimizer).__name__)
        pylab.xlabel("Eval count")
        pylab.ylabel("Energy")
        pylab.title("Energy convergence for various optimizers")
        pylab.legend(loc="upper right")

        # Sample most liklely eigenstate
        x = max_cut.sample_most_likely(algo_result.eigenstate)
        # Energy Gap = E_{g} / E_{opt}
        energy_gap = 1 - algo_result.eigenvalue.real / optimal_result.eigenvalue.real
        print("Final energy:", algo_result.eigenvalue.real)
        print("time:", algo_result.optimizer_time)
        print("max-cut objective:", algo_result.eigenvalue.real + offset)
        print("solution:", x)
        print("solution objective:", qp.objective.evaluate(x))
        print("energy_gap:", energy_gap)

        if track_mlflow:
            print(
                f"\n{'-'*50}\n Logging Results for {instance_type_logging}\n{'-'*50}\n"
            )            
            mlflow.log_metric(f"energy_gap_{instance_type_logging}", energy_gap)
            mlflow.log_metric(f"final_energy_{instance_type_logging}", algo_result.eigenvalue.real)
            mlflow.log_metric(f"maxcut_objective_{instance_type_logging}", algo_result.eigenvalue.real + offset)
            mlflow.log_metric(f"solution_objective_{instance_type_logging}",  qp.objective.evaluate(x))

            with make_temp_directory() as temp_dir:
                # Plot Network Graph
                graph_plot_fn = f"network_plot_{instance_type_logging}.png"
                graph_plot_fn = os.path.join(temp_dir, graph_plot_fn)
                colors = ["r" if x[i] == 0 else "c" for i in range(n)]
                draw_graph(G, colors, pos)
                pylab.savefig(graph_plot_fn)
                pylab.clf()
                mlflow.log_artifact(graph_plot_fn)

                # Plot convergence
                convergence_plot_fn = f"convergence_plot_{instance_type_logging}.png"
                convergence_plot_fn = os.path.join(temp_dir, convergence_plot_fn)            
                pylab.rcParams["figure.figsize"] = (12, 8)
                pylab.plot(total_counts, values, label=type(optimizer).__name__)
                pylab.xlabel("Eval count")
                pylab.ylabel("Energy")
                pylab.title("Energy convergence for various optimizers")
                pylab.legend(loc="upper right")
                pylab.savefig(convergence_plot_fn)
                pylab.axhline(y=algo_result.eigenvalue.real, ls='--', c='red')
                mlflow.log_artifact(convergence_plot_fn)


    print(f"\n{'-'*10}\n\n{'-'*10} Run Complete \n{'-'*10}\n\n{'-'*10}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    print(args)
    t1 = time.time()
    main(track_mlflow=args["track_mlflow"])
    t2 = time.time()
    print("Result found in: {} seconds".format(round(t2 - t1, 3)))
