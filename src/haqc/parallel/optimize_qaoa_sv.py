import numpy as np
import mlflow
from numpy.core.fromnumeric import argsort

from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit import Aer
from qiskit.aqua.algorithms import QAOA

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.haqc.exp_utils import make_temp_directory


def run_qaoa_parallel_sv(args):
    optimizer, budget, op, p, mlflow_tracking = (
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
    )
    print("\r Running Optimizer: {} in parallel".format(type(optimizer).__name__))
    backend = Aer.get_backend("statevector_simulator")
    counts = []
    values = []
    # Run energy and results
    run_energy = []
    run_results = []
    global global_count
    global_count = 0
    n_restart = 0

    def store_intermediate_result(eval_count, parameters, mean, std):
        """Function for storing intermediary result during optimisation"""
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
            backend, seed_simulator=seed, seed_transpiler=seed
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
        if mlflow_tracking:
            mlflow.log_metric(
                key=f"{type(optimizer).__name__}_p_{p}", value=min(run_energy)
            )

        min_energy_ind = run_energy.index(min(run_energy))
        min_energy_state = run_results[min_energy_ind]

    print(
        "\r Ending run for  Optimizer: {} in parallel".format(type(optimizer).__name__)
    )
    results = {
        "optimizer": type(optimizer).__name__,
        "layers": p,
        "min_energy": min(run_energy),
        "min_energy_state": min_energy_state,
        "converge_cnts": np.asarray(counts),
        "converge_vals": np.asarray(values),
    }

    return results


def run_qaoa_parallel_control_max_restarts_sv(args):
    optimizer, max_restarts, op, p, mlflow_tracking = (
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
    )
    print(
        f"\r Running Optimizer: {type(optimizer).__name__} in parallel with {p} layers and {max_restarts} restarts"
    )
    backend = Aer.get_backend("aer_simulator_statevector")
    counts = []
    values = []
    # Run energy and results
    run_energy = []
    run_results = []
    global global_count
    global_count = 0
    n_restart = 0

    def store_intermediate_result(eval_count, parameters, mean, std):
        """Function for storing intermediary result during optimisation"""
        global global_count
        global_count += 1
        counts.append(eval_count)
        values.append(mean)

    while n_restart < max_restarts:
        # Increment n_restarts
        n_restart += 1
        print(f"Starting restart n={n_restart} for layer(s) {p}")
        # Initiate a random point uniformly from [0,1]
        initial_point = [np.random.uniform(0, 1) for i in range(2 * p)]
        # Set random seed
        aqua_globals.random_seed = np.random.default_rng(123)
        seed = 10598
        # Initate quantum instance
        quantum_instance = QuantumInstance(
            backend, seed_simulator=seed, seed_transpiler=seed
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
        if mlflow_tracking:
            mlflow.log_metric(
                key=f"{type(optimizer).__name__}_p_{p}", value=min(run_energy)
            )

        min_energy_ind = run_energy.index(min(run_energy))
        min_energy_state = run_results[min_energy_ind]

    # Produce plots for optimizations at each layer
    if mlflow_tracking:
        # Construct data for plot
        d = pd.DataFrame(list(zip(counts, values)), columns=["feval", "energy"])
        # Add counter columns
        inds = [i + 1 for i in range(len(counts))]
        d["counts"] = inds

        # Produce Seaborn chart
        g = sns.relplot(data=d, x="counts", y="energy", kind="line")
        plt.axhline(y=-180, ls="--", color="grey")

        # Build and store on MLFLow
        with make_temp_directory() as temp_dir:
            layer_opt_fn = f"optimization_plot_layer_{p}.png"
            g.savefig(layer_opt_fn)
            mlflow.log_artifact(layer_opt_fn)

    print(
        "\r Ending run for  Optimizer: {} in parallel".format(type(optimizer).__name__)
    )
    results = {
        "optimizer": type(optimizer).__name__,
        "layers": p,
        "min_energy": min(run_energy),
        "min_energy_state": min_energy_state,
        "converge_cnts": np.asarray(counts),
        "converge_vals": np.asarray(values),
    }

    return results
