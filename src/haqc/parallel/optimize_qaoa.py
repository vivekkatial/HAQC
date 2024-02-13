import numpy as np
import os
import mlflow
from numpy.core.fromnumeric import argsort

from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit import Aer
from qiskit.aqua.algorithms import QAOA

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.haqc.exp_utils import make_temp_directory
from src.haqc.initialisation.initialisation import Initialisation


def run_qaoa_parallel_control_max_restarts(args):
    optimizer, max_restarts, op, p, mlflow_tracking, InitialPoint = (
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
    )
    print(
        f"\r Running Optimizer: {type(optimizer).__name__} in parallel with {p} layers and {max_restarts} restarts"
    )
    backend = Aer.get_backend("aer_simulator_matrix_product_state")
    counts = []
    values = []
    # Run energy and results
    run_energy = []
    run_optimal_params = []
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
        print(
            f"Starting restart n={n_restart} for layer(s) {p}, init method: {InitialPoint.initialisation_method}"
        )
        # Initiate a random point uniformly from [0,1]
        initialisation_method = getattr(
            InitialPoint, InitialPoint.initialisation_method
        )
        InitialPoint.initial_point = initialisation_method(
            p=p, previous_layer_initial_point=InitialPoint.initial_point
        )
        # Set random seed
        aqua_globals.random_seed = np.random.default_rng(123)
        seed = 10598
        # Initate quantum instance
        print(InitialPoint)
        quantum_instance = QuantumInstance(
            backend, seed_simulator=seed, seed_transpiler=seed
        )

        # Initialise QAOA
        qaoa = QAOA(
            operator=op,
            optimizer=optimizer,
            callback=store_intermediate_result,
            reps=p,
            initial_point=InitialPoint.initial_point,
            quantum_instance=quantum_instance,
        )

        # Compute the QAOA result
        result = qaoa.compute_minimum_eigenvalue(operator=op)

        # Store in temp run info
        run_energy.append(result.eigenvalue.real)
        run_results.append(result.eigenstate)
        run_optimal_params.append(result.optimal_point)

        # Append the minimum value from temp run info into doc
        if mlflow_tracking:
            mlflow.log_metric(
                key=f"{type(optimizer).__name__}_p_{p}", value=min(run_energy)
            )

        min_energy_ind = run_energy.index(min(run_energy))
        min_energy_state = run_results[min_energy_ind]
        min_energy_point = run_optimal_params[min_energy_ind]

    # Produce plots for optimizations at each layer
    if mlflow_tracking:
        # Construct data for plot
        df = pd.DataFrame(list(zip(counts, values)), columns=["feval", "energy"])
        # Add counter columns
        inds = [i + 1 for i in range(len(counts))]
        df["counts"] = inds

        # Produce Seaborn chart
        g = sns.relplot(data=df, x="counts", y="energy", kind="line")
        plt.axhline(y=-180, ls="--", color="grey")
        plt.title(f"Init Method: {InitialPoint.pprint_method()}")

        with make_temp_directory() as temp_dir:
            # Build and store on MLFLow
            layer_opt_fn = f"optimization_plot_layer_{p}_method_{InitialPoint.initialisation_method}.png"
            layer_opt_fn = os.path.join(temp_dir, layer_opt_fn)
            g.savefig(layer_opt_fn)
            mlflow.log_artifact(layer_opt_fn)

    print(
        "\r Ending run for  Optimizer: {} in parallel, init:\t{}".format(
            type(optimizer).__name__, InitialPoint.initialisation_method
        )
    )

    results = {
        "optimizer": type(optimizer).__name__,
        "layers": p,
        "initialisation_method": InitialPoint.initialisation_method,
        "min_energy": min(run_energy),
        "min_energy_state": min_energy_state,
        "min_energy_point": min_energy_point,
        "converge_cnts": np.asarray(counts),
        "converge_vals": np.asarray(values),
    }

    return results
