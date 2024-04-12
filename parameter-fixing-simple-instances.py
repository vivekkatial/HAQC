import os
import argparse
import json
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from qiskit import Aer
from qiskit.algorithms.optimizers import ADAM, COBYLA, SLSQP, NELDER_MEAD, SPSA, L_BFGS_B, GradientDescent
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.utils import QuantumInstance
from qiskit_optimization.applications import Maxcut
from qiskit.circuit import Parameter, ParameterVector
from scipy.optimize import minimize



# Custom imports
from haqc.generators.graph_instance import create_graphs_from_all_sources, GraphInstance
from haqc.exp_utils import (
    str2bool,
    to_snake_case,
    make_temp_directory,
    check_boto3_credentials,
)
from haqc.features.graph_features import get_graph_features
from haqc.generators.parameter import get_optimal_parameters
from haqc.solutions.solutions import compute_max_cut_brute_force, compute_distance
from haqc.parallel.landscape_parallel import parallel_computation, parallel_computation_n_layers_fixed
from haqc.plot.utils import *
from haqc.algorithms.custom_optimizers import OptimiseLayerGreaterThanP, OptimizationTracker
from haqc.plot.landscape import plot_landscape
from haqc.utils import adjacency_matrix_to_graph
from haqc.initialisation.initialisation import lookup_optimal_values
from haqc.parallel.landscape_parallel import build_landscape_plot


import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas.core.frame')


# Logger setup
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
qiskit_logger = logging.getLogger('qiskit')
qiskit_logger.setLevel(logging.CRITICAL)
logging.info('Script started')
seed = 1024182
json_file_path = "data/ml-model-landscape.json"

MAX_LAYERS = 15  # Maximum number of layers to optimize for
MESH_SIZE = 100  # Mesh size for the landscape plot

with open(json_file_path, "r") as file:
    data = json.load(file)

# Load instance based on `graph_type` from the JSON file and argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--graph_type', type=str, default=None, help='Graph type to process')
args = parser.parse_args()



if __name__ == "__main__":

    # Access instance that matches the graph type
    instance = [d for d in data if d["graph_type"] == args.graph_type][0]
    print(f"Processing instance: {instance['graph_type']}")

    # Create a temporary directory to store the plots based on the graph type
    directory = f".temp/{instance['graph_type']}"
    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)
        
    backend = Aer.get_backend('aer_simulator_statevector')
    quantum_instance = QuantumInstance(
        backend, 
        seed_simulator=seed, 
        seed_transpiler=seed
    )

    G = adjacency_matrix_to_graph(instance["graph"]["adjacency_matrix"])
    plt.figure()  
    plt.title(instance["graph_type"])
    nx.draw(G)
    plt.savefig(f"{directory}/graph.png")


    # Solving for Exact Ground State using Brute Force for reference
    max_cut_partition, max_cut_value = compute_max_cut_brute_force(G)
    print(f"MAXCUT Partition: {max_cut_partition}")
    print(f"MAXCUT Value: {max_cut_value}")

    # Placeholder for optimal parameters for each layer
    optimal_params = []
    # Initialize the tracker object outside your optimization loop
    tracker = OptimizationTracker()


    for N_LAYERS in range(2, MAX_LAYERS + 1):
        print(f"\n {'-'*10} Processing {N_LAYERS} layers {'-'*10}\n\n")
        
        tracker.set_layer(N_LAYERS)
        
        if N_LAYERS == 2:
            # For layer 2, fetch initial optimal parameters
            optimal_params_n_layer = lookup_optimal_values(instance["graph_type"])
            print(f"Initial optimal parameters for layer {N_LAYERS}: {optimal_params_n_layer}")
            fixed_betas = [optimal_params_n_layer[0]]
            fixed_gammas = [optimal_params_n_layer[1]]
        else:
            # For layers > 2, use the last optimal parameters from the previous layer
            optimal_params_n_layer = optimal_params[-1]  # Last entry from the optimal parameters list
            print(f"Using best optimized parameters from Layer {N_LAYERS - 1}: {optimal_params_n_layer}")
            half_length = len(optimal_params_n_layer) // 2
            fixed_betas = optimal_params_n_layer[:half_length]
            fixed_gammas = optimal_params_n_layer[half_length:]
        
        ws_init_point = fixed_betas + [np.random.rand()] + fixed_gammas + [np.random.rand()]
        # Define the optimizers for QAOA
        algos_optimizers = [
            ('QAOA', 'Layer-Fix', OptimiseLayerGreaterThanP(
                fixed_betas=fixed_betas, 
                fixed_gammas=fixed_gammas,
            ), None),
            ('QAOA', 'Random', NELDER_MEAD(maxfev=1000), 'random'),
            ('QAOA', 'Layer-WS', NELDER_MEAD(maxfev=1000), ws_init_point)
        ]
        
        for optimizer in algos_optimizers:
            print(f"Solving with {optimizer[1]} strategy")
            
            
            tracker.set_optimizer_name(optimizer[1])
            
            if optimizer[1] == "Random":
                initial_point = np.concatenate([np.random.uniform(-np.pi/4, np.pi/4, N_LAYERS), np.random.uniform(-np.pi, np.pi, N_LAYERS)]).tolist()
            else:
                initial_point = optimizer[3]
                
            
            print(f"Initial Point: {initial_point}")
            
            qaoa = QAOA(
                optimizer=optimizer[2],
                reps=N_LAYERS,
                initial_point=initial_point,
                callback=tracker.store_intermediate_result,  # Use method from the tracker instance
                quantum_instance=quantum_instance
            )
            
            adjacency_matrix = nx.adjacency_matrix(G)
            max_cut = Maxcut(adjacency_matrix)
            qubitOp, offset = max_cut.to_quadratic_program().to_ising()
            qaoa.compute_minimum_eigenvalue(qubitOp)
            
            # Extract the last values (optimal parameters) from the tracker for each optimizer
            last_optimal = tracker.intermediate_values[-1]['parameters']
            min_energy = tracker.intermediate_values[-1]['energy']
            print(f"Minimum energy found {min_energy}")
            print(f"Minimum energy parameters for {optimizer[1]}: {last_optimal}")
            
            
            print(f"Optimization Complete for {optimizer[1]} \n \n \n")
        
        print(f"Solved layer {N_LAYERS} using all techniques")
        layer_n_data = [d for d in tracker.intermediate_values if d['layer'] == N_LAYERS]
        best_result = min(layer_n_data, key=lambda x: x['energy'])
        best_result['parameters'] = best_result['parameters'].tolist()
        print(f"Minimum Energy Found: {best_result['energy']} \n")
        print(json.dumps(best_result, indent=4))
        
        # Update optimal parameters for the current layer with the best results from the last optimizer run
        optimal_params.append(best_result['parameters'])

        results_df = pd.DataFrame(tracker.intermediate_values)
        print(results_df.head())
        print(results_df.optimizer.value_counts())

        results_df['parameters_str'] = results_df['parameters'].apply(lambda x: ', '.join(map(str, x)))

        # Clear existing plots
        plt.clf()
        # Create the FacetGrid
        g = sns.FacetGrid(results_df, col='layer', hue='optimizer', col_wrap=4, sharey=True, legend_out=True)
        g = g.map(plt.plot, 'iteration', 'energy').add_legend()
        
        plt.savefig(f"{directory}/optimization_{N_LAYERS}_layers.png")

        # Group by 'layer' and find the index of the minimum 'energy' in each group
        idx_min = results_df.groupby('layer')['energy'].idxmin()
        # Extract the rows corresponding to minimum energy and focus on 'parameters' column
        min_energy_details = results_df.loc[idx_min, ['parameters', 'optimizer']]
        print(min_energy_details)

        # Save min_energy_details to a CSV file
        min_energy_details.to_csv(f"{directory}/min_energy_details.csv", index=False)

        # Build landscape plot based on the optimal parameters
        print("Building landscape plot using optimal parameters: \n")
        print("Betas: ", fixed_betas)
        print("Gammas: ", fixed_gammas)

        landscape_data_storage = build_landscape_plot(
                G, 
                mesh_size=MESH_SIZE, 
                beta_lb=-np.pi/2, 
                beta_ub=np.pi/2, 
                gamma_lb=-np.pi, 
                gamma_ub=np.pi, 
                layers=N_LAYERS,
                fixed_betas=fixed_betas,
                fixed_gammas=fixed_gammas
        )

        plt = plot_landscape(
            landscape_data_storage, 
            source=instance["graph_type"]
        )

        plt.savefig(f"{directory}/landscape_{N_LAYERS}_layers.png")

        # Save all data to JSON file for each layer
        json_data = {
            "graph": instance["graph"],
            "graph_type": instance["graph_type"],
            "layers": N_LAYERS,
            "optimization_results": tracker.intermediate_values,
            "optimal_parameters": optimal_params,
            "min_energy_details": min_energy_details.to_dict(orient='records'),
            "landscape_data": landscape_data_storage
        }

        # Ensure the JSON data is serializable from numpy arrays (min_energy_details)
        for record in json_data["optimization_results"]:
            record["parameters"] = list(record["parameters"])

        # Convert the landscape data to lists
        for key in json_data["landscape_data"]:
            json_data["landscape_data"][key] = json_data["landscape_data"][key].tolist()

        with open(f"{directory}/optimization_results_{N_LAYERS}_layers.json", "w") as file:
            json.dump(json_data, file, indent=4)
