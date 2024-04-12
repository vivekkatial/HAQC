import numpy as np
import networkx as nx
from qiskit_optimization.applications import Maxcut
from qiskit.circuit import ParameterVector

from qiskit import Aer
from qiskit.algorithms.optimizers import SLSQP
from qiskit.algorithms import QAOA
from qiskit.quantum_info import Statevector
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Function to compute expectation value in parallel
def compute_expectation_value(beta_val, gamma_val, qubitOp, qaoa):
    """Compute expectation value in parallel
    beta_val: beta value
    gamma_val: gamma value
    qubitOp: qubit operator
    qaoa: qaoa instance

    Returns: expectation value
    """
    qc = qaoa.construct_circuit([beta_val, gamma_val], operator=qubitOp)[0]
    backend = Aer.get_backend('aer_simulator')
    statevector = Statevector.from_instruction(qc)
    expectation = statevector.expectation_value(qubitOp).real
    return expectation


# Parallel computation of the objective function values
def parallel_computation(beta, gamma, qubitOp, qaoa):
    """Parallel computation of the objective function values
    beta: array of beta values
    gamma: array of gamma values
    qubitOp: qubit operator
    qaoa: qaoa instance

    Returns: array of objective function values
    """
    obj_vals = np.zeros((len(gamma), len(beta)))

    with ProcessPoolExecutor() as executor:
        futures = {}
        for i, gamma_val in enumerate(gamma):
            for j, beta_val in enumerate(beta):
                future = executor.submit(
                    compute_expectation_value, beta_val, gamma_val, qubitOp, qaoa
                )
                futures[future] = (i, j)

        for future in tqdm(as_completed(futures), desc="Progress", total=len(futures)):
            i, j = futures[future]
            obj_vals[i, j] = future.result()

    return obj_vals


# Function to compute expectation value in parallel
def compute_expectation_value_n_layers(beta, gamma, qubitOp, qaoa):
    qaoa_params = beta + gamma
    qc = qaoa.construct_circuit(qaoa_params, operator=qubitOp)[0]
    backend = Aer.get_backend('aer_simulator')
    statevector = Statevector.from_instruction(qc)
    expectation = statevector.expectation_value(qubitOp).real
    return expectation


def non_parallel_computation_n_layers_fixed(beta, gamma, fixed_betas, fixed_gammas, qubitOp, qaoa):

    obj_vals = np.zeros((len(gamma), len(beta)))
    for i, gamma_val in enumerate(gamma):
        for j, beta_val in enumerate(beta):
            beta = fixed_betas + [beta_val]
            gamma = fixed_gammas + [gamma_val]
            obj_vals[i, j] = compute_expectation_value_n_layers(
                beta, gamma, qubitOp, qaoa
            )

    return obj_vals




def parallel_computation_n_layers_fixed(
    beta, gamma, fixed_betas,  fixed_gammas, qubitOp, qaoa
):
    """Parallel computation of the objective function values for fixed layers
    gamma: array of gamma values
    beta: array of beta values
    fixed_gammas: array of fixed gamma values
    fixed_betas: array of fixed beta values
    qubitOp: qubit operator
    qaoa: QAOA instance

    Returns: array of objective function values
    """
    obj_vals = np.zeros((len(gamma), len(beta)))

    with ProcessPoolExecutor() as executor:
        futures = {}
        for i, gamma_val in enumerate(gamma):
            for j, beta_val in enumerate(beta):
                # Append the current values to the fixed values
                current_beta = fixed_betas + [beta_val]
                current_gamma = fixed_gammas + [gamma_val]

                # Submit the task to the executor
                future = executor.submit(
                    compute_expectation_value_n_layers,
                    current_beta,
                    current_gamma,
                    qubitOp,
                    qaoa,
                )
                futures[future] = (i, j)

        # Collecting results with a progress bar
        for future in tqdm(as_completed(futures), desc="Progress", total=len(futures)):
            i, j = futures[future]
            obj_vals[i, j] = future.result()

    return obj_vals


def build_landscape_plot(
    G, 
    mesh_size: int = 10, 
    beta_lb: float = -2 * np.pi, 
    beta_ub: float = 2 * np.pi, 
    gamma_lb: float = -2 * np.pi, 
    gamma_ub: float = 2 * np.pi,
    layers: int = 1,
    **kwargs
) -> dict:
    """
    Build a landscape plot for the QAOA applied to the Maxcut problem on a given graph.

    Parameters:
    G : networkx.Graph
        The graph for which the QAOA landscape is being computed.
    mesh_size : int
        The number of points in the mesh grid for beta and gamma.
    beta_lb : float, default -2*pi
        The lower bound for the beta parameter.
    beta_ub : float, default 2*pi
        The upper bound for the beta parameter.
    gamma_lb : float, default -2*pi
        The lower bound for the gamma parameter.
    gamma_ub : float, default 2*pi
        The upper bound for the gamma parameter.
    layers : int, default 1
        Number of QAOA layers
        
    Returns:
    dict
        A dictionary containing the beta, gamma, and objective values for the QAOA applied to Maxcut.
    """
    # Convert graph to adjacency matrix and setup Maxcut
    adjacency_matrix = nx.adjacency_matrix(G)
    max_cut = Maxcut(adjacency_matrix)
    qubitOp, offset = max_cut.to_quadratic_program().to_ising()

    # Define QAOA parameters
    beta = ParameterVector('β', length=layers)
    gamma = ParameterVector('γ', length=layers)
    
    # If layers > 1 then fix those layers
    if layers > 1:
        # Check **kwargs if fixed values exist
        if 'fixed_betas' in kwargs and 'fixed_gammas' in kwargs:
            fixed_betas = kwargs['fixed_betas']
            fixed_gammas = kwargs['fixed_gammas']
        else:
            raise TypeError("Provide Fixed Values for Beta and Gamma as a List")
        
    # Initialize the QAOA circuit with the parameters
    qaoa = QAOA(optimizer=SLSQP(), reps=layers, initial_point=[beta, gamma])

    # Create linspace for beta and gamma
    beta_vals = np.linspace(beta_lb, beta_ub, mesh_size)
    gamma_vals = np.linspace(gamma_lb, gamma_ub, mesh_size)
    
    # Check if layers > 1
    if layers > 1:
        print("DOING MORE THAN 1 LAYER")
        obj_vals = parallel_computation_n_layers_fixed(beta_vals, gamma_vals, fixed_betas, fixed_gammas, qubitOp, qaoa)
        # obj_vals = non_parallel_computation_n_layers_fixed(beta_vals, gamma_vals, fixed_betas, fixed_gammas, qubitOp, qaoa)
    else:
        obj_vals = parallel_computation(beta_vals, gamma_vals, qubitOp, qaoa)
    
    return {'beta': beta_vals, 'gamma': gamma_vals, 'obj_vals': obj_vals}