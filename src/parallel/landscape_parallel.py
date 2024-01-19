import numpy as np
from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA
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
def parallel_computation(gamma, beta, qubitOp, qaoa):
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


def non_parallel_computation_n_layers_fixed(gamma, beta, qubitOp, qaoa):
    obj_vals = np.zeros((len(gamma), len(beta)))
    for i, gamma_val in enumerate(gamma):
        for j, beta_val in enumerate(beta):
            fixed_beta = [0.1, 0.2, 0.3]
            fixed_beta.append(beta_val)
            beta = fixed_beta

            fixed_gamma = [0.3, 0.2, 0.1]
            fixed_gamma.append(gamma_val)
            gamma = fixed_gamma

            obj_vals[i, j] = compute_expectation_value_n_layers(
                beta, gamma, qubitOp, qaoa
            )

    return obj_vals


def parallel_computation_n_layers_fixed(
    gamma, beta, fixed_gammas, fixed_betas, qubitOp, qaoa
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
