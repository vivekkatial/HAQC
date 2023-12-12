import numpy as np
from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.algorithms import QAOA
from qiskit.quantum_info import Statevector
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Function to compute expectation value in parallel
def compute_expectation_value(gamma_val, beta_val, qubitOp, qaoa):
    """ Compute expectation value in parallel
    gamma_val: gamma value
    beta_val: beta value
    qubitOp: qubit operator
    qaoa: qaoa instance

    Returns: expectation value
    """
    qc = qaoa.construct_circuit([gamma_val, beta_val], operator=qubitOp)[0]
    backend = Aer.get_backend('aer_simulator')
    statevector = Statevector.from_instruction(qc)
    expectation = statevector.expectation_value(qubitOp).real
    return expectation

# Parallel computation of the objective function values
def parallel_computation(gamma, beta, qubitOp, qaoa):
    """ Parallel computation of the objective function values
    gamma: array of gamma values
    beta: array of beta values
    qubitOp: qubit operator
    qaoa: qaoa instance

    Returns: array of objective function values
    """
    obj_vals = np.zeros((len(gamma), len(beta)))

    with ProcessPoolExecutor() as executor:
        futures = {}
        for i, gamma_val in enumerate(gamma):
            for j, beta_val in enumerate(beta):
                future = executor.submit(compute_expectation_value, gamma_val, beta_val, qubitOp, qaoa)
                futures[future] = (i, j)

        for future in tqdm(as_completed(futures), desc="Progress", total=len(futures)):
            i, j = futures[future]
            obj_vals[i, j] = future.result()

    return obj_vals
