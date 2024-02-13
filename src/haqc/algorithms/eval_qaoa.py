import numpy as np
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import NELDER_MEAD
from qiskit.utils import QuantumInstance
import logging

def eval_qaoa(initial_point, qubitOp):
    """
    Evaluates the Quantum Approximate Optimization Algorithm (QAOA) for a given problem.

    Parameters:
    initial_point (numpy.ndarray): An array of initial values for the QAOA parameters.
                                 The length of the array must be twice the number of layers.
    qubitOp (OperatorBase): The operator representing the problem Hamiltonian.

    Returns:
    dict: A dictionary containing the results of the QAOA computation.
          It includes the minimum eigenvalue, optimal parameters, and other information.

    This function initializes the QAOA algorithm with the specified parameters and
    runs the optimization to find the minimum eigenvalue of the given Hamiltonian.

    The QAOA algorithm is configured with the Nelder-Mead optimizer, a quantum simulator backend,
    and a callback function to store intermediate results.

    Note: The QAOA algorithm's performance heavily depends on the choice of initial parameters,
          number of layers, and the problem Hamiltonian.

    Example:
    initial_point = np.random.rand(4)  # Random initial parameters for 2 layers
    qaoa_result = eval_qaoa(initial_point, qubitOp)
    print(qaoa_result)
    """    

    # These are done to mimic the parameter fixing paper
    optimizer = NELDER_MEAD(maxfev=1000, xatol=0.0001)
    backend = Aer.get_backend("aer_simulator_statevector")
    n_layers = len(initial_point) // 2
    quantum_instance = QuantumInstance(backend)

    # Callback function to store intermediate values
    intermediate_values = []

    def store_intermediate_result(eval_count, parameters, mean, std):
        if eval_count % 100 == 0:
            print(
                f"{type(optimizer).__name__} iteration {eval_count} \t cost function {mean}"
            )
        betas = parameters[:n_layers]  # Extracting beta values
        gammas = parameters[n_layers:]  # Extracting gamma values
        intermediate_values.append(
            {
                'eval_count': eval_count,
                'parameters': {'gammas': gammas, 'betas': betas},
                'mean': mean,
                'std': std,
            }
        )

    qaoa = QAOA(
        optimizer=optimizer,
        reps=n_layers,
        initial_point=initial_point,
        callback=store_intermediate_result,
        quantum_instance=quantum_instance,
        include_custom=True,
    )
    algo_result = qaoa.compute_minimum_eigenvalue(qubitOp)

    return algo_result
