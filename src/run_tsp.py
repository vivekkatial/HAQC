import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import TravelingSalesmanProblem

# Define the list of cities and their coordinates
cities = ['A', 'B', 'C', 'D']
coordinates = [(0, 0), (1, 0), (1, 1), (0, 1)]

# Create the TSP instance
tsp = TravelingSalesmanProblem(cities, coordinates)

# Formulate the TSP as a QUBO
qp = tsp.to_quadratic_program()

# Define the quantum instance
seed = 123
algorithm_globals.random_seed = seed
qi = QuantumInstance(
    Aer.get_backend('qasm_simulator'),
    shots=8192,
    seed_simulator=seed,
    seed_transpiler=seed,
)

# Solve the QUBO using QAOA
optimizer = COBYLA()
qaoa_mes = QAOA(optimizer, reps=1, quantum_instance=qi)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qp)

# Decode the solution
tsp.interpret(result)
