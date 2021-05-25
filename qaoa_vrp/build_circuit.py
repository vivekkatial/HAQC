import base64
import uuid
from collections import defaultdict
from itertools import count
from qiskit.providers.aer import QasmSimulator
import networkx as nx
import numpy as np
from qiskit import BasicAer, execute
from qiskit.providers.aer import QasmSimulator
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.aqua.components.optimizers import ADAM, AQGD, COBYLA, NELDER_MEAD
from qiskit.circuit import Parameter
from qiskit.finance.applications.ising import portfolio
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.converters import QuadraticProgramToQubo
from qiskit.optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
)


def build_qubos(clusters, depot_info, A=30):
    """A function to build QUBO formulations using qiskit
        clusters (list): A list of `networkX` graph objects that contain the clusters (including depot)
        depot_info (dict): A dictionary consisting of the depot information
        A (int): A penalty (defualt is `A=30` as discussed in Feld)
    Returns:
        list: A list of QUBO formulations
    """

    qubos = []
    for subgraph in clusters:
        cluster = clusters[subgraph]
        constrained_qp = QuadraticProgram()
        connected_elems = list(cluster.edges)
        no_nodes = len(cluster.nodes)
        # vars_lookup =  create_vars_lookup(cluster, depot_id) Create vars_look_up for indexing
        # Create binary variables for each node at each timestep
        binary_vars = []
        for node in cluster.nodes:
            if node == depot_info["id"]:  # Not including depot
                continue
            for i in range(no_nodes - 1):  # no_timesteps = no_nodes - depot
                binary_vars.append("X" + str(node) + str(i + 1))
        for var in binary_vars:
            constrained_qp.binary_var(var)
        # Calculate constraint coefficients (linear and quadratic terms)
        linear = {}
        quadratic = {}
        # Linear cost for travelling from depot to a node in the first and last step
        for edge in connected_elems:
            if edge[0] == depot_info["id"]:
                # Starting node
                start_var = "X" + str(edge[1]) + str(1)
                linear[start_var] = cluster[edge[0]][edge[1]]["cost"]
                # Last node
                last_var = "X" + str(edge[1]) + str(no_nodes - 1)
                linear[last_var] = cluster[edge[0]][edge[1]]["cost"]
            # Allowing for having the depot as the 2nd node on the ordered edge pair (so just reversing the code above)
            elif edge[1] == depot_info["id"]:
                # Starting node
                start_var = "X" + str(edge[0]) + str(1)
                linear[start_var] = cluster[edge[0]][edge[1]]["cost"]
                # Last node
                last_var = "X" + str(edge[0]) + str(no_nodes - 1)
                linear[last_var] = cluster[edge[0]][edge[1]]["cost"]
            else:  # Now quadratic cost for travelling between nodes apart from depot
                for j in range(no_nodes - 2):
                    pairing = (
                        "X" + str(edge[0]) + str(j + 1),
                        "X" + str(edge[1]) + str(j + 2),
                    )
                    quadratic[pairing] = cluster[edge[0]][edge[1]]["cost"]
                    # Backwards directions
                    pairing = (
                        "X" + str(edge[1]) + str(j + 1),
                        "X" + str(edge[0]) + str(j + 2),
                    )
                    quadratic[pairing] = cluster[edge[0]][edge[1]]["cost"]
        for node in cluster.nodes:
            if node == depot_info["id"]:  # Not depot
                continue
            # If node is not connected to the depot, increase cost when starting at that node
            if (depot_info["id"], node) not in connected_elems:
                var = "X" + str(node) + str(1)
                if var in linear:
                    linear[var] += A
                else:
                    linear[var] = A
                # Likewise if the ending node is not connected to the depot
                var = "X" + str(node) + str(no_nodes - 1)
                if var in linear:
                    linear[var] += A
                else:
                    linear[var] = A
            for node2 in cluster.nodes:
                if (
                    node2 != depot_info["id"]
                    and node2 != node
                    and (node, node2) not in cluster.edges
                ):  # Not depot, and different node, and if the two nodes are not connected, add penalty when travelling between them
                    for j in range(no_nodes - 2):
                        # Adding cost for travelling from node to node2,
                        pairing = (
                            "X" + str(node) + str(j + 1),
                            "X" + str(node2) + str(j + 2),
                        )
                        if pairing in quadratic:
                            quadratic[pairing] += A
                        else:
                            quadratic[pairing] = A
                        # Reverse Direction
                        pairing = (
                            "X" + str(node2) + str(j + 1),
                            "X" + str(node) + str(j + 2),
                        )
                        if pairing in quadratic:
                            quadratic[pairing] += A
                        else:
                            quadratic[pairing] = A
            # Input linear and quadratic terms for minimizing qubo objective function
            constrained_qp.minimize(linear=linear, quadratic=quadratic)
            # Now add constraints to make sure each node is visited exactly once:
            node_constraint = {}
            for r in range(no_nodes - 1):
                var = "X" + str(node) + str(r + 1)
                node_constraint[var] = 1
            constrained_qp.linear_constraint(
                linear=node_constraint,
                sense="==",
                rhs=1,
                name="visit_node{}_once".format(node),
            )
        # Now add constraints to make sure each vehicle is only at one node for each timestep:
        for r in range(no_nodes - 1):
            timestep_constraint = {}
            for node in cluster.nodes:
                if node == depot_info["id"]:
                    continue
                var = "X" + str(node) + str(r + 1)
                timestep_constraint[var] = 1
            constrained_qp.linear_constraint(
                linear=timestep_constraint,
                sense="==",
                rhs=1,
                name="timestep{}_one_node".format(r + 1),
            )
        # Convert unconstrained to QUBO
        converter = QuadraticProgramToQubo(penalty=A)
        qubo = converter.convert(constrained_qp)
        # Append each QP to QUBO
        qubos.append(qubo)
    return qubos


def qubo_to_qaoa(qubo, num_layers):
    """Function to build QAOA circuit from QUBO formulations

    Args:
        qubo (object): Qiskit `QuadraticProgram` object
        num_layers (int): Number of layers (p)

    Returns:
        [type]: A tuple where element [0] is a QAOA qiskit circuit object, and [1] is
    """
    op, offset = qubo.to_ising()

    quantum_instance = QuantumInstance(
        QasmSimulator(method='matrix_product_state'),
        seed_simulator=aqua_globals.random_seed,
        seed_transpiler=aqua_globals.random_seed,
    )

    qaoa = QAOA(
        operator=op,
        p=num_layers,
        quantum_instance=quantum_instance,
        initial_point=list(2 * np.pi * np.random.random(2 * num_layers)),
    )

    # Create QAOA parameter list:
    params_expr = []

    for a in range(num_layers):  # "gamma_n" parameters
        params_expr.append(Parameter("a" + str(a + 1)))

    for b in range(num_layers):  # "beta_n" parameters
        params_expr.append(Parameter("b" + str(b + 1)))

    # Create qaoa circuit with arbitrary parameters
    circuit = qaoa.construct_circuit(params_expr[0 : 2 * num_layers])[0]

    return qaoa, circuit, params_expr


def measure_energy(qubo, qc):
    """Measure QAOA Circuit Energy

    Args:
        qubo (object): qiskit QUBO obejct
        qc (object): qiskit QuantumCircuit object

    Returns:
        float : Energy of QC
    """
    state_vector = (
        execute(qc, backend=Aer.get_backend("statevector_simulator"))
        .result()
        .get_statevector()
    )
    hamiltonian = qubo.to_ising()[0].to_matrix(massive=True)
    ham_state = np.matmul(hamiltonian, state_vector)
    energy = np.dot(ham_state, np.conjugate(state_vector))
    energy = energy.real
    return energy


def solve_qubo_qaoa(qubo, p, points):
    """
    Create QAOA from given qubo, and solves for both the exact value and the QAOA

    Args:
        qubo (object): qiskit QUBO object
        p (int): the number of layers in the QAOA circuit (p value)

    Returns:
        exact_result (dict): the exact result of the MinimumEigenOptimizer
        qaoa_result (dict): the result of running the QAOA
    """

    exact_mes = NumPyMinimumEigensolver()
    exact = MinimumEigenOptimizer(exact_mes)
    exact_result = exact.solve(qubo)

    op, offset = qubo.to_ising()

    qaoa = QAOA(operator=op, p=p,  initial_point=list(2*np.pi*np.random.random(2*p)), optimizer=NELDER_MEAD())
    # qaoa = QAOA(operator=op, p=p, initial_point=points, optimizer=NELDER_MEAD())

    quantum_instance = QuantumInstance(
        QasmSimulator(method='matrix_product_state'),
        seed_simulator=aqua_globals.random_seed,
        seed_transpiler=aqua_globals.random_seed,
    )

    qaoa_result = qaoa.run(quantum_instance)

    return qaoa_result, exact_result, offset


def interp_point(optimal_point):
    """Method to interpolate to next point from the optimal point found from the previous layer

    Args:
        optimal_point (np.array): Optimal point from previous layer

    Returns:
        point (list): the informed next point
    """
    optimal_point = list(optimal_point)
    p = int(len(optimal_point) / 2)
    gammas = [0] + optimal_point[0:p] + [0]
    betas = [0] + optimal_point[p : 2 * p] + [0]
    interp_gammas = [0] + gammas
    interp_betas = [0] + betas
    for i in range(1, p + 2):
        interp_gammas[i] = gammas[i - 1] * (i - 1) / p + gammas[i] * (p + 1 - i) / p
        interp_betas[i] = betas[i - 1] * (i - 1) / p + betas[i] * (p + 1 - i) / p

    point = interp_gammas[1 : p + 2] + interp_betas[1 : p + 2]

    return point


def get_fourier_points(last_expectation_value, p):
    """"""

    points = (
        list(last_expectation_value[:p]) + [0] + list(last_expectation_value[p:]) + [0]
    )

    print(points)

    return points


def index_to_selection(i, num_assets):
    """
    Creates an index for the string value suggestion (used in print_result)

    Args:
        i (int): the index of the given string
        num_assets (int): the number of qubits in the given index string

    Returns:
        x (dict): dictionary result of the given index in binary
    """

    s = "{0:b}".format(i).rjust(num_assets)
    x = np.array([1 if s[i] == "1" else 0 for i in reversed(range(num_assets))])
    return x


def print_result(qubo, result, num_qubits, exact_value):
    """
    Prints the results of the QAOA in a nice form

    Args:
        qubo (object): qiskit QUBO object
        result (dict): the result of the QAOA
        num_qubits (int): the number of qubits in the QAOA circuit
    """
    # import pdb
    # pdb.set_trace()
    # eigenvector = (
    #     result.eigenstate
    #     if isinstance(result.eigenstate, np.ndarray)
    #     else result.eigenstate.to_matrix()
    # )
    # probabilities = np.abs(eigenvector) ** 2
    probabilities = []
    for probability in result.eigenstate.values():
        probabilities.append(float(probability)/1024)
    # import pdb
    # pdb.set_trace()
    i_sorted = reversed(np.argsort(probabilities))
    print("----------------- Full result ---------------------")
    print("index\tselection\t\tvalue\t\tprobability")
    print("---------------------------------------------------")
    exact_probs = []
    solution_data = {}
    for index, i in enumerate(i_sorted):
        x = index_to_selection(i, num_qubits)
        probability = probabilities[i]
        if index == 0 or index == 1:
            print(
                "%d\t%10s\t%.4f\t\t%.4f"
                % (index, x, qubo.objective.evaluate(x), probability)
            )
        if qubo.objective.evaluate(x) == exact_value:
            print(
                "%d\t%10s\t%.4f\t\t%.4f"
                % (index, x, qubo.objective.evaluate(x), probability)
            )
            exact_probs.append(probability)
        solution_data[f"{x}"] = {
            "index": index,
            "energy": qubo.objective.evaluate(x),
            "probability": probability,
        }
    print("\n")
    return exact_probs, solution_data


def assign_parameters(circuit, params_expr, params):
    """
    Args:
        circuit ([type]): [description]
        params_expr ([type]): [description]
        params ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Assign params_expr -> params
    circuit2 = circuit.assign_parameters(
        {params_expr[i]: params[i] for i in range(len(params))}, inplace=False
    )
    return circuit2


def to_hamiltonian_dicts(quadratic_program: QuadraticProgram):
    """
    Converts a Qiskit QuadraticProgram for QAOA to pair of dictionaries representing the
    Hamiltonian. Based on qiskit.optimization.QuadraticProgram.to_ising.

    Args:
        quadratic_program (QuadraticProgram): Qiskit QuadraticProgram representing a
            QAOA problem

    Returns:
        num_nodes (int): Integer number of qubits
        linear_terms (defaultdict[int, float]): Coefficients of Z_i terms in the
            Hamiltonian.
        quadratic_terms (defaultdict[Tuple[int, int], float]): Coefficients of Z_i Z_j
            terms in the Hamiltonian
    """

    # if problem has variables that are not binary, raise an error
    if quadratic_program.get_num_vars() > quadratic_program.get_num_binary_vars():
        raise ValueError(
            "The type of variable must be a binary variable. "
            "Use a QuadraticProgramToQubo converter to convert "
            "integer variables to binary variables. "
            "If the problem contains continuous variables, "
            "currently we can not apply VQE/QAOA directly. "
            "you might want to use an ADMM optimizer "
            "for the problem. "
        )

    # if constraints exist, raise an error
    if quadratic_program.linear_constraints or quadratic_program.quadratic_constraints:
        raise ValueError(
            "An constraint exists. "
            "The method supports only model with no constraints. "
            "Use a QuadraticProgramToQubo converter. "
            "It converts inequality constraints to equality "
            "constraints, and then, it converters equality "
            "constraints to penalty terms of the object function."
        )

    # initialize Hamiltonian.
    num_nodes = quadratic_program.get_num_vars()

    linear_terms = defaultdict(float)
    quadratic_terms = defaultdict(float)

    # set a sign corresponding to a maximized or minimized problem.
    # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
    sense = quadratic_program.objective.sense.value

    # convert linear parts of the object function into Hamiltonian.
    for i, coeff in quadratic_program.objective.linear.to_dict().items():
        linear_terms[i] -= sense * coeff / 2

    # create Pauli terms
    for pair, coeff in quadratic_program.objective.quadratic.to_dict().items():
        weight = sense * coeff / 4

        i, j = sorted(pair)
        if i != j:
            quadratic_terms[i, j] += weight
        linear_terms[i] -= weight
        linear_terms[j] -= weight

    return num_nodes, linear_terms, quadratic_terms


def generate_circuit_pb(n, linear_terms, quadratic_terms, gamma, beta):
    """
    Constructs a QAOA circuit in the Protobuf format used by QUI.

    Args:
        n (int): Integer number of qubits
        linear_terms (defaultdict[int, float]): Coefficients of Z_i terms in the
            Hamiltonian.
        quadratic_terms (defaultdict[Tuple[int, int], float]): Coefficients of Z_i Z_j
            terms in the Hamiltonian
        gamma (List[float]): gamma values to use in the QAOA circuit, one for each layer
        beta (List[float]): beta values to use in the QAOA circuit, one for each layer

    Returns:
        pb_string (str): base64 string representation of the QAOA circuit in QUI's
            Protobuf format
    """
    if len(gamma) != len(beta):
        raise ValueError("Incorrect number of optimisation parameters provided.")

    import proto.circuit_pb2 as circuit_pb2

    circuit = circuit_pb2.Circuit()
    circuit.id = str(uuid.uuid4())
    circuit.num_qubits = n

    time_block_itercount = count()

    def new_time_block():
        time_block = circuit.time_blocks.add()
        time_block.id = str(uuid.uuid4())
        return next(time_block_itercount), time_block

    _, hadamard_time_block = new_time_block()
    for i in range(n):
        component = hadamard_time_block.components.add()
        component.type = circuit_pb2.Component.GATE
        component.gate.type = circuit_pb2.Gate.H
        component.qubits[:] = [i]

    def new_time_block_group(name, initial_tb_idx, final_tb_idx, expanded=False):
        group = circuit.time_block_groups.add()
        group.name = name
        group.start_time_block_idx = initial_tb_idx
        group.end_time_block_idx = final_tb_idx
        group.expanded = expanded

    def linear_layer(g):
        _, time_block = new_time_block()
        for i, coeff in linear_terms.items():
            if coeff:
                component = time_block.components.add()
                component.type = circuit_pb2.Component.GATE
                component.gate.type = circuit_pb2.Gate.ARB
                component.qubits[:] = [i]
                component.gate.arb_params.xyz_axes.x.numeric = 0
                component.gate.arb_params.xyz_axes.y.numeric = 0
                component.gate.arb_params.xyz_axes.z.numeric = 1
                component.gate.arb_params.rotation_angle.numeric = 2 * g * coeff
                component.gate.arb_params.global_phase.numeric = 0

    def quadratic_layer(g):
        def cnot(i, j):
            _, time_block = new_time_block()
            component = time_block.components.add()
            component.type = circuit_pb2.Component.GATE
            component.gate.type = circuit_pb2.Gate.NOT
            component.qubits[:] = [j]
            component.controls_one[:] = [i]

        for (i, j), coeff in quadratic_terms.items():
            if coeff:
                cnot(i, j)

                _, time_block = new_time_block()
                component = time_block.components.add()
                component.type = circuit_pb2.Component.GATE
                component.gate.type = circuit_pb2.Gate.ARB
                component.qubits[:] = [j]
                component.gate.arb_params.xyz_axes.x.numeric = 0
                component.gate.arb_params.xyz_axes.y.numeric = 0
                component.gate.arb_params.xyz_axes.z.numeric = 1
                component.gate.arb_params.rotation_angle.numeric = 2 * g * coeff
                component.gate.arb_params.global_phase.numeric = 0

                cnot(i, j)

    def x_layer(b):
        _, x_time_block = new_time_block()
        for i in range(n):
            component = x_time_block.components.add()
            component.type = circuit_pb2.Component.GATE
            component.gate.type = circuit_pb2.Gate.ARB
            component.qubits[:] = [i]
            component.gate.arb_params.xyz_axes.x.numeric = 1
            component.gate.arb_params.xyz_axes.y.numeric = 0
            component.gate.arb_params.xyz_axes.z.numeric = 0
            component.gate.arb_params.rotation_angle.numeric = 2 * b
            component.gate.arb_params.global_phase.numeric = 0

    def new_qaoa_layer(p, b, g):
        initial_tb_idx, _ = new_time_block()
        linear_layer(g)
        quadratic_layer(g)
        x_layer(b)
        final_tb_idx, _ = new_time_block()
        new_time_block_group(f"p = {i}", initial_tb_idx, final_tb_idx)

    for p, (b, g) in enumerate(zip(beta, gamma)):
        new_qaoa_layer(p, b, g)

    _, measurement_time_block = new_time_block()
    for i in range(n):
        component = measurement_time_block.components.add()
        component.id = str(uuid.uuid4())
        component.type = circuit_pb2.Component.GATE
        component.gate.type = circuit_pb2.Gate.MEASURE
        component.qubits[:] = [i]

    return str(base64.b64encode(circuit.SerializeToString()), "utf-8")
