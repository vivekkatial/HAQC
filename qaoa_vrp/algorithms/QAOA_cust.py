"""
Description:
    Defines the QAOACustom and CircuitSamplerCustom classes that replace the 
    qiskit QAOA and CircuitSampler classes respectively.
    It is more easily customised than qiskit's built in ones and includes a variety of helper methods.

Author: Gary Mooney
    Adapted from Qiskit 0.26.2 documentation

Example 1: Full usage example.
    from QAOAEx import (QAOACustom, convert_to_fourier_point, print_qaoa_solutions, 
                    get_quadratic_program_from_ising_hamiltonian_terms, 
                    output_ising_graph, get_ising_graph_from_ising_hamiltonian_terms,
                    convert_from_fourier_point)

    backend = Aer.get_backend('aer_simulator_matrix_product_state')
    quantum_instance = QuantumInstance(backend, shots=8192)
    optimizer = NELDER_MEAD()
    couplings = [(0, 1, -1.0), (0, 2, 1.0), (1, 2, 1.0), (2, 3, -1.0), (0, 3, 0.5)] # formatted as List[Tuple[int, int, float]]
    local_fields = {0: 0.2, 1: -0.3, 2: 0.0, 3: 0.5} # formatted as Mapping[int, float]
    constant_term = 1.0 # formatted as float
    quadratic_program = get_quadratic_program_from_ising_hamiltonian_terms(couplings = couplings, 
                                                                        local_fields = local_fields, 
                                                                        constant_term = constant_term,
                                                                        output_ising_graph_filename = "example-ising_graph")
    qaoa_instance = QAOACustom(quantum_instance = quantum_instance, 
                                reps = 2, 
                                force_shots = False, 
                                optimizer = optimizer, 
                                qaoa_name = "example_qaoa")

    operator, offset = quadratic_program.to_ising()

    initial_point = [0.40784, 0.73974, -0.53411, -0.28296]
    print()
    print("Solving QAOA...")
    qaoa_results = qaoa_instance.solve(operator, initial_point)

    qaoa_results_eigenstate = qaoa_results.eigenstate
    print("optimal_value:", qaoa_results.optimal_value)
    print("optimal_parameters:", qaoa_results.optimal_parameters)
    print("optimal_point:", qaoa_results.optimal_point)
    print("optimizer_evals:", qaoa_results.optimizer_evals)

    solutions = qaoa_instance.get_optimal_solutions_from_statevector(qaoa_results_eigenstate, quadratic_program)
    print_qaoa_QuantumCircuit
    # initial Fourier space point, will be converted to a typical point using 
    # 'convert_from_fourier_point' as per previous line
    initial_fourier_point = [0.5, 0.7]

    # bounds used for the optimiser
    bounds = [(-1, 1)] * len(initial_fourier_point)

    qaoa_results = qaoa_instance.solve(operator, initial_fourier_point, bounds)
    optimal_parameterised_point = qaoa_instance.latest_parameterised_point

Example 3: Post process raw data. This is how QREM could be applied.
           Before the line 'qaoa_results = qaoa_instance.solve(operator, initial_point)' 
           in Example 1, add the following.
    
    # Define a method to process the counts dict. In this case it simply calculates and prints the shot counts.
    def print_shot_count(raw_counts_data):
        shot_count = None
        if len(raw_counts_data) > 0:
            if isinstance(raw_counts_data[0], dict):
                shot_count = sum(raw_counts_data[0].values())
            elif isinstance(raw_counts_data[0], list) and len(raw_counts_data[0]) > 0:
                shot_count = sum(raw_counts_data[0][0].values())
            else:
                raise Exception("Error: Wrong format 'raw_counts_data', execting List[Dict] or List[List[Dict]]")
        
        print("Raw data shot count:", shot_count)

        return raw_counts_data

    # set the raw data processing method. If using a statevector simulator 
    # with force_shot = False, (in qaoa_instance) then raw processing will not be used.
    qaoa_instance.set_post_process_raw_data(print_shot_count)
"""

import logging
import time as time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx  # tool to handle general Graphs
import numpy as np
import math as math
from qiskit import QiskitError
from qiskit.algorithms import QAOA
from qiskit.algorithms.exceptions import AlgorithmError
from qiskit.algorithms.minimum_eigen_solvers.minimum_eigen_solver import MinimumEigensolverResult
from qiskit.algorithms.minimum_eigen_solvers.vqe import VQE
from qiskit.algorithms.optimizers import Optimizer
from qiskit.algorithms.variational_algorithm import VariationalAlgorithm, VariationalResult
from qiskit.circuit import ClassicalRegister, Parameter, QuantumCircuit
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
#from qiskit.algorithms.minimum_eigen_solvers
from qiskit.opflow import (CircuitSampler, CircuitStateFn, DictStateFn,
                           ExpectationBase, I, OperatorBase, StateFn)
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.gradients import GradientBase
from qiskit.providers import Backend, BaseBackend
from qiskit.quantum_info import Statevector
from qiskit.tools.visualization import circuit_drawer
from qiskit.utils import algorithm_globals
from qiskit.utils.backend_utils import is_aer_provider
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.utils.validation import validate_min
from qiskit.visualization import plot_histogram
from qiskit_optimization import QuadraticProgram

logger = logging.getLogger(__name__)

###############
### Classes
###############

class CircuitSamplerCustom(CircuitSampler):
    # a function pointer that processes returned results from execution when sample_circuits is called.
    # post_process_raw_data(Result) -> Result
    _post_process_raw_data: Optional[Callable[[Union[List[Dict[str, int]], List[List[Dict[str, int]]]]], Union[List[Dict[str, int]], List[List[Dict[str, int]]]]]] = None
    _shots = None
    _log_text = print
    _force_shots = False
    _sampler_name = ""
    _output_circuit_when_sample = False

    def __init__(self,
                 backend: Union[Backend, BaseBackend, QuantumInstance],
                 statevector: Optional[bool] = None,
                 param_qobj: bool = False,
                 attach_results: bool = False,
                 caching: str = 'last',
                 sampler_name: str = "",
                 force_shots: bool = False,
                 output_circuit_when_sample: bool = False,
                 log_text: Optional[Callable[..., Any]] = print
                ) -> None:
        """
        Args:
            backend: The quantum backend or QuantumInstance to use to sample the circuits.
            statevector: If backend is a statevector backend, whether to replace the
                CircuitStateFns with DictStateFns (from the counts) or VectorStateFns (from the
                statevector). ``None`` will set this argument automatically based on the backend.
            param_qobj: Whether to use Aer's parameterized Qobj capability to avoid re-assembling
                the circuits.
            attach_results: Whether to attach the data from the backend ``Results`` object for
                a given ``CircuitStateFn``` to an ``execution_results`` field added the converted
                ``DictStateFn`` or ``VectorStateFn``.
            caching: The caching strategy. Can be `'last'` (default) to store the last operator
                that was converted, set to `'all'` to cache all processed operators.
            sampler_name: Name used when outputting text or files to help identify CircuitSamplerCustom instance.
            force_shots: If quantum instance returns a statevector, then convert into shots instead.
            output_circuit_when_sample: Whether to output circuit using circuit_drawer whenever circuit is sampled.
            log_text: Used for text output, replacement to the default print method to make logging easy.
                If None, no text output can occur.
        Raises:
            ValueError: Set statevector or param_qobj True when not supported by backend.
        """
        super().__init__(backend=backend,
                         statevector=statevector,
                         param_qobj=param_qobj,
                         attach_results=attach_results,
                         caching=caching)

        self._sampler_name = sampler_name
        self._log_text = log_text
        # determines whether to use the statevectors directly from simulations is available
        # If true, counts are sampled from statevector (default 8192)
        self._force_shots = force_shots
        self._output_circuit_when_sample = output_circuit_when_sample

    def set_post_process_raw_data(self, 
                                  post_process_raw_data_method: Optional[Callable[[Union[List[Dict[str, int]], List[List[Dict[str, int]]]]], Union[List[Dict[str, int]], List[List[Dict[str, int]]]]]]
                                 ) -> None:
        """ Uses the specified method to process the raw sampled data executed on the backened whenever circuits are sampled.
        Args:
            post_process_raw_data_method: The method to process the data. 
               Inputs a list f counts dicts List[Dict[str, int]] and outputs the processed list of count dicts List[Dict[str, int]].
               The data could potentially be formatted as a list of a list of dictionaries List[List[Dict[str, int]]]. However, this
               will likely not happen withouth modifying QAOA to do so.
               Each dictionary has the counts for each qubit with the keys containing a string in binary format and separated
               according to the registers in circuit (e.g. ``0100 1110``). The string is little-endian (cr[0] on the right hand side).
               However there will likely only be a single register without modifying QAOA, so the state bitstring should have no spaces. 
        """
        self._post_process_raw_data = post_process_raw_data_method

    def sample_circuits(self,
                        circuit_sfns: Optional[List[CircuitStateFn]] = None,
                        param_bindings: Optional[List[Dict[Parameter, float]]] = None
                       ) -> Dict[int, List[StateFn]]:
        r"""
            Samples the CircuitStateFns and returns a dict associating their ``id()`` values to their
            replacement DictStateFn or VectorStateFn. If param_bindings is provided,
            the CircuitStateFns are broken into their parameterizations, and a list of StateFns is
            returned in the dict for each circuit ``id()``. Note that param_bindings is provided here
            in a different format than in ``convert``, and lists of parameters within the dict is not
            supported, and only binding dicts which are valid to be passed into Terra can be included
            in this list. (Overides method)
        Args:
            circuit_sfns: The list of CircuitStateFns to sample.
            param_bindings: The parameterizations to bind to each CircuitStateFn.
        Returns:
            The dictionary mapping ids of the CircuitStateFns to their replacement StateFns.
        Raises:
            OpflowError: if extracted circuits are empty.
        """
        if not circuit_sfns and not self._transpiled_circ_cache:
            raise OpflowError('CircuitStateFn is empty and there is no cache.')

        #############
        # NOTE:
        # Can modify circuits before execution here.
        # can even manually transpile to specific qubit layout.
        #############


        if circuit_sfns:
            self._transpiled_circ_templates = None
            if self._statevector:
                circuits = [op_c.to_circuit(meas=False) for op_c in circuit_sfns]
            else:
                circuits = [op_c.to_circuit(meas=True) for op_c in circuit_sfns]

            ####### Saving circuit
            if self._output_circuit_when_sample == True:
                filename = "quantum-circuit-" + self._sampler_name + "-params"
                for _, value in param_bindings[0].items():
                    filename += "-" + str(int(1000*value))
                if self._log_text != None:
                    self._log_text("Saving circuit '" + filename + "'...")
                fig = circuit_drawer(circuits[0], filename=filename, output='mpl')
                plt.close(fig)
            #######

            try:
                self._transpiled_circ_cache = self.quantum_instance.transpile(circuits)
            except QiskitError:
                logger.debug(r'CircuitSampler failed to transpile circuits with unbound '
                             r'parameters. Attempting to transpile only when circuits are bound '
                             r'now, but this can hurt performance due to repeated transpilation.')
                self._transpile_before_bind = False
                self._transpiled_circ_cache = circuits
        else:
            circuit_sfns = list(self._circuit_ops_cache.values())

        if param_bindings is not None:
            # if fourier method, then convert param_bindings to another param_bindings, usually larger.
            if self._param_qobj:
                start_time = time.time()
                ready_circs = self._prepare_parameterized_run_config(param_bindings)
                end_time = time.time()
                logger.debug('Parameter conversion %.5f (ms)', (end_time - start_time) * 1000)
            else:
                start_time = time.time()
                ready_circs = [circ.assign_parameters(CircuitSamplerCustom._filter_params(circ, binding))
                               for circ in self._transpiled_circ_cache
                               for binding in param_bindings]
                end_time = time.time()
                logger.debug('Parameter binding %.5f (ms)', (end_time - start_time) * 1000)
        else:
            ready_circs = self._transpiled_circ_cache

        results = self.quantum_instance.execute(ready_circs,
                                                had_transpiled=self._transpile_before_bind)

        if param_bindings is not None and self._param_qobj:
            self._clean_parameterized_run_config()

        # Wipe parameterizations, if any
        # self.quantum_instance._run_config.parameterizations = None


        #############
        # NOTE:
        # Can apply QREM here. But we need to know which qubits were used in order to apply...
        # results.get_counts(circ_index)
        # will need to convert results in case it's a statevector.
        #############

        counts_dicts = []
        for i, op_c in enumerate(circuit_sfns):
            # Taking square root because we're replacing a statevector
            # representation of probabilities.
            reps = len(param_bindings) if param_bindings is not None else 1
            c_statefns = []
            for j in range(reps):
                circ_index = (i * reps) + j

                #counts_dicts[circ_index] = results.get_counts(circ_index)
                circ_results = results.data(circ_index)
                #statevector = results.get_statevector(circ_index)

                if 'expval_measurement' in circ_results.get('snapshots', {}).get(
                        'expectation_value', {}):

                    if self.quantum_instance.run_config.shots != None:
                        shots = self.quantum_instance.run_config.shots
                    else:
                        shots = 8192
                    counts_dicts.append(Statevector(results.get_statevector(circ_index)).sample_counts(shots))
                    #print("DEBUG: From statevector (1): " + str(shots) + " shots")
                elif self._statevector:
                    if self.quantum_instance.run_config.shots != None:
                        shots = self.quantum_instance.run_config.shots
                    else:
                        shots = 8192
                    counts_dicts.append(Statevector(results.get_statevector(circ_index)).sample_counts(shots))
                    #print("counts_dicts[circ_index]", counts_dicts[circ_index])
                    #if self._force_shots == True:
                    #    print("DEBUG: From statevector (2): " + str(shots) + " shots")
                    #else:
                    #    print("DEBUG: From statevector (2) - using statevector")
                else:
                    counts_dicts.append(results.get_counts(circ_index))
                    #print("counts_dicts[circ_index]", counts_dicts[circ_index])
                    shots = 0
                    for count in counts_dicts[circ_index].values():
                        shots += count
                    #print("DEBUG: From counts: " + str(shots) + " shots")
        #print("counts_dicts:", counts_dicts)








        #############
        ### Post process raw counts
        ### NOTE: counts_dicts could be formatted as 
        ###     List[Dict[str, int]] or List[List[Dict[str, int]]]: a list of dictionaries or a list of
        ###     a list of dictionaries. A dictionary has the counts for each qubit with
        ###     the keys containing a string in binary format and separated
        ###     according to the registers in circuit (e.g. ``0100 1110``).
        ###     The string is little-endian (cr[0] on the right hand side).
        ###     
        ###     However the format will most likely always be List[Dict[str, int]]
        ###     with a single register, so the state bitstring will have no spaces. 
        #############
        counts_dicts_new = None
        if self._post_process_raw_data != None:
            if self._force_shots == False and self._statevector and self._log_text != None:
                self._log_text("WARNING: post_process_raw_data method cannot execute on statevector, set force_shots to True or don't use the stavevector simulator.")
            counts_dicts_new = self._post_process_raw_data(counts_dicts)
        else:
            counts_dicts_new = counts_dicts
        #############



        sampled_statefn_dicts = {}
        for i, op_c in enumerate(circuit_sfns):
            # Taking square root because we're replacing a statevector
            # representation of probabilities.
            reps = len(param_bindings) if param_bindings is not None else 1
            c_statefns = []
            for j in range(reps):
                circ_index = (i * reps) + j
                circ_results = results.data(circ_index)

                if self._force_shots == False:
                    if 'expval_measurement' in circ_results.get('snapshots', {}).get(
                            'expectation_value', {}):
                        snapshot_data = results.data(circ_index)['snapshots']
                        avg = snapshot_data['expectation_value']['expval_measurement'][0]['value']
                        if isinstance(avg, (list, tuple)):
                            # Aer versions before 0.4 use a list snapshot format
                            # which must be converted to a complex value.
                            avg = avg[0] + 1j * avg[1]
                        # Will be replaced with just avg when eval is called later
                        num_qubits = circuit_sfns[0].num_qubits
                        result_sfn = DictStateFn('0' * num_qubits,
                                                 is_measurement=op_c.is_measurement) * avg
                    elif self._statevector:
                        result_sfn = StateFn(op_c.coeff * results.get_statevector(circ_index),
                                             is_measurement=op_c.is_measurement)
                    else:
                        shots = self.quantum_instance._run_config.shots
                        result_sfn = StateFn({b: (v / shots) ** 0.5 * op_c.coeff
                                             for (b, v) in counts_dicts_new[circ_index].items()},
                                             is_measurement=op_c.is_measurement)
                else:
                    #result_sfn = ConvertCountsToStateFunction(counts_dicts_new[circ_index], shots=None, op_c=op_c)
                    shots = 0
                    for _, count in counts_dicts_new[circ_index].items():
                        shots += count

                    result_sfn = StateFn({b: (v / shots) ** 0.5 * op_c.coeff
                                          for (b, v) in counts_dicts_new[circ_index].items()},
                                          is_measurement=op_c.is_measurement)
                    # use statefn instead of dictstatefn
                    if self._statevector:
                        result_sfn = result_sfn.to_matrix_op(massive=True)

                if self._attach_results:
                    result_sfn.execution_results = circ_results
                c_statefns.append(result_sfn)
            sampled_statefn_dicts[id(op_c)] = c_statefns
        return sampled_statefn_dicts






class QAOACustom(QAOA):
    # a function pointer that processes returned results from execution when sample_circuits is called.
    # post_process_raw_data(Result) -> Result
    _post_process_raw_data: Optional[Callable[[Union[List[Dict[str, int]], List[List[Dict[str, int]]]]], Union[List[Dict[str, int]], List[List[Dict[str, int]]]]]] = None

    _qaoa_name = ""
    _force_shots = False
    _log_text = print
    _output_circuit_when_sample = False
    _reps = 1
    _mixer = None
    _initial_state = None
    _optimiser_parameter_bounds = None
    _parameterise_point_for_energy_evaluation: Callable[[Union[List[float], np.ndarray], int], List[float]] = None
    
    # After solving/optimising using a custom parameterisation, the member 'latest_parameterised_point' should 
    # contain the solution parameterised point returned by the optimiser.
    latest_parameterised_point = None
    

    def __init__(self,
                 optimizer: Optimizer = None,
                 reps: int = 1,
                 initial_state: Optional[QuantumCircuit] = None,
                 mixer: Union[QuantumCircuit, OperatorBase] = None,
                 initial_point: Union[List[float], np.ndarray, None] = None,
                 gradient: Optional[Union[GradientBase, Callable[[Union[np.ndarray, List]],
                                                                 List]]] = None,
                 expectation: Optional[ExpectationBase] = None,
                 include_custom: bool = False,
                 max_evals_grouped: int = 1,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 quantum_instance: Optional[
                     Union[QuantumInstance, BaseBackend, Backend]] = None,
                 qaoa_name: str = "",
                 force_shots: bool = False,
                 output_circuit_when_sample: bool = False,
                 log_text: Optional[Callable[..., Any]] = print
                ) -> None:
        """
        Args:
            optimizer: A classical optimizer.
            reps: the integer parameter :math:`p` as specified in https://arxiv.org/abs/1411.4028,
                Has a minimum valid value of 1.
            initial_state: An optional initial state to prepend the QAOA circuit with
            mixer: the mixer Hamiltonian to evolve with or a custom quantum circuit. Allows support
                of optimizations in constrained subspaces as per https://arxiv.org/abs/1709.03489
                as well as warm-starting the optimization as introduced
                in http://arxiv.org/abs/2009.10095.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then it will simply compute a random one.
                QAOA parameters (a list ordered as: [all_ZZ_gamma_values] + [all_X_beta_values]).
            gradient: An optional gradient operator respectively a gradient function used for
                      optimization.
            expectation: The Expectation converter for taking the average value of the
                Observable over the ansatz state function. When None (the default) an
                :class:`~qiskit.opflow.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to True (defaults to False).
            include_custom: When `expectation` parameter here is None setting this to True will
                allow the factory to include the custom Aer pauli expectation.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Ignored if a gradient operator or function is
                given.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.
            quantum_instance: Quantum Instance or Backend
            qaoa_name: Name to identify this QAOAEx instance when logging or outputting files.
            force_shots: If quantum instance returns a statevector, then convert into shots instead.
            output_circuit_when_sample: Whether to output circuit using circuit_drawer whenever circuit is sampled.
            log_text: Used for text output, replacement to the default print method to make logging easy.
                If None, no text output can occur.
        """
        validate_min('reps', reps, 1)

        self._qaoa_name = qaoa_name
        self._reps = reps
        self._mixer = mixer
        self._initial_state = initial_state
        self._force_shots = force_shots
        self._log_text = log_text
        self._output_circuit_when_sample = output_circuit_when_sample

        # VQE will use the operator setter, during its constructor, which is overridden below and
        # will cause the var form to be built
        super(QAOA, self).__init__(ansatz=None,
                                   optimizer=optimizer,
                                   initial_point=initial_point,
                                   gradient=gradient,
                                   expectation=expectation,
                                   include_custom=include_custom,
                                   max_evals_grouped=max_evals_grouped,
                                   callback=callback,
                                   quantum_instance=quantum_instance)

    @VariationalAlgorithm.quantum_instance.setter
    def quantum_instance(self,
                         quantum_instance: Union[QuantumInstance, BaseBackend, Backend]
                        ) -> None:
        """ set quantum_instance. (Overides method)"""
        super(VQE, self.__class__).quantum_instance.__set__(self, quantum_instance)

        self._circuit_sampler = CircuitSamplerCustom(
                                    self._quantum_instance,
                                    param_qobj = is_aer_provider(self._quantum_instance.backend),
                                    sampler_name = self._qaoa_name,
                                    force_shots = self._force_shots,
                                    output_circuit_when_sample = self._output_circuit_when_sample,
                                    log_text = self._log_text)
        self._circuit_sampler.set_post_process_raw_data(self._post_process_raw_data)

    def find_minimum(self,
                     initial_point: Optional[np.ndarray] = None,
                     ansatz: Optional[QuantumCircuit] = None,
                     cost_fn: Optional[Callable] = None,
                     optimizer: Optional[Optimizer] = None,
                     gradient_fn: Optional[Callable] = None) -> 'VariationalResult':
        """Optimize to find the minimum cost value.

        Args:
            initial_point: If not `None` will be used instead of any initial point supplied via
                constructor. If `None` and `None` was supplied to constructor then a random
                point will be used if the optimizer requires an initial point.
            ansatz: If not `None` will be used instead of any ansatz supplied via constructor.
            cost_fn: If not `None` will be used instead of any cost_fn supplied via
                constructor.
            optimizer: If not `None` will be used instead of any optimizer supplied via
                constructor.
            gradient_fn: Optional gradient function for optimizer

        Returns:
            dict: Optimized variational parameters, and corresponding minimum cost value.

        Raises:
            ValueError: invalid input
        """
        initial_point = initial_point if initial_point is not None else self.initial_point
        ansatz = ansatz if ansatz is not None else self.ansatz
        cost_fn = cost_fn if cost_fn is not None else self._cost_fn
        optimizer = optimizer if optimizer is not None else self.optimizer

        if ansatz is None:
            raise ValueError('Ansatz neither supplied to constructor nor find minimum.')
        if cost_fn is None:
            raise ValueError('Cost function neither supplied to constructor nor find minimum.')
        if optimizer is None:
            raise ValueError('Optimizer neither supplied to constructor nor find minimum.')

        nparms = ansatz.num_parameters
        
        if self._optimiser_parameter_bounds == None:
            if hasattr(ansatz, 'parameter_bounds') and ansatz.parameter_bounds is not None:
                bounds = ansatz.parameter_bounds
            else:
                bounds = [(None, None)] * len(self.initial_point)
        else:
            bounds = self._optimiser_parameter_bounds

        #if initial_point is not None and len(initial_point) != nparms:
        #    raise ValueError(
        #        'Initial point size {} and parameter size {} mismatch'.format(
        #            len(initial_point), nparms))
        if len(bounds) != len(self.initial_point):
            bounds = [(None, None)] * len(self.initial_point)
            print("WARNING: Ansatz bounds size does not match parameter size (len(self.initial_point)), setting bounds to (None, None)")
            #raise ValueError('Ansatz bounds size does not match parameter size (len(self.initial_point))')

        # If *any* value is *equal* in bounds array to None then the problem does *not* have bounds
        problem_has_bounds = not np.any(np.equal(bounds, None))
        # Check capabilities of the optimizer
        if problem_has_bounds:
            if not optimizer.is_bounds_supported:
                raise ValueError('Problem has bounds but optimizer does not support bounds')
        else:
            if optimizer.is_bounds_required:
                raise ValueError('Problem does not have bounds but optimizer requires bounds')
        if initial_point is not None:
            if not optimizer.is_initial_point_supported:
                raise ValueError('Optimizer does not support initial point')
        else:
            if optimizer.is_initial_point_required:
                if hasattr(ansatz, 'preferred_init_points'):
                    # Note: default implementation returns None, hence check again after below
                    initial_point = ansatz.preferred_init_points

                if initial_point is None:  # If still None use a random generated point
                    low = [(l if l is not None else -2 * np.pi) for (l, u) in bounds]
                    high = [(u if u is not None else 2 * np.pi) for (l, u) in bounds]
                    initial_point = algorithm_globals.random.uniform(low, high)

        start = time.time()
        if not optimizer.is_gradient_supported:  # ignore the passed gradient function
            gradient_fn = None
        else:
            if not gradient_fn:
                gradient_fn = self._gradient

        logger.info('Starting optimizer.\nbounds=%s\ninitial point=%s', bounds, initial_point)
        opt_params, opt_val, num_optimizer_evals = optimizer.optimize(len(self.initial_point),
                                                                          cost_fn,
                                                                          variable_bounds=bounds,
                                                                          initial_point=initial_point,
                                                                          gradient_function=gradient_fn)
                                   
        if self._parameterise_point_for_energy_evaluation != None:
            self.latest_parameterised_point = self._parameterise_point_for_energy_evaluation(opt_params, nparms)

        eval_time = time.time() - start

        result = VariationalResult()
        result.optimizer_evals = num_optimizer_evals
        result.optimizer_time = eval_time
        result.optimal_value = opt_val
        result.optimal_point = opt_params
        result.optimal_parameters = dict(zip(self._ansatz_params, opt_params))

        return result
    
    def eigenvector_to_solutions(self, 
                                  eigenvector: Union[dict, np.ndarray, StateFn],
                                  quadratic_program: QuadraticProgram,
                                  min_probability: float = 1e-6
                                 ) -> List[Tuple[str, float, float]]:
        """ Convert the eigenvector to a list of solution 3-tuples (bitstrings, quadratic_function_objective_value, probability). (Overides method)
        Args:
            eigenvector: The eigenvector from which the solution states are extracted.
            quadratic_program: The quadatic program to evaluate at the bitstring.
            min_probability: Only consider states where the amplitude exceeds this threshold.
        Returns:
            A list with elements for each computational basis state contained in the eigenvector.
            Each element is a 3-tuple:
            (state as bitstring (str), 
             quadatic program evaluated at that bitstring (float), 
             probability of sampling this bitstring from the eigenvector (float)
            ).
        Raises:
            TypeError: If the type of eigenvector is not supported.
        """
        if isinstance(eigenvector, DictStateFn):
            eigenvector = {bitstr: val ** 2 for (bitstr, val) in eigenvector.primitive.items()}
        elif isinstance(eigenvector, StateFn):
            eigenvector = eigenvector.to_matrix()

        solutions = []
        if isinstance(eigenvector, dict):
            # iterate over all samples
            for bitstr, amplitude in eigenvector.items():
                sampling_probability = amplitude * amplitude
                # add the bitstring, if the sampling probability exceeds the threshold
                if sampling_probability > 0:
                    if sampling_probability >= min_probability:
                        # I've reversed the qubits here, I think they were the wrong order.
                        value = quadratic_program.objective.evaluate([int(bit) for bit in bitstr[::-1]])
                        solutions += [(bitstr[::-1], value, sampling_probability)]

        elif isinstance(eigenvector, np.ndarray):
            num_qubits = int(np.log2(eigenvector.size))
            probabilities = np.abs(eigenvector * eigenvector.conj())

            # iterate over all states and their sampling probabilities
            for i, sampling_probability in enumerate(probabilities):

                # add the i-th state if the sampling probability exceeds the threshold
                if sampling_probability > 0:
                    if sampling_probability >= min_probability:
                        bitstr = '{:b}'.format(i).rjust(num_qubits, '0')[::-1]
                        value = quadratic_program.objective.evaluate([int(bit) for bit in bitstr])
                        solutions += [(bitstr, value, sampling_probability)]

        else:
            raise TypeError('Unsupported format of eigenvector. Provide a dict or numpy.ndarray.')

        return solutions

    def _energy_evaluation(self,
                           parameters: Union[List[float], np.ndarray]
                          ) -> Union[float, List[float]]:
        """ Evaluate energy at given parameters for the ansatz. This is the objective function 
        to be passed to the optimizer that is used for evaluation. (Overides method)
        Args:
            parameters: The parameters for the ansatz.
        Returns:
            Energy of the hamiltonian of each parameter.
        Raises:
            RuntimeError: If the ansatz  has no parameters.
        """
        num_parameters = self.ansatz.num_parameters

        if self._parameterise_point_for_energy_evaluation != None:
            self.latest_parameterised_point = parameters
            parameters = self._parameterise_point_for_energy_evaluation(parameters, num_parameters)

        if self._ansatz.num_parameters == 0:
            raise RuntimeError('The ansatz cannot have 0 parameters.')

        parameter_sets = np.reshape(parameters, (-1, num_parameters))
        # Create dict associating each parameter with the lists of parameterization values for it

        param_bindings = dict(zip(self._ansatz_params,
                                  parameter_sets.transpose().tolist()))  # type: Dict

        start_time = time.time()
        #self._log_text("self._expect_op:", self._expect_op)
        sampled_expect_op = self._circuit_sampler.convert(self._expect_op, params=param_bindings)
        means = np.real(sampled_expect_op.eval())

        if self._callback is not None:
            variance = np.real(self._expectation.compute_variance(sampled_expect_op))
            estimator_error = np.sqrt(variance / self.quantum_instance.run_config.shots)
            for i, param_set in enumerate(parameter_sets):
                self._eval_count += 1
                self._callback(self._eval_count, param_set, means[i], estimator_error[i])
        else:
            self._eval_count += len(means)

        end_time = time.time()
        logger.info('Energy evaluation returned %s - %.5f (ms), eval count: %s',
                    means, (end_time - start_time) * 1000, self._eval_count)

        return means if len(means) > 1 else means[0]

    def _prepare_for_optisation(self,
                               operator: OperatorBase,
                               aux_operators: Optional[List[Optional[OperatorBase]]] = None
                              ) -> None:
        """ Prepares the QAOA instance to perform simulation without needing to run the optimisation loop. (New method)
        Args:
            operator: The operator (usually obtained from QuadraticProgram.to_ising()).
        """
        #super(VQE, self).compute_minimum_eigenvalue(operator, aux_operators)
        if self.quantum_instance is None:
            raise AlgorithmError("A QuantumInstance or Backend "
                                 "must be supplied to run the quantum algorithm.")

        if operator is None:
            raise AlgorithmError("The operator was never provided.")
        #operator = self._check_operator(operator)
        # The following code "operator = self._check_operator(operator)" was not working correctly here since it is meant to replace the operator.
        # So instead, using below code to manually update the ansatz.
        self.ansatz = QAOAAnsatz(operator,
                                 self._reps,
                                 initial_state = self._initial_state,
                                 mixer_operator = self._mixer)
        # We need to handle the array entries being Optional i.e. having value None
        if aux_operators:
            zero_op = I.tensorpower(operator.num_qubits) * 0.0
            converted = []
            for op in aux_operators:
                if op is None:
                    converted.append(zero_op)
                else:
                    converted.append(op)

            # For some reason Chemistry passes aux_ops with 0 qubits and paulis sometimes.
            aux_operators = [zero_op if op == 0 else op for op in converted]
        else:
            aux_operators = None
        self._quantum_instance.circuit_summary = True

        self._eval_count = 0

        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        if self._gradient:
            if isinstance(self._gradient, GradientBase):
                self._gradient = self._gradient.gradient_wrapper(
                    ~StateFn(operator) @ StateFn(self._ansatz),
                    bind_params=self._ansatz_params,
                    backend=self._quantum_instance)
        #if not self._expect_op:
        self._expect_op = self.construct_expectation(self._ansatz_params, operator)

    def calculate_statevector_at_point(self,
                                       operator: OperatorBase,
                                       point: Union[List[float], np.ndarray],
                                       force_shots: bool = False,
                                       sample_shots: int = 8192
                                      ) -> Union[Dict[str, float], List[float], np.ndarray]:
        """ Prepares for QAOA simulation and calculates the statevector for the given point. (New method)
        Args:
            operator: The operator (usually obtained from QuadraticProgram.to_ising()).
            point: The QAOA parameters (a list ordered as: [all_ZZ_gamma_values] + [all_X_beta_values]).
            force_shots: If simulating using a statevector, should a new statevector be formed by sampling from it?
            sample_shots: If force_shots is True, how many shots to sample?
        Returns:
            The resulting statevector. Might be a dict or an ndarray, depending on which
            simulator is used and whether the statevector is being sampled or not.
            When statevector sim is used, returns an ndarray, otherwise returns a dict.
        """
        from qiskit.utils.run_circuits import find_regs_by_name

        self._prepare_for_optisation(operator)

        qc = self.ansatz.assign_parameters(point)
        statevector = {}
        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            statevector = ret.get_statevector(qc)
            if force_shots == True:
                counts = Statevector(ret.get_statevector(qc)).sample_counts(sample_shots)
                statevector = {}
                for state in counts.keys():
                    statevector[state] = (counts[state]/sample_shots) ** 0.5
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            ret = self._quantum_instance.execute(qc)
            counts = ret.get_counts(qc)
            shots = self._quantum_instance._run_config.shots
            statevector = {b: (v / shots) ** 0.5 for (b, v) in counts.items()}
        return statevector

    def execute_at_point(self, 
                         point: Union[List[float], np.ndarray], 
                         quadratic_program: QuadraticProgram, 
                         optimal_function_value: float = None, 
                         log_text: Optional[Callable[..., Any]] = print
                        ) -> Dict[str, Any]:
        """ Runs QAOA without the optimization loop. Evaluates a single set of qaoa parameters. (New method)
        Args:
            point: The QAOA parameters (a list ordered as: [all_ZZ_gamma_values] + [all_X_beta_values]).
            quadratic_program: The quadratic program to obtain the operator from and to evaluate the solution state bitstrings with.
            optimal_function_value: The optimal value for which the solution states return in the quadratic_program. 
                Useful in rare cases where the solutions have zero probability. 
                If None, the best function_value among solutions will be used.
            log_text: Used for text output, replacement to the default print method to make logging easy.
                If None, no text output can occur.
        Returns:
            A dict containing the results. Keys are: 'energy', 'point', 'solutions', 'solution_probability', 'eigenstate', 'function_value'.
        """
        
        op_custom, offset = quadratic_program.to_ising()

        results_dict = {}

        # no need to call "self.prepare_for_optisation(op_custom)" because the
        # methods "self.calculate_statevector_at_point(op_custom, point)" and 
        # "self.evaluate_energy(op_custom, point)" already do.
        eigenstate = self.calculate_statevector_at_point(op_custom, point)
        energy = self.evaluate_energy_at_point(op_custom, point)

        solutions = self.get_optimal_solutions_from_statevector(eigenstate, quadratic_program, min_probability=10 ** -6, optimal_function_value=optimal_function_value)

        solution_probability = 0
        for sol in solutions:
            solution_probability += sol["probability"]
        results_dict["energy"] = energy
        results_dict["point"] = point
        results_dict["solutions"] = solutions
        results_dict["solution_probability"] = solution_probability
        results_dict["eigenstate"] = eigenstate
        if len(solutions) > 0:
            results_dict["function_value"] = solutions[0]["function_value"]
        else:
            if log_text != None:
                log_text("WARNING: No solutions were found.")
        return results_dict

    def evaluate_energy_at_point(self,
                                 operator: OperatorBase,
                                 point: Union[List[float], np.ndarray]
                                ) -> Union[float, List[float]]:
        """Evaluate energy at given parameters for the operator ansatz. (New method)
        Args:
            operator: The operator (usually obtained from QuadraticProgram.to_ising()).
            point: The QAOA parameters (a list ordered as: [all_ZZ_gamma_values] + [all_X_beta_values]).
        Returns:
            Energy of the hamiltonian of each parameter.
        Raises:
            RuntimeError: If the ansatz has no parameters.
        """
        self._prepare_for_optisation(operator)
        return self._energy_evaluation(point)

    def get_optimal_solutions_from_statevector(self,
                                               eigenvector: Union[dict, np.ndarray, StateFn],
                                               quadratic_program: QuadraticProgram,
                                               min_probability: float = 1e-6,
                                               optimal_function_value: float = None
                                              ) -> List[Tuple[str, float, float]]:
        """ Extract the solution state information from the eigenvector. (New method)
        Args:
            eigenvector: The eigenvector from which the solution states are extracted.
            quadratic_program: The QUBO to evaluate at the bitstring.
            min_probability: Only consider states where the amplitude exceeds this threshold.
            optimal_function_value: The optimal value for which the solution states return in the quadratic_program. Useful in rare cases where the solutions have zero probability.
        Returns:
            A list of all solutions. Each solution is a dict of length 3: "state": the state bitstring, "function_value": the function value, and "probability": the state probability.
        Raises:
            TypeError: If the type of eigenvector is not supported.
        """

        samples = self.eigenvector_to_solutions(eigenvector,
                                                 quadratic_program,
                                                 min_probability)
        samples.sort(key=lambda x: quadratic_program.objective.sense.value * x[1])
        fval = samples[0][1]
        if optimal_function_value != None:
            fval = optimal_function_value
        solution_samples = []
        for i in range(len(samples)):
            if samples[i][1] == fval:
                solution = {}
                solution["state"] = samples[i][0]
                solution["function_value"] = samples[i][1]
                solution["probability"] = samples[i][2]
                solution_samples.append(solution)

        return solution_samples

    def reset_reps(self, reps: int) -> None:
        """ Reset the number of reps when performing QAOA.
        Args:
            reps: The number of layers in QAOA (the 'p' value)
        """
        validate_min('reps', reps, 1)
        self._reps = reps

    def set_optimiser_parameter_bounds(self, 
                                       optimiser_parameter_bounds: Optional[List[Tuple[Optional[float], Optional[float]]]]
                                      ) -> None:
        self._optimiser_parameter_bounds = optimiser_parameter_bounds

    def set_parameterise_point_for_energy_evaluation(self,
                                                     parameterise_point_for_optimisation: Callable[[Union[List[float], np.ndarray], int], List[float]]
                                                    ) -> None:
        self._parameterise_point_for_energy_evaluation = parameterise_point_for_optimisation
    
    def set_post_process_raw_data(self, 
                                  post_process_raw_data_method: Optional[Callable[[Union[List[Dict[str, int]], List[List[Dict[str, int]]]]], Union[List[Dict[str, int]], List[List[Dict[str, int]]]] ]]
                                 ) -> None:
        """ Uses the specified method to process the raw sampled data executed on the backened whenever circuits are sampled.
        Args:
            post_process_raw_data_method: The method to process the data. 
               Inputs a list f counts dicts List[Dict[str, int]] and outputs the processed list of count dicts List[Dict[str, int]].
               The data could potentially be formatted as a list of a list of dictionaries List[List[Dict[str, int]]]. However, this
               will likely not happen withouth modifying QAOA to do so.
               Each dictionary has the counts for each qubit with the keys containing a string in binary format and separated
               according to the registers in circuit (e.g. ``0100 1110``). The string is little-endian (cr[0] on the right hand side).
               However there will likely only be a single register without modifying QAOA, so the state bitstring should have no spaces. 
        """
        self._post_process_raw_data = post_process_raw_data_method
        if self._circuit_sampler != None:
            self._circuit_sampler.set_post_process_raw_data(self._post_process_raw_data)

    def solve(self, 
              ising_hamiltonian_operator: Union[OperatorBase, nx.Graph],
              initial_point: Union[List[float], np.ndarray],
              bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
             ) -> MinimumEigensolverResult:
        if isinstance(ising_hamiltonian_operator, nx.Graph):
            couplings, local_fields = get_ising_hamiltonian_terms_from_ising_graph(ising_hamiltonian_operator)
            quadratic_program = get_quadratic_program_from_ising_hamiltonian_terms(couplings, local_fields, 0, None, None)
            ising_hamiltonian_operator, _ = quadratic_program.to_ising()


        self.initial_point = initial_point
        self.set_optimiser_parameter_bounds(bounds)
        return self.compute_minimum_eigenvalue(ising_hamiltonian_operator)

    def solve_from_ising_hamiltonian_terms(self,
                                           couplings: List[Tuple[int, int, float]],
                                           local_fields: Mapping[int, float],
                                           constant_term: float,
                                           initial_point: Union[List[float], np.ndarray],
                                           bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
                                          ) -> MinimumEigensolverResult:
        quadratic_program = get_quadratic_program_from_ising_hamiltonian_terms(couplings, local_fields, constant_term, None, None)
        ising_hamiltonian_operator, _ = quadratic_program.to_ising()
        
        self.initial_point = initial_point
        self.set_optimiser_parameter_bounds(bounds)
        return self.compute_minimum_eigenvalue(ising_hamiltonian_operator)


###############
### Helper Methods
###############

def convert_from_fourier_point(fourier_point: List[float], num_params_in_point: int) -> List[float]:
    """ Converts a point in Fourier space back to QAOA angles.
    Args:
        fourier_point: The point in Fourier space to convert.
        num_params_in_point: The length of the resulting point. Must be even.
    Returns:
        The converted point in the form of QAOA rotation angles.
    """
    new_point = [0] * num_params_in_point
    reps = int(num_params_in_point / 2) # num_params_in_result should always be even
    max_frequency = int(len(fourier_point) / 2) # fourier_point should always be even
    for i in range(reps):
        new_point[i] = 0
        for k in range(max_frequency):
            new_point[i] += fourier_point[k] * math.sin((k + 0.5) * (i + 0.5) * math.pi / reps)

        new_point[i + reps] = 0
        for k in range(max_frequency):
            new_point[i + reps] += fourier_point[k + max_frequency] * math.cos((k + 0.5) * (i + 0.5) * math.pi / reps)
    return new_point

def convert_to_fourier_point(point: List[float], num_params_in_fourier_point: int) -> List[float]:
    """ Converts a point to fourier space.
    Args:
        point: The point to convert.
        num_params_in_fourier_point: The length of the resulting fourier point. Must be even.
    Returns:
        The converted point in fourier space.
    """
    fourier_point = [0] * num_params_in_fourier_point
    reps = int(len(point) / 2) # point should always be even
    max_frequency = int(num_params_in_fourier_point / 2) # num_params_in_fourier_point should always be even
    for i in range(max_frequency):
        fourier_point[i] = 0
        for k in range(reps):
            fourier_point[i] += point[k] * math.sin((k + 0.5) * (i + 0.5) * math.pi / max_frequency)
        fourier_point[i] = 2 * fourier_point[i] / reps

        fourier_point[i + max_frequency] = 0
        for k in range(reps):
            fourier_point[i + max_frequency] += point[k + reps] * math.cos((k + 0.5) * (i + 0.5) * math.pi / max_frequency)
        fourier_point[i + max_frequency] = 2 * fourier_point[i + max_frequency] / reps
    return fourier_point

def get_ising_graph_from_ising_hamiltonian_terms(couplings: List[Tuple[int, int, float]], 
                                                 local_fields: Mapping[int, float]
                                                ) -> nx.Graph:
    """ Constructs a networkx graph with node and edge weights corresponding to the coefficients 
        of the local field and coupling strengths of the Ising Hamiltonian respectively.
    Args:
        couplings: A list of couplings for the Ising graph (or Hamiltonian). 
            Couplings are in the form of a 3-tuple e.g. 
            (spin_1, spin_2, coupling_strength).
        local_fields: The local field strengths for the Ising graph (or Hamiltonian) 
            A Dict with keys: spin numbers and values: field strengths.
    Returns:
        The Ising graph as an instance of a networkx Graph object with node and edge weights.
    """
    G = nx.Graph()
    for local_field in local_fields.keys():    
        G.add_node(local_field, weight=local_fields[local_field])
        
    G.add_weighted_edges_from(couplings)

    return G

def get_ising_hamiltonian_terms_from_ising_graph(ising_graph: nx.Graph) -> Tuple[List[Tuple[int, int, float]], Dict[int, float]]:
    """ Constructs a networkx graph with node and edge weights corresponding to the coefficients 
        of the local field and coupling strengths of the Ising Hamiltonian respectively.
    Args:
        couplings: A list of couplings for the Ising graph (or Hamiltonian). 
            Couplings are in the form of a 3-tuple e.g. 
            (spin_1, spin_2, coupling_strength).
        local_fields: The local field strengths for the Ising graph (or Hamiltonian) 
            A Dict with keys: spin numbers and values: field strengths.
    Returns:
        The Ising graph as an instance of a networkx Graph object with node and edge weights.
    """
    local_fields = {}
    for i in range(len(ising_graph.nodes)):
        local_fields[i] = ising_graph.nodes[i]['weight']

    couplings = []

    edge_data = ising_graph.edges(data=True)
    for edge in edge_data:
        couplings.append((edge[0], edge[1], edge[3]['weight']))

    return couplings, local_fields

def get_quadratic_program_from_ising_hamiltonian_terms(couplings: List[Tuple[int, int, float]], 
                                                       local_fields: Mapping[int, float], 
                                                       constant_term: float,
                                                       output_ising_graph_filename: Optional[str] = None,
                                                       log_text: Optional[Callable[..., Any]] = print
                                                      ) -> QuadraticProgram:
    """ Constructs and returns the quadratic program corresponding to the input Hamiltonian terms.
        Applies the transformation -> Z = 2b - 1, since Ising Hamiltonian spins have {+-1} values 
        while the quadratic program is binary.
    Args:
        couplings: A list of couplings for the Ising graph (or Hamiltonian). 
            Couplings are in the form of a 3-tuple e.g. 
            (spin_1, spin_2, coupling_strength).
            Negative coupling strengths are Ferromagnetic (spin states want to be the same).
        local_fields: The local field strengths for the Ising graph (or Hamiltonian) 
            A Dict with keys: spin numbers and values: field strengths.
            Using convention with negative sign on local fields. So a negative local field makes the spin want to be +1.
        constant_term: the constant for the Ising Hamiltonian.
        output_ising_graph_filename: Filename to save ising graph file with.
            If None, will not output ising graph to file.
        log_text: Used for text output, replacement to the default print method to make logging easy.
            If None, no text output will occur.
    Returns:
        The binary quadratic program corresponding to the Hamiltonian
    """

    if output_ising_graph_filename != None:
        ising_graph = get_ising_graph_from_ising_hamiltonian_terms(couplings, local_fields)
        output_ising_graph(ising_graph, custom_filename_no_ext = output_ising_graph_filename, log_text=log_text)

    quadratic_program = QuadraticProgram()
    for local_field in local_fields.keys():
        quadratic_program.binary_var('c' + str(local_field))

    new_constant_term = 0
    new_linear_terms = {}
    for car_number in local_fields.keys():
        new_linear_terms[car_number] = 0.0
    new_quadratic_terms = {}
    
    # transform constant term
    new_constant_term = constant_term
    
    # transform local fields
    for car_number in local_fields.keys():
        new_linear_terms[car_number] = 2*local_fields[car_number]
        new_constant_term -= local_fields[car_number]

    # transform couplings
    for coupling in couplings:
        if ('c' + str(coupling[0]), 'c' + str(coupling[1])) in new_quadratic_terms:
            new_quadratic_terms[('c' + str(coupling[0]), 'c' + str(coupling[1]))] += 4*coupling[2]
        else:
            new_quadratic_terms[('c' + str(coupling[0]), 'c' + str(coupling[1]))] = 4*coupling[2]
        new_linear_terms[coupling[0]] -= 2*coupling[2]
        new_linear_terms[coupling[1]] -= 2*coupling[2]
        new_constant_term += coupling[2]
        
    quadratic_program.minimize(constant = new_constant_term, linear=[new_linear_terms[lf] for lf in new_linear_terms.keys()], quadratic=new_quadratic_terms)

    return quadratic_program

def output_ising_graph(ising_graph: nx.Graph, 
                       custom_filename_no_ext: Optional[str] = None, 
                       log_text: Optional[Callable[..., Any]] = print
                      ) -> None:
    """ Outputs the networkx graph to file in PNG format 
    Args:
        ising_graph: A networkx graph with node and edge weights specified.
            Nodes have attribute 'weight' that corresponds to a local field strength.
            Edges have attribute 'weight' corresponding to the coupling strength.
        custom_filename_no_ext: The filename to save the figure to. 
            Defaults to "Ising_graph" if None.
        log_text: Used for text output, replacement to the default print method to make logging easy.
            If None, no text output will occur.
    """
    # Generate plot of the Graph
    colors       = ['r' for node in ising_graph.nodes()]
    default_axes = plt.axes(frameon=False)
    default_axes.set_axis_off()
    default_axes.margins(0.1) 
    pos          = nx.circular_layout(ising_graph)
    labels = {n: str(n) + ';   ' + str(ising_graph.nodes[n]['weight']) for n in ising_graph.nodes}

    nx.draw_networkx(ising_graph, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos, labels=labels)
    edge_labels = nx.get_edge_attributes(ising_graph, 'weight')
    nx.draw_networkx_edge_labels(ising_graph, pos=pos,edge_labels=edge_labels)
    if custom_filename_no_ext == None:
        filename = "Ising_graph.png"
    else:
        filename = custom_filename_no_ext + '.png'
    if log_text != None:
        log_text("Saving Ising graph '" + filename + "'...")
    
    plt.savefig(filename, format="PNG", bbox_inches=0)
    plt.close()

def print_qaoa_solutions(solutions: List[Mapping[str, Any]], 
                         log_text: Callable[..., Any] = print
                        ) -> None:
    """ Pretty prints (pprint) a list of solutions followed by their summed probability.
    Args:
        solutions: List of solutions, they are each formatted as a dict with (key, value):
            'state', state bitstring (str)
            'function_value', binary quadratic program objective value (float)
            'probability', probability (float)
        log_text: Used for text output, replacement to the default print method to make logging easy.
    """
    import pprint
    if len(solutions) > 0:
        log_text("function value (quadratic program):", str(solutions[0]["function_value"]))
        solutions_string = pprint.pformat([[solutions[x]["state"], solutions[x]["probability"]] for x in range(len(solutions))], indent=2)
        log_text(solutions_string)
        initial_solution_probability = 0
        for x in range(len(solutions)):
            initial_solution_probability += solutions[x]["probability"]
        log_text("total probability:", initial_solution_probability)
    else:
        log_text("total probability: 0")
