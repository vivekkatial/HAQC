import unittest
from unittest.mock import patch
import numpy as np

from haqc.initialisation.parameter_fixing import (
    get_optimal_parameters_from_parameter_fixing,
)


class TestQAOAParameterFixing(unittest.TestCase):
    @patch('haqc.algorithms.eval_qaoa')
    def test_get_optimal_parameters_from_parameter_fixing(self, mock_eval_qaoa):
        # Setup the mock to return a structured object with the required attributes
        mock_result = type(
            'test', (object,), {'optimal_value': 0, 'cost_function_evals': 1}
        )()

        # Mock 'eval_qaoa' to return the mock result
        mock_eval_qaoa.return_value = mock_result

        # Call the function with test values
        n_layers = 2
        n = 5
        qaoa_circuit = 'mock_circuit'  # This would be a mock or placeholder value

        # Run the test
        (
            best_params,
            best_expectation,
            total_fevals,
        ) = get_optimal_parameters_from_parameter_fixing(n_layers, n, qaoa_circuit)

        # Check the total function evaluations
        # self.assertEqual(total_fevals, n_layers * n)
        # More assertions could be added to check the correctness of best_params and best_expectation
        # For example:
        self.assertEqual(len(best_params), 2 * n_layers)
        # self.assertTrue(-np.inf < best_expectation <= 0)


if __name__ == '__main__':
    unittest.main()
