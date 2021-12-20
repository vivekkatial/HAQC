import unittest
import numpy as np
from unittest.mock import patch

from unittest.case import expectedFailure
from qaoa_vrp.initialisation.initialisation import Initialisation
from PIL.Image import init


@patch('numpy.random.uniform', return_value=0.1)
class TestRampedUpInitialPoint(unittest.TestCase):
    def test_ramped_up_initial_point_for_p_is_greater_than_one(self, mock_uniform):
        # Set a random seed to make result predictable
        np.random.seed(123)
        p = 3
        actual = Initialisation().ramped_up_initialisation(p=p, growth=0.1)
        # Expect to see alpha_i = growth*i + \error
        # Expect to see beta_i = growth*(p - i) + \error
        expected = np.array(
            [0.1 + 0.1, 0.2 + 0.1, 0.3 + 0.1, 0.3 + 0.1, 0.2 + 0.1, 0.1 + 0.1]
        )
        np.testing.assert_allclose(expected, actual)

    def test_ramped_up_initial_point_for_p_is_equal_to_one(self, mock_uniform):
        # Set a random seed to make result predictable
        np.random.seed(123)
        p = 1
        actual = Initialisation().ramped_up_initialisation(p=p, growth=0.1)
        # Expect to see alpha_i = growth*i + \error
        # Expect to see beta_i = growth*(p - i) + \error
        expected = np.array([0.2, 0.2])
        np.testing.assert_allclose(expected, actual)


class TestPerturbFromPreviousLayer(unittest.TestCase):
    @patch('numpy.random.uniform', return_value=[0.1, 0.1])
    def test_perturb_from_previous_layer_initial_point_for_p_equals_one_and_p_not_specified(
        self, mock_uniform
    ):
        np.random.seed(123)
        initial_point = [0.2, 0.3]
        expected = np.array([0.1, 0.1])
        actual = Initialisation().perturb_from_previous_layer(
            previous_layer_initial_point=initial_point
        )
        # Check that they are some what close
        np.testing.assert_allclose(expected, actual)

    @patch('numpy.random.uniform', return_value=[0.1, 0.1])
    def test_perturb_from_previous_layer_initial_point_for_p_equals_one_and_p_is_specified(
        self, mock_uniform
    ):
        np.random.seed(123)
        p = 1
        initial_point = [0.2, 0.3]
        actual = Initialisation().perturb_from_previous_layer(
            previous_layer_initial_point=initial_point, p=p
        )
        expected = [0.1, 0.1]
        # Check that they are some what close
        np.testing.assert_allclose(expected, actual)

    @patch('numpy.random.uniform', return_value=0.1)
    def test_perturb_initial_point_fails_when_invalid_length_inital_point(
        self, mock_uniform
    ):
        initial_point = [0.2, 0.4, 0.3]
        self.assertRaises(
            ValueError, Initialisation().perturb_from_previous_layer, initial_point
        )


class TestFourierTransform(unittest.TestCase):
    def test_fourier_transform_is_working(self):
        initial_point = [0.2, 0.3]
        p = 1
        # Check that they are some what close
        fourier_point = Initialisation().fourier_transform(
            previous_layer_initial_point=initial_point, p=2
        )
        assert len(fourier_point) == (len(initial_point) + 2)


@patch('numpy.random.uniform', return_value=0.1)
class TestTQA(unittest.TestCase):
    def test_tqa_initialisation_for_when_p_greater_than_one(self, mock_uniform):
        p = 5
        actual = Initialisation(evolution_time=5).trotterized_quantum_annealing(p=p)
        expected = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2, 0.0])
        np.testing.assert_allclose(expected, actual)

    def test_tqa_initialisation_for_when_p_equals_one(self, mock_uniform):
        p = 1
        actual = Initialisation(evolution_time=5).trotterized_quantum_annealing(p=p, evolution_time=5)
        expected = np.array([5.1, 0.1])
        np.testing.assert_allclose(expected, actual)
