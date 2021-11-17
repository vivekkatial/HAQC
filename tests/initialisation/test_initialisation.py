import unittest
import numpy as np
from unittest.mock import patch

from unittest.case import expectedFailure
from qaoa_vrp.initialisation.initialisation import Initialisation
from PIL.Image import init

@patch('numpy.random.uniform', return_value=0.1)
class TestPeturbFromLastRestart(unittest.TestCase):
    def test_perturb_from_last_restart_when_first_restart(self, mock_uniform):
        # Initial points
        p = 1
        restart = 0
        initial_point = [np.random.uniform(0, 1) for i in range(2 * p)]
        # Execute Perturbation
        actual = Initialisation.perturb_from_last_restart(initial_point, restart=restart)
        expected = [0.1, 0.1]
        # Check they're equal
        assert actual == expected

    def test_perturb_from_second_restart(self, mock_uniform):
        # Set a random seed to make result predictable
        np.random.seed(123)
        p = 1
        # Ensure not the first restart
        restart = 1
        initial_point = [0.1, 0.1]
        actual = Initialisation.perturb_from_last_restart(initial_point, restart=restart)
        # Use the random pertubation from 0.1 (so this means that each element needs to be added by 0.1)
        expected = [0.2, 0.2]
        assert actual == expected


@patch('numpy.random.uniform', return_value=0.1)
class TestRampedUpInitialPoint(unittest.TestCase):
    def test_ramped_up_initial_point_for_p_equals_one(self, mock_uniform):
        # Set a random seed to make result predictable
        np.random.seed(123)
        p = 3
        actual = Initialisation.ramped_up_initialisation(p=p)
        # Expect to see alpha_0, alpha_0 + growth, alpha_0 + 2*growth
        # Expect to see beta_0, beta_0 - growth, beta_0 - 2*growth
        expected = np.array([ 0.1,  0.2,  0.3,  0.1,  0. , -0.1])
        np.testing.assert_allclose(expected, actual)

@patch('numpy.random.uniform', return_value=0.1)
class TestPerturbFromPreviousLayer(unittest.TestCase):
    def test_perturb_from_previous_layer_initial_point_for_p_equals_one(self, mock_uniform):
        np.random.seed(123)
        initial_point = [0.2, 0.3]
        expected = np.array([0.3, 0.0, 0.4, 0.0])
        actual = Initialisation.perturb_from_previous_layer(previous_layer_initial_point=initial_point)
        # Check that they are some what close
        np.testing.assert_allclose(expected, actual)

    def test_perturb_initial_point_fails_when_invalid_length_inital_point(self, mock_uniform):
        initial_point = [0.2, 0.4, 0.3]
        self.assertRaises(ValueError, Initialisation.perturb_from_previous_layer, initial_point)


class TestFourierTransform(unittest.TestCase):
    def test_fourier_transform_is_working(self):
        initial_point = [0.2, 0.3]
        # Check that they are some what close
        fourier_point = Initialisation.fourier_transform(initial_point=initial_point)
        assert len(fourier_point) == (len(initial_point) + 2)
