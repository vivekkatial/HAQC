import unittest
from unittest.case import expectedFailure
import numpy as np
from qaoa_vrp.initialisation.initialisation import *


class TestPeturbFromLastRestart:
    def test_perturb_from_last_restart_when_first_restart(self):
        # Initial points
        p = 1
        restart = 0
        initial_point = [np.random.uniform(0, 1) for i in range(2 * p)]
        # Execute Perturbation
        actual = perturb_from_last_restart(initial_point, restart=restart)
        expected = initial_point
        # Check they're equal
        assert actual == expected

    def test_perturb_from_second_restart(self):
        # Set a random seed to make result predictable
        np.random.seed(123)
        p = 1
        # Ensure not the first restart
        restart = 1
        initial_point = [0.04, -0.78]
        actual = perturb_from_last_restart(initial_point, restart=restart)
        # Use the random pertubation
        expected = [0.07035308144021385, -0.708613933495038]
        assert actual == expected


class TestRampedUpInitialPoint(unittest.TestCase):
    def test_ramped_up_initial_point_for_p_equals_one(self):
        initial_point = [0.2, 0.3]
        expected = np.array([0.22, 0., 0.33, 0.])
        actual = ramped_up_from_previous_layer(previous_layer_initial_point=initial_point)
        # Check that they are some what close
        np.testing.assert_allclose(expected, actual)

    def test_ramped_up_initial_point_for_p_equals_two(self):
        initial_point = [0.2, 0.4, 0.3, 0.5]
        expected = np.array([0.22, 0.44, 0., 0.33, 0.55, 0.])
        actual = ramped_up_from_previous_layer(previous_layer_initial_point=initial_point)
        # Check that they are some what close
        np.testing.assert_allclose(expected, actual)

    def test_ramped_up_works_for_different_growth_params(self):
        growth = 0.2
        initial_point = [0.1, 0.2]
        expected = np.array([0.12, 0., 0.24, 0.])
        actual = ramped_up_from_previous_layer(previous_layer_initial_point=initial_point, growth=growth)
        # Check that they are some what close
        np.testing.assert_allclose(expected, actual)

    def test_ramped_up_initial_point_fails_when_invalid_length_inital_point(self):
        initial_point = [0.2, 0.4, 0.3]
        self.assertRaises(ValueError, ramped_up_from_previous_layer, initial_point)

class TestPerturbFromPreviousLayer(unittest.TestCase):
    def test_perturb_from_previous_layer_initial_point_for_p_equals_one(self):
        np.random.seed(123)
        initial_point = [0.2, 0.3]
        expected = np.array([0.23035308, 0., 0.37138607, 0.])
        actual = perturb_from_previous_layer(previous_layer_initial_point=initial_point)
        # Check that they are some what close
        np.testing.assert_allclose(expected, actual)

    def test_ramped_up_initial_point_fails_when_invalid_length_inital_point(self):
        initial_point = [0.2, 0.4, 0.3]
        self.assertRaises(ValueError, ramped_up_from_previous_layer, initial_point)