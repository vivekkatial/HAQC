from typing import List
import numpy as np

from qaoa_vrp.algorithms.QAOA_cust import (
    convert_to_fourier_point,
    convert_from_fourier_point,
)


class Initialisation:
    """Initialisation Class for the method for initialisations"""
    def __init__(self, p: int = None, initial_point: List[float] = [None], method: str=None) -> None:
        self.p = p
        self.initial_point = initial_point        
        self.initialisation_method = method

        # Computed attributes
        self.previous_layer_point = None

    def random_initialisation(self, p: int | None = None, **kw_args) -> List[float]:
        """A function to randomly initialise a point in layer 'p'

        Args:
            p (int): Number of integer layers

        Returns:
            List[float]: New initial point in the parameter space for layer p
        """
        return np.random.uniform(-2 * np.pi, 2 * np.pi, 2*p)


    def perturb_from_previous_layer(
        self, previous_layer_initial_point: List[float], **kw_args
    ) -> List[float]:
        """A function to perturb data from the previous layer

        Args:
            previous_layer_initial_point (List[float]): Initial optimized point from the previous layer

        Returns:
            List[float]: New initial point in the parameter space for layer p+1
        """
        if len(previous_layer_initial_point) % 2 != 0:
            raise ValueError("Must be an even number of params for alpha and beta (2*p)")

        p = int(len(previous_layer_initial_point) / 2)
        alphas = previous_layer_initial_point[:p]
        betas = previous_layer_initial_point[p:]

        # Ramp up alpha and beta based on growth
        perturbed_alphas = [alpha + np.random.uniform(1e-1, 0) for alpha in alphas]
        perturbed_betas = [beta + np.random.uniform(1e-1, 0) for beta in betas]

        # Add zeros
        perturbed_alphas.append(0.0)
        perturbed_betas.append(0.0)

        perturbed_params = [perturbed_alphas, perturbed_betas]
        perturbed_params = np.concatenate(perturbed_params)
        return perturbed_params


    def ramped_up_initialisation(self, p: int, growth: float = 0.1, **kw_args) -> List[float]:

        # Initialise arrays
        alphas = np.zeros(p)
        betas = np.zeros(p)

        # Initialise alpha and beta
        alpha_init = np.random.uniform(-2 * np.pi, 2 * np.pi)
        beta_init = np.random.uniform(-2 * np.pi, 2 * np.pi)

        # Validate initial \alpha_0 and \beta_0
        while (alpha_init + p * growth > 2 * np.pi) or (
            beta_init - p * growth < -2 * np.pi
        ):
            # Initialise alpha and beta
            alpha_init = np.random.uniform(-2 * np.pi, 2 * np.pi)
            beta_init = np.random.uniform(-2 * np.pi, 2 * np.pi)

        # Ramp up alpha and beta based on growth
        ramped_alphas = [alpha_init + i * growth for i, _ in enumerate(alphas)]
        ramped_betas = [beta_init - i * growth for i, _ in enumerate(betas)]

        ramped_params = [ramped_alphas, ramped_betas]
        ramped_params = np.concatenate(ramped_params)
        return ramped_params


    def fourier_transform(self, initial_point: List[float], **kw_args) -> List[float]:
        # Initalise point in fourier space
        fourier_point = convert_to_fourier_point(
            initial_point, num_params_in_fourier_point=len(initial_point)
        )
        # Convert back to the angle space
        new_initial_parameters = convert_from_fourier_point(
            fourier_point=fourier_point, num_params_in_point=(len(initial_point) + 2)
        )
        return new_initial_parameters
