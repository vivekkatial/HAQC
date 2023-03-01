from typing import List, Optional
import numpy as np
from numpy.random.mtrand import random, uniform
import math


def convert_to_fourier_point(
    point: List[float], num_params_in_fourier_point: int
) -> List[float]:
    """Converts a point to fourier space.
    Args:
        point: The point to convert.
        num_params_in_fourier_point: The length of the resulting fourier point. Must be even.
    Returns:
        The converted point in fourier space.
    """
    fourier_point = [0] * num_params_in_fourier_point
    reps = int(len(point) / 2)  # point should always be even
    max_frequency = int(
        num_params_in_fourier_point / 2
    )  # num_params_in_fourier_point should always be even
    for i in range(max_frequency):
        fourier_point[i] = 0
        for k in range(reps):
            fourier_point[i] += point[k] * math.sin(
                (k + 0.5) * (i + 0.5) * math.pi / max_frequency
            )
        fourier_point[i] = 2 * fourier_point[i] / reps

        fourier_point[i + max_frequency] = 0
        for k in range(reps):
            fourier_point[i + max_frequency] += point[k + reps] * math.cos(
                (k + 0.5) * (i + 0.5) * math.pi / max_frequency
            )
        fourier_point[i + max_frequency] = 2 * fourier_point[i + max_frequency] / reps
    return fourier_point


def convert_from_fourier_point(
    fourier_point: List[float], num_params_in_point: int
) -> List[float]:
    """Converts a point in Fourier space back to QAOA angles.
    Args:
        fourier_point: The point in Fourier space to convert.
        num_params_in_point: The length of the resulting point. Must be even.
    Returns:
        The converted point in the form of QAOA rotation angles.
    """
    new_point = [0] * num_params_in_point
    reps = int(num_params_in_point / 2)  # num_params_in_result should always be even
    max_frequency = int(len(fourier_point) / 2)  # fourier_point should always be even
    for i in range(reps):
        new_point[i] = 0
        for k in range(max_frequency):
            new_point[i] += fourier_point[k] * math.sin(
                (k + 0.5) * (i + 0.5) * math.pi / reps
            )

        new_point[i + reps] = 0
        for k in range(max_frequency):
            new_point[i + reps] += fourier_point[k + max_frequency] * math.cos(
                (k + 0.5) * (i + 0.5) * math.pi / reps
            )
    return new_point


class Initialisation:
    """Initialisation Class for the method for initialisations"""

    def __init__(
        self,
        p: int = None,
        initial_point: List[float] = [None],
        method: str = None,
        evolution_time: float = 5.0,
    ) -> None:
        self.p = p
        self.initial_point = initial_point
        self.initialisation_method = method
        # Computed attributes
        self.previous_layer_point = None
        self.evolution_time = evolution_time

    def __str__(self) -> str:
        return f"Initialisation Method: {self.initialisation_method}\nLayer: \t {self.p}\nInitial Point: \t{self.initial_point}"

    def pprint_method(self) -> str:
        """Pretty print the method name

        Returns:
            str: Print out of method
        """
        method_str = self.initialisation_method
        print(method_str.replace("_", " ").title())
        return 0

    def random_initialisation(
        self, p: int, previous_layer_initial_point: Optional[List[float]] = None
    ) -> List[float]:
        """A function to randomly initialise a point in layer 'p'

        Args:
            p (int): Number of integer layers

        Returns:
            List[float]: New initial point in the parameter space for layer p
        """
        return np.random.uniform(-2 * np.pi, 2 * np.pi, 2 * p)

    def perturb_from_previous_layer(
        self,
        previous_layer_initial_point: List[float],
        noise: float = 0.1,
        p: Optional[int] = None,
    ) -> List[float]:
        """A function to perturb data from the previous layer

        Args:
            previous_layer_initial_point (List[float]): Initial optimized point from the previous layer
            p (Optional[int]): The number of points in the initial layer

        Returns:
            List[float]: New initial point in the parameter space for layer p+1
        """
        if len(previous_layer_initial_point) % 2 != 0:
            raise ValueError(
                "Must be an even number of params for alpha and beta (2*p)"
            )
        if p is None:
            p = len(previous_layer_initial_point) / 2

        if p == 1:
            return np.random.uniform(-2 * np.pi, 2 * np.pi, 2 * p)

        # For cases when we're doing restarts use the previous layer initial point (with some perturbation)
        if 2 * p == len(previous_layer_initial_point):
            return [
                x + np.random.uniform(noise, -noise)
                for x in previous_layer_initial_point
            ]

        p = int(len(previous_layer_initial_point) / 2)
        alphas = previous_layer_initial_point[:p]
        betas = previous_layer_initial_point[p:]

        # Ramp up alpha and beta based on growth
        perturbed_alphas = [
            alpha + np.random.uniform(noise, -noise) for alpha in alphas
        ]
        perturbed_betas = [beta + np.random.uniform(noise, -noise) for beta in betas]

        # Add zeros
        perturbed_alphas.append(0.0)
        perturbed_betas.append(0.0)

        perturbed_params = [perturbed_alphas, perturbed_betas]
        perturbed_params = np.concatenate(perturbed_params)
        return perturbed_params

    def ramped_up_initialisation(
        self,
        p: int,
        growth: float = 0.1,
        noise: float = 0.1,
        previous_layer_initial_point: Optional[List[float]] = None,
    ) -> List[float]:

        # Handle initial p=1
        if p == 1:
            return [
                growth + np.random.uniform(noise, -noise),
                p * growth + np.random.uniform(noise, -noise),
            ]

        # Initialise arrays
        alphas = np.zeros(p)
        betas = np.zeros(p)

        # Initialise alpha and beta
        alpha_init = growth
        beta_init = p * growth

        # Validate initial \alpha_0 and \beta_0
        while (alpha_init + p * growth > 2 * np.pi) or (
            beta_init - p * growth < -2 * np.pi
        ):
            # Initialise alpha and beta
            alpha_init = growth
            beta_init = p * growth

        # Ramp up alpha and beta based on growth
        ramped_alphas = [
            alpha_init + i * growth + np.random.uniform(noise, -noise)
            for i, _ in enumerate(alphas)
        ]
        ramped_betas = [
            beta_init - i * growth + np.random.uniform(noise, -noise)
            for i, _ in enumerate(betas)
        ]

        ramped_params = [ramped_alphas, ramped_betas]
        ramped_params = np.concatenate(ramped_params)
        return ramped_params

    def fourier_transform(
        self,
        previous_layer_initial_point: List[float],
        noise: float = 0.1,
        p: Optional[int] = None,
    ) -> List[float]:
        if p == 1:
            return np.random.uniform(-2 * np.pi, 2 * np.pi, 2 * p)
        # For cases when we're doing restarts use the previous layer initial point (with some perturbation)
        if 2 * p == len(previous_layer_initial_point):
            return [
                x + np.random.uniform(noise, -noise)
                for x in previous_layer_initial_point
            ]
        # Initalise point in fourier space
        fourier_point = convert_to_fourier_point(
            previous_layer_initial_point,
            num_params_in_fourier_point=len(previous_layer_initial_point),
        )
        # Convert back to the angle space
        new_initial_parameters = convert_from_fourier_point(
            fourier_point=fourier_point,
            num_params_in_point=(len(previous_layer_initial_point) + 2),
        )
        return new_initial_parameters

    def trotterized_quantum_annealing(
        self,
        p: int,
        noise: float = 0.1,
        evolution_time: float = 5.0,
        previous_layer_initial_point: Optional[List[float]] = None,
    ) -> List[float]:
        """Implementing Trotterized Quantum Anealing based on https://arxiv.org/pdf/2101.05742.pdf
        results.

        Args:
            p (int): Number of layers in QAOA circuit
            noise (float): Noise applied to initial point when doing a restart. Defaults to 0.1.
            evolution_time (float): Evolution Time. Defaults to 5.
            previous_layer_initial_point (Optional[List[float]], optional): Previous point when doing a random restart. Defaults to None.

        Returns:
            List[float]: Initial point which has been initialised based on TQA
        """

        # Handle initial p=1
        if p == 1:
            return [
                self.evolution_time + np.random.uniform(noise, -noise),
                np.random.uniform(noise, -noise),
            ]

        # For cases when we're doing restarts use the previous layer initial point (with some perturbation)
        if previous_layer_initial_point is not None and 2 * p == len(
            previous_layer_initial_point
        ):
            return [
                x + np.random.uniform(noise, -noise)
                for x in previous_layer_initial_point
            ]

        # Initialise delta_t based on T and p
        dt = self.evolution_time / p

        # Initialise alphas
        alpha_0 = 1 / p * dt
        alpha_N = dt
        alphas = np.linspace(start=alpha_0, stop=alpha_N, num=p)

        # Initialise betas
        beta_0 = (1 - 1 / p) * dt
        beta_N = 0
        betas = np.linspace(start=beta_0, stop=beta_N, num=p)
        # Combine and add
        tqa_params = [alphas, betas]
        tqa_params = np.concatenate(tqa_params)
        return tqa_params
