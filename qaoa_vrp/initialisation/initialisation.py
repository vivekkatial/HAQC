from typing import List
import numpy as np
from qaoa_vrp.algorithms.QAOA_cust import convert_to_fourier_point

def perturb_from_last_restart(initial_point: List[float], restart: int) -> List[float]:
    """Function to make pertubation from previous initial point

    Args:
        initial_point (list): Initial point from previous restart
        restart (int): Restart index

    Returns:
        list: New initial point
    """

    if restart == 0:
        return initial_point
    else:
        initial_point = [x + np.random.uniform(1e-1, 0) for x in initial_point]
        return initial_point

def perturb_from_previous_layer(previous_layer_initial_point: List[float]) -> List[float]:
    """A function to perturb data from the previous layer

    Args:
        previous_layer_initial_point (List[float]): Initial optimized point from the previous layer

    Returns:
        List[float]: New initial point in the parameter space for layer p+1
    """
    if len(previous_layer_initial_point) % 2 != 0:
        raise ValueError("Must be an even number of params for alpha and beta (2*p)")

    p = int(len(previous_layer_initial_point)/2)
    alphas = previous_layer_initial_point[:p]
    betas = previous_layer_initial_point[p:]

    # Ramp up alpha and beta based on growth
    perturbed_alphas = [alpha + np.random.uniform(1e-1, 0) for alpha in alphas]
    perturbed_betas = [beta + np.random.uniform(1e-1, 0) for beta in betas]

    # Add zeros
    perturbed_alphas.append(0.)
    perturbed_betas.append(0.)

    perturbed_params = [perturbed_alphas, perturbed_betas]
    perturbed_params = np.concatenate(perturbed_params)
    return perturbed_params


def ramped_up_from_previous_layer(
    previous_layer_initial_point: List[float], growth: float = 0.10
) -> List[float]:
    """A function to ramp up the initial points

    Args:
        previous_layer_initial_point (list): A list for `alpha` and `beta` in previous state
        growth (float): The growth parameter for how much

    Returns:
        list: Initial point at new layer
    """
    if len(previous_layer_initial_point) % 2 != 0:
        raise ValueError("Must be an even number of params for alpha and beta (2*p)")

    p = int(len(previous_layer_initial_point)/2)
    alphas = previous_layer_initial_point[:p]
    betas = previous_layer_initial_point[p:]

    # Add zero for next layer
    alphas.append(0) 
    betas.append(0)

    # Ramp up alpha and beta based on growth
    ramped_alphas = [alpha * (1 + growth) for alpha in alphas]
    ramped_betas = [beta * (1 + growth) for beta in betas]

    ramped_params = [ramped_alphas, ramped_betas]
    ramped_params = np.concatenate(ramped_params)
    return ramped_params

def fourier_transform(initial_point: List[float])->List[float]:
    return convert_to_fourier_point(initial_point)
    