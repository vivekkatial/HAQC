import json
import numpy as np
import math


def np_encoder(object):
    """Numpy encoder"""
    if isinstance(object, np.generic):
        return object.item()


def second_largest(numbers):
    count = 0
    m1 = m2 = float("-inf")
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None


def distance(i, j):
    """Function to compute distances"""
    if isinstance(i, tuple) and isinstance(j, tuple):
        dx = j[0] - i[0]
        dy = j[1] - i[1]
        return math.sqrt(dx * dx + dy * dy)
    else:
        raise TypeError("Incorrect Type - Please feed a tuple of coordinates")


def read_instance(filename):
    """Function to read an instance file from test data

    Args:
        filename (str): Path to file for instance
    """

    with open(filename) as f:
        data = json.load(f)
        network_data = data["graph"]
        num_vehicles = data["numVehicles"]
    return network_data, num_vehicles


def create_qaoa_params(threshold, n_max, p):
    qaoa_dict = {"q_threshold": threshold, "q_n_max": n_max, "q_p": p}
    return qaoa_dict


def get_direction():
    direction = np.random.randint(-1, 2)
    if direction == 0:
        return get_direction()
    else:
        return direction
