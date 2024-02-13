from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.optimization.applications.ising import tsp
import seaborn as sns
import numpy as np
import networkx as nx


def conver_bitstr_to_state(bitstr):
    """Convert bitstr to numpy array"""
    return np.array([float(i) for i in bitstr])


def check_constraint(state_vec, cns_type="city"):
    """Check constraints for TSP feasibility

    Args:
        state_vec (str): A bit string representing the state
        cns_type (str, optional): [description]. Defaults to "city".

    Raises:
        ValueError: If incorrect constraint type requested

    Returns:
        bool: `True` if constraint valid, `False` if constraint violated
    """
    if cns_type not in ["city", "time"]:
        raise ValueError("Can only handle 'city' or 'time' constraints")
    state_vec = conver_bitstr_to_state(state_vec)
    # Find num nodes
    dim = int(np.sqrt(len(state_vec)))
    # Matrix Representation
    matrix_rep = state_vec.reshape(dim, dim)
    # All row sums must equal to 1. for time constraints
    time_sum = np.sum(matrix_rep, axis=1)
    # All column sums must equal to 1. for city constraints
    city_sum = np.sum(matrix_rep, axis=0)

    if cns_type == "city":
        return np.array_equal(city_sum, np.ones(dim))
    else:
        return np.array_equal(time_sum, np.ones(dim))


def generate_feasibility_results(eigenstate, exact_result):
    """Plotting function for a graph of feasibility

    Args:
        eigenstate (dict): Dictionary of MPS counts for each state
        exact_result (np.Array): Exact result based on classical solver
    """
    feasible_count = 0
    infeasible_count = 0
    solution_count = 0
    vio_city_cnts = 0
    vio_time_cnts = 0
    vio_both_cnts = 0

    feasibility_results = {
        "feasible_count": None,
        "infeasible_count": None,
        "vio_city_cnts": None,
        "vio_time_cnts": None,
        "vio_both_cnts": None,
        "solution_count": None,
        "random_guess": None,
    }

    num_feasible = len([x for x in eigenstate.keys() if tsp.tsp_feasible(x)])
    x = sample_most_likely(exact_result.eigenstate)
    exact_sol_state = "".join([str(i) for i in x])
    for state in eigenstate.keys():
        # Confirm feasible count
        if (
            check_constraint(state, cns_type="city")
            and check_constraint(state, cns_type="time")
            and tsp.tsp_feasible(state)
        ):
            feasible_count += eigenstate[state]
        # Confirm city constraints are violated
        if check_constraint(state, cns_type="time") and not check_constraint(
            state, cns_type="city"
        ):
            vio_city_cnts += eigenstate[state]
            infeasible_count += eigenstate[state]
        if check_constraint(state, cns_type="city") and not check_constraint(
            state, cns_type="time"
        ):
            vio_time_cnts += eigenstate[state]
            infeasible_count += eigenstate[state]
        if not check_constraint(state, cns_type="city") and not check_constraint(
            state, cns_type="time"
        ):
            vio_both_cnts += eigenstate[state]
            infeasible_count += eigenstate[state]
        if exact_sol_state == state:
            solution_count = eigenstate[state]

    feasibility_results["feasible_count"] = feasible_count
    feasibility_results["infeasible_count"] = infeasible_count
    feasibility_results["solution_count"] = solution_count
    feasibility_results["vio_city_cnts"] = vio_city_cnts
    feasibility_results["vio_time_cnts"] = vio_time_cnts
    feasibility_results["vio_both_cnts"] = vio_both_cnts
    feasibility_results["random_guess"] = (
        feasible_count + infeasible_count
    ) / 2 ** len(state)
    feasibility_results["random_feasible_guess"] = (
        feasible_count + infeasible_count
    ) / 6

    return feasibility_results


def plot_feasibility_results(feasibility_results):
    """Produce plot of feasibility results

    Args:
        feasibility_results (dict): Dictionary of feasibility results for barplto

    """
    keys = list(feasibility_results.keys())
    # # get values in the same order as keys, and parse percentage values
    vals = [feasibility_results[k] for k in keys]

    sns.set_color_codes("muted")
    feasible_plot = sns.barplot(x=keys, y=vals)
    feasible_plot.set_xticklabels(
        feasible_plot.get_xticklabels(), rotation=90, horizontalalignment="right"
    )

    return feasible_plot
