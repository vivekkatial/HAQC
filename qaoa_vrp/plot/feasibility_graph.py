from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.optimization.applications.ising import tsp
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feasibility(eigenstate, exact_result):
    """Plotting function for a graph of feasibility

    Args:
        eigenstate ([type]): [description]
        exact_result ([type]): [description]
    """
    feasible_count = 0
    infeasible_count = 0
    solution_count = 0
    feasibility_results = {
        "feasible_count": None,
        "infeasible_count": None,
        "solution_count": None,
        "random_guess": None
    }

    num_feasible = len([x for x in eigenstate.keys() if tsp.tsp_feasible(x)])

    for state in eigenstate.keys():
        x = sample_most_likely(exact_result.eigenstate)
        exact_sol_state = ''.join([str(i) for i in x])
        if tsp.tsp_feasible(state):
            feasible_count += eigenstate[state]
        else:
            infeasible_count += eigenstate[state]
        if exact_sol_state == state:
            solution_count = eigenstate[state]

    feasibility_results["feasible_count"]=feasible_count
    feasibility_results["infeasible_count"]=infeasible_count
    feasibility_results["solution_count"]=solution_count
    feasibility_results["random_guess"]=(feasible_count+infeasible_count)/num_feasible
    feasibility_results["random_feasible_guess"]=(feasible_count+infeasible_count)/num_feasible
    keys = list(feasibility_results.keys())
    # # get values in the same order as keys, and parse percentage values
    vals = [feasibility_results[k] for k in keys]
    feasible_plot = sns.barplot(x=keys, y=vals)
    feasible_plot.set_xticklabels(feasible_plot.get_xticklabels(), 
                            rotation=90, 
                            horizontalalignment='right')
    
    return feasible_plot
                        