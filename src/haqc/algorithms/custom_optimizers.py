from qiskit.algorithms.optimizers import Optimizer, OptimizerResult
from scipy.optimize import minimize
import numpy as np

class OptimiseLayerGreaterThanP(Optimizer):
    """Custom optimizer for QAOA that fixes the first layer's parameters (beta_0, gamma_0) and optimizes the rest using Nelder-Mead."""
    
    def __init__(self, fixed_betas, fixed_gammas):
        """
        Initializes the optimizer with fixed parameters for the first layer's beta and gamma.
        
        Parameters:
            fixed_betas (float): Fixed parameters for beta.
            fixed_gammas (float): Fixed parameters for gamma.
        """
        self.fixed_betas = fixed_betas
        self.fixed_gammas = fixed_gammas
        super().__init__()
    
    def get_support_level(self):
        """Return the support level dictionary indicating gradient and bound support."""
        return {
            'gradient': 0,  # Nelder-Mead does not use gradients
            'bounds': 0,    # Nelder-Mead does not inherently support bounds
            'initial_point': 1,  # Initial points are used
        }
    
    def minimize(self, fun, x0, jac=None, bounds=None):
        """
        Custom minimize function for optimizing only the second set of parameters (beta_1 and gamma_1) using Nelder-Mead.
        """

        
        def fun_wrapper(params):
            # Construct the parameter list for the function with fixed and optimized parameters
            # import pdb; pdb.set_trace()
            all_params = np.concatenate([self.fixed_betas, [params[0]], self.fixed_gammas, [params[1]]])
            return fun(all_params)
    
        # Assuming x0[N] and x0[3] are initial guesses for beta_i and gamma_j, respectively
        n_layers = len(x0) // 2
        
        # beta_{n} index
        beta_n_ind = n_layers - 1
        gamma_n_ind = n_layers*2 - 1
        
        initial_guess = [x0[beta_n_ind], x0[gamma_n_ind]]

        # Perform optimization using Nelder-Mead, optimizing only for beta_1 and gamma_1
        result = minimize(fun_wrapper, initial_guess, method='Nelder-Mead')
    
        # Construct the full parameter list with optimized values
        optimized_params = [*self.fixed_betas, result.x[0], *self.fixed_gammas, result.x[1]]

    
        optimizer_result = OptimizerResult()
        optimizer_result.x = optimized_params  # Complete parameter set including fixed and optimized
        optimizer_result.fun = result.fun  # Objective function value at the optimized parameters
        optimizer_result.nfev = result.nfev  # Number of function evaluations
        optimizer_result.success = result.success  # Whether the optimization was successful
    
        return optimizer_result
    

class OptimizationTracker:
    def __init__(self):
        self.intermediate_values = []
        self.optimizer_name = ""
        self.layer = None
    
    def set_optimizer_name(self, name):
        self.optimizer_name = name
    
    def set_layer(self, layer):
        self.layer = layer

    def store_intermediate_result(self, eval_count, parameters, mean, std):
        self.intermediate_values.append({
            'iteration': eval_count,
            'parameters': parameters,
            'energy': mean,
            'std': std,
            'optimizer': self.optimizer_name,
            'layer': self.layer
        })
