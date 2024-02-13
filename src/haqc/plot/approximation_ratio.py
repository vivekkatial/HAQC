import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Any


def plot_approx_ratio_vs_iterations_for_layers(results_df, max_layers, filename):
    '''Plot the approximation ratio vs. iterations for selected algorithms: 
    every 5 layers and always include the first layer. Every 5th layer (and the first layer) 
    has a highlighted thickness, and only these layers are shown in the legend.'''

    fig, ax = plt.subplots()

    # Plot every 5 layers and always include the first layer
    for layer in range(1, max_layers+1):
        if layer == 1 or layer % 5 == 0:
            layer_df = results_df[results_df['algo'] == layer]
            # Set a higher linewidth for the first layer and every 5th layer
            linewidth = 1
            label = f'Algo {layer}'
            ax.plot(layer_df['eval_count'], layer_df['approx_ratio'], label=label, linewidth=linewidth)

    # Calculate the acceptable approximation ratio
    max_approx_ratio = results_df['approx_ratio'].max()
    acceptable_approx_ratio = 0.95 * max_approx_ratio

    # Add dotted lines for the acceptable approximation ratio and an approximation ratio of 1
    ax.axhline(y=acceptable_approx_ratio, color='r', linestyle='--', label='Acceptable Approx. Ratio')
    ax.axhline(y=1, color='g', linestyle='--', label='Approx. Ratio of 1')

    # Labeling the plot
    ax.set_xlabel('Iterations (eval_count)')
    ax.set_ylabel('Approximation Ratio')
    ax.set_title('Approximation Ratio vs Iterations ')
    ax.legend()
    # Save the plot to a file
    plt.savefig(filename)


def plot_approx_ratio_vs_iterations_for_optimizers(results_df: pd.DataFrame, 
                                                   acceptable_approx_ratio: float, 
                                                   filename: str) -> None:
    """
    Generates a grid plot of approximation ratio versus iterations for each optimizer 
    present in the results DataFrame. It also adds horizontal lines for the acceptable 
    approximation ratio.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing the optimization results with columns 
                                 for 'optimizer', 'total_count', and 'approximation_ratio'.
    - acceptable_approx_ratio (float): The value of the acceptable approximation ratio to 
                                       be indicated on the plots.
    - filename (str): The filename for the output plot image (without file extension).

    Returns:
    - None: This function does not return a value. It saves the grid plot to a file.
    """

    # Get a list of unique optimizers
    unique_optimizers = results_df['optimizer'].unique()
    n_optimizers = len(unique_optimizers)

    # Calculate the number of rows/columns for the grid
    n_cols = int(np.ceil(np.sqrt(n_optimizers)))
    n_rows = int(np.ceil(n_optimizers / n_cols))

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()  # Flatten the array to make it easier to iterate over

    # Loop through each optimizer and create a subplot
    for i, optimizer in enumerate(unique_optimizers):
        # Filter the DataFrame for the current optimizer
        optimizer_df = results_df[results_df['optimizer'] == optimizer]
        
        # Plot approximation_ratio vs total_count
        axes[i].plot(optimizer_df['total_count'], optimizer_df['approximation_ratio'])
        
        # Add a horizontal line for the acceptable approximation ratio
        axes[i].axhline(y=acceptable_approx_ratio, color='r', linestyle='--')
        axes[i].axhline(y=1, color='g', linestyle='--', label='Approx. Ratio of 1')
        
        # Title and labels for the subplot
        axes[i].set_title(f'Optimizer: {optimizer}')
        axes[i].set_xlabel('Total Count')
        axes[i].set_ylabel('Approximation Ratio')

    # If there are more subplots than optimizers, remove the empty subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the adjusted grid plot as a PNG file
    plt.savefig(filename)

