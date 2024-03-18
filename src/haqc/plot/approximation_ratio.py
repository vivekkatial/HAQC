import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Any


def plot_approx_ratio_vs_iterations_for_layers(results_df, max_layers, filename):
    '''Plot the approximation ratio vs. iterations for selected algorithms: 
    every 5 layers and always include the first layer. Every 5th layer (and the first layer) 
    has a highlighted thickness, and only these layers are shown in the legend.'''
    
    # Group by 'algo' and 'restart', then calculate the max 'approx_ratio' for each group
    max_ratio_per_restart = results_df.groupby(['algo', 'restart'])['approx_ratio'].max().reset_index()

    # Calculate median and IQR for each algo
    median_max_ratios = max_ratio_per_restart.groupby('algo')['approx_ratio'].median().reset_index(name='median')
    iqr_max_ratios = max_ratio_per_restart.groupby('algo')['approx_ratio'].quantile([0.25, 0.75]).unstack().reset_index()
    iqr_max_ratios.columns = ['algo', 'q1', 'q3']

    # Merge median and IQR data
    plot_data = pd.merge(median_max_ratios, iqr_max_ratios, on='algo')

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.errorbar(plot_data['algo'], plot_data['median'], 
                yerr=[plot_data['median'] - plot_data['q1'], plot_data['q3'] - plot_data['median']],
                fmt='o-', capsize=5, capthick=2, ecolor='black', color='black', mec='grey', mew=2, ms=10)

    plt.title('Best Approx Ratio')
    plt.xlabel('Layer')
    plt.ylabel('Approx Ratio')
    plt.ylim(0, 1)  # Setting y-axis limits
    plt.xticks(plot_data['algo'])  # Ensure all algo are shown
    plt.grid(True)

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

