import pandas as pd
import json


def get_optimal_parameters(source, n_layers, df):
    """Get optimal parameters for a given source and number of layers."""
    # Check if the dataframe is empty
    if df.empty:
        return "No data available."

    # Allowed source values
    allowed_sources = [
        "four_regular_graph",
        "geometric",
        "nearly_complete_bi_partite",
        "power_law_tree",
        "three_regular_graph",
        "uniform_random",
        "watts_strogatz_small_world",
    ]
    # Check if the source is valid
    if source not in allowed_sources:
        return "Invalid source. Please choose from the allowed values."

    # Filter the dataframe for the specific source and number of layers
    filtered_df = df[(df['Source'] == source) & (df['params.n_layers'] == n_layers)]

    # Check if the filtered dataframe is not empty
    if not filtered_df.empty:
        # Initialize lists for beta and gamma values
        beta_values = []
        gamma_values = []

        # Extract relevant beta and gamma values
        for i in range(1, n_layers + 1):
            beta_key = 'median_beta_' + str(i)
            gamma_key = 'median_gamma_' + str(i)
            beta_values.append(filtered_df.iloc[0][beta_key])
            gamma_values.append(filtered_df.iloc[0][gamma_key])

        # Creating the final result
        params = {
            'beta': beta_values,
            'gamma': gamma_values,
            'Source': source,
            'params.n_layers': n_layers,
        }
        return params
    else:
        return "No data available for the specified source and number of layers."
