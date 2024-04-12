import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def plot_landscape(landscape_data, ax=None, **kwargs):
    colors = [
        "#0000A3",
        "#7282ee",
        "#B0C7F9",
        "#e2d9d4",
        "#F6BFA6",
        "#de4d4d",
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_red_blue_darker_ends", colors, N=256)

    beta = landscape_data['beta'] / np.pi
    gamma = landscape_data['gamma'] / np.pi
    obj_vals = landscape_data['obj_vals']

    # Create a new figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_colorbar = True
    else:
        show_colorbar = 'colorbar' in kwargs and kwargs['colorbar']

    im = ax.imshow(
        obj_vals.T,
        origin='lower',
        cmap=custom_cmap,
        extent=(beta[0], beta[-1], gamma[0], gamma[-1]),
    )

    ax.set_xlabel(r'$\frac{\beta}{\pi}$')
    ax.set_ylabel(r'$\frac{\gamma}{\pi}$')

    beta_ticks = np.linspace(beta[0], beta[-1], 5)
    gamma_ticks = np.linspace(gamma[0], gamma[-1], 5)
    ax.set_xticks(beta_ticks)
    ax.set_yticks(gamma_ticks)

    if 'source' in kwargs:
        ax.set_title(kwargs['source'])

    if 'point' in kwargs:
        point = kwargs['point']
        point = (point[0] / np.pi, point[1] / np.pi)
        ax.plot(point[0], point[1], 'r*')

    if show_colorbar:
        plt.colorbar(im, ax=ax)

    

    return plt