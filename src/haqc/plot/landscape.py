import matplotlib.pyplot as plt
import numpy as np


def plot_landscape(landscape_data, **kwargs):
    beta = landscape_data['beta'] / np.pi
    gamma = landscape_data['gamma'] / np.pi
    obj_vals = landscape_data['obj_vals']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        obj_vals.T,
        origin='lower',
        cmap='viridis',
        extent=(beta[0], beta[-1], gamma[0], gamma[-1]),
    )

    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\gamma$')

    beta_ticks = np.linspace(beta[0], beta[-1], 5)
    gamma_ticks = np.linspace(gamma[0], gamma[-1], 5)
    ax.set_xticks(beta_ticks)
    ax.set_yticks(gamma_ticks)

    # Add title for Source if source in kwargs
    if 'source' in kwargs:
        ax.set_title(kwargs['source'])

    plt.colorbar(im, ax=ax)
    plt.show()
