import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fspb.simulation.model_simulation import generate_time_grid


def plot_sample_paths(
    paths: NDArray[np.floating], path_name: str = ""
) -> tuple[plt.Figure, plt.Axes]:
    """Plot sample paths.

    Args:
        paths: The paths to plot.
        path_name: The name of the path. Will be displayed on the y-axis.

    Returns:
        fig: The figure.
        ax: The axis.

    """
    n_samples, n_points = paths.shape
    time_grid = generate_time_grid(n_points)

    fig, ax = plt.subplots(figsize=(10, 6))
    for path in paths:
        ax.plot(time_grid, path, alpha=0.7)

    ax.set_xlabel("Time (t)")
    ax.set_ylabel(path_name)
    return fig, ax
