from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fspb.config import SRC, BLD_FIGURES
from fspb.simulation.model_simulation import (
    CovarianceType,
    generate_default_time_grid,
    simulate_from_model,
)
from pytask import Product

PAPER_TEXT_WIDTH = 8.5 - 2  # us-letter width in inches minus margin
FIG_FONT_SIZE = 10


def task_visualize_outcome(
    _script: Path = SRC / "simulation" / "model_simulation.py",
    path_figure: Annotated[dict[str, Path], Product] = {
        "stationary": BLD_FIGURES / "outcomes_stationary.pdf",
        "non_stationary": BLD_FIGURES / "outcomes_non_stationary.pdf",
    },
) -> None:
    """Save Figure XXX for paper."""
    data = _generate_outcome_figure_data()
    figures = _create_outcome_figures(data)
    for key, figure in figures.items():
        figure.savefig(path_figure[key], bbox_inches="tight")
        plt.close(figure)


def _create_outcome_figure(
    data: dict[str, NDArray[np.floating]],
    key: str,
    color: str,
) -> plt.Figure:
    """Create figure showing stationary and non-stationary outcomes."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif=["Computer Modern Roman"])

    fig, ax = plt.subplots()

    ax.plot(
        data["time_grid"],
        data[key],
        color=color,
        linewidth=0.9,
        alpha=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(labelsize=FIG_FONT_SIZE)
    ax.grid(visible=True, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.95)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

    ax.set_xlabel("$t$", fontsize=FIG_FONT_SIZE)
    fig.text(0, 0.50, "$Y_i(t)$", fontsize=FIG_FONT_SIZE, rotation=0)

    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.set_size_inches(PAPER_TEXT_WIDTH, 1.5)
    fig.subplots_adjust(hspace=0)
    return fig


def _create_outcome_figures(
    data: dict[str, NDArray[np.floating]],
) -> dict[str, plt.Figure]:
    """Create figure showing stationary and non-stationary outcomes."""
    tableau_blue = "#5778a4"
    tableau_orange = "#e49444"

    fig_stationary = _create_outcome_figure(
        data, key="stationary_outcomes", color=tableau_blue
    )
    fig_non_stationary = _create_outcome_figure(
        data, key="non_stationary_outcomes", color=tableau_orange
    )
    return {
        "stationary": fig_stationary,
        "non_stationary": fig_non_stationary,
    }


def _generate_outcome_figure_data() -> dict[str, NDArray[np.floating]]:
    """Generate data for Figure XXX for paper."""
    rng = np.random.default_rng(13221)

    time_grid = generate_default_time_grid()
    stationary_data = simulate_from_model(
        n_samples=15,
        time_grid=time_grid,
        dof=5,
        covariance_type=CovarianceType.STATIONARY,
        length_scale=0.4,
        rng=rng,
    )
    non_stationary_data = simulate_from_model(
        n_samples=15,
        time_grid=time_grid,
        dof=5,
        covariance_type=CovarianceType.NON_STATIONARY,
        length_scale=0.4,
        rng=rng,
    )
    return {
        "time_grid": time_grid,
        "stationary_outcomes": stationary_data.y.T,
        "non_stationary_outcomes": non_stationary_data.y.T,
    }
