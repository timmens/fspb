from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fspb.config import BLD, SRC
from fspb.model_simulation import (
    CovarianceType,
    generate_time_grid,
    simulate_from_model,
)
from pytask import Product

PAPER_TEXT_WIDTH = 8.5 - 2  # us-letter width in inches minus margin
FIG_FONT_SIZE = 10


def task_visualize_outcome(
    _script: Path = SRC / "model_simulation.py",
    path_figure: Annotated[Path, Product] = BLD / "figures" / "outcomes.pdf",
) -> None:
    """Save Figure XXX for paper."""
    data = _generate_outcome_figure_data()
    fig = _create_outcome_figure(data)
    fig.savefig(path_figure)


def _create_outcome_figure(data: dict[str, NDArray[np.float64]]) -> plt.Figure:
    """Create figure showing stationary and non-stationary outcomes."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif=["Computer Modern Roman"])

    tableau_blue = "#5778a4"
    tableau_orange = "#e49444"

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

    axes[0].plot(
        data["time_grid"],
        data["stationary_outcomes"],
        color=tableau_blue,
        linewidth=0.9,
        alpha=0.8,
    )
    axes[1].plot(
        data["time_grid"],
        data["non_stationary_outcomes"],
        color=tableau_orange,
        linewidth=0.9,
        alpha=0.8,
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(labelsize=FIG_FONT_SIZE)
        ax.grid(visible=True, linestyle="--", alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.25, 1.8)

    fig.text(
        0.1,
        0.93,
        "(a) Stationary",
        fontsize=FIG_FONT_SIZE,
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
            "boxstyle": "round,pad=0.4",
            "edgecolor": "gray",
        },
    )
    fig.text(
        0.1,
        0.48,
        "(b) Non-stationary",
        fontsize=FIG_FONT_SIZE,
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
            "boxstyle": "round,pad=0.4",
            "edgecolor": "gray",
        },
    )

    axes[1].set_xlabel("$t$", fontsize=FIG_FONT_SIZE)
    fig.text(0, 0.53, "$Y_i(t)$", fontsize=FIG_FONT_SIZE, rotation=0)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.set_size_inches(PAPER_TEXT_WIDTH, 3)
    return fig


def _generate_outcome_figure_data() -> dict[str, NDArray[np.float64]]:
    """Generate data for Figure XXX for paper."""
    rng = np.random.default_rng(13221)

    time_grid = generate_time_grid(n_points=100)
    stationary_outcomes, _ = simulate_from_model(
        n_samples=20,
        time_grid=time_grid,
        dof=15,
        covariance_type=CovarianceType.STATIONARY,
        rng=rng,
    )
    non_stationary_outcomes, _ = simulate_from_model(
        n_samples=20,
        time_grid=time_grid,
        dof=15,
        covariance_type=CovarianceType.NON_STATIONARY,
        rng=rng,
    )
    return {
        "time_grid": time_grid,
        "stationary_outcomes": stationary_outcomes.T,
        "non_stationary_outcomes": non_stationary_outcomes.T,
    }
