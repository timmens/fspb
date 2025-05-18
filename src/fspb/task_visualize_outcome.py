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
FIG_FONT_SIZE = 11


def task_visualize_outcome(
    _script: Path = SRC / "simulation" / "model_simulation.py",
    path_figure: Annotated[Path, Product] = BLD_FIGURES / "outcomes.pdf",
) -> None:
    """Save Figure XXX for paper."""
    data = _generate_outcome_figure_data()
    fig = _create_outcome_figures(data)
    fig.savefig(path_figure, bbox_inches="tight")


def _create_outcome_figures(
    data: dict[str, NDArray[np.floating]],
) -> plt.Figure:
    """Create figure showing stationary and non-stationary outcomes."""

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif=["Computer Modern Roman"])

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(PAPER_TEXT_WIDTH, 3),
        sharex=True,
        sharey=True,
    )

    tableau_blue = "#5778a4"
    tableau_orange = "#e49444"

    plot_data = {
        "stationary": {
            "title": "(a) Stationary",
            "ax": axes[0],
            "color": tableau_blue,
            "time_grid": data["time_grid"],
            "data": data["stationary_outcomes"],
        },
        "non_stationary": {
            "title": "(b) Non-stationary",
            "ax": axes[1],
            "color": tableau_orange,
            "time_grid": data["time_grid"],
            "data": data["non_stationary_outcomes"],
        },
    }

    for name, pdata in plot_data.items():
        ax = pdata["ax"]

        ax.plot(
            pdata["time_grid"],
            pdata["data"],
            color=pdata["color"],
            linewidth=0.9,
            alpha=0.8,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(labelsize=FIG_FONT_SIZE)
        ax.grid(visible=True, linestyle="--", alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.9, 1.95)
        ax.set_xticks([0, 1 / 3, 2 / 3, 1])
        ax.set_xticklabels(["$0$", "$1/3$", "$2/3$", "$1$"])
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["$-1$", "$0$", "$1$"])
        ax.set_xlabel("$t$", fontsize=FIG_FONT_SIZE)
        ax.set_title(pdata["title"], fontsize=FIG_FONT_SIZE)

    # fig.text(0, 0.50, "$Y_i(t)$", fontsize=FIG_FONT_SIZE, rotation=0)
    # fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.set_size_inches(PAPER_TEXT_WIDTH, 1.5)
    fig.subplots_adjust(hspace=0)
    return fig


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
