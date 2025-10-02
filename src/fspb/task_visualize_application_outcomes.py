from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.typing import NDArray

from fspb.config import BLD_APPLICATION, PAPER_TEXT_WIDTH
from fspb.simulation.model_simulation import generate_time_grid
from pytask import Product

FIG_FONT_SIZE = 11


def task_visualize_outcome(
    data_path_y_train: Path = BLD_APPLICATION / "y_train.pickle",
    data_path_y_pred: Path = BLD_APPLICATION / "y_pred.pickle",
    path_figure: Annotated[Path, Product] = BLD_APPLICATION
    / "application-outcomes.pdf",
) -> None:
    """Save Figure XXX for paper."""
    y_train = pd.read_pickle(data_path_y_train)
    y_pred = pd.read_pickle(data_path_y_pred)
    fig = _create_outcome_figures(y_train, y_pred)
    fig.savefig(path_figure, bbox_inches="tight")


def _create_outcome_figures(
    y_train: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> plt.Figure:
    """Create figure showing training and prediction outcomes."""

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif=["Computer Modern Roman"])

    figsize = (0.7 * PAPER_TEXT_WIDTH, 0.7 * 0.6 * PAPER_TEXT_WIDTH)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=figsize,
    )

    time_grid = generate_time_grid(n_points=y_train.shape[1])

    ax.plot(
        time_grid,
        y_train.T,
        color="gray",
        linewidth=1,
        alpha=0.2,
    )

    ax.plot(
        time_grid,
        y_pred.T,
        color="#E15759",  # tableau red
        linewidth=2,
        alpha=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(labelsize=FIG_FONT_SIZE)
    ax.grid(visible=True, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 1)
    # ax.set_ylim(-1.9, 2.05)
    ax.set_xticks([0, 1 / 3, 2 / 3, 1])
    ax.set_xticklabels(["$0$", "$1/3$", "$2/3$", "$1$"])
    # ax.set_yticks([-1, 0, 1])
    # ax.set_yticklabels(["$-1$", "$0$", "$1$"])
    ax.set_xlabel("$t$", fontsize=FIG_FONT_SIZE)
    ax.set_ylabel("Force $(N/kg)$", fontsize=FIG_FONT_SIZE)

    # fig.text(0, 0.50, "$Y_i(t)$", fontsize=FIG_FONT_SIZE, rotation=0)
    # fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.set_size_inches(*figsize)
    # fig.subplots_adjust(hspace=0)
    return fig
