import numpy as np
import pytask
from pathlib import Path
from typing import Annotated
from pytask import Product
import pandas as pd
import json
from fspb.types import (
    BandType,
    CIPredictionMethod,
    DistributionType,
    EstimationMethod,
)
from fspb.config import SRC, SKIP_R, BLD_APPLICATION, PAPER_TEXT_WIDTH

import matplotlib.pyplot as plt
from fspb.bands.band import Band

SIGNFICANCE_LEVEL = 0.1


def task_fit_min_width_band(
    _scripts: list[Path] = [
        SRC / "bands" / "band.py",
        SRC / "bands" / "covariance.py",
        SRC / "bands" / "critical_values.py",
    ],
    y_train_path: Path = BLD_APPLICATION / "y_train.pickle",
    x_train_path: Path = BLD_APPLICATION / "x_train.pickle",
    x_new_path: Path = BLD_APPLICATION / "x_pred.pickle",
    result_path: Annotated[Path, Product] = BLD_APPLICATION / "min_width_bands.pickle",
) -> None:
    y_train = pd.read_pickle(y_train_path)
    x_train = pd.read_pickle(x_train_path)
    x_new = pd.read_pickle(x_new_path)
    bands = []
    for amputee_index in range(len(x_new)):
        _x_new = x_new[amputee_index]
        band = Band.fit(
            y=y_train,
            x=x_train,
            x_new=_x_new,
            band_type=BandType.PREDICTION,
            time_grid=np.linspace(0, 1, 101),
            interval_cutoffs=np.array([0, 1 / 3, 2 / 3, 1]),
            significance_level=SIGNFICANCE_LEVEL,
            distribution_type=DistributionType.STUDENT_T,
            method=EstimationMethod.MIN_WIDTH,
        )
        bands.append(band)
    pd.to_pickle(bands, result_path)


# Convert application data to json
# ======================================================================================


@pytask.mark.skipif(SKIP_R, reason="Not running R analysis.")
def task_export_application_data_to_json(
    y_train_path: Path = BLD_APPLICATION / "y_train.pickle",
    x_train_path: Path = BLD_APPLICATION / "x_train.pickle",
    y_new_path: Path = BLD_APPLICATION / "y_pred.pickle",
    x_new_path: Path = BLD_APPLICATION / "x_pred.pickle",
    json_path: Annotated[Path, Product] = (
        BLD_APPLICATION / "data_for_conformal_inference.json"
    ),
) -> None:
    y_train = pd.read_pickle(y_train_path)
    x_train = pd.read_pickle(x_train_path)
    y_new = pd.read_pickle(y_new_path)
    x_new = pd.read_pickle(x_new_path)

    data = []
    for amputee_index in range(len(x_new)):
        _x_new = x_new[amputee_index]
        _y_new = y_new[amputee_index]
        item = {
            "y": y_train.tolist(),
            "x": x_train.tolist(),
            "time_grid": np.linspace(0, 1, 101).tolist(),
            "new_y": _y_new.tolist(),
            "new_x": _x_new.tolist(),
        }
        data.append(item)

    with open(json_path, "w") as file:
        json.dump(data, file)


# Run conformal inference in R
# ======================================================================================


@pytask.mark.skipif(SKIP_R, reason="Not running R analysis.")
@pytask.mark.r(script=SRC / "R" / "conformal_prediction.R")
def task_run_conformal_inference_on_application_data(
    _scripts: list[Path] = [
        SRC / "R" / "functions.R",
    ],
    functions_script_path: Path = SRC / "R" / "functions.R",
    simulation_data_path: Path = (
        BLD_APPLICATION / "data_for_conformal_inference.json"
    ),
    significance_level: float = SIGNFICANCE_LEVEL,
    fit_method: str = str(CIPredictionMethod.MEAN),
    results_path: Annotated[Path, Product] = (
        BLD_APPLICATION / "conformal_inference_results.json"
    ),
) -> None:
    pass


@pytask.mark.skipif(SKIP_R, reason="Not running R analysis.")
def task_process_conformal_inference_results_to_bands(
    their_result_path: Path = BLD_APPLICATION / "conformal_inference_results.json",
    processed_path: Annotated[Path, Product] = (
        BLD_APPLICATION / "conformal_inference_bands.pickle"
    ),
) -> None:
    their_results = pd.read_json(their_result_path)
    bands = []
    for amputee_index in range(len(their_results)):
        _their_result = their_results.iloc[amputee_index]
        band = Band(
            estimate=np.array(_their_result["estimate"]),
            lower=np.array(_their_result["lower"]),
            upper=np.array(_their_result["upper"]),
        )
        bands.append(band)
    pd.to_pickle(bands, processed_path)


# Visualize band
# ======================================================================================


def task_visualize_application_bands(
    min_width_band_paths: Path = BLD_APPLICATION / "min_width_bands.pickle",
    conformal_inference_band_paths: Path = (
        BLD_APPLICATION / "conformal_inference_bands.pickle"
    ),
    y_new_path: Path = BLD_APPLICATION / "y_pred.pickle",
    y_nearest_neighbors_path: Path = BLD_APPLICATION / "y_nearest_neighbors.pickle",
    figure_paths: Annotated[list[Path], Product] = [
        BLD_APPLICATION / f"amputee_band_{amputee_index}.pdf"  # type: ignore[name-defined]
        for amputee_index in range(7)
    ],
) -> None:
    min_width_bands = pd.read_pickle(min_width_band_paths)
    conformal_inference_bands = pd.read_pickle(conformal_inference_band_paths)
    y_new = pd.read_pickle(y_new_path)
    y_nearest_neighbors = pd.read_pickle(y_nearest_neighbors_path)
    for amputee_index, processed_path in enumerate(figure_paths):
        min_width_band = min_width_bands[amputee_index]
        conformal_inference_band = conformal_inference_bands[amputee_index]
        fig = visualize_bands(
            min_width_band=min_width_band,
            conformal_inference_band=conformal_inference_band,
            new_y=y_new[amputee_index],
            nearest_neighbor_y=y_nearest_neighbors[amputee_index],
        )
        fig.savefig(processed_path, bbox_inches="tight")


# ======================================================================================
# Visualization
# ======================================================================================


def visualize_bands(
    min_width_band: Band,
    conformal_inference_band: Band,
    new_y: np.ndarray,
    nearest_neighbor_y: np.ndarray,
) -> plt.Figure:
    """Visualize the bands for the stationary and non-stationary cases."""
    FIG_FONT_SIZE = 11

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif=["Computer Modern Roman"])

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(PAPER_TEXT_WIDTH, 3),
        sharex=True,
        sharey=True,
    )

    ax = _visualize_bands(
        min_width_band=min_width_band,
        conformal_inference_band=conformal_inference_band,
        ax=ax,
        new_y=new_y,
        nearest_neighbor_y=nearest_neighbor_y,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(labelsize=FIG_FONT_SIZE)
    ax.grid(visible=True, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 1)
    # ax.set_ylim(-2.9, 3.9)
    ax.set_xticks([0, 1 / 3, 2 / 3, 1])
    ax.set_xticklabels(["$0$", "$1/3$", "$2/3$", "$1$"])
    # ax.set_yticks([-2, 0, 2])
    # ax.set_yticklabels(["$-2$", "$0$", "$2$"])
    ax.set_ylabel("Force $(N/kg)$", fontsize=FIG_FONT_SIZE)
    ax.set_xlabel("$t$", fontsize=FIG_FONT_SIZE)

    # set legend
    handles, _ = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        [
            "Min.-Width",
            "Conformal inference",
            r"$Y_{\textsf{amputee}}(t)$",
            r"$Y_{\textsf{nearest-neighbor}}(t)$",
            r"$X_{\textsf{amputee}}(t)^{\mathsf{T}} \hat{\beta}(t)$",
        ],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.52, -0.04),
        fontsize=FIG_FONT_SIZE,
    )

    # fig.text(0, 0.48, r"$Y(t)$", fontsize=FIG_FONT_SIZE, rotation=0)
    fig.tight_layout(rect=(0.01, 0.03, 1, 1))
    fig.set_size_inches(PAPER_TEXT_WIDTH, PAPER_TEXT_WIDTH * 0.75)
    return fig


def _visualize_bands(
    min_width_band: Band,
    conformal_inference_band: Band,
    new_y: np.ndarray,
    nearest_neighbor_y: np.ndarray,
    ax: plt.Axes,
    set_legend: bool = False,
) -> plt.Axes:
    """Create figure showing stationary or non-stationary outcomes."""

    time_grid = np.linspace(0, 1, 101)
    new_y_hat = min_width_band.estimate

    tableau = {
        "blue": "#5778a4",
        "orange": "#e78230",
        "pink": "#f1a2a9",
        "green": "#6a9f58",
        "yellow": "#e7ca60",
        "brown": "#967662",
    }

    # order in which these plots are drawn matters, since otherwise the order of the
    # legend labels is not correct
    ax.fill_between(
        time_grid,
        min_width_band.lower,
        min_width_band.upper,
        label="Min.-Width",
        alpha=0.6,
        color=tableau["blue"],
        zorder=2,
    )
    ax.fill_between(
        time_grid,
        conformal_inference_band.lower,
        conformal_inference_band.upper,
        label="CI (linear)",
        alpha=0.5,
        color=tableau["green"],
        zorder=1,
    )
    ax.plot(
        time_grid,
        new_y,
        label="True",
        color=tableau["yellow"],
        linewidth=3,
        zorder=4,
        alpha=0.9,
    )
    ax.plot(
        time_grid,
        nearest_neighbor_y,
        label="NN",
        color=tableau["orange"],
        linewidth=2,
        zorder=3,
        alpha=0.9,
    )
    ax.plot(
        time_grid,
        new_y_hat,
        label="Estimate",
        color="black",
        alpha=0.6,
        linestyle="--",
        linewidth=1.5,
        zorder=3,
    )

    return ax
