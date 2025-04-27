import pytask
from pathlib import Path
from typing import Annotated
from pytask import Product
import pandas as pd
import json
from fspb.simulation.processing import their_results_to_simulation_results_object
from fspb.types import BandType, CovarianceType, ConformalInferencePredictionMethod
from fspb.bands.band import BAND_OPTIONS, BandOptions
from fspb.config import Scenario
from fspb.simulation.simulation_study import (
    simulation_study,
    SimulationOptions,
    SingleSimulationResult,
    SimulationResult,
)
from fspb.config import SRC, BLD, BLD_VISUALIZATION, SKIP_R

import matplotlib.pyplot as plt
from fspb.bands.band import Band

N_TRIALS = 1

scenarios = Scenario.from_lists(
    n_samples=[30],
    dof=[15],
    covariance_type=[CovarianceType.STATIONARY, CovarianceType.NON_STATIONARY],
    band_type=[BandType.PREDICTION],
)

for scenario in scenarios:
    our_results_path = (
        BLD_VISUALIZATION / "data" / f"our_{scenario.covariance_type}.pkl"
    )

    # Scenario-specific Options
    # ==================================================================================

    simulation_options = SimulationOptions(
        n_samples=scenario.n_samples,
        dof=scenario.dof,
        covariance_type=scenario.covariance_type,
        length_scale=0.4,
    )

    band_options = BAND_OPTIONS[scenario.band_type]

    # Simulate data and run simulation study
    # ==================================================================================

    @pytask.task(id=str(scenario.covariance_type))
    def task_simulation_study_our_method_for_visualize(
        _scripts: list[Path] = [
            SRC / "bands" / "band.py",
            SRC / "bands" / "covariance.py",
            SRC / "bands" / "fair_algorithm.py",
            SRC / "bands" / "min_width_algorithm.py",
            SRC / "simulation" / "simulation_study.py",
            SRC / "simulation" / "model_simulation.py",
        ],
        result_path: Annotated[Path, Product] = our_results_path,
        simulation_options: SimulationOptions = simulation_options,
        band_options: BandOptions = band_options,
    ) -> None:
        results = simulation_study(
            n_simulations=N_TRIALS,
            simulation_options=simulation_options,
            band_options=band_options,
            n_cores=10,
            seed=3,
        )
        pd.to_pickle(results, result_path)

    # Skip confidence band simulation in R
    # ==================================================================================

    skip_r_analysis = SKIP_R or scenario.band_type == BandType.CONFIDENCE

    # Convert simulation data to json
    # ==================================================================================

    simulation_data_path = (
        BLD_VISUALIZATION / "data" / f"data_{scenario.covariance_type}.json"
    )

    @pytask.task(id=str(scenario.covariance_type))
    @pytask.mark.skipif(skip_r_analysis, reason="Not running R analysis.")
    def task_export_simulation_data_to_json_for_visualize(
        our_simulation_results_path: Path = our_results_path,
        json_path: Annotated[Path, Product] = simulation_data_path,
    ) -> None:
        sim_result: SimulationResult = pd.read_pickle(our_simulation_results_path)
        results: list[SingleSimulationResult] = sim_result.simulation_results

        data = []
        for r in results:
            item = {
                "y": r.data.y.tolist(),
                "x": r.data.x.tolist(),
                "time_grid": r.data.time_grid.tolist(),
                "new_y": r.new_data.y.tolist(),
                "new_x": r.new_data.x.tolist(),
            }
            data.append(item)

        with open(json_path, "w") as file:
            json.dump(data, file)

    # Run conformal inference in R
    # ==================================================================================

    conformal_inference_results_path = (
        BLD_VISUALIZATION
        / "data"
        / f"conformal_inference_{scenario.covariance_type}.json"
    )

    @pytask.mark.skipif(skip_r_analysis, reason="Not running R analysis.")
    @pytask.task(id=str(scenario.covariance_type))
    @pytask.mark.r(script=SRC / "R" / "conformal_prediction.R")
    def task_simulation_study_conformal_inference_for_visualize(
        _scripts: list[Path] = [
            SRC / "R" / "functions.R",
        ],
        functions_script_path: Path = SRC / "R" / "functions.R",
        simulation_data_path: Path = simulation_data_path,
        significance_level: float = band_options.significance_level,
        fit_method: str = str(ConformalInferencePredictionMethod.LINEAR),
        results_path: Annotated[Path, Product] = conformal_inference_results_path,
    ) -> None:
        pass

    conformal_inference_processed_path = (
        BLD_VISUALIZATION
        / "data"
        / f"conformal_inference_{scenario.covariance_type}.pkl"
    )

    @pytask.mark.skipif(skip_r_analysis, reason="Not running R analysis.")
    @pytask.task(id=f"{scenario.covariance_type}")
    def task_process_conformal_inference_simulation_results_for_visualize(
        our_result_path: Path = our_results_path,
        their_result_path: Path = conformal_inference_results_path,
        processed_path: Annotated[Path, Product] = conformal_inference_processed_path,
    ) -> None:
        their_results = pd.read_json(their_result_path)
        our_results = pd.read_pickle(our_result_path)
        processed = their_results_to_simulation_results_object(
            their_results=[their_results],
            our_results=[our_results],
            scenarios=[scenario],
        )
        pd.to_pickle(processed[0], processed_path)

    # Visualize band
    # ======================================================================================

    @pytask.task(id=str(scenario.covariance_type))
    def task_visualize_band(
        our_result_path: Path = our_results_path,
        conformal_inference_result_path: Path = conformal_inference_processed_path,
        processed_paths: Annotated[list[Path], Product] = [
            BLD / "visualization" / f"seed_{seed}_{scenario.covariance_type}.pdf"
            for seed in range(N_TRIALS)
        ],
    ) -> None:
        our_result = pd.read_pickle(our_result_path)
        conformal_inference_results = pd.read_pickle(conformal_inference_result_path)
        for seed, processed_path in zip(range(N_TRIALS), processed_paths):
            ours = our_result.simulation_results[seed]
            conformal_inference = conformal_inference_results.simulation_results[seed]
            fig = _visualize_bands(
                our_sim_result=ours,
                conformal_inference_band=conformal_inference.band,
            )
            fig.savefig(processed_path, bbox_inches="tight")


# ======================================================================================
# Visualization
# ======================================================================================


def _visualize_bands(
    our_sim_result: SingleSimulationResult,
    conformal_inference_band: Band,
) -> plt.Figure:
    """Create figure showing stationary and non-stationary outcomes."""
    PAPER_TEXT_WIDTH = 8.5 - 2  # us-letter width in inches minus margin
    FIG_FONT_SIZE = 10

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", serif=["Computer Modern Roman"])

    time_grid = our_sim_result.data.time_grid
    new_y = our_sim_result.new_data.y
    new_y_hat = our_sim_result.band.estimate

    tableau = {
        "blue": "#5778a4",
        "orange": "#e78230",
        "pink": "#f1a2a9",
        "green": "#6a9f58",
        "yellow": "#e7ca60",
    }

    fig, ax = plt.subplots()

    conformal_inference_band_ax = ax.fill_between(
        time_grid,
        conformal_inference_band.lower,
        conformal_inference_band.upper,
        label="CI (linear)",
        alpha=0.5,
        color=tableau["green"],
    )

    our_sim_result_ax = ax.fill_between(
        time_grid,
        our_sim_result.band.lower,
        our_sim_result.band.upper,
        label="Our",
        alpha=0.6,
        color=tableau["blue"],
    )

    y_new_hat_ax = ax.plot(
        time_grid,
        new_y_hat,
        label="Estimate",
        color="black",
        alpha=0.6,
        linestyle="--",
        linewidth=1.5,
    )
    y_new_ax = ax.plot(
        time_grid, new_y, label="True", color=tableau["yellow"], linewidth=2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(labelsize=FIG_FONT_SIZE)
    ax.grid(visible=True, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, 3.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

    ax.set_xlabel("$t$", fontsize=FIG_FONT_SIZE)
    fig.text(0, 0.50, r"$Y(t)$", fontsize=FIG_FONT_SIZE, rotation=0)

    fig.tight_layout(rect=(0.01, 0.03, 1, 1))
    fig.set_size_inches(PAPER_TEXT_WIDTH, 3)

    ax.legend(
        [
            our_sim_result_ax,
            conformal_inference_band_ax,
            y_new_ax[0],
            y_new_hat_ax[0],
        ],
        [
            "Our",
            "Conformal inference",
            r"$Y_{\textsf{new}}(t)$",
            r"$X_{\textsf{new}}(t)^{\mathsf{T}} \hat{\beta}(t)$",
        ],
        frameon=False,
        ncols=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.275),
    )

    return fig
