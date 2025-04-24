import pytask
from pathlib import Path
from typing import Annotated
from pytask import Product
import pandas as pd
import json
from fspb.simulation.processing import their_results_to_simulation_results_object
from fspb.types import BandType, CovarianceType
from fspb.bands.band import BAND_OPTIONS, BandOptions
from fspb.config import Scenario
from fspb.simulation.simulation_study import (
    simulation_study,
    SimulationOptions,
    SingleSimulationResult,
    SimulationResult,
)
from fspb.config import SRC, BLD

import matplotlib.pyplot as plt


scenario = Scenario(
    n_samples=30,
    dof=15,
    covariance_type=CovarianceType.STATIONARY,
    band_type=BandType.PREDICTION,
)

pickle_path = BLD / "visualization" / "data" / f"{scenario.to_str()}.pkl"

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


@pytask.task(id=scenario.to_str())
def task_simulation_study(
    _scripts: list[Path] = [
        SRC / "bands" / "band.py",
        SRC / "bands" / "covariance.py",
        SRC / "bands" / "fair_algorithm.py",
        SRC / "bands" / "min_width_algorithm.py",
        SRC / "simulation" / "simulation_study.py",
        SRC / "simulation" / "model_simulation.py",
    ],
    result_path: Annotated[Path, Product] = pickle_path,
    simulation_options: SimulationOptions = simulation_options,
    band_options: BandOptions = band_options,
) -> None:
    results = simulation_study(
        n_simulations=20,
        simulation_options=simulation_options,
        band_options=band_options,
        n_cores=4,
        seed=0,
    )
    pd.to_pickle(results, result_path)


# Convert simulation data to json
# ==================================================================================

json_path = BLD / "visualization" / "data" / f"{scenario.to_str()}_raw.json"


@pytask.task(id=scenario.to_str())
def task_export_simulation_data_to_json(
    _scripts: Path = SRC / "config.py",
    simulation_data_path: Path = pickle_path,
    json_path: Annotated[Path, Product] = json_path,
) -> None:
    sim_result: SimulationResult = pd.read_pickle(simulation_data_path)
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

product_path = BLD / "visualization" / "data" / f"{scenario.to_str()}.json"


@pytask.task(id=scenario.to_str())
@pytask.mark.r(script=SRC / "R" / "conformal_prediction.R")
def task_simulation_R(
    _scripts: list[Path] = [
        SRC / "config.py",
        SRC / "R" / "functions.R",
    ],
    functions_script_path: Path = SRC / "R" / "functions.R",
    simulation_data_path: Path = json_path,
    results_path: Annotated[Path, Product] = product_path,
) -> None:
    pass


# Process simulation results
# ======================================================================================


def task_process_their_simulation_results(
    their_result_path: Path = product_path,
    our_result_path: Path = pickle_path,
    processed_path: Annotated[Path, Product] = BLD
    / "visualization"
    / "data"
    / "their_results_as_simulation_results.pkl",
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


def task_visualize_band(
    our_result_path: Path = pickle_path,
    their_result_path: Path = BLD
    / "visualization"
    / "data"
    / "their_results_as_simulation_results.pkl",
    processed_paths: Annotated[list[Path], Product] = [
        BLD / "visualization" / f"seed_{seed}.png" for seed in range(20)
    ],
) -> None:
    our_result = pd.read_pickle(our_result_path)
    their_result = pd.read_pickle(their_result_path)
    for seed, processed_path in zip(range(20), processed_paths):
        ours = our_result.simulation_results[seed]
        theirs = their_result.simulation_results[seed]
        fig = visualize_band(ours, theirs)
        fig.savefig(processed_path)


# ======================================================================================
# Visualization
# ======================================================================================


def visualize_band(
    our_result: SingleSimulationResult,
    their_result: SingleSimulationResult,
) -> None:
    fig, ax = plt.subplots()

    time_grid = our_result.data.time_grid

    for result, color, label in [
        (our_result, "tab:blue", "Our band"),
        (their_result, "tab:orange", "Their band"),
    ]:
        ax.fill_between(
            time_grid,
            result.band.lower,
            result.band.upper,
            label=label,
            alpha=0.2,
            color=color,
        )

    ax.plot(time_grid, result.new_data.y, label="True", color="black")
    ax.legend()

    return fig
