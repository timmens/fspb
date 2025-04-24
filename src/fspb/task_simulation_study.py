import pytask
from pathlib import Path
from typing import Annotated
from pytask import Product
import pandas as pd
import json
from fspb.types import BandType
from fspb.bands.band import BAND_OPTIONS, BandOptions
from fspb.config import PREDICTION_SCENARIOS, CONFIDENCE_SCENARIOS, SKIP_R
from fspb.simulation.simulation_study import (
    simulation_study,
    SimulationOptions,
    SingleSimulationResult,
    SimulationResult,
)
from fspb.config import SRC, BLD


# ======================================================================================
# Tasks
# ======================================================================================


for scenario in PREDICTION_SCENARIOS + CONFIDENCE_SCENARIOS:
    pickle_path = BLD / "simulation" / "pickle" / f"{scenario.to_str()}.pkl"

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
            n_simulations=500,
            simulation_options=simulation_options,
            band_options=band_options,
            n_cores=10,
            seed=None,
        )
        pd.to_pickle(results, result_path)

    # Skip confidence band simulation in R
    # ==================================================================================

    skip_r_analysis = SKIP_R or scenario.band_type == BandType.CONFIDENCE

    # Convert simulation data to json
    # ==================================================================================

    json_path = BLD / "simulation" / "json" / f"{scenario.to_str()}.json"

    @pytask.mark.skipif(skip_r_analysis, reason="Not running R analysis.")
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

    product_path = BLD / "simulation" / "R" / f"{scenario.to_str()}.json"

    @pytask.mark.skipif(skip_r_analysis, reason="Not running R analysis.")
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
