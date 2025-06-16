import pytask
from pathlib import Path
from typing import Annotated
from pytask import Product
import pandas as pd
import json
from fspb.types import EstimationMethod, BandType, CIPredictionMethod
from fspb.bands.band import BandOptions
from fspb.config import (
    SRC,
    PREDICTION_SCENARIOS,
    CONFIDENCE_SCENARIOS,
    SKIP_R,
    BLD_SIMULATION,
    N_SIMULATIONS,
    N_JOBS,
)
from fspb.simulation.simulation_study import (
    simulation_study,
    SimulationOptions,
    SingleSimulationResult,
    SimulationResult,
)


# ======================================================================================
# Tasks
# ======================================================================================


for scenario in PREDICTION_SCENARIOS + CONFIDENCE_SCENARIOS:
    # Scenario-specific Options
    # ==================================================================================

    simulation_options = SimulationOptions(
        n_samples=scenario.n_samples,
        dof=scenario.dof,
        covariance_type=scenario.covariance_type,
        length_scale=1.0,
    )

    band_options = BandOptions.from_scenario(scenario)

    # Simulate data and run simulation study
    # ==================================================================================
    common_script_deps: list[Path] = [
        SRC / "bands" / "band.py",
        SRC / "bands" / "covariance.py",
        SRC / "simulation" / "simulation_study.py",
        SRC / "simulation" / "model_simulation.py",
    ]

    for method in [EstimationMethod.FAIR, EstimationMethod.MIN_WIDTH]:

        @pytask.task(id=f"{method}/{scenario.to_str()}")
        def task_simulation_study_our_method(
            _scripts: list[Path] = common_script_deps
            + [
                SRC / "bands" / f"{method.lower()}_algorithm.py",
            ],
            result_path: Annotated[Path, Product] = (
                BLD_SIMULATION / method / f"{scenario.to_str()}.pkl"
            ),
            simulation_options: SimulationOptions = simulation_options,
            band_options: BandOptions = band_options,
            method: EstimationMethod = method,
        ) -> None:
            results = simulation_study(
                n_simulations=N_SIMULATIONS,
                simulation_options=simulation_options,
                band_options=band_options,
                estimation_method=method,
                n_cores=N_JOBS,
                seed=1239487,
            )
            pd.to_pickle(results, result_path)

    # Skip confidence band simulation in R
    # ==================================================================================

    skip_r_analysis = SKIP_R or scenario.band_type == BandType.CONFIDENCE

    # Convert simulation data to json
    # ==================================================================================

    simulation_data_path = BLD_SIMULATION / "data" / f"{scenario.to_str()}.json"

    # Path to our simulation results. Since we fix the seed, it does not matter whether
    # we use FAIR or MIN_WIDTH method here.
    _sim_res_path = BLD_SIMULATION / EstimationMethod.FAIR / f"{scenario.to_str()}.pkl"

    @pytask.mark.skipif(skip_r_analysis, reason="Not running R analysis.")
    @pytask.task(id=scenario.to_str())
    def task_export_simulation_data_to_json(
        our_simulation_results_path: Path = _sim_res_path,
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
        BLD_SIMULATION / "ci" / f"{scenario.to_str()}.json"
    )

    @pytask.mark.skipif(skip_r_analysis, reason="Not running R analysis.")
    @pytask.task(id=scenario.to_str())
    @pytask.mark.r(script=SRC / "R" / "conformal_prediction.R")
    def task_simulation_study_conformal_inference(
        _scripts: list[Path] = [
            SRC / "R" / "functions.R",
        ],
        functions_script_path: Path = SRC / "R" / "functions.R",
        simulation_data_path: Path = simulation_data_path,
        significance_level: float = band_options.significance_level,
        fit_method: str = str(CIPredictionMethod.LINEAR),
        results_path: Annotated[Path, Product] = conformal_inference_results_path,
    ) -> None:
        pass
