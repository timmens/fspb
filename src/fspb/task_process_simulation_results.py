from fspb.config import (
    PREDICTION_SCENARIOS,
    CONFIDENCE_SCENARIOS,
    Scenario,
    BLD_SIMULATION_OUR,
    BLD_SIMULATION_CONFORMAL_INFERENCE,
    BLD_SIMULATION_PROCESSED,
)
from pathlib import Path
from typing import Annotated
import pandas as pd
import pytask
from pytask import Product
from fspb.types import ConformalInferencePredictionMethod
from fspb.simulation.processing import (
    process_our_simulation_results,
    process_conformal_inference_simulation_results,
)

# ======================================================================================
# Results processing
# ======================================================================================

OUR_RESULTS_PATHS = [
    BLD_SIMULATION_OUR / f"{scenario.to_str()}.pkl"
    for scenario in PREDICTION_SCENARIOS + CONFIDENCE_SCENARIOS
]

CONFORMAL_INFERENCE_RESULTS_PATHS = {
    prediction_method: [
        BLD_SIMULATION_CONFORMAL_INFERENCE
        / str(prediction_method)
        / f"{scenario.to_str()}.json"
        for scenario in PREDICTION_SCENARIOS
    ]
    for prediction_method in ConformalInferencePredictionMethod
}


def task_process_our_simulation_results(
    our_result_paths: list[Path] = OUR_RESULTS_PATHS,
    processed_path: Annotated[Path, Product] = BLD_SIMULATION_PROCESSED / "our.pkl",
) -> None:
    our_results = [pd.read_pickle(path) for path in our_result_paths]
    scenarios = [Scenario.from_str(path.stem) for path in our_result_paths]
    processed = process_our_simulation_results(our_results, scenarios)
    processed.to_pickle(processed_path)


for prediction_method in ConformalInferencePredictionMethod:

    @pytask.task(id=prediction_method.value)
    def task_process_conformal_inference_simulation_results(
        conformal_inference_result_paths: list[
            Path
        ] = CONFORMAL_INFERENCE_RESULTS_PATHS[prediction_method],
        our_result_paths: list[Path] = OUR_RESULTS_PATHS,
        processed_path: Annotated[Path, Product] = BLD_SIMULATION_PROCESSED
        / f"conformal_inference_{prediction_method}.pkl",
    ) -> None:
        conformal_inference_results = [
            pd.read_json(path) for path in conformal_inference_result_paths
        ]
        our_results = [pd.read_pickle(path) for path in our_result_paths]
        scenarios = [
            Scenario.from_str(path.stem) for path in conformal_inference_result_paths
        ]
        processed = process_conformal_inference_simulation_results(
            conformal_inference_results=conformal_inference_results,
            our_results=our_results,
            scenarios=scenarios,
        )
        processed.to_pickle(processed_path)


# ======================================================================================
# Consolidation
# ======================================================================================


def task_consolidate_simulation_results(
    our_results_path: Path = BLD_SIMULATION_PROCESSED / "our.pkl",
    conformal_inference_mean_results_path: Path = BLD_SIMULATION_PROCESSED
    / "conformal_inference_mean.pkl",
    conformal_inference_linear_results_path: Path = BLD_SIMULATION_PROCESSED
    / "conformal_inference_linear.pkl",
    consolidated_path: Annotated[Path, Product] = BLD_SIMULATION_PROCESSED
    / "consolidated.pkl",
) -> None:
    our_results = pd.read_pickle(our_results_path)
    conformal_inference_mean_results = pd.read_pickle(
        conformal_inference_mean_results_path
    )
    conformal_inference_linear_results = pd.read_pickle(
        conformal_inference_linear_results_path
    )
    out = pd.concat(
        [
            our_results,
            conformal_inference_mean_results,
            conformal_inference_linear_results,
        ],
        keys=["Ours", "CI (Mean)", "CI (Linear)"],
        names=["Method"],
    )
    out.to_pickle(consolidated_path)
