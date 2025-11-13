from fspb.config import (
    PREDICTION_SCENARIOS,
    CONFIDENCE_SCENARIOS,
    Scenario,
    BLD_SIMULATION,
    BLD_SIMULATION_PROCESSED,
)
from pathlib import Path
from typing import Annotated
import pandas as pd
import pytask
from pytask import Product
from fspb.types import EstimationMethod
from fspb.simulation.processing import (
    process_our_simulation_results,
    process_conformal_inference_simulation_results,
)

# ======================================================================================
# Results processing
# ======================================================================================

OUR_RESULTS_PATHS = {
    method: [
        BLD_SIMULATION / method / f"{scenario.to_str()}.pkl"
        for scenario in PREDICTION_SCENARIOS + CONFIDENCE_SCENARIOS
    ]
    for method in EstimationMethod
}

# Only process FAIR method results
for method in [EstimationMethod.FAIR]:

    @pytask.task(id=method)
    def task_process_our_simulation_results(
        our_result_paths: list[Path] = OUR_RESULTS_PATHS[method],
        processed_path: Annotated[Path, Product] = BLD_SIMULATION_PROCESSED
        / f"{method}.pkl",
    ) -> None:
        our_results = [pd.read_pickle(path) for path in our_result_paths]
        scenarios = [Scenario.from_str(path.stem) for path in our_result_paths]
        processed = process_our_simulation_results(our_results, scenarios)
        processed.to_pickle(processed_path)


CI_RESULTS_PATHS = [
    BLD_SIMULATION / "ci" / f"{scenario.to_str()}.json"
    for scenario in PREDICTION_SCENARIOS
]


def task_process_ci_simulation_results(
    conformal_inference_result_paths: list[Path] = CI_RESULTS_PATHS,
    our_result_paths: list[Path] = OUR_RESULTS_PATHS[EstimationMethod.FAIR],
    processed_path: Annotated[Path, Product] = BLD_SIMULATION_PROCESSED / "ci.pkl",
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
    results_paths: dict[str, Path] = {
        "fair": BLD_SIMULATION_PROCESSED / "fair.pkl",
        "ci": BLD_SIMULATION_PROCESSED / "ci.pkl",
    },
    consolidated_path: Annotated[Path, Product] = BLD_SIMULATION_PROCESSED
    / "consolidated.pkl",
) -> None:
    results = {method: pd.read_pickle(path) for method, path in results_paths.items()}
    out = pd.concat(results, names=["method"])
    out.to_pickle(consolidated_path)
