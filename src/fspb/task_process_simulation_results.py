from fspb.config import BLD, ALL_SCENARIOS, Scenario
from pathlib import Path
from typing import Annotated
import pandas as pd
from pytask import Product
from fspb.simulation.processing import (
    process_our_simulation_results,
    process_their_simulation_results,
    prepare_consolidated_results_for_publication,
)


OUR_RESULTS_PATHS = [
    BLD / "simulation" / "pickle" / f"{scenario.to_str()}.pkl"
    for scenario in ALL_SCENARIOS
]

THEIR_RESULTS_PATHS = [
    BLD / "simulation" / "R" / f"{scenario.to_str()}.json" for scenario in ALL_SCENARIOS
]


def task_process_our_simulation_results(
    our_result_paths: list[Path] = OUR_RESULTS_PATHS,
    processed_path: Annotated[Path, Product] = BLD
    / "simulation"
    / "our_results_processed.pkl",
) -> None:
    our_results = [pd.read_pickle(path) for path in our_result_paths]
    scenarios = [Scenario.from_str(path.stem) for path in our_result_paths]
    processed = process_our_simulation_results(our_results, scenarios)
    processed.to_pickle(processed_path)


def task_process_their_simulation_results(
    their_result_paths: list[Path] = THEIR_RESULTS_PATHS,
    our_result_paths: list[Path] = OUR_RESULTS_PATHS,
    processed_path: Annotated[Path, Product] = BLD
    / "simulation"
    / "their_results_processed.pkl",
) -> None:
    their_results = [pd.read_json(path) for path in their_result_paths]
    our_results = [pd.read_pickle(path) for path in our_result_paths]
    scenarios = [Scenario.from_str(path.stem) for path in their_result_paths]
    processed = process_their_simulation_results(
        their_results=their_results,
        our_results=our_results,
        scenarios=scenarios,
    )
    processed.to_pickle(processed_path)


def task_consolidate_simulation_results(
    our_results_path: Path = BLD / "simulation" / "our_results_processed.pkl",
    their_results_path: Path = BLD / "simulation" / "their_results_processed.pkl",
    consolidated_path: Annotated[Path, Product] = BLD
    / "simulation"
    / "consolidated_results.pkl",
) -> None:
    our_results = pd.read_pickle(our_results_path)
    their_results = pd.read_pickle(their_results_path)
    out = pd.concat(
        [our_results, their_results], keys=["Ours", "Their"], names=["Method"]
    )
    out.to_pickle(consolidated_path)


def task_processed_results_to_latex(
    consolidated_path: Path = BLD / "simulation" / "consolidated_results.pkl",
    latex_paths: Annotated[dict[str, Path], Product] = {
        "confidence": BLD / "simulation" / "consolidated_results_confidence.tex",
        "prediction": BLD / "simulation" / "consolidated_results_prediction.tex",
    },
) -> None:
    consolidated = pd.read_pickle(consolidated_path)
    tables = prepare_consolidated_results_for_publication(consolidated)
    for band_type, latex_path in latex_paths.items():
        tables[band_type].to_latex(latex_path)
