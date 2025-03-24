from fspb.config import BLD, ALL_SCENARIOS, Scenario
from pathlib import Path
from typing import Annotated
import pandas as pd
from pytask import Product
from fspb.process_monte_carlo_results import process_monte_carlo_results

ALL_RESULTS_PATHS = [
    BLD / "monte_carlo" / "raw" / f"{scenario.to_str()}.pkl"
    for scenario in ALL_SCENARIOS
]


def task_process_monte_carlo_study(
    result_paths: list[Path] = ALL_RESULTS_PATHS,
    processed_path: Annotated[Path, Product] = BLD / "monte_carlo" / "processed.pkl",
) -> None:
    results = [pd.read_pickle(path) for path in result_paths]
    scenarios = [Scenario.from_str(path.stem) for path in result_paths]
    processed = process_monte_carlo_results(results, scenarios)
    processed.to_pickle(processed_path)
