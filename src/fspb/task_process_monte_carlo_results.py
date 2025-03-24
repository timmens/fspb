from fspb.config import BLD, ALL_SCENARIOS, Scenario
from pathlib import Path
from typing import Annotated
import pandas as pd
from pytask import Product
import pytask
from fspb.process_monte_carlo_results import process_monte_carlo_results
from fspb.monte_carlo import SingleSimulationResult, MonteCarloSimulationResult
import numpy as np

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


for result_path in ALL_RESULTS_PATHS:
    scenario = Scenario.from_str(result_path.stem)
    product_path = BLD / "monte_carlo" / "R" / f"{scenario.to_str()}.npz"

    @pytask.task(id=scenario.to_str())
    def task_prepare_simulation_data_for_R(
        result_path: Path = result_path,
        product_path: Annotated[Path, Product] = product_path,
    ) -> None:
        mc_result: MonteCarloSimulationResult = pd.read_pickle(result_path)
        results: list[SingleSimulationResult] = mc_result.simulation_results

        y = [r.simulation_data.y for r in results]
        x = [r.simulation_data.x for r in results]
        time_grid = [r.simulation_data.time_grid for r in results]

        new_y = [r.new_data.y for r in results]
        new_x = [r.new_data.x for r in results]

        np.savez_compressed(
            file=product_path,
            y=y,
            x=x,
            time_grid=time_grid,
            new_y=new_y,
            new_x=new_x,
        )
