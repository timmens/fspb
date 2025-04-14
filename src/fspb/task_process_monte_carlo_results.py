from fspb.config import BLD, ALL_SCENARIOS, Scenario
from pathlib import Path
from typing import Annotated
import pandas as pd
from pytask import Product
import pytask
from fspb.monte_carlo import SingleSimulationResult, MonteCarloSimulationResult
import json

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
    product_path = BLD / "monte_carlo" / "R" / f"{scenario.to_str()}.json"

    @pytask.task(id=scenario.to_str())
    def task_prepare_simulation_data_for_R(
        result_path: Path = result_path,
        product_path: Annotated[Path, Product] = product_path,
    ) -> None:
        mc_result: MonteCarloSimulationResult = pd.read_pickle(result_path)
        results: list[SingleSimulationResult] = mc_result.simulation_results

        data = []
        for r in results:
            item = {
                "y": r.simulation_data.y.tolist(),
                "x": r.simulation_data.x.tolist(),
                "time_grid": r.simulation_data.time_grid.tolist(),
                "new_y": r.new_data.y.tolist(),
                "new_x": r.new_data.x.tolist(),
            }
            data.append(item)

        with open(product_path, "w") as file:
            json.dump(data, file)


# ======================================================================================
# Processing code
# ======================================================================================


def process_monte_carlo_results(
    results: list[MonteCarloSimulationResult],
    scenarios: list[Scenario],
) -> pd.DataFrame:
    """Process the results of a Monte Carlo simulation.

    Args:
        results: The results of a Monte Carlo simulation.
        scenarios: The scenarios of the Monte Carlo simulation.

    """
    processed: list[pd.Series[pd.Float64Dtype]] = []
    for result, scenario in zip(results, scenarios):
        sr = pd.Series(scenario.to_dict())
        sr["coverage"] = result.coverage
        sr["maximum_width_statistic"] = result.maximum_width_statistic
        sr["interval_score"] = result.interval_score
        processed.append(sr)
    index_cols = scenarios[0]._fields()
    return pd.concat(processed, axis=1).T.set_index(index_cols)
