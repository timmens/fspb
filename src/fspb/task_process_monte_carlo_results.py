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


def task_processed_results_to_latex(
    processed_path: Path = BLD / "monte_carlo" / "processed.pkl",
    latex_paths: Annotated[dict[str, Path], Product] = {
        "Confidence": BLD / "monte_carlo" / "processed_confidence.tex",
        "Prediction": BLD / "monte_carlo" / "processed_prediction.tex",
    },
) -> None:
    processed = pd.read_pickle(processed_path)
    publication_table = prepare_processed_data_for_publication(processed)
    for band_type, latex_path in latex_paths.items():
        out = publication_table.xs(band_type, level="Band Type").reset_index()
        out.to_latex(latex_path, index=False)


for result_path in ALL_RESULTS_PATHS:
    scenario = Scenario.from_str(result_path.stem)
    product_path = BLD / "monte_carlo" / "R" / f"{scenario.to_str()}.json"

    @pytask.mark.skip(reason="Slow.")
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
                "y": r.data.y.tolist(),
                "x": r.data.x.tolist(),
                "time_grid": r.data.time_grid.tolist(),
                "new_y": r.new_data.y.tolist(),
                "new_x": r.new_data.x.tolist(),
            }
            data.append(item)

        with open(product_path, "w") as file:
            json.dump(data, file)


# ======================================================================================
# Processing code
# ======================================================================================


def prepare_processed_data_for_publication(processed: pd.DataFrame) -> pd.DataFrame:
    df = processed.map(lambda x: f"{x:.3f}")
    column_groups = [
        "coverage",
        "maximum_width_statistic",
        "interval_score",
    ]

    combined = {}

    for column in column_groups:
        mean_col = df[column]
        std_col = df[f"{column}_std"]

        combined[column] = mean_col.astype(str) + " (" + std_col.astype(str) + ")"

    result = pd.DataFrame(combined)

    val_rename_mapping = {
        "covariance_type": {
            "non_stationary": "NS",
            "stationary": "S",
        },
        "band_type": {
            "confidence": "Confidence",
            "prediction": "Prediction",
        },
    }

    var_rename_mapping = {
        "coverage": "Coverage",
        "maximum_width_statistic": "Max. Width",
        "interval_score": "Interval Score",
        "n_samples": "$n$",
        "dof": r"$\nu$",
        "covariance_type": r"$\gamma_{st}$",
        "band_type": "Band Type",
    }

    result = result.reset_index()
    result = result.replace(val_rename_mapping)  # type: ignore[arg-type]
    result = result.rename(columns=var_rename_mapping)

    return result.set_index(["$n$", r"$\nu$", r"$\gamma_{st}$", "Band Type"])


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
        data = result.report() | scenario.to_dict()
        processed.append(pd.Series(data))
    index_cols = scenarios[0]._fields()
    return (
        pd.concat(processed, axis=1).T.set_index(index_cols).sort_index().astype(float)
    )
