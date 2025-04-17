from __future__ import annotations

import numpy as np
import pandas as pd
from fspb.bands.band import Band, BAND_OPTIONS
from fspb.simulation.simulation_study import SimulationResult, SingleSimulationResult
from fspb.config import Scenario


def prepare_consolidated_results_for_publication(
    consolidated: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    rounded = consolidated.map(lambda x: f"{x:.3f}").drop(columns="coverage_std")

    column_groups = [
        "maximum_width_statistic",
        "interval_score",
    ]
    combined = {"coverage": rounded["coverage"]}

    for column in column_groups:
        mean_col = rounded[column]
        std_col = rounded[f"{column}_std"]

        combined[column] = mean_col.astype(str) + " (" + std_col.astype(str) + ")"

    result = pd.DataFrame(combined)

    val_rename_mapping = {
        "covariance_type": {
            "non_stationary": "NS",
            "stationary": "S",
        },
    }

    var_rename_mapping = {
        "coverage": "Coverage",
        "maximum_width_statistic": "Maximum Width",
        "interval_score": "Interval Score",
        "n_samples": "$n$",
        "dof": r"$\nu$",
        "covariance_type": r"$\gamma_{st}$",
    }

    result = result.reset_index()
    result = result.replace(val_rename_mapping)  # type: ignore[arg-type]
    result = result.rename(columns=var_rename_mapping)
    result = result.set_index(
        ["Method", "band_type", "$n$", r"$\nu$", r"$\gamma_{st}$"]
    )
    result = result.unstack(level="Method")  # type: ignore[assignment]

    return {
        "prediction": result.xs("prediction", level="band_type"),  # type: ignore[dict-item]
        "confidence": result.xs("confidence", level="band_type"),  # type: ignore[dict-item]
    }


def process_our_simulation_results(
    results: list[SimulationResult],
    scenarios: list[Scenario],
) -> pd.DataFrame:
    """Process the results of our simulation study results.

    Args:
        results: The results of a simulation.
        scenarios: The scenarios of the simulation.

    Returns:
        A pandas DataFrame with the processed results.

    """
    processed: list[pd.Series[pd.Float64Dtype]] = []
    for result, scenario in zip(results, scenarios):
        data = result.report() | scenario.to_dict()
        processed.append(pd.Series(data))
    index_cols = scenarios[0]._fields()
    return (
        pd.concat(processed, axis=1).T.set_index(index_cols).sort_index().astype(float)
    )


def process_their_simulation_results(
    their_results: list[pd.DataFrame],
    our_results: list[SimulationResult],
    scenarios: list[Scenario],
) -> pd.DataFrame:
    """Process the results of their simulation study results.

    Since their simulation results are created in R and saved as json, we need to
    recover the additional information that is not saved in the json file, and then
    process the results like our own simulation results.

    Args:
        their_results: The results of their simulation study.
        our_results: The results of our simulation study.
        scenarios: The scenarios of the simulation.

    """
    simulation_results = [
        _result_and_scenario_to_simulation_result(
            their_result=their_result,
            our_result=our_result,
            scenario=scenario,
        )
        for their_result, our_result, scenario in zip(
            their_results, our_results, scenarios
        )
    ]
    return process_our_simulation_results(simulation_results, scenarios)


def _result_and_scenario_to_simulation_result(
    their_result: pd.DataFrame,
    our_result: SimulationResult,
    scenario: Scenario,
) -> SimulationResult:
    """Convert a scenario result DataFrame and a scenario to a SimulationResult object.

    Args:
        their_result: A DataFrame with the scenario results.
        our_result: The results of our simulation study.
        scenario: The scenario of the simulation.

    Returns:
        A SimulationResult object.

    """
    simulation_results = _scenario_results_to_single_simulation_results(
        their_result=their_result,
        our_result=our_result,
    )
    return SimulationResult(
        simulation_results=simulation_results,
        band_options=BAND_OPTIONS[scenario.band_type],
    )


def _scenario_results_to_single_simulation_results(
    their_result: pd.DataFrame,
    our_result: SimulationResult,
) -> list[SingleSimulationResult]:
    bands = _scenario_results_to_bands(their_result)
    return [
        SingleSimulationResult(
            band=band,
            data=our_result.data,
            new_data=our_result.new_data,
        )
        for band, our_result in zip(bands, our_result.simulation_results)
    ]


def _scenario_results_to_bands(scenario_results: pd.DataFrame) -> list[Band]:
    """Convert a scenario results DataFrame to a list of Band objects.

    The scenario results DataFrame has one row per simulation. It has the columns
    "estimate", "lower", and "upper". Each column entry is a list of floats, with length
    equal to the number of time points in the simulation.

    Args:
        scenario_results: A DataFrame with the scenario results.

    Returns:
        A list of Band objects.

    """
    return scenario_results.apply(_row_to_band, axis=1).to_list()  # type: ignore[call-overload]


def _row_to_band(row: pd.Series[pd.Float64Dtype]) -> Band:
    """Convert a row of a pandas DataFrame to a Band object.

    The data frame must have the columns "estimate", "lower", and "upper".

    Args:
        row: A row of a pandas DataFrame.

    Returns:
        A Band object.

    """
    return Band(**{k: np.array(v) for k, v in row.to_dict().items()})
