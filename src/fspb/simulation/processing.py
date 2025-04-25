from __future__ import annotations

import numpy as np
import pandas as pd
from fspb.bands.band import Band, BAND_OPTIONS
from fspb.simulation.simulation_study import SimulationResult, SingleSimulationResult
from fspb.config import Scenario


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


def their_results_to_simulation_results_object(
    their_results: list[pd.DataFrame],
    our_results: list[SimulationResult],
    scenarios: list[Scenario],
) -> list[SimulationResult]:
    return [
        _result_and_scenario_to_simulation_result(their_result, our_result, scenario)
        for their_result, our_result, scenario in zip(
            their_results, our_results, scenarios
        )
    ]


def process_conformal_inference_simulation_results(
    conformal_inference_results: list[pd.DataFrame],
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
    simulation_results = their_results_to_simulation_results_object(
        their_results=conformal_inference_results,
        our_results=our_results,
        scenarios=scenarios,
    )
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
