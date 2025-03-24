import pandas as pd
from fspb.monte_carlo import MonteCarloSimulationResult
from fspb.config import Scenario


def process_monte_carlo_results(
    results: list[MonteCarloSimulationResult],
    scenarios: list[Scenario],
) -> pd.DataFrame:
    """Process the results of a Monte Carlo simulation.

    Args:
        results: The results of a Monte Carlo simulation.
        scenarios: The scenarios of the Monte Carlo simulation.

    """
    processed: list[pd.Series] = []
    for result, scenario in zip(results, scenarios):
        ser = pd.Series(scenario.to_dict())
        ser["coverage"] = result.coverage
        processed.append(ser)
    index_cols = scenarios[0]._fields()
    return pd.concat(processed, axis=1).T.set_index(index_cols)
