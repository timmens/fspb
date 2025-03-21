import pandas as pd
from fspb.monte_carlo import MonteCarloSimulationResult


def process_monte_carlo_results(
    results: list[MonteCarloSimulationResult],
    scenarios: list[dict[str, str]],
) -> pd.DataFrame:
    """Process the results of a Monte Carlo simulation.

    Args:
        results: The results of a Monte Carlo simulation.
        scenarios: The scenarios of the Monte Carlo simulation.

    """
    processed = []
    for result, scenario in zip(results, scenarios):
        df = pd.Series(scenario)
        df["coverage"] = result.coverage
        processed.append(df)
    index_cols = list(scenarios[0].keys())
    return pd.concat(processed, axis=1).T.set_index(index_cols)
