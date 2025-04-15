import pytask
from pathlib import Path
from typing import Annotated
from pytask import Product
import pandas as pd
import numpy as np

from fspb.config import ALL_SCENARIOS
from fspb.bands import BandType
from fspb.monte_carlo import monte_carlo_simulation, BandOptions, SimulationOptions
from fspb.config import SRC, BLD
from fspb.fair_algorithm import DistributionType


band_options = BandOptions(
    band_type=BandType.CONFIDENCE,
    interval_cutoffs=np.array([0, 1 / 3, 2 / 3, 1]),
    significance_level=0.1,
    distribution_type=DistributionType.STUDENT_T,
)

for scenario in ALL_SCENARIOS:
    result_path = BLD / "monte_carlo" / "raw" / f"{scenario.to_str()}.pkl"

    simulation_options = SimulationOptions(
        n_samples=scenario.n_samples,
        dof=scenario.dof,
        covariance_type=scenario.covariance_type,
        length_scale=0.1,
    )

    @pytask.task(id=scenario.to_str())
    def task_run_monte_carlo_study(
        _scripts: list[Path] = [
            SRC / "monte_carlo.py",
            SRC / "bands.py",
            SRC / "fair_algorithm.py",
            SRC / "covariance.py",
        ],
        result_path: Annotated[Path, Product] = result_path,
        simulation_options: SimulationOptions = simulation_options,
    ) -> None:
        results = monte_carlo_simulation(
            n_simulations=500,
            simulation_options=simulation_options,
            band_options=band_options,
            n_cores=10,
            seed=None,
        )
        pd.to_pickle(results, result_path)
