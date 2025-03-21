from fspb.monte_carlo import monte_carlo_simulation
from fspb.config import SRC, BLD
import pytask
from pathlib import Path
from typing import Annotated
from pytask import Product
import pandas as pd
from fspb.config import ALL_SCENARIOS, id_from_scenario


for scenario in ALL_SCENARIOS:
    _id = id_from_scenario(scenario)

    @pytask.task(id=_id)
    def task_run_monte_carlo_study(
        _script: Path = SRC / "monte_carlo.py",
        result_path: Annotated[Path, Product] = BLD
        / "monte_carlo"
        / "raw"
        / f"{_id}.pkl",
    ) -> None:
        results = monte_carlo_simulation(
            n_simulations=100,
            **scenario,
            significance_level=0.05,
            n_cores=10,
            seed=None,
        )
        pd.to_pickle(results, result_path)
