from fspb.monte_carlo import monte_carlo_simulation
from fspb.config import SRC, BLD
import pytask
from pathlib import Path
from typing import Annotated
from pytask import Product
import pandas as pd
from fspb.config import ALL_SCENARIOS


for scenario in ALL_SCENARIOS:
    result_path = BLD / "monte_carlo" / "raw" / f"{scenario.to_str()}.pkl"

    @pytask.task(id=scenario.to_str())
    def task_run_monte_carlo_study(
        _script: Path = SRC / "monte_carlo.py",
        result_path: Annotated[Path, Product] = result_path,
    ) -> None:
        results = monte_carlo_simulation(
            n_simulations=100,
            **scenario.to_dict(),
            significance_level=0.05,
            n_cores=10,
            seed=None,
        )
        pd.to_pickle(results, result_path)
