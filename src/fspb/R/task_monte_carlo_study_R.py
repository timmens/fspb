from fspb.config import BLD, ALL_SCENARIOS
from pathlib import Path
from typing import Annotated
from pytask import Product
import pytask
from fspb.config import SKIP_R, SRC


for scenario in ALL_SCENARIOS:
    data_path = BLD / "monte_carlo" / "R" / "raw" / f"{scenario.to_str()}.json"
    product_path = BLD / "monte_carlo" / "R" / "processed" / f"{scenario.to_str()}.json"

    @pytask.mark.skipif(SKIP_R, reason="Not running R analysis.")
    @pytask.task(id=scenario.to_str())
    @pytask.mark.r(script=SRC / "R" / "conformal_prediction.R")
    def task_monte_carlo_study_R(
        _scripts: list[Path] = [
            SRC / "config.py",
            SRC / "R" / "functions.R",
        ],
        functions_path: Path = SRC / "R" / "functions.R",
        data_path: Path = data_path,
        product_path: Annotated[Path, Product] = product_path,
    ) -> None:
        pass
