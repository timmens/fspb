import shutil
from fspb.config import (
    MOVE_RESULTS_TO_PAPER_DIR,
    PAPER_BLD,
    BLD_VISUALIZATION,
    BLD_FIGURES,
    BLD_TABLES,
)
import pytask
from pathlib import Path
from pytask import Product
from typing import Annotated

from fspb.types import CovarianceType

for covariance_type in ("stationary", "non_stationary"):
    outcome_figure_path = BLD_FIGURES / f"outcomes_{covariance_type}.pdf"
    to_path = PAPER_BLD / f"outcomes_{covariance_type}.pdf"

    @pytask.task
    @pytask.mark.skipif(
        not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
    )
    def task_move_outcome_figure(
        outcome_figure_path: Path = outcome_figure_path,
        to_path: Annotated[Path, Product] = to_path,
    ) -> None:
        shutil.copy(outcome_figure_path, to_path)


for covariance_type in CovarianceType:

    @pytask.mark.skipif(
        not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
    )
    @pytask.task(id=str(covariance_type))
    def task_move_band_figure(
        band_figure_path: Path = BLD_VISUALIZATION / f"seed_0_{covariance_type}.pdf",
        to_path: Annotated[Path, Product] = PAPER_BLD / f"band_{covariance_type}.pdf",
    ) -> None:
        shutil.copy(band_figure_path, to_path)


@pytask.mark.skipif(
    not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
)
@pytask.task(id="confidence")
def task_move_confidence_simulation_results_table(
    simulation_results_table_path: Path = BLD_TABLES / "confidence.tex",
    to_path: Annotated[Path, Product] = PAPER_BLD / "confidence.tex",
) -> None:
    shutil.copy(simulation_results_table_path, to_path)


for covariance_type in ("stationary", "non_stationary"):

    @pytask.mark.skipif(
        not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
    )
    @pytask.task(id=f"prediction_{covariance_type}")
    def task_move_prediction_simulation_results_table(
        simulation_results_table_path: Path = BLD_TABLES
        / f"prediction_{covariance_type}.tex",
        to_path: Annotated[Path, Product] = PAPER_BLD
        / f"prediction_{covariance_type}.tex",
    ) -> None:
        shutil.copy(simulation_results_table_path, to_path)
