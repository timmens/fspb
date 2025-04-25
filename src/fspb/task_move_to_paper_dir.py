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


@pytask.mark.skipif(
    not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
)
def task_move_outcome_figure(
    outcome_figure_path: Path = BLD_FIGURES / "outcomes.pdf",
    to_path: Annotated[Path, Product] = PAPER_BLD / "outcomes.pdf",
) -> None:
    shutil.copy(outcome_figure_path, to_path)


for covariance_type in CovarianceType:

    @pytask.mark.skipif(
        not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
    )
    @pytask.task(id=str(covariance_type))
    def task_move_band_figure(
        band_figure_path: Path = BLD_VISUALIZATION / f"seed_0_{covariance_type}.png",
        to_path: Annotated[Path, Product] = PAPER_BLD / f"band_{covariance_type}.png",
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


for metric in ("coverage", "maximum_width", "band_score"):

    @pytask.mark.skipif(
        not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
    )
    @pytask.task(id=f"prediction_{metric}")
    def task_move_prediction_simulation_results_table(
        simulation_results_table_path: Path = BLD_TABLES / f"prediction_{metric}.tex",
        to_path: Annotated[Path, Product] = PAPER_BLD / f"prediction_{metric}.tex",
    ) -> None:
        shutil.copy(simulation_results_table_path, to_path)
