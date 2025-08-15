import shutil
from fspb.config import (
    MOVE_RESULTS_TO_PAPER_DIR,
    PAPER_BLD,
    BLD_FIGURES,
    BLD_TABLES,
    BLD_APPLICATION,
)
import pytask
from pathlib import Path
from pytask import Product
from typing import Annotated


@pytask.mark.skipif(
    not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
)
def task_move_outcome_figure(
    outcome_figure_path: Path = BLD_FIGURES / "outcomes.pdf",
    to_path: Annotated[Path, Product] = PAPER_BLD / "outcomes.pdf",
) -> None:
    shutil.copy(outcome_figure_path, to_path)


@pytask.mark.skipif(
    not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
)
def task_move_application_outcome_figure(
    outcome_figure_path: Path = BLD_APPLICATION / "application-outcomes.pdf",
    to_path: Annotated[Path, Product] = PAPER_BLD / "application-outcomes.pdf",
) -> None:
    shutil.copy(outcome_figure_path, to_path)


@pytask.mark.skipif(
    not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
)
def task_move_band_figure(
    band_figure_path: Path = BLD_FIGURES / "band_seed_0.pdf",
    to_path: Annotated[Path, Product] = PAPER_BLD / "band.pdf",
) -> None:
    shutil.copy(band_figure_path, to_path)


for covariance_type in ("stationary", "non_stationary"):

    @pytask.mark.skipif(
        not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
    )
    @pytask.task(id=covariance_type)
    def task_move_prediction_simulation_results_table_min_width_vs_ci(
        simulation_results_table_path: Path = BLD_TABLES
        / f"prediction_{covariance_type}.tex",
        to_path: Annotated[Path, Product] = PAPER_BLD
        / f"prediction_{covariance_type}.tex",
    ) -> None:
        shutil.copy(simulation_results_table_path, to_path)

    @pytask.mark.skipif(
        not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
    )
    @pytask.task(id=covariance_type)
    def task_move_prediction_simulation_results_table_min_width_vs_fair(
        simulation_results_table_path: Path = BLD_TABLES
        / f"prediction_{covariance_type}_min_width_vs_fair.tex",
        to_path: Annotated[Path, Product] = PAPER_BLD
        / f"prediction_{covariance_type}_min_width_vs_fair.tex",
    ) -> None:
        shutil.copy(simulation_results_table_path, to_path)

    @pytask.mark.skipif(
        not MOVE_RESULTS_TO_PAPER_DIR, reason="Not moving results to paper directory."
    )
    @pytask.task(id=covariance_type)
    def task_move_confidence_simulation_results_table(
        simulation_results_table_path: Path = BLD_TABLES
        / f"confidence_{covariance_type}.tex",
        to_path: Annotated[Path, Product] = PAPER_BLD
        / f"confidence_{covariance_type}.tex",
    ) -> None:
        shutil.copy(simulation_results_table_path, to_path)
