from fspb.config import BLD_TABLES, BLD_SIMULATION_PROCESSED, SRC
from fspb.simulation.results_tables import (
    produce_confidence_publication_table,
    produce_prediction_publication_table,
    fill_template,
)
from pathlib import Path
from typing import Annotated
import pandas as pd
from pytask import Product
import pytask


for covariance_type in ("stationary", "non_stationary"):

    @pytask.task(id=covariance_type)
    def task_produce_prediction_table_fair_vs_ci(
        _script: Path = SRC / "simulation" / "results_tables.py",
        consolidated_path: Path = BLD_SIMULATION_PROCESSED / "consolidated.pkl",
        product_path: Annotated[Path, Product] = BLD_TABLES
        / f"prediction_{covariance_type}.tex",
        covariance_type: str = covariance_type,
    ) -> None:
        consolidated: pd.DataFrame = pd.read_pickle(consolidated_path)
        prediction_results = (
            consolidated.xs("prediction", level="band_type")
            .xs(covariance_type, level="covariance_type")
            .query("method in ('fair', 'ci')")
        )
        table = produce_prediction_publication_table(prediction_results)
        latex_str = fill_template(table, type="prediction")
        product_path.write_text(latex_str)

    @pytask.task(id=covariance_type)
    def task_produce_confidence_table(
        _script: Path = SRC / "simulation" / "results_tables.py",
        consolidated_path: Path = BLD_SIMULATION_PROCESSED / "consolidated.pkl",
        product_path: Annotated[Path, Product] = BLD_TABLES
        / f"confidence_{covariance_type}.tex",
        covariance_type: str = covariance_type,
    ) -> None:
        consolidated: pd.DataFrame = pd.read_pickle(consolidated_path)
        confidence_results = (
            consolidated.xs("confidence", level="band_type")
            .xs(covariance_type, level="covariance_type")
            .query("method == 'fair'")
        )
        table = produce_confidence_publication_table(confidence_results)
        latex_str = fill_template(table, type="confidence")
        product_path.write_text(latex_str)
