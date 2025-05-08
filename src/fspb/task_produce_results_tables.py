from fspb.config import BLD_TABLES, BLD_SIMULATION_PROCESSED
from fspb.simulation.results_tables import produce_publication_table
from pathlib import Path
from typing import Annotated
import pandas as pd
from pytask import Product
import pytask


for covariance_type in ("stationary", "non_stationary"):

    @pytask.task(id=covariance_type)
    def task_produce_prediction_table(
        consolidated_path: Path = BLD_SIMULATION_PROCESSED / "consolidated.pkl",
        product_path: Annotated[Path, Product] = BLD_TABLES
        / f"prediction_{covariance_type}.tex",
        covariance_type: str = covariance_type,
    ) -> None:
        consolidated: pd.DataFrame = pd.read_pickle(consolidated_path)
        prediction_results = (
            consolidated.xs("prediction", level="band_type")
            .xs(covariance_type, level="covariance_type")
            .query("Method in ('Ours', 'CI (Linear)')")
        )
        table = produce_publication_table(prediction_results)  # type: ignore[arg-type]
        table.to_latex(
            product_path,
            escape=False,
            multicolumn=True,
            multicolumn_format="c",
            column_format="l" + "c" * (len(table.columns) + 1),
        )

    # @pytask.task(id=file_type)
    # def task_produce_confidence_table(
    #     consolidated_path: Path = BLD_SIMULATION_PROCESSED / "consolidated.pkl",
    #     product_path: Annotated[Path, Product] = BLD_TABLES / f"confidence.{file_type}",
    #     write_method: str = write_method,
    # ) -> None:
    #     consolidated: pd.DataFrame = pd.read_pickle(consolidated_path)
    #     confidence_results = consolidated.xs("confidence", level="band_type")
    #     table = produce_publication_table(confidence_results)  # type: ignore[arg-type]
    #     getattr(table, write_method)(product_path)
