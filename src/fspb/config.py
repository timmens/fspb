from __future__ import annotations

import socket
from pathlib import Path
import itertools
from typing import Literal, TypedDict
from dataclasses import dataclass, fields
from fspb.types import CovarianceType, parse_enum_type, BandType

SRC = Path(__file__).parent.resolve()

ROOT = SRC.parent.parent.resolve()

BLD = ROOT / "bld"
BLD_SIMULATION = BLD / "simulation"
BLD_SIMULATION_OUR = BLD_SIMULATION / "our"
BLD_SIMULATION_CONFORMAL_INFERENCE = BLD_SIMULATION / "conformal_inference"
BLD_SIMULATION_PROCESSED = BLD_SIMULATION / "processed"
BLD_TABLES = BLD / "tables"
BLD_FIGURES = BLD / "figures"

SKIP_R = False

N_SIMULATIONS = 1_000
N_JOBS = 10
LENGTH_SCALE = 0.4

# If running on Tim's laptop (thinky), move results to paper directory
if socket.gethostname() == "thinky":
    MOVE_RESULTS_TO_PAPER_DIR = True
    PAPER_BLD = Path(
        "/home/tim/sciebo/PRJ_Creutzinger_Liebl_Mensinger_Sharp/bld"
    ).resolve()
else:
    MOVE_RESULTS_TO_PAPER_DIR = False
    PAPER_BLD = Path("Not implemented.")


class ScenarioDict(TypedDict):
    n_samples: int
    dof: int
    covariance_type: Literal["STATIONARY", "NON_STATIONARY"]
    band_type: Literal["CONFIDENCE", "PREDICTION"]


@dataclass
class Scenario:
    n_samples: int
    dof: int
    covariance_type: CovarianceType
    band_type: BandType

    def _fields(self) -> list[str]:
        return [f.name for f in fields(self)]

    def to_str(self) -> str:
        return f"n={self.n_samples}-d={self.dof}-c={self.covariance_type.value}-b={self.band_type.value}"

    def to_dict(self) -> ScenarioDict:
        return ScenarioDict(
            n_samples=self.n_samples,
            dof=self.dof,
            covariance_type=self.covariance_type.value,  # type: ignore[typeddict-item]
            band_type=self.band_type.value,  # type: ignore[typeddict-item]
        )

    @classmethod
    def from_str(cls, string: str) -> Scenario:
        n_samples, dof, cov_type_str, band_type_str = [
            s.split("=")[1] for s in string.split("-")
        ]
        covariance_type = parse_enum_type(cov_type_str, CovarianceType)
        band_type = parse_enum_type(band_type_str, BandType)
        return Scenario(int(n_samples), int(dof), covariance_type, band_type)

    @classmethod
    def from_lists(
        cls,
        n_samples: list[int],
        dof: list[int],
        covariance_type: list[CovarianceType],
        band_type: list[BandType],
    ) -> list[Scenario]:
        return [
            cls(n, d, c, b)
            for n, d, c, b in itertools.product(
                n_samples,
                dof,
                covariance_type,
                band_type,
            )
        ]


PREDICTION_SCENARIOS = Scenario.from_lists(
    n_samples=[30, 100],
    dof=[5, 15],
    covariance_type=[CovarianceType.STATIONARY, CovarianceType.NON_STATIONARY],
    band_type=[BandType.PREDICTION],
)

CONFIDENCE_SCENARIOS = Scenario.from_lists(
    n_samples=[30, 100],
    dof=[5, 15],
    covariance_type=[CovarianceType.STATIONARY, CovarianceType.NON_STATIONARY],
    band_type=[BandType.CONFIDENCE],
)
