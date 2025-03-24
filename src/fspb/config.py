from __future__ import annotations

from pathlib import Path
import itertools
from typing import Literal, TypedDict
from dataclasses import dataclass, fields

SRC = Path(__file__).parent.resolve()

ROOT = SRC.parent.parent.resolve()

BLD = ROOT / "bld"


class ScenarioDict(TypedDict):
    n_samples: int
    dof: int
    covariance_type: Literal["stationary", "non_stationary"]


@dataclass
class Scenario:
    n_samples: int
    dof: int
    covariance_type: Literal["stationary", "non_stationary"]

    def _fields(self) -> list[str]:
        return [f.name for f in fields(self)]

    def to_str(self) -> str:
        return f"n={self.n_samples}-d={self.dof}-c={self.covariance_type}"

    def to_dict(self) -> ScenarioDict:
        return ScenarioDict(
            n_samples=self.n_samples,
            dof=self.dof,
            covariance_type=self.covariance_type,
        )

    @classmethod
    def from_str(cls, string: str) -> Scenario:
        n_samples, dof, covariance_type = [s.split("=")[1] for s in string.split("-")]
        if covariance_type not in ("stationary", "non_stationary"):
            raise ValueError(f"Unknown covariance type: {covariance_type}")
        return Scenario(int(n_samples), int(dof), covariance_type)  # type: ignore[arg-type]

    @classmethod
    def from_lists(
        cls,
        n_samples: list[int],
        dof: list[int],
        covariance_type: list[Literal["stationary", "non_stationary"]],
    ) -> list[Scenario]:
        return [
            cls(n, d, c)
            for n, d, c in itertools.product(n_samples, dof, covariance_type)
        ]


ALL_SCENARIOS = Scenario.from_lists(
    n_samples=[30, 100],
    dof=[5, 15],
    covariance_type=["stationary", "non_stationary"],
)
