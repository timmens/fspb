from pathlib import Path

SRC = Path(__file__).parent.resolve()

ROOT = SRC.parent.parent.resolve()

BLD = ROOT / "bld"
import itertools
from typing import Any


SCENARIO_VALUES = {
    "n_samples": [30, 100],
    "dof": [5, 15],
    "covariance_type": ["stationary", "non_stationary"],
}

ALL_SCENARIOS = [
    {
        "n_samples": n_samples,
        "dof": dof,
        "covariance_type": covariance_type,
    }
    for n_samples, dof, covariance_type in itertools.product(*SCENARIO_VALUES.values())
]


def id_from_scenario(scenario: dict[str, Any]) -> str:
    return "--".join([f"{k}-{v}" for k, v in scenario.items()])


def scenario_from_id(id: str) -> dict[str, str]:
    return dict([option.split("-") for option in id.split("--")])
