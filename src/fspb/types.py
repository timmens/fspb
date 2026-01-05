from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TypeVar

from fspb.bands.linear_model import ConcurrentLinearModel

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=StrEnum)


class BandType(StrEnum):
    CONFIDENCE = auto()
    PREDICTION = auto()


class EstimationMethod(StrEnum):
    FAIR = auto()
    MIN_WIDTH = auto()
    CI = auto()


class CovarianceType(StrEnum):
    STATIONARY = auto()
    NON_STATIONARY = auto()


class DistributionType(StrEnum):
    GAUSSIAN = auto()
    STUDENT_T = auto()


class CIPredictionMethod(StrEnum):
    MEAN = auto()
    LINEAR = auto()


def parse_enum_type(type_field: T | str, enum_type: type[T]) -> T:
    if isinstance(type_field, enum_type):
        return type_field
    try:
        return enum_type[type_field.upper()]
    except (KeyError, AttributeError):
        try:
            return enum_type(type_field)
        except ValueError:
            raise ValueError(f"Invalid type: {type_field}")


@dataclass
class SimulationData:
    """The data from a simulation."""

    y: NDArray[np.floating]
    x: NDArray[np.floating]
    time_grid: NDArray[np.floating]
    model: ConcurrentLinearModel
