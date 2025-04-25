from enum import StrEnum, auto
from typing import TypeVar

T = TypeVar("T", bound=StrEnum)


class BandType(StrEnum):
    CONFIDENCE = auto()
    PREDICTION = auto()


class CovarianceType(StrEnum):
    STATIONARY = auto()
    NON_STATIONARY = auto()


class DistributionType(StrEnum):
    GAUSSIAN = auto()
    STUDENT_T = auto()


class ConformalInferencePredictionMethod(StrEnum):
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
