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


def parse_enum_type(type_field: T | str, enum_type: type[T]) -> T:
    if not isinstance(type, enum_type):
        try:
            type_field = enum_type[type_field.upper()]
        except KeyError:
            raise ValueError(f"Invalid type: {type_field}")
    return type_field
