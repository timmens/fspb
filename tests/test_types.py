import pytest
from fspb.types import CovarianceType, parse_enum_type, DistributionType


def test_parse_covariance_type_valid():
    assert parse_enum_type("stationary", CovarianceType) == CovarianceType.STATIONARY
    assert parse_enum_type("STATIONARY", CovarianceType) == CovarianceType.STATIONARY
    assert (
        parse_enum_type("non_stationary", CovarianceType)
        == CovarianceType.NON_STATIONARY
    )
    assert (
        parse_enum_type(CovarianceType.STATIONARY, CovarianceType)
        == CovarianceType.STATIONARY
    )


def test_parse_covariance_type_invalid():
    with pytest.raises(ValueError):
        parse_enum_type("not_a_field", CovarianceType)


def test_parse_distribution_type_valid():
    assert parse_enum_type("gaussian", DistributionType) == DistributionType.GAUSSIAN
    assert parse_enum_type("GAUSSIAN", DistributionType) == DistributionType.GAUSSIAN
