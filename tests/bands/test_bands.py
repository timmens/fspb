from fspb.bands.band import Band
import pytest
import numpy as np


@pytest.fixture
def band():
    estimate = np.zeros(3)
    lower = np.array([-1, -2, -3])
    upper = np.array([1, 0, 2])
    return Band(estimate=estimate, lower=lower, upper=upper)


def test_contains(band):
    assert band.contains(np.array([0, -1, 1]))
    assert band.contains(np.array([-1, -2, 0]))
    assert not band.contains(np.array([-2, -1, 1]))
    assert not band.contains(np.array([0, -3, 1]))
    assert not band.contains(np.array([0, -1, 3]))


def test_maximum_width_statistic(band):
    assert band.maximum_width_statistic == 5


def test_interval_score(band):
    assert np.allclose(band.interval_score(np.zeros(3), signifance_level=0.2), 5)
    assert np.allclose(
        band.interval_score(np.array([-3, -2, -1]), signifance_level=0.2), 5.2
    )
