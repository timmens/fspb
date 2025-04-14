from fspb.fair_algorithm import _calculate_piecewise_integrals
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_calculate_piecewise_integrals_linear():
    cutoffs = np.array([0, 0.5, 1])
    time_grid = np.linspace(0, 1, 1_000)
    values = 2 * time_grid
    expected = np.array([0.25, 0.75])
    got = _calculate_piecewise_integrals(cutoffs, values, time_grid)
    assert_array_almost_equal(got, expected, decimal=3)


def test_calculate_piecewise_integrals_quadratic():
    time_grid = np.linspace(0, 1, 1_000)
    values = time_grid**2  # Integral of t^2 is t^3 / 3
    cutoffs = np.array([0, 0.2, 0.7, 1], dtype=np.float64)
    expected = np.array([(0.2**3) / 3, (0.7**3) / 3 - (0.2**3) / 3, (1 - 0.7**3) / 3])
    got = _calculate_piecewise_integrals(cutoffs, values, time_grid=time_grid)
    assert_array_almost_equal(got, expected, decimal=3)
