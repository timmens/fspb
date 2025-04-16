from fspb.fair_algorithm import (
    calculate_piecewise_integrals,
    fair_critical_value_selection,
    GaussianAlgorithm,
    StudentTAlgorithm,
    DistributionType,
)
from scipy.stats import norm, t
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_calculate_piecewise_integrals_linear():
    cutoffs = np.array([0, 0.5, 1])
    time_grid = np.linspace(0, 1, 1_000)
    values = 2 * time_grid
    expected = np.array([0.25, 0.75])
    got = calculate_piecewise_integrals(cutoffs, values, time_grid)
    assert_array_almost_equal(got, expected, decimal=3)


def test_calculate_piecewise_integrals_quadratic():
    time_grid = np.linspace(0, 1, 1_000)
    values = time_grid**2  # Integral of t^2 is t^3 / 3
    cutoffs = np.array([0, 0.2, 0.7, 1], dtype=np.float64)
    expected = np.array([(0.2**3) / 3, (0.7**3) / 3 - (0.2**3) / 3, (1 - 0.7**3) / 3])
    got = calculate_piecewise_integrals(cutoffs, values, time_grid=time_grid)
    assert_array_almost_equal(got, expected, decimal=3)


def test_fair_critical_value_selection_gaussian():
    interval_cutoffs = np.array([0, 0.5, 1], dtype=float)
    time_grid = np.linspace(0, 1, 100)
    roughness = np.ones_like(time_grid)
    result = fair_critical_value_selection(
        significance_level=0.05,
        interval_cutoffs=interval_cutoffs,
        time_grid=time_grid,
        roughness=roughness,
        distribution_type=DistributionType.GAUSSIAN,
    )
    assert result.shape == (2,)


def test_gaussian_algorithm_cdf():
    algo = GaussianAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
    )
    # norm.cdf(0) should be 0.5
    assert abs(algo._cdf(0.0) - 0.5) < 1e-7


def test_gaussian_algorithm_cdf_gradient():
    algo = GaussianAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
    )
    # norm.pdf(0) ~ 0.3989
    expected = norm.pdf(0.0)
    assert abs(algo._cdf_gradient(0.0) - expected) < 1e-7


def test_gaussian_algorithm_scaling():
    algo = GaussianAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
    )
    # scaling = norm.pdf(x). At x=0 => ~0.3989
    expected = norm.pdf(0.0)
    assert abs(algo._scaling(0.0) - expected) < 1e-7


def test_studentt_algorithm_cdf():
    algo = StudentTAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
        degrees_of_freedom=10,
    )
    # t.cdf(0, df=10) == 0.5
    assert abs(algo._cdf(0.0) - 0.5) < 1e-7


def test_studentt_algorithm_pdf():
    algo = StudentTAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
        degrees_of_freedom=10,
    )
    expected = t.pdf(0.0, df=10)
    assert abs(algo._cdf_gradient(0.0) - expected) < 1e-7
