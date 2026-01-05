from fspb.bands.critical_values import (
    calculate_piecewise_integrals,
    solve_for_critical_values,
    GaussianAlgorithm,
    StudentTAlgorithm,
    DistributionType,
)
from fspb.types import BandType, EstimationMethod
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
    sd_diag = np.ones_like(time_grid)  # Not used for Fair estimation method
    result = solve_for_critical_values(
        significance_level=0.05,
        interval_cutoffs=interval_cutoffs,
        time_grid=time_grid,
        covariance_diag=sd_diag,
        roughness=roughness,
        n_samples=10,
        band_type=BandType.CONFIDENCE,
        distribution_type=DistributionType.GAUSSIAN,
        degrees_of_freedom=10,
        estimation_method=EstimationMethod.FAIR,
    )
    assert result.shape == (100,)  # Now returns array same length as time_grid
    # Check that critical values are constant within each interval
    first_interval_mask = time_grid < 0.5
    second_interval_mask = time_grid >= 0.5
    assert np.allclose(result[first_interval_mask], result[first_interval_mask][0])
    assert np.allclose(result[second_interval_mask], result[second_interval_mask][0])


def test_gaussian_algorithm_cdf():
    algo = GaussianAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
        covariance_diag_integrals=np.array([1], dtype=float),
        sample_size_factor=1.0,
    )
    # norm.cdf(0) should be 0.5
    assert abs(algo._cdf(0.0) - 0.5) < 1e-7


def test_gaussian_algorithm_cdf_gradient():
    algo = GaussianAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
        covariance_diag_integrals=np.array([1], dtype=float),
        sample_size_factor=1.0,
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
        covariance_diag_integrals=np.array([1], dtype=float),
        sample_size_factor=1.0,
    )
    # scaling = norm.pdf(x) / np.sqrt(2 * np.pi). At x=0 => ~0.1592
    expected = norm.pdf(0.0) / np.sqrt(2 * np.pi)
    assert abs(algo._scaling(0.0) - expected) < 1e-7


def test_studentt_algorithm_cdf():
    algo = StudentTAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
        covariance_diag_integrals=np.array([1], dtype=float),
        sample_size_factor=1.0,
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
        covariance_diag_integrals=np.array([1], dtype=float),
        sample_size_factor=1.0,
        degrees_of_freedom=10,
    )
    expected = t.pdf(0.0, df=10)
    assert abs(algo._cdf_gradient(0.0) - expected) < 1e-7


def test_gaussian_algorithm_scaling_values():
    """Test the scaling function at multiple points to verify mathematical correctness."""
    algo = GaussianAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
        covariance_diag_integrals=np.array([1], dtype=float),
        sample_size_factor=1.0,
    )

    # Test multiple points
    test_values = [0.0, 1.0, -1.0, 2.0, -2.0]
    for x in test_values:
        expected = norm.pdf(x) / np.sqrt(2 * np.pi)
        result = algo._scaling(x)
        assert abs(result - expected) < 1e-10, (
            f"Failed at x={x}: got {result}, expected {expected}"
        )


def test_gaussian_algorithm_scaling_gradient():
    """Test the scaling gradient function matches mathematical derivative."""
    algo = GaussianAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
        covariance_diag_integrals=np.array([1], dtype=float),
        sample_size_factor=1.0,
    )

    # Test multiple points
    test_values = [0.0, 1.0, -1.0, 0.5, -0.5]
    for x in test_values:
        expected = -x * norm.pdf(x) / np.sqrt(2 * np.pi)
        result = algo._scaling_gradient(x)
        assert abs(result - expected) < 1e-10, (
            f"Failed at x={x}: got {result}, expected {expected}"
        )


def test_gaussian_algorithm_scaling_symmetry():
    """Test that scaling function has correct symmetry properties."""
    algo = GaussianAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
        covariance_diag_integrals=np.array([1], dtype=float),
        sample_size_factor=1.0,
    )

    # Scaling function should be even: S(x) = S(-x)
    test_values = [1.0, 2.0, 0.5, 1.5]
    for x in test_values:
        assert abs(algo._scaling(x) - algo._scaling(-x)) < 1e-15

    # Scaling gradient should be odd: S'(x) = -S'(-x)
    for x in test_values:
        assert abs(algo._scaling_gradient(x) + algo._scaling_gradient(-x)) < 1e-15


def test_gaussian_algorithm_numerical_consistency():
    """Test numerical consistency of the equation and gradient."""
    algo = GaussianAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 0.5, 1], dtype=float),
        roughness_integrals=np.array([0.5, 0.5], dtype=float),
        interval_lengths=np.array([0.5, 0.5], dtype=float),
        covariance_diag_integrals=np.array([0.5, 0.5], dtype=float),
        sample_size_factor=1.0,
    )

    # Test that the equation evaluates correctly
    x = 1.96  # Critical value for 95% confidence
    interval_idx = 0

    # The equation is: cdf(-x) + scaling(x) * roughness_integral - (significance_level/2) * interval_length
    expected_eq = (
        norm.cdf(-x) + (norm.pdf(x) / np.sqrt(2 * np.pi)) * 0.5 - (0.05 / 2) * 0.5
    )

    result_eq = algo._fair_equation(x, interval_idx)
    assert abs(result_eq - expected_eq) < 1e-10

    # Test gradient: -cdf_gradient(-x) + scaling_gradient(x) * roughness_integral
    expected_grad = -norm.pdf(-x) + (-x * norm.pdf(x) / np.sqrt(2 * np.pi)) * 0.5

    result_grad = algo._fair_equation_gradient(x, interval_idx)
    assert abs(result_grad - expected_grad) < 1e-10


def test_gaussian_scaling_normalization():
    """Test that the scaling function has the correct normalization factor."""
    algo = GaussianAlgorithm(
        significance_level=0.05,
        interval_cutoffs=np.array([0, 1], dtype=float),
        roughness_integrals=np.array([1], dtype=float),
        interval_lengths=np.array([1], dtype=float),
        covariance_diag_integrals=np.array([1], dtype=float),
        sample_size_factor=1.0,
    )

    # At x=0, norm.pdf(0) = 1/sqrt(2*pi), so scaling should be 1/(2*pi)
    expected_at_zero = 1 / (2 * np.pi)
    result_at_zero = algo._scaling(0.0)
    assert abs(result_at_zero - expected_at_zero) < 1e-15
