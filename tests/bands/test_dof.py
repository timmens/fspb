from fspb.bands.dof import estimate_dof
import pytest
import numpy as np


@pytest.fixture
def data():
    residuals = np.array(
        [
            [-1, -2],
            [0, 0],
            [1, 2],
        ]
    )
    return residuals


def test_estimate_dof(data):
    """Test DOF estimation using Singh method."""
    residuals = data
    # residuals = [[-1, -2], [0, 0], [1, 2]]
    # mean_squared_residuals = [2/3, 8/3]
    # mean_residuals_to_the_fourth = [2/3, 32/3]
    # kurtosis = mean_residuals_to_the_fourth / mean_squared_residuals**2 = [1.5, 1.5]
    # Both kurtosis values are <= 3.1, so we return 30.0

    expected = 30.0
    got = estimate_dof(residuals)
    assert got == expected


def test_estimate_dof_edge_cases():
    """Test DOF estimation with edge cases."""
    # Case 1: All zeros - kurtosis becomes NaN, no values > 3.1, returns 30.0
    residuals_zeros = np.zeros((5, 3))
    dof = estimate_dof(residuals_zeros)
    # When residuals are zero, kurtosis becomes NaN but mask.any() is False, so returns 30.0
    assert dof == 30.0

    # Case 2: Constant non-zero values - kurtosis = 1.0 for all points
    residuals_constant = np.ones((4, 2))
    dof = estimate_dof(residuals_constant)
    # mean_squared = [1, 1], mean_fourth = [1, 1], kurtosis = [1, 1]
    # Since kurtosis values are < 3.1, returns 30.0
    assert dof == 30.0

    # Case 3: Extreme case where kurtosis exceeds 3.1
    # Most values near zero with extreme outliers
    residuals_extreme = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [10, 0], [0, 10]])
    dof = estimate_dof(residuals_extreme)
    # This has kurtosis = 6.0 > 3.1, so dof_candidates = 6/(6-3) + 4 = 6.0
    assert dof == 6.0
