import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy.typing import NDArray


def calculate_roughness_on_grid(
    cov: NDArray[np.floating],
    time_grid: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Calculate the roughness function on a grid.

    Args:
        time_grid: The time grid.
        cov: The estimated covariance matrix.

    Returns:
        The roughness function on the grid.

    """
    corr = _cov_to_corr(cov)
    roughness_numerical = _calculate_roughness_on_grid_numerical_diff(corr, time_grid)
    roughness_splines = _calculate_roughness_on_grid_splines(corr, time_grid)
    return (roughness_numerical + roughness_splines) / 2


def _calculate_roughness_on_grid_numerical_diff(
    corr: NDArray[np.floating],
    time_grid: NDArray[np.floating],
) -> NDArray[np.floating]:
    dx = time_grid[1] - time_grid[0]

    corr_dx = np.gradient(corr, dx, axis=0)
    corr_dxdy = np.gradient(corr_dx, dx, axis=1)

    return np.sqrt(np.diag(corr_dxdy))


def _calculate_roughness_on_grid_splines(
    corr: NDArray[np.floating],
    time_grid: NDArray[np.floating],
) -> NDArray[np.floating]:
    spline = RectBivariateSpline(time_grid, time_grid, corr, kx=4, ky=4)
    partial_derivative = spline(time_grid, time_grid, dx=1, dy=1)
    roughness_squared = np.diag(partial_derivative)
    return np.sqrt(roughness_squared)


def _cov_to_corr(cov: NDArray[np.floating]) -> NDArray[np.floating]:
    standard_errors = np.sqrt(np.diag(cov))
    corr = cov.copy()
    corr /= standard_errors.reshape(1, -1)
    corr /= standard_errors.reshape(-1, 1)
    return corr
