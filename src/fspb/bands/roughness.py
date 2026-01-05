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
    sq_roughness_numerical = _calculate_squared_roughness_on_grid_numerical_diff(
        corr, time_grid
    )
    sq_roughness_splines = _calculate_squared_roughness_on_grid_splines(corr, time_grid)
    sq_roughness = (sq_roughness_numerical + sq_roughness_splines) / 2
    return np.sqrt(np.clip(sq_roughness, a_min=1e-12, a_max=None))


def _calculate_squared_roughness_on_grid_numerical_diff(
    corr: NDArray[np.floating],
    time_grid: NDArray[np.floating],
) -> NDArray[np.floating]:
    dx = time_grid[1] - time_grid[0]
    corr_dx = np.gradient(corr, dx, axis=0)
    corr_dxdy = np.gradient(corr_dx, dx, axis=1)
    return np.diag(corr_dxdy)


def _calculate_squared_roughness_on_grid_splines(
    corr: NDArray[np.floating],
    time_grid: NDArray[np.floating],
) -> NDArray[np.floating]:
    spline = RectBivariateSpline(time_grid, time_grid, corr, kx=4, ky=4)
    partial_derivative = spline(time_grid, time_grid, dx=1, dy=1)
    return np.diag(partial_derivative)


def _cov_to_corr(cov: NDArray[np.floating]) -> NDArray[np.floating]:
    standard_errors = np.sqrt(np.diag(cov))
    corr = cov.copy()
    corr /= standard_errors.reshape(1, -1)
    corr /= standard_errors.reshape(-1, 1)
    return corr
