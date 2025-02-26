import numpy as np
from scipy.ndimage import gaussian_filter
from numpy.typing import NDArray
from typing import Any


def calculate_roughness_on_grid(
    cov: NDArray[np.float64],
    time_grid: NDArray[np.float64],
    *,
    smooth: bool = True,
    smooth_kwargs: dict[str, Any] | None = None,
) -> NDArray[np.float64]:
    """Calculate the roughness function on a grid.

    Args:
        time_grid: The time grid.
        cov: The estimated covariance matrix.
        smooth: Whether to smooth the covariance matrix using a Gaussian filter.
        smooth_kwargs: Keyword arguments for the Gaussian filter.

    Returns:
        The roughness function on the grid.

    """
    if smooth:
        if smooth_kwargs is None:
            smooth_kwargs = {"sigma": 0.1}
        elif not isinstance(smooth_kwargs, dict):
            raise ValueError("smooth_kwargs must be a dictionary.")

        cov = gaussian_filter(cov, **smooth_kwargs).astype(np.float64)

    corr = _cov_to_corr(cov)

    dx = time_grid[1] - time_grid[0]

    corr_dx = np.gradient(corr, dx, axis=0)
    corr_dxdy = np.gradient(corr_dx, dx, axis=1)

    return np.sqrt(np.diag(corr_dxdy))


def _cov_to_corr(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    standard_errors = np.sqrt(np.diag(cov))
    corr = cov.copy()
    corr /= standard_errors.reshape(1, -1)
    corr /= standard_errors.reshape(-1, 1)
    return corr
