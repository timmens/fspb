import numpy as np
from numpy.typing import NDArray


def estimate_dof(residuals: NDArray[np.floating]) -> float:
    """Estimate the degrees of freedom of the residuals.

    Args:
        residuals: The residuals of the model. Has shape (n_samples, n_time_points).

    Returns:
        The estimated degrees of freedom.

    """
    return _dof_estimate_singh(residuals)


def _dof_estimate_singh(residuals: NDArray[np.floating]) -> float:
    """Estimate the degrees of freedom of the residuals using the Singh method.

    This estimation method is based on the Singh (1988) method, but extended to the
    functional data setting.

    Args:
        residuals: The residuals of the model. Has shape (n_samples, n_time_points).

    Returns:
        The estimated degrees of freedom.

    """
    mean_residuals_to_the_fourth = np.mean(residuals**4, axis=0)
    mean_squared_residuals = np.mean(residuals**2, axis=0)
    kurtosis = mean_residuals_to_the_fourth / mean_squared_residuals**2
    mask = kurtosis > 3.1
    if not mask.any():
        return 30.0
    dof_candidates = 6 / (kurtosis[mask] - 3) + 4
    return np.min(dof_candidates)
