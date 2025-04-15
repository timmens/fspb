from fspb.config import BandType
from numpy.typing import NDArray
import numpy as np
from enum import Enum


class ErrorAssumption(Enum):
    HOMOSKEDASTIC = "homoskedastic"
    HETEROSKEDASTIC = "heteroskedastic"


def calculate_covariance(
    residuals: NDArray[np.floating],
    x: NDArray[np.floating],
    x_new: NDArray[np.floating],
    band_type: BandType,
) -> NDArray[np.floating]:
    """Estimate the asymptotic covariance of the residuals.

    Args:
        residuals: The residuals of the model. Has shape (n_samples, n_time_points).
        x: The design matrix. Has shape (n_samples, n_features, n_time_points).
        x_new: The new design matrix for which to calculate the covariance. Has shape
            (n_features, n_time_points).

    Returns:
        The covariance of the residuals on a grid. Has shape
        (n_time_points, n_time_points).

    """
    if band_type == BandType.CONFIDENCE:
        return _calculate_covariance_confidence_band(residuals, x, x_new)
    elif band_type == BandType.PREDICTION:
        return _calculate_covariance_prediction_band(residuals, x, x_new)
    else:
        raise ValueError(f"Unknown band type: {band_type}")


def _calculate_covariance_confidence_band(
    residuals: NDArray[np.floating],
    x: NDArray[np.floating],
    x_new: NDArray[np.floating],
    *,
    error_assumption: ErrorAssumption = ErrorAssumption.HOMOSKEDASTIC,
) -> NDArray[np.floating]:
    if error_assumption == ErrorAssumption.HETEROSKEDASTIC:
        raise NotImplementedError(
            "Heteroskedastic error assumption is not implemented yet."
        )

    sigma_x_inv = _calculate_sigma_x_inv(x)
    sigma_error = _calculate_error_covariance(residuals)
    xt_sigma_x_inv_xt = _multiply_x_new_sigma_x_inv_x_newT(x_new, sigma_x_inv)
    return sigma_error * xt_sigma_x_inv_xt


def _calculate_covariance_prediction_band(
    residuals: NDArray[np.floating],
    x: NDArray[np.floating],
    x_new: NDArray[np.floating],
    *,
    error_assumption: ErrorAssumption = ErrorAssumption.HOMOSKEDASTIC,
) -> NDArray[np.floating]:
    if error_assumption == ErrorAssumption.HETEROSKEDASTIC:
        raise NotImplementedError(
            "Heteroskedastic error assumption is not implemented yet."
        )

    sigma_CB = _calculate_covariance_confidence_band(residuals, x=x, x_new=x_new)
    sigma_Z = _estimate_scaling_covariance_and_dof(residuals)
    sigma_PB = sigma_CB / len(residuals) + sigma_Z
    return sigma_PB


def _calculate_sigma_x_inv(x: NDArray[np.floating]) -> NDArray[np.floating]:
    sigma_x = np.tensordot(x, x, axes=([0], [0])).transpose(1, 3, 2, 0) / len(x)
    return np.linalg.inv(sigma_x).astype(np.float64)


def _calculate_error_covariance(
    residuals: NDArray[np.floating],
) -> NDArray[np.floating]:
    return residuals.T @ residuals / (len(residuals) - 1)


def _multiply_x_new_sigma_x_inv_x_newT(
    x_new: NDArray[np.floating], sigma_x_inv: NDArray[np.floating]
) -> NDArray[np.floating]:
    return np.einsum("is,stij,jt->st", x_new, sigma_x_inv, x_new)


def _estimate_scaling_covariance_and_dof(
    residuals: NDArray[np.floating],
) -> NDArray[np.floating]:
    sigma_error = _calculate_error_covariance(residuals)
    dof = dof_estimate(residuals)
    return sigma_error * (dof - 2) / dof


def dof_estimate(residuals: NDArray[np.floating]) -> float:
    mean_squared_residuals = np.mean(residuals**2, axis=0)
    mean_residuals_to_the_fourth = np.mean(residuals**4, axis=0)
    a_hat = mean_residuals_to_the_fourth / mean_squared_residuals**2
    dof_candidates = 2 * (2 * a_hat - 3) / (a_hat - 3)
    return np.maximum(dof_candidates.min(), 4.001)
