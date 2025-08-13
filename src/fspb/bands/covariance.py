from fspb.config import BandType
from numpy.typing import NDArray
import numpy as np
from enum import StrEnum, auto
from fspb.bands.dof import estimate_dof


class ErrorAssumption(StrEnum):
    HOMOSKEDASTIC = auto()
    HETEROSKEDASTIC = auto()


def calculate_covariance(
    residuals: NDArray[np.floating],
    x: NDArray[np.floating],
    x_new: NDArray[np.floating],
    band_type: BandType,
    *,
    error_assumption: ErrorAssumption = ErrorAssumption.HOMOSKEDASTIC,
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
    if error_assumption == ErrorAssumption.HETEROSKEDASTIC:
        raise NotImplementedError("Heteroskedastic error assumption not implemented.")
    if band_type == BandType.CONFIDENCE:
        return _calculate_covariance_confidence_band(
            residuals, x, x_new, error_assumption=error_assumption
        )
    elif band_type == BandType.PREDICTION:
        return _calculate_covariance_prediction_band(
            residuals, x, x_new, error_assumption=error_assumption
        )
    else:
        raise ValueError(f"Unknown band type: {band_type}")


def _calculate_covariance_confidence_band(
    residuals: NDArray[np.floating],
    x: NDArray[np.floating],
    x_new: NDArray[np.floating],
    *,
    error_assumption: ErrorAssumption,
) -> NDArray[np.floating]:
    if error_assumption == ErrorAssumption.HOMOSKEDASTIC:
        return _calculate_covariance_confidence_band_homoskedastic(
            residuals=residuals, x=x, x_new=x_new
        )
    elif error_assumption == ErrorAssumption.HETEROSKEDASTIC:
        return _calculate_covariance_confidence_band_heteroskedastic(
            residuals, x, x_new
        )
    else:
        raise ValueError(f"Unknown error assumption: {error_assumption}")


def _calculate_covariance_confidence_band_homoskedastic(
    residuals: NDArray[np.floating],
    x: NDArray[np.floating],
    x_new: NDArray[np.floating],
) -> NDArray[np.floating]:
    sigma_x_inv = _calculate_sigma_x_inv(x)
    sigma_error = _calculate_error_covariance(residuals, n_parameters=x.shape[1])
    xt_sigma_x_inv_xt = _multiply_a_B_a(x_new, sigma_x_inv)
    return sigma_error * xt_sigma_x_inv_xt


def _calculate_covariance_confidence_band_heteroskedastic(
    residuals: NDArray[np.floating],
    x: NDArray[np.floating],
    x_new: NDArray[np.floating],
) -> NDArray[np.floating]:
    sigma_x_inv = _calculate_sigma_x_inv(x)
    sigma_x_error = _calculate_sigma_x_error(residuals, x)
    xt_sigma_x_inv = _multiply_a_B(x_new, sigma_x_inv)
    xt_sigma_x_inv_xt_sigma_x_error = _multiply_c_B(xt_sigma_x_inv, sigma_x_error)
    xt_sigma_x_inv_xt_sigma_x_error_sigma_x_inv = _multiply_c_B(
        xt_sigma_x_inv_xt_sigma_x_error, sigma_x_inv
    )
    return _multiply_c_a(xt_sigma_x_inv_xt_sigma_x_error_sigma_x_inv, x_new)


def _calculate_covariance_prediction_band(
    residuals: NDArray[np.floating],
    x: NDArray[np.floating],
    x_new: NDArray[np.floating],
    *,
    error_assumption: ErrorAssumption,
) -> NDArray[np.floating]:
    sigma_CB = _calculate_covariance_confidence_band(
        residuals, x=x, x_new=x_new, error_assumption=error_assumption
    )
    sigma_Z = _estimate_scaling_covariance_and_dof(residuals, n_parameters=x.shape[1])
    sigma_PB = sigma_CB / len(residuals) + sigma_Z
    return sigma_PB


def _calculate_sigma_x_error(
    residuals: NDArray[np.floating], x: NDArray[np.floating]
) -> NDArray[np.floating]:
    x_error = residuals[:, np.newaxis, :] * x
    return _calculate_sigma_x(x_error)


def _calculate_sigma_x(x: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.tensordot(x, x, axes=([0], [0])).transpose(1, 3, 2, 0) / len(x)


def _calculate_sigma_x_inv(x: NDArray[np.floating]) -> NDArray[np.floating]:
    sigma_x = _calculate_sigma_x(x)
    return np.linalg.pinv(sigma_x).astype(np.float64)


def _calculate_error_covariance(
    residuals: NDArray[np.floating],
    n_parameters: int,
) -> NDArray[np.floating]:
    return residuals.T @ residuals / (len(residuals) - n_parameters)


def _multiply_a_B_a(
    a: NDArray[np.floating], B: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Multiply a vector by a matrix by a vector.

    Args:
        a: The vector to multiply. Has shape (n_features, n_time_points).
        B: The matrix to multiply. Has shape (n_time_points, n_time_points, n_features, n_features).

    Returns:
        The result of the multiplication. Has shape (n_time_points, n_time_points).

    """
    return np.einsum("ps,stpk,kt->st", a, B, a, optimize=True)


def _multiply_a_B(
    a: NDArray[np.floating], B: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Multiply a vector by a matrix.

    Args:
        a: The vector to multiply. Has shape (n_features, n_time_points).
        B: The matrix to multiply. Has shape (n_time_points, n_time_points, n_features, n_features).

    Returns:
        The result of the multiplication. Has shape (n_time_points, n_time_points, n_features).

    """
    return np.einsum("ps,stpp->stp", a, B, optimize=True)


def _multiply_c_B(
    c: NDArray[np.floating], B: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Multiply a vector by a matrix.

    Args:
        c: The vector to multiply. Has shape (n_time_points, n_time_points, n_features).
        B: The matrix to multiply. Has shape (n_time_points, n_time_points, n_features, n_features).

    Returns:
        The result of the multiplication. Has shape (n_time_points, n_time_points, n_features).

    """
    return np.einsum("stp,stpk->stk", c, B, optimize=True)


def _multiply_c_a(
    c: NDArray[np.floating], a: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Multiply a vector by a matrix.

    Args:
        c: The vector to multiply. Has shape (n_time_points, n_time_points, n_features).
        a: The matrix to multiply. Has shape (n_features, n_time_points).

    Returns:
        The result of the multiplication. Has shape (n_time_points, n_time_points).

    """
    return np.einsum("stp,ps->st", c, a)


def _estimate_scaling_covariance_and_dof(
    residuals: NDArray[np.floating], n_parameters: int
) -> NDArray[np.floating]:
    sigma_error = _calculate_error_covariance(residuals, n_parameters=n_parameters)
    dof = estimate_dof(residuals)
    return sigma_error * (dof - 2) / dof
