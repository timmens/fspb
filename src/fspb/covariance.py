from numpy.typing import NDArray
import numpy as np


def calculate_covariance_on_grid(
    residuals: NDArray[np.floating],
    x: NDArray[np.floating],
    x_new: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Calculate the covariance of the residuals on a grid.

    Args:
        residuals: The residuals of the model. Has shape (n_samples, n_time_points).
        x: The design matrix. Has shape (n_samples, n_features, n_time_points).
        x_new: The new design matrix for which to calculate the covariance. Has shape
            (n_features, n_time_points).

    """
    sigma_x_inv = _calculate_sigma_inv(x)
    sigma_error = _calculate_homoskedastic_error(residuals)
    xt_sigma_x_inv_xt = _multiply_x_new_sigma_x_inv_x_newT(x_new, sigma_x_inv)
    return sigma_error * xt_sigma_x_inv_xt


def _calculate_sigma_inv(x: NDArray[np.floating]) -> NDArray[np.floating]:
    sigma_x = np.tensordot(x, x, axes=([0], [0])).transpose(1, 3, 2, 0) / x.shape[0]
    return np.linalg.inv(sigma_x).astype(np.float64)


def _calculate_homoskedastic_error(
    residuals: NDArray[np.floating],
) -> NDArray[np.floating]:
    return residuals.T @ residuals / len(residuals)


def _multiply_x_new_sigma_x_inv_x_newT(
    x_new: NDArray[np.floating], sigma_x_inv: NDArray[np.floating]
) -> NDArray[np.floating]:
    return np.einsum("is,stij,jt->st", x_new, sigma_x_inv, x_new)
