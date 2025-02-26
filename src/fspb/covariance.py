from numpy.typing import NDArray
import numpy as np


def calculate_covariance_on_grid(
    residuals: NDArray[np.float64],
    x: NDArray[np.float64],
    x_new: NDArray[np.float64],
) -> NDArray[np.float64]:
    sigma_x = np.tensordot(x, x, axes=([0], [0])).transpose(1, 3, 2, 0) / x.shape[0]

    sigma_x_inv = np.linalg.inv(sigma_x)

    sigma_error = residuals.T @ residuals / len(residuals)

    xt_sigma_x_inv_xt = np.einsum("is,stij,jt->st", x_new, sigma_x_inv, x_new)

    return sigma_error * xt_sigma_x_inv_xt
