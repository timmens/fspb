import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class ConcurrentLinearModel:
    """Functional concurrent linear model.

    Attributes:
        y: Has shape (n_samples, n_time_points)
        x: Has shape (n_samples, 2, n_time_points)

    """

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Fit the model coefficients."""
        if len(y) != len(x):
            raise ValueError("y and x must have the same number of samples")
        if y.shape[1] != x.shape[2]:
            raise ValueError("y and x must have the same number of time points")

        self.coefs = _fit(y, x)
        self.x_shape = x.shape

    def predict(self, x_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict the outcome for new data.

        Args:
            x_new: Has shape (n_samples, 2, n_time_points) or (2, n_time_points).

        Returns:
            Predicted outcome for new data. Has shape (n_samples, n_time_points) if
            x_new has shape (n_samples, 2, n_time_points). Otherwise, has shape
            (n_time_points,).

        """
        if x_new.ndim == 2:
            x_new = x_new[np.newaxis]

        if x_new.ndim != 3 or x_new.shape[1:] != self.x_shape[1:]:
            raise ValueError(f"x_new has invalid shape: {x_new.shape}")

        return np.squeeze((x_new * self.coefs).sum(axis=1))


def _fit(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit functional concurrent linear model.

    Args:
        y: Has shape (n_samples, n_time_points)
        x: Has shape (n_samples, 2, n_time_points)

    Returns:
        Coefficients of shape (n_features, n_time_points)

    """
    features = x.transpose(2, 0, 1)

    coefs_list = []

    for _y, _x in zip(y.T, features):
        _coef = np.linalg.lstsq(_x, _y, rcond=None)[0]
        coefs_list.append(_coef)

    return np.array(coefs_list).T
