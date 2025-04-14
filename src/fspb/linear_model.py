import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field


@dataclass
class ConcurrentLinearModel:
    """Functional concurrent linear model.

    Attributes:
        y: Has shape (n_samples, n_time_points)
        x: Has shape (n_samples, 2, n_time_points)

    """

    intercept: NDArray[np.floating] = field(default_factory=lambda: np.array([np.nan]))
    slope: NDArray[np.floating] = field(default_factory=lambda: np.array([np.nan]))
    x_shape: tuple[int, ...] = tuple()

    def fit(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        """Fit the model coefficients."""
        if len(y) != len(x):
            raise ValueError("y and x must have the same number of samples")
        if y.shape[1] != x.shape[2]:
            raise ValueError("y and x must have the same number of time points")

        intercept, slope = _fit(y, x)
        self.intercept = intercept
        self.slope = slope
        self.x_shape = x.shape

    def predict(self, x_new: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict the outcome for new data.

        Args:
            x_new: Has shape (n_samples, 2, n_time_points) or (2, n_time_points).

        Returns:
            Predicted outcome for new data. Has shape (n_samples, n_time_points) if
            x_new has shape (n_samples, 2, n_time_points). Otherwise, has shape
            (n_time_points,).

        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Fit model first.")

        if self.intercept is None or self.slope is None or self.x_shape is None:
            raise ValueError("No model information available yet. Fit model.")

        if x_new.ndim == 2:
            x_new = x_new[np.newaxis]

        if x_new.ndim != 3 or x_new.shape[1:] != self.x_shape[1:]:
            raise ValueError(f"x_new has invalid shape: {x_new.shape}")

        pred = self.intercept * x_new[:, 0, :] + self.slope * x_new[:, 1, :]
        return np.squeeze(pred)

    @property
    def is_fitted(self) -> bool:
        """Check if the model is fitted."""
        return not (
            np.isnan(self.intercept).any()
            or np.isnan(self.slope).any()
            or not self.x_shape
        )


def _fit(
    y: NDArray[np.floating],
    x: NDArray[np.floating],
) -> NDArray[np.floating]:
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
