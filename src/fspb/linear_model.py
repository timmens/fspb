import numpy as np
from numpy.typing import NDArray


def fit_concurrent_model(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    *,
    fit_intercept: bool = True,
) -> NDArray[np.float64]:
    """Fit functional concurrent linear model.

    Args:
        y: Has shape (n_samples, n_time_points)
        x: Has shape (n_samples, n_time_points)
        fit_intercept: bool = True

    Returns:
        Coefficients of shape (n_features, n_time_points)

    """
    xt = x.T
    if fit_intercept:
        features = np.stack([np.ones_like(xt), xt], axis=2)
    else:
        features = np.stack([xt], axis=2)

    coefs_list = []

    for _y, _x in zip(y.T, features):
        _coef = np.linalg.lstsq(_x, _y, rcond=None)[0]
        coefs_list.append(_coef)

    return np.asarray(coefs_list).T
