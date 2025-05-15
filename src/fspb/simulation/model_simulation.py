import numpy as np
import scipy as sp
from dataclasses import dataclass
from numpy.typing import NDArray

from fspb.bands.linear_model import ConcurrentLinearModel
from fspb.types import CovarianceType


@dataclass
class SimulationData:
    """The data from a simulation."""

    y: NDArray[np.floating]
    x: NDArray[np.floating]
    time_grid: NDArray[np.floating]
    model: ConcurrentLinearModel


def simulate_from_model(
    n_samples: int,
    time_grid: NDArray[np.floating],
    dof: int,
    covariance_type: CovarianceType,
    length_scale: float,
    rng: np.random.Generator | None = None,
) -> SimulationData:
    """Simulate from the model.

    Args:
        n_samples: The number of samples to simulate.
        time_grid: The time grid to simulate the model for. Has shape (n_points,).
        dof: The degrees of freedom of the Student's t distribution.
        covariance_type: The type of covariance to use.
        length_scale: The length scale of the covariance.
        rng: The random state to use for the simulation.

    Returns:
        A SimulationData object.

    """
    if rng is None:
        rng = np.random.default_rng()

    x = _simulate_predictor(
        time_grid=time_grid,
        n_samples=n_samples,
        rng=rng,
    )

    error = _simulate_error(
        n_samples=n_samples,
        time_grid=time_grid,
        dof=dof,
        covariance_type=covariance_type,
        rng=rng,
        length_scale=length_scale,
    )

    model = ConcurrentLinearModel(
        intercept=np.zeros_like(time_grid),
        slope=_slope_function(time_grid),
        x_shape=(n_samples, 2, len(time_grid)),
    )

    y = model.predict(x) + error

    if y.shape[0] == 1:
        y = np.squeeze(y, axis=0)

    if x.shape[0] == 1:
        x = np.squeeze(x, axis=0)

    return SimulationData(
        y=y,
        x=x,
        time_grid=time_grid,
        model=model,
    )


# ======================================================================================
# Simulate Outcome
# ======================================================================================


def _slope_function(time_grid: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the slope function.

    Args:
        time_grid: The time grid to compute the slope parameter for.

    Returns:
        The slope function.

    """
    return time_grid * np.sin(2 * np.pi * time_grid)


# ======================================================================================
# Simulate Predictor
# ======================================================================================


def _simulate_predictor(
    time_grid: NDArray[np.floating], n_samples: int, rng: np.random.Generator
) -> NDArray[np.floating]:
    """Compute the predictor grid.

    Args:
        time_grid: The time grid to compute the predictor grid for. Has shape
            (n_points,).
        n_samples: The number of samples to compute the predictor grid for.
        rng: The random number generator to use for the simulation.

    Returns:
        The predictor grid. Has shape (n_samples, 2, n_points).

    """
    binary_covariate = _simulate_binary_covariate(n_samples=n_samples, rng=rng)
    scaling = rng.uniform(0.75, 1.25, size=n_samples)
    x = _predictor_function(
        time_grid=time_grid, binary_covariate=binary_covariate, scaling=scaling
    )
    ones = np.ones_like(x)
    return np.stack([ones, x], axis=1)


def _simulate_binary_covariate(
    n_samples: int, rng: np.random.Generator
) -> NDArray[np.int_]:
    return rng.binomial(1, 0.5, size=n_samples)


def _predictor_function(
    time_grid: NDArray[np.floating],
    binary_covariate: NDArray[np.int_],
    scaling: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the predictor variable, given the binary covariate B.

    Args:
        time_grid: The time grid to compute the predictor variable for. Has shape
          (n_points,).
        binary_covariate: The binary covariates. Has shape (n_samples,).

    Returns:
        The predictor variables. Has shape (n_samples, n_points).

    """
    curve = scaling.reshape(-1, 1) * np.cos(2 * np.pi * time_grid)
    upper = curve + 1.25
    lower = curve - 1.25
    binary_covariate_boolean_reshaped = binary_covariate.astype(bool).reshape(-1, 1)
    return np.where(binary_covariate_boolean_reshaped, upper, lower)


# ======================================================================================
# Simulate Error
# ======================================================================================


def _simulate_error(
    n_samples: int,
    time_grid: NDArray[np.floating],
    dof: int,
    covariance_type: CovarianceType,
    rng: np.random.Generator,
    length_scale: float,
) -> NDArray[np.floating]:
    """Simulate the error processes from a multivariate Student's t distribution.

    For each sample:
      - Sample u ~ chi-square(dof)
      - Sample z ~ N(0, cov_matrix)
      - Return error = z * sqrt(dof / u)

    Args:
        n_samples: The number of samples to simulate.
        time_grid: The time grid to simulate the errors for.
        x: The predictor grid. Has shape (n_samples, 2, n_points).
        dof: The degrees of freedom of the Student's t distribution.
        covariance_type: The type of covariance to use.
        rng: The random number generator to use for the simulation.
        length_scale: The length scale of the covariance.

    Returns:
        The simulated error grid. Has shape (n_samples, n_points).

    """
    cov_matrix = _matern_covariance(
        time_grid, covariance_type=covariance_type, length_scale=length_scale
    )

    u = rng.chisquare(dof, size=n_samples)
    scales = np.sqrt(dof / u)

    z = rng.multivariate_normal(
        mean=np.zeros_like(time_grid), cov=cov_matrix, size=n_samples
    )

    return scales[:, np.newaxis] * z


def _matern_covariance(
    time_grid: NDArray[np.floating],
    covariance_type: CovarianceType,
    length_scale: float,
    sigma: float = 1 / 4,
) -> NDArray[np.floating]:
    """Compute the Matern covariance matrix for the given time grid.

    Args:
        time_grid: The time grid to compute the covariance matrix for. Has shape
          (n_points,).
        covariance_type: The type of covariance to use.
        length_scale: The length scale of the covariance.
        sigma: The sigma of the covariance.

    Returns:
        The covariance matrix.

    """
    time_mesh_t, time_mesh_s = np.meshgrid(time_grid, time_grid)
    absolute_distance = np.abs(time_mesh_t - time_mesh_s)

    if covariance_type == CovarianceType.STATIONARY:
        nu = np.full(absolute_distance.shape, 3 / 2)
    elif covariance_type == CovarianceType.NON_STATIONARY:
        nu = 2 + np.sqrt(np.maximum(time_mesh_t, time_mesh_s)) * (1 / 4 - 2)
    else:
        raise ValueError("Invalid covariance type.")

    factor = np.sqrt(2 * nu) * absolute_distance / length_scale

    cov = np.empty_like(absolute_distance)
    mask = absolute_distance > 0

    cov[mask] = (
        sigma**2
        * (2 ** (1 - nu[mask]) / sp.special.gamma(nu[mask]))
        * (factor[mask]) ** (nu[mask])
        * sp.special.kv(nu[mask], factor[mask])
    )
    cov[~mask] = sigma**2

    return cov


# ======================================================================================
# Generate Time Grid
# ======================================================================================


def generate_default_time_grid() -> NDArray[np.floating]:
    return generate_time_grid(101)


def generate_time_grid(n_points: int) -> NDArray[np.floating]:
    """Generate a time grid between 0 and 1."""
    return np.linspace(0, 1, n_points, dtype=np.float64)
