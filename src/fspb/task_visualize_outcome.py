from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fspb.config import BLD, SRC
from fspb.model_simulation import (
    CovarianceType,
    generate_time_grid,
    simulate_from_model,
)
from pytask import Product


def task_visualize_outcome(
    _script: Path = SRC / "model_simulation.py",
    path_figure: Annotated[Path, Product] = BLD / "figures" / "outcomes.png",
) -> None:
    """Save Figure XXX for paper."""
    fig = _create_outcome_figure()
    fig.savefig(path_figure)


def _create_outcome_figure() -> plt.Figure:
    """Create Figure XXX for paper."""
    time_grid = generate_time_grid(n_points=100)
    stationary_outcomes = _generate_stationary_outcomes(
        n_samples=20, time_grid=time_grid
    )
    non_stationary_outcomes = _generate_non_stationary_outcomes(
        n_samples=20, time_grid=time_grid
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].plot(stationary_outcomes, color="red")
    ax[1].plot(non_stationary_outcomes, color="blue")
    return fig


def _generate_stationary_outcomes(
    n_samples: int, time_grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Generate some stationary outcomes.

    Returned outcomes have the shape: (n_points, n_samples)

    """
    outcomes, predictors = simulate_from_model(
        n_samples=n_samples,
        time_grid=time_grid,
        dof=15,
        covariance_type=CovarianceType.STATIONARY,
    )
    return outcomes.T


def _generate_non_stationary_outcomes(
    n_samples: int, time_grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Generate some non-stationary outcomes.

    Returned outcomes have the shape: (n_points, n_samples)

    """
    outcomes, predictors = simulate_from_model(
        n_samples=n_samples,
        time_grid=time_grid,
        dof=15,
        covariance_type=CovarianceType.NON_STATIONARY,
    )
    return outcomes.T
