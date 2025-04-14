from __future__ import annotations

from joblib import Parallel, delayed
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass


from fspb.bands import Band, confidence_band
from fspb.fair_algorithm import DistributionType
from fspb.model_simulation import (
    CovarianceType,
    SimulationData,
    generate_time_grid,
    simulate_from_model,
)
from typing import TypedDict


@dataclass
class MonteCarloSimulationResult:
    """The result of a Monte Carlo simulation."""

    simulation_results: list[SingleSimulationResult]
    signficance_level: float

    @property
    def coverage(self) -> NDArray[np.floating]:
        """The coverage of the Monte Carlo simulation."""
        true_ys = [
            result.new_data.model.predict(result.new_data.x)
            for result in self.simulation_results
        ]
        contained = [
            result.band.contains(true_y)
            for result, true_y in zip(self.simulation_results, true_ys)
        ]
        return np.array(contained, dtype=np.int8).mean()

    @property
    def maximum_width_statistic(self) -> NDArray[np.floating]:
        """The maximum width statistic of the Monte Carlo simulation."""
        widths = [
            result.band.maximum_width_statistic for result in self.simulation_results
        ]
        return np.array(widths).mean()

    @property
    def interval_score(self) -> NDArray[np.floating]:
        """The interval scores of the Monte Carlo simulation."""
        true_ys = [
            result.new_data.model.predict(result.new_data.x)
            for result in self.simulation_results
        ]
        scores = [
            result.band.interval_score(true_y, signifance_level=self.signficance_level)
            for result, true_y in zip(self.simulation_results, true_ys)
        ]
        return np.array(scores).mean()


@dataclass
class SingleSimulationResult:
    """The result of a single Monte Carlo simulation."""

    simulation_data: SimulationData
    new_data: SimulationData
    band: Band


class SimulationOptions(TypedDict):
    n_samples: int
    dof: int
    covariance_type: CovarianceType | str
    length_scale: float
    time_grid: NDArray[np.floating]


class BandOptions(TypedDict):
    interval_cutoffs: NDArray[np.floating]
    significance_level: float
    distribution_type: DistributionType | str
    time_grid: NDArray[np.floating]
    dof: int


def monte_carlo_simulation(
    n_simulations: int,
    # Simulation parameters
    n_samples: int,
    *,
    dof: int = 15,
    covariance_type: CovarianceType | str = CovarianceType.STATIONARY,
    length_scale: float = 0.1,
    time_grid: NDArray[np.floating] | None = None,
    seed: int | None = None,
    # Confidence band parameters
    interval_cutoffs: NDArray[np.floating] | None = None,
    significance_level: float = 0.05,
    distribution_type: DistributionType | str = DistributionType.GAUSSIAN,
    # Parallelization parameters
    n_cores: int = 1,
) -> MonteCarloSimulationResult:
    """Run a Monte Carlo simulation.

    Args:
        n_simulations: The number of simulations to run.
        n_samples: The number of samples to simulate.
        time_grid: The time grid to simulate the model for. Has shape (n_points,).
        dof: The degrees of freedom of the Student's t distribution.
        covariance_type: The type of covariance to use.
        length_scale: The length scale of the covariance.
        rng: The random state to use for the simulation.
        n_cores: The number of cores to use for the simulation.

    Returns:
        A MonteCarloSimulationResult object.

    """
    if time_grid is None:
        time_grid = generate_time_grid(n_points=101)

    if interval_cutoffs is None:
        interval_cutoffs = np.linspace(0, 1, 4)

    band_options = BandOptions(
        interval_cutoffs=interval_cutoffs,
        significance_level=significance_level,
        distribution_type=distribution_type,
        time_grid=time_grid,
        dof=dof,
    )

    sim_options = SimulationOptions(
        n_samples=n_samples,
        time_grid=time_grid,
        dof=dof,
        covariance_type=covariance_type,
        length_scale=length_scale,
    )

    if seed is None:
        seed = np.random.default_rng().integers(0, 1_000_000)

    rng_per_simulation = [
        np.random.default_rng(seed + loop_seed) for loop_seed in range(n_simulations)
    ]

    if n_cores < 1:
        raise ValueError("n_cores must be at least 1")
    elif n_cores == 1:
        results = [
            _single_simulation(**sim_options, band_kwargs=band_options, rng=rng)
            for rng in rng_per_simulation
        ]
    else:
        results = Parallel(n_jobs=n_cores)(
            delayed(_single_simulation)(
                **sim_options, band_kwargs=band_options, rng=rng
            )
            for rng in rng_per_simulation
        )

    return MonteCarloSimulationResult(results, signficance_level=significance_level)


def _single_simulation(
    n_samples: int,
    time_grid: NDArray[np.floating],
    dof: int,
    covariance_type: CovarianceType | str,
    length_scale: float,
    rng: np.random.Generator,
    band_kwargs: BandOptions,
) -> SingleSimulationResult:
    """Run a single Monte Carlo simulation."""
    simulation_data = simulate_from_model(
        n_samples=n_samples,
        time_grid=time_grid,
        dof=dof,
        covariance_type=covariance_type,
        length_scale=length_scale,
        rng=rng,
    )

    new_data = simulate_from_model(
        n_samples=1,
        time_grid=time_grid,
        dof=dof,
        covariance_type=covariance_type,
        length_scale=length_scale,
        rng=rng,
    )

    band = confidence_band(
        y=simulation_data.y,
        x=simulation_data.x,
        x_new=new_data.x,
        **band_kwargs,
    )

    return SingleSimulationResult(
        simulation_data=simulation_data,
        new_data=new_data,
        band=band,
    )
