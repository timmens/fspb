from __future__ import annotations

from joblib import Parallel, delayed
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass

from functools import partial

from fspb.bands import Band, BandType
from fspb.fair_algorithm import DistributionType
from fspb.model_simulation import (
    CovarianceType,
    SimulationData,
    generate_default_time_grid,
    simulate_from_model,
)


@dataclass
class MonteCarloSimulationResult:
    """The result of a Monte Carlo simulation."""

    simulation_results: list[SingleSimulationResult]
    band_options: BandOptions

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
            result.band.interval_score(
                true_y, signifance_level=self.band_options.significance_level
            )
            for result, true_y in zip(self.simulation_results, true_ys)
        ]
        return np.array(scores).mean()


@dataclass
class SingleSimulationResult:
    """The result of a single Monte Carlo simulation."""

    data: SimulationData
    new_data: SimulationData
    band: Band


@dataclass
class SimulationOptions:
    n_samples: int
    dof: int
    covariance_type: CovarianceType
    length_scale: float


@dataclass
class BandOptions:
    band_type: BandType
    interval_cutoffs: NDArray[np.floating]
    significance_level: float
    distribution_type: DistributionType


def monte_carlo_simulation(
    *,
    n_simulations: int,
    simulation_options: SimulationOptions,
    band_options: BandOptions,
    n_cores: int = 1,
    seed: int | None = None,
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
    if n_cores < 1:
        raise ValueError("n_cores must be at least 1")

    if seed is None:
        seed = np.random.default_rng().integers(0, 1_000_000)

    time_grid = generate_default_time_grid()

    rng_per_simulation = [
        np.random.default_rng(seed + loop_seed) for loop_seed in range(n_simulations)
    ]

    single_simulation_partialled = partial(
        _single_simulation,
        simulation_options=simulation_options,
        band_options=band_options,
        time_grid=time_grid,
    )

    if n_cores == 1:
        results = [single_simulation_partialled(rng=rng) for rng in rng_per_simulation]
    else:
        results = Parallel(n_jobs=n_cores)(
            delayed(single_simulation_partialled)(rng=rng) for rng in rng_per_simulation
        )

    return MonteCarloSimulationResult(results, band_options)


def _single_simulation(
    simulation_options: SimulationOptions,
    band_options: BandOptions,
    time_grid: NDArray[np.floating],
    rng: np.random.Generator,
) -> SingleSimulationResult:
    """Run a single Monte Carlo simulation."""

    data = simulate_from_model(
        n_samples=simulation_options.n_samples,
        time_grid=time_grid,
        dof=simulation_options.dof,
        covariance_type=simulation_options.covariance_type,
        length_scale=simulation_options.length_scale,
        rng=rng,
    )

    new_data = simulate_from_model(
        n_samples=1,
        time_grid=time_grid,
        dof=simulation_options.dof,
        covariance_type=simulation_options.covariance_type,
        length_scale=simulation_options.length_scale,
        rng=rng,
    )

    band = Band.fit(
        y=data.y,
        x=data.x,
        x_new=new_data.x,
        band_type=band_options.band_type,
        time_grid=time_grid,
        interval_cutoffs=band_options.interval_cutoffs,
        significance_level=band_options.significance_level,
        distribution_type=band_options.distribution_type,
    )

    return SingleSimulationResult(
        data=data,
        new_data=new_data,
        band=band,
    )
