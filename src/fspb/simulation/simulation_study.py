from __future__ import annotations

from joblib import Parallel, delayed
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass

from functools import partial

from fspb.bands.band import Band, BandType, BandOptions
from fspb.types import CovarianceType
from fspb.simulation.model_simulation import (
    SimulationData,
    generate_default_time_grid,
    simulate_from_model,
)


@dataclass
class SimulationResult:
    """The result of a simulation."""

    simulation_results: list[SingleSimulationResult]
    band_options: BandOptions

    def report(self) -> dict[str, float]:
        coverage_mean, coverage_std = self.coverage()
        maximum_width_statistic_mean, maximum_width_statistic_std = (
            self.maximum_width_statistic()
        )
        band_score_mean, band_score_std = self.band_score()
        return {
            "coverage": float(coverage_mean),
            "coverage_std": float(coverage_std),
            "maximum_width_statistic": float(maximum_width_statistic_mean),
            "maximum_width_statistic_std": float(maximum_width_statistic_std),
            "band_score": float(band_score_mean),
            "band_score_std": float(band_score_std),
        }

    def coverage(self) -> tuple[np.floating, np.floating]:
        """The coverage of the simulation."""
        contained_list = [
            result.band.contains(true_f)
            for result, true_f in zip(self.simulation_results, self.band_center_func())
        ]
        contained_arr = np.array(contained_list, dtype=np.float64)
        return nan_mean_and_std(contained_arr)

    def maximum_width_statistic(self) -> tuple[np.floating, np.floating]:
        """The maximum width statistic of the simulation."""
        widths_list = [
            result.band.maximum_width_statistic for result in self.simulation_results
        ]
        widths_arr = np.array(widths_list, dtype=np.float64)
        return nan_mean_and_std(widths_arr)

    def band_score(self) -> tuple[np.floating, np.floating]:
        """The band scores of the simulation."""
        scores_list = [
            result.band.band_score(
                true_f, signifance_level=self.band_options.significance_level
            )
            for result, true_f in zip(self.simulation_results, self.band_center_func())
        ]
        scores_arr = np.array(scores_list, dtype=np.float64)
        return nan_mean_and_std(scores_arr)

    def band_center_func(self) -> list[NDArray[np.floating]]:
        if self.band_options.band_type == BandType.CONFIDENCE:
            return [
                result.new_data.model.predict(result.new_data.x)
                for result in self.simulation_results
            ]
        elif self.band_options.band_type == BandType.PREDICTION:
            return [result.new_data.y for result in self.simulation_results]
        else:
            raise ValueError(f"Band type {self.band_options.band_type} not supported.")


def nan_mean_and_std(a: NDArray[np.floating]) -> tuple[np.floating, np.floating]:
    """Calculate the mean and standard deviation of an array, ignoring NaNs."""
    valid_mask = ~np.isnan(a)
    return np.mean(a, where=valid_mask), np.std(a, where=valid_mask)


@dataclass
class SingleSimulationResult:
    """The result of a single simulation."""

    data: SimulationData
    new_data: SimulationData
    band: Band


@dataclass
class SimulationOptions:
    n_samples: int
    dof: int
    covariance_type: CovarianceType
    length_scale: float


def simulation_study(
    *,
    n_simulations: int,
    simulation_options: SimulationOptions,
    band_options: BandOptions,
    n_cores: int = 1,
    seed: int | None = None,
) -> SimulationResult:
    """Run a simulation.

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
        A SimulationResult object.

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

    return SimulationResult(results, band_options)


def _single_simulation(
    simulation_options: SimulationOptions,
    band_options: BandOptions,
    time_grid: NDArray[np.floating],
    rng: np.random.Generator,
) -> SingleSimulationResult:
    """Run a single simulation."""

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
        norm_order=band_options.norm_order,
        method=band_options.method,
    )

    return SingleSimulationResult(
        data=data,
        new_data=new_data,
        band=band,
    )
