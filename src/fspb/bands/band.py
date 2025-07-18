from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from fspb.bands.linear_model import ConcurrentLinearModel
from fspb.bands.roughness import calculate_roughness_on_grid
from fspb.bands.fair_algorithm import fair_critical_value_selection
from fspb.types import DistributionType, EstimationMethod
from fspb.bands.min_width_algorithm import min_width_critical_value_selection
from fspb.bands.covariance import calculate_covariance, dof_estimate
from fspb.config import BandType, Scenario


@dataclass
class Band:
    estimate: NDArray[np.floating]
    lower: NDArray[np.floating]
    upper: NDArray[np.floating]

    def contains(self, func: NDArray[np.floating]) -> bool | float:
        """Check if the band contains another function.

        Args:
            func: Has shape (n_time_points, )

        Returns:
            True if the band contains func, False otherwise.

        """
        if _is_invalid(self.lower, self.upper):
            return np.nan
        return bool(np.all(func >= self.lower) and np.all(func <= self.upper))

    @property
    def maximum_width_statistic(self) -> float:
        """Calculate the maximum width statistic of the band.

        Returns:
            The maximum width statistic of the band.

        """
        if _is_invalid(self.lower, self.upper):
            return np.nan
        return np.max(self.upper - self.lower)

    def band_score(
        self, func: NDArray[np.floating], significance_level: float
    ) -> float:
        """Calculate the band score of the band.

        Args:
            func: Has shape (n_time_points, )
            signifance_level: The significance level of the band.

        Returns:
            The band score of the band.

        """
        if _is_invalid(self.lower, self.upper):
            return np.nan
        mws = self.maximum_width_statistic
        max_low_to_func = np.max((self.lower - func) * (func < self.lower))
        max_func_to_high = np.max((func - self.upper) * (func > self.upper))
        return mws + (2 / significance_level) * (max_low_to_func + max_func_to_high)

    @classmethod
    def fit(
        cls,
        y: NDArray[np.floating],
        x: NDArray[np.floating],
        x_new: NDArray[np.floating],
        *,
        band_type: BandType,
        time_grid: NDArray[np.floating],
        interval_cutoffs: NDArray[np.floating],
        significance_level: float,
        distribution_type: DistributionType,
        norm_order: float,
        method: EstimationMethod,
    ) -> Band:
        model = ConcurrentLinearModel()
        model.fit(x, y)

        residuals = y - model.predict(x)

        covariance = calculate_covariance(
            residuals,
            x=x,
            x_new=x_new,
            band_type=band_type,
        )

        dof_hat = dof_estimate(residuals)

        roughness = calculate_roughness_on_grid(
            cov=covariance,
            time_grid=time_grid,
        )
        sd_diag = np.sqrt(np.diag(covariance))

        if method == EstimationMethod.FAIR:
            critical_value_per_interval = fair_critical_value_selection(
                significance_level=significance_level,
                interval_cutoffs=interval_cutoffs,
                roughness=roughness,
                time_grid=time_grid,
                distribution_type=distribution_type,
                degrees_of_freedom=dof_hat,
            )
        elif method == EstimationMethod.MIN_WIDTH:
            critical_value_per_interval = min_width_critical_value_selection(
                significance_level=significance_level,
                interval_cutoffs=interval_cutoffs,
                roughness=roughness,
                distribution_type=distribution_type,
                degrees_of_freedom=dof_hat,
                time_grid=time_grid,
                sd_diag=sd_diag,
                norm_order=norm_order,
                n_samples=len(y),
                band_type=band_type,
            )
        else:
            raise ValueError(f"Unknown band method: {method}")

        # Critical values are defined per interval section, so we need to map the
        # values over the time grid interval.
        scaling_idx = np.searchsorted(interval_cutoffs[1:-1], time_grid)
        critical_values = critical_value_per_interval[scaling_idx]

        if band_type == BandType.CONFIDENCE:
            scaling = critical_values / np.sqrt(len(y))
        elif band_type == BandType.PREDICTION:
            scaling = critical_values
        else:
            raise ValueError(f"Unknown band type: {band_type}")

        y_pred = model.predict(x_new)

        return Band(
            estimate=y_pred,
            lower=y_pred - scaling * sd_diag,
            upper=y_pred + scaling * sd_diag,
        )


@dataclass
class BandOptions:
    band_type: BandType
    interval_cutoffs: NDArray[np.floating]
    significance_level: float
    distribution_type: DistributionType
    norm_order: float

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario,
        distribution_type: DistributionType | None = None,
        interval_cutoffs: NDArray[np.floating] | None = None,
        significance_level: float | None = None,
        norm_order: float | None = None,
    ):
        if distribution_type is None:
            distribution_type = {
                BandType.CONFIDENCE: DistributionType.GAUSSIAN,
                BandType.PREDICTION: DistributionType.STUDENT_T,
            }[scenario.band_type]

        if interval_cutoffs is None:
            interval_cutoffs = np.array([0, 1 / 3, 2 / 3, 1])

        if significance_level is None:
            significance_level = 0.1

        if norm_order is None:
            norm_order = 2.0

        return cls(
            band_type=scenario.band_type,
            interval_cutoffs=interval_cutoffs,
            significance_level=significance_level,
            distribution_type=distribution_type,
            norm_order=norm_order,
        )


def _is_invalid(lower: NDArray[np.floating], upper: NDArray[np.floating]) -> bool:
    return (
        np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.isinf(lower).any()
        or np.isinf(upper).any()
    )
