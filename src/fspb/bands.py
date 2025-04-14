import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from fspb.linear_model import ConcurrentLinearModel
from fspb.roughness import calculate_roughness_on_grid
from fspb.fair_algorithm import fair_critical_value_selection, DistributionType
from fspb.covariance import calculate_covariance_on_grid


@dataclass
class Band:
    estimate: NDArray[np.floating]
    lower: NDArray[np.floating]
    upper: NDArray[np.floating]

    def contains(self, func: NDArray[np.floating]) -> bool:
        """Check if the band contains another function.

        Args:
            func: Has shape (n_time_points, )

        Returns:
            True if the band contains func, False otherwise.

        """
        return bool(np.all(func >= self.lower) and np.all(func <= self.upper))

    @property
    def maximum_width_statistic(self) -> float:
        """Calculate the maximum width statistic of the band.

        Returns:
            The maximum width statistic of the band.

        """
        return np.max(self.upper - self.lower)

    def interval_score(
        self, func: NDArray[np.floating], signifance_level: float
    ) -> float:
        """Calculate the interval score of the band.

        Args:
            func: Has shape (n_time_points, )
            signifance_level: The significance level of the band.

        Returns:
            The interval score of the band.

        """
        maximum_width_statistic = self.maximum_width_statistic
        maximum_low_to_func = np.max((self.lower - func) * (func < self.lower))
        maximum_func_to_high = np.max((self.upper - func) * (func > self.upper))
        return (
            maximum_width_statistic
            + signifance_level / 2 * maximum_low_to_func
            + signifance_level / 2 * maximum_func_to_high
        )


# ======================================================================================
# Confidence bands
# ======================================================================================


def confidence_band(
    y: NDArray[np.floating],
    x: NDArray[np.floating],
    x_new: NDArray[np.floating],
    *,
    time_grid: NDArray[np.floating],
    interval_cutoffs: NDArray[np.floating],
    significance_level: float = 0.05,
    dof: int = 15,
    distribution_type: DistributionType | str = "gaussian",
) -> Band:
    """Confidence band.

    Args:
        y: Has shape (n_samples, n_time_points)
        x: Has shape (n_samples, n_time_points)

    Returns:
        Confidence band.

    """
    model = ConcurrentLinearModel()
    model.fit(x, y)

    residuals = y - model.predict(x)

    covariance = calculate_covariance_on_grid(residuals, x=x, x_new=x_new)

    roughness = calculate_roughness_on_grid(
        cov=covariance,
        time_grid=time_grid,
    )

    sd_diag = np.sqrt(np.diag(covariance))

    roots = fair_critical_value_selection(
        significance_level=significance_level,
        interval_cutoffs=interval_cutoffs,
        roughness=roughness,
        time_grid=time_grid,
        distribution_type=distribution_type,
        degrees_of_freedom=dof,
    )

    scaling_idx = np.searchsorted(interval_cutoffs[1:-1], time_grid)
    scalings = roots[scaling_idx] / np.sqrt(len(y))

    y_pred = model.predict(x_new)

    return Band(
        estimate=y_pred,
        lower=y_pred - scalings * sd_diag,
        upper=y_pred + scalings * sd_diag,
    )
