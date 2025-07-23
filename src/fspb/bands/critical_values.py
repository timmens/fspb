import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.optimize import root_scalar, RootResults
from scipy.stats import norm, t
from scipy import integrate
from abc import ABC, abstractmethod
from fspb.types import DistributionType, parse_enum_type, BandType, EstimationMethod


def solve_for_critical_values(
    significance_level: float,
    interval_cutoffs: NDArray[np.floating],
    time_grid: NDArray[np.floating],
    sd_diag: NDArray[np.floating],
    roughness: NDArray[np.floating],
    distribution_type: DistributionType,
    n_samples: int,
    band_type: BandType,
    degrees_of_freedom: float | None = None,
    norm_order: float = 2,
    *,
    estimation_method: EstimationMethod,
    raise_on_error: bool = True,
) -> NDArray[np.floating]:
    # if estimation_method == EstimationMethod.FAIR:
    return _fair_critical_value_selection(
        significance_level=significance_level,
        interval_cutoffs=interval_cutoffs,
        time_grid=time_grid,
        roughness=roughness,
        distribution_type=distribution_type,
        degrees_of_freedom=degrees_of_freedom,
        raise_on_error=raise_on_error,
    )
    # elif estimation_method == EstimationMethod.MIN_WIDTH:
    #     return _min_width_critical_value_selection(
    #         significance_level=significance_level,
    #         interval_cutoffs=interval_cutoffs,
    #         time_grid=time_grid,
    #         sd_diag=sd_diag,
    #         roughness=roughness,
    #         distribution_type=distribution_type,
    #         degrees_of_freedom=degrees_of_freedom,
    #         norm_order=norm_order,
    #         n_samples=n_samples,
    #         band_type=band_type,
    #         raise_on_error=raise_on_error,
    #     )
    # else:
    #     raise ValueError(f"Unknown estimation method: {estimation_method}")


def _fair_critical_value_selection(
    significance_level: float,
    interval_cutoffs: NDArray[np.floating],
    time_grid: NDArray[np.floating],
    roughness: NDArray[np.floating],
    distribution_type: DistributionType | str,
    degrees_of_freedom: float | None = None,
    *,
    raise_on_error: bool = True,
) -> NDArray[np.floating]:
    """Select critical values for fair band estimation.

    This implements Algorithm 1 from our paper. The CDF F and the roughness scaling
    S, are defined for each distribution type. Therefore there exists a Gaussian and a
    Student-t algorithm.

    Returns:
        An array of one critical value per interval section.

    """
    distribution_type = parse_enum_type(distribution_type, DistributionType)

    roughness_integrals = calculate_piecewise_integrals(
        interval_cutoffs, values=roughness, time_grid=time_grid
    )
    # Assuming that the intervals lengths are equal
    interval_lengths = interval_cutoffs[1:] - interval_cutoffs[:-1]

    algo: Algorithm

    if distribution_type == DistributionType.GAUSSIAN:
        algo = GaussianAlgorithm(
            significance_level=significance_level,
            interval_cutoffs=interval_cutoffs,
            roughness_integrals=roughness_integrals,
            interval_lengths=interval_lengths,
        )
    elif distribution_type == DistributionType.STUDENT_T:
        if degrees_of_freedom is None:
            raise ValueError(
                "Degrees of freedom must be provided for Student-t distribution"
            )

        algo = StudentTAlgorithm(
            significance_level=significance_level,
            interval_cutoffs=interval_cutoffs,
            roughness_integrals=roughness_integrals,
            interval_lengths=interval_lengths,
            degrees_of_freedom=degrees_of_freedom,
        )

    root_results = algo.solve()

    roots = []

    for k, root_result in enumerate(root_results):
        if raise_on_error and not root_result.converged:
            raise ValueError(f"Root for interval {k} did not converge")
        roots.append(root_result.root)

    critical_values_per_interval = np.array(roots, dtype=np.float64)

    return _map_values_per_interval_onto_grid(
        interval_cutoffs=interval_cutoffs,
        time_grid=time_grid,
        values=critical_values_per_interval,
    )


@dataclass(frozen=True)
class Algorithm(ABC):
    significance_level: float
    interval_cutoffs: NDArray[np.floating]
    roughness_integrals: NDArray[np.floating]
    interval_lengths: NDArray[np.floating]

    def solve(self) -> list[RootResults]:
        interval_ids = range(len(self.interval_cutoffs) - 1)
        return [self._solve(interval_id) for interval_id in interval_ids]

    def _solve(self, interval_index: int) -> RootResults:
        try:
            result = root_scalar(
                f=self._equation,
                fprime=self._equation_gradient,
                x0=1.0,
                args=(interval_index,),
                method="newton",
            )
            if result.converged:
                return result
        except Exception:
            pass  # Catch rare numerical errors

        # Fallback to brentq if Newton fails
        return root_scalar(  # type: ignore[call-overload]
            f=self._equation,
            args=(interval_index,),
            bracket=[0, 10],
            method="brentq",
        )

    def _equation(self, x: float, interval_index: int) -> float:
        return (
            self._cdf(-x)
            + self._scaling(x) * self.roughness_integrals[interval_index]
            - (self.significance_level / 2) * self.interval_lengths[interval_index]
        )

    def _equation_gradient(self, x: float, interval_index: int) -> float:
        return (
            -self._cdf_gradient(-x)
            + self._scaling_gradient(x) * self.roughness_integrals[interval_index]
        )

    @abstractmethod
    def _cdf(self, x: float) -> float:
        pass

    @abstractmethod
    def _cdf_gradient(self, x: float) -> float:
        pass

    @abstractmethod
    def _scaling(self, x: float) -> float:
        pass

    @abstractmethod
    def _scaling_gradient(self, x: float) -> float:
        pass


@dataclass(frozen=True)
class GaussianAlgorithm(Algorithm):
    def _cdf(self, u: float) -> float:
        return norm.cdf(u)

    def _cdf_gradient(self, u: float) -> float:
        return norm.pdf(u)

    def _scaling(self, u: float) -> float:
        return norm.pdf(u) / np.sqrt(2 * np.pi)

    def _scaling_gradient(self, u: float) -> float:
        return -u * norm.pdf(u) / np.sqrt(2 * np.pi)


@dataclass(frozen=True)
class StudentTAlgorithm(Algorithm):
    degrees_of_freedom: float

    def _cdf(self, u: float) -> float:
        return t.cdf(u, df=self.degrees_of_freedom)

    def _cdf_gradient(self, u: float) -> float:
        return t.pdf(u, df=self.degrees_of_freedom)

    def _scaling(self, u: float) -> float:
        v = self.degrees_of_freedom
        return (1 + u**2 / v) ** (-v / 2) / (2 * np.pi)

    def _scaling_gradient(self, u: float) -> float:
        v = self.degrees_of_freedom
        return -u * (1 + u**2 / v) ** (-v / 2 - 1) / (2 * np.pi)


def calculate_piecewise_integrals(
    interval_cutoffs: NDArray[np.floating],
    values: NDArray[np.floating],
    time_grid: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the integral over subintervals using the Simpson rule.

    Args:
        interval_cutoffs: array of increasing cutoff points defining subintervals.
        values: array of function values sampled at `time_grid`.
        time_grid: array of time points corresponding to `values`.

    Returns:
        Array of integrals over each subinterval.

    """
    integrals = np.empty(len(interval_cutoffs) - 1)
    idx = np.searchsorted(time_grid, interval_cutoffs)

    for i in range(len(integrals)):
        left = idx[i]
        right = idx[i + 1] + (1 if i == len(integrals) - 1 else 0)
        f_segment = values[left:right]
        t_segment = time_grid[left:right]
        integrals[i] = integrate.simpson(f_segment, t_segment)

    return integrals


def _map_values_per_interval_onto_grid(
    interval_cutoffs: NDArray[np.floating],
    time_grid: NDArray[np.floating],
    values: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Map values defined per section onto the time grid."""
    scaling_idx = np.searchsorted(interval_cutoffs[1:-1], time_grid)
    return values[scaling_idx]
