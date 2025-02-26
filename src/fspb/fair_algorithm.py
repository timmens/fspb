import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable
from enum import Enum, auto
from dataclasses import dataclass
from scipy.optimize import root_scalar, RootResults
from scipy.stats import norm, t
from scipy.integrate import quad
from abc import ABC, abstractmethod
from joblib import Parallel, delayed


class DistributionType(Enum):
    GAUSSIAN = auto()
    STUDENT_T = auto()


@dataclass(frozen=True)
class Algorithm(ABC):
    significance_level: float
    interval_cutoffs: NDArray[np.float64]
    roughness_integrals: NDArray[np.float64]
    interval_lengths: NDArray[np.float64]

    def solve(self, method: str = "brentq", n_cores: int = 1) -> list[RootResults]:
        interval_ids = range(len(self.interval_cutoffs) - 1)

        if n_cores == 1:
            return [self._solve(interval_id, method) for interval_id in interval_ids]
        else:
            return Parallel(n_jobs=n_cores)(
                delayed(self._solve)(interval_id, method)
                for interval_id in interval_ids
            )

    def _solve(self, interval_index: int, method: str = "brentq") -> RootResults:
        return root_scalar(  # type: ignore[call-overload]
            f=self._equation,
            fprime=self._equation_gradient,
            fprime2=self._equation_hessian,
            args=(interval_index,),
            bracket=[-30, 30],
            method=method,
        )

    def _equation(self, x: float, interval_index: int) -> float:
        return (
            self._cdf(-x)
            + self._scaling(x) * self.roughness_integrals[interval_index]
            - self.significance_level / 2 * self.interval_lengths[interval_index]
        )

    def _equation_gradient(self, x: float, interval_index: int) -> float:
        return (
            -self._cdf_gradient(-x)
            + self._scaling_gradient(x) * self.roughness_integrals[interval_index]
        )

    def _equation_hessian(self, x: float, interval_index: int) -> float:
        return (
            self._cdf_hessian(-x)
            + self._scaling_hessian(x) * self.roughness_integrals[interval_index]
        )

    @abstractmethod
    def _cdf(self, x: float) -> float:
        pass

    @abstractmethod
    def _cdf_gradient(self, x: float) -> float:
        pass

    @abstractmethod
    def _cdf_hessian(self, x: float) -> float:
        pass

    @abstractmethod
    def _scaling(self, x: float) -> float:
        pass

    @abstractmethod
    def _scaling_gradient(self, x: float) -> float:
        pass

    @abstractmethod
    def _scaling_hessian(self, x: float) -> float:
        pass


@dataclass(frozen=True)
class GaussianAlgorithm(Algorithm):
    def _cdf(self, x: float) -> float:
        return norm.cdf(x)

    def _cdf_gradient(self, x: float) -> float:
        return norm.pdf(x)

    def _cdf_hessian(self, x: float) -> float:
        return -x * norm.pdf(x)

    def _scaling(self, x: float) -> float:
        return norm.pdf(x)

    def _scaling_gradient(self, x: float) -> float:
        return -x * norm.pdf(x)

    def _scaling_hessian(self, x: float) -> float:
        return (x**2 - 1) * norm.pdf(x)


@dataclass(frozen=True)
class StudentTAlgorithm(Algorithm):
    degrees_of_freedom: int

    def _cdf(self, x: float) -> float:
        return t.cdf(x, df=self.degrees_of_freedom)

    def _cdf_gradient(self, x: float) -> float:
        return t.pdf(x, df=self.degrees_of_freedom)

    def _cdf_hessian(self, x: float) -> float:
        dof = self.degrees_of_freedom
        return -(dof + 1) * x / (dof + x**2) * t.pdf(x, df=dof)

    def _scaling(self, x: float) -> float:
        return 1 + x**2 / self.degrees_of_freedom

    def _scaling_gradient(self, x: float) -> float:
        return 2 * x / self.degrees_of_freedom

    def _scaling_hessian(self, x: float) -> float:
        return 2 / self.degrees_of_freedom


def fair_critical_value_selection(
    significance_level: float,
    interval_cutoffs: NDArray[np.float64],
    roughness_function: Callable[[float], float],
    distribution_type: DistributionType | str,
    degrees_of_freedom: int | None = None,
    method: str = "brentq",
    n_cores: int = 1,
    quad_kwargs: dict[str, Any] = {},
) -> list[RootResults]:
    if not isinstance(distribution_type, DistributionType):
        try:
            distribution_type = DistributionType[distribution_type.upper()]
        except ValueError:
            raise ValueError(f"Invalid distribution type: {distribution_type}")

    roughness_integrals = _piecewise_integrals(
        interval_cutoffs, roughness_function, quad_kwargs
    )
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

    return algo.solve(method=method, n_cores=n_cores)


def _piecewise_integrals(
    interval_cutoffs: NDArray[np.float64],
    func: Callable[[float], float],
    quad_kwargs: dict[str, Any] = {},
) -> NDArray[np.float64]:
    integrals = np.empty(len(interval_cutoffs) - 1)
    for i in range(len(interval_cutoffs) - 1):
        integrals[i] = quad(
            func, interval_cutoffs[i], interval_cutoffs[i + 1], **quad_kwargs
        )[0]
    return integrals
