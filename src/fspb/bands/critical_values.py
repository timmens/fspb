import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.optimize import root_scalar
from scipy.stats import norm, t
from scipy import integrate
from abc import ABC, abstractmethod
from fspb.types import DistributionType, parse_enum_type, BandType, EstimationMethod
import scipy.optimize as sp_opt

MAX_CRITICAL_VALUE = 20.0  # Maximum critical value for the band


def solve_for_critical_values(
    significance_level: float,
    interval_cutoffs: NDArray[np.floating],
    time_grid: NDArray[np.floating],
    covariance_diag: NDArray[np.floating],
    roughness: NDArray[np.floating],
    distribution_type: DistributionType,
    n_samples: int,
    band_type: BandType,
    degrees_of_freedom: float,
    *,
    estimation_method: EstimationMethod,
) -> NDArray[np.floating]:
    distribution_type = parse_enum_type(distribution_type, DistributionType)

    roughness_integrals = calculate_piecewise_integrals(
        interval_cutoffs, values=roughness, time_grid=time_grid
    )
    covariance_diag_integrals = calculate_piecewise_integrals(
        interval_cutoffs, values=covariance_diag, time_grid=time_grid
    )
    # Assuming that the intervals lengths are equal
    interval_lengths = interval_cutoffs[1:] - interval_cutoffs[:-1]

    SAMPLE_SIZE_FACTOR_LOOKUP = {
        BandType.CONFIDENCE: 1 / n_samples,
        BandType.PREDICTION: 1.0,
    }

    MAX_ITER = {
        BandType.CONFIDENCE: 10,
        BandType.PREDICTION: 10,
    }

    algo: Algorithm

    if distribution_type == DistributionType.GAUSSIAN:
        algo = GaussianAlgorithm(
            significance_level=significance_level,
            interval_cutoffs=interval_cutoffs,
            roughness_integrals=roughness_integrals,
            covariance_diag_integrals=covariance_diag_integrals,
            interval_lengths=interval_lengths,
            sample_size_factor=SAMPLE_SIZE_FACTOR_LOOKUP[band_type],
        )
    elif distribution_type == DistributionType.STUDENT_T:
        algo = StudentTAlgorithm(
            significance_level=significance_level,
            interval_cutoffs=interval_cutoffs,
            roughness_integrals=roughness_integrals,
            covariance_diag_integrals=covariance_diag_integrals,
            interval_lengths=interval_lengths,
            degrees_of_freedom=degrees_of_freedom,
            sample_size_factor=SAMPLE_SIZE_FACTOR_LOOKUP[band_type],
        )
    else:
        msg = f"Unsupported distribution type: {distribution_type}"
        raise ValueError(msg)

    if estimation_method == EstimationMethod.FAIR:
        critical_values_per_interval = algo.fair_solve()
    elif estimation_method == EstimationMethod.MIN_WIDTH:
        critical_values_per_interval = algo.min_width_solve(maxiter=MAX_ITER[band_type])
    else:
        raise ValueError(f"Unknown estimation method: {estimation_method}")

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
    covariance_diag_integrals: NDArray[np.floating]
    sample_size_factor: float

    # Fair solution
    # ==================================================================================
    def fair_solve(self) -> NDArray[np.floating]:
        interval_ids = range(len(self.interval_cutoffs) - 1)
        roots = [self._fair_solve_interval(interval_id) for interval_id in interval_ids]
        return np.array(roots, dtype=np.float64)

    def _fair_solve_interval(self, interval_index: int) -> float:
        try:
            root_result = root_scalar(
                f=self._fair_equation,
                fprime=self._fair_equation_gradient,
                x0=1.0,
                args=(interval_index,),
                method="newton",
            )
            if root_result.converged:
                return root_result.root
        except Exception:
            pass  # Catch rare numerical errors

        # Fallback to brentq if Newton fails
        root_result = root_scalar(  # type: ignore[call-overload]
            f=self._fair_equation,
            args=(interval_index,),
            bracket=[0, MAX_CRITICAL_VALUE],
            method="brentq",
        )
        raise_error = False  # Add this as argument to Algorithm if needed
        if not root_result.converged and raise_error:
            raise ValueError(f"Root for interval {interval_index} did not converge")
        elif not root_result.converged:
            return MAX_CRITICAL_VALUE

        return root_result.root

    def _fair_equation(self, x: float, interval_index: int) -> float:
        return (
            self._cdf(-x)
            + self._scaling(x) * self.roughness_integrals[interval_index]
            - (self.significance_level / 2) * self.interval_lengths[interval_index]
        )

    def _fair_equation_gradient(self, x: float, interval_index: int) -> float:
        return (
            -self._cdf_gradient(-x)
            + self._scaling_gradient(x) * self.roughness_integrals[interval_index]
        )

    # Min.-Width solution
    # ==================================================================================
    def _is_feasible(
        self, u: NDArray[np.floating], constraint: sp_opt.NonlinearConstraint
    ) -> bool:
        return constraint.lb <= constraint.fun(u) <= constraint.ub  # type: ignore[operator, return-value, arg-type]

    def make_feasible(
        self,
        u0: NDArray[np.floating],
        constraint: sp_opt.NonlinearConstraint,
        bounds: sp_opt.Bounds,
    ) -> NDArray[np.floating]:
        # Define violation measure (0 if feasible, positive otherwise)
        constraint_center = (constraint.lb + constraint.ub) / 2

        def violation(x):  # type: ignore[no-untyped-def]
            return (constraint.fun(x) - constraint_center) ** 2  # type: ignore[operator]

        def violation_gradient(x):  # type: ignore[no-untyped-def]
            grad = constraint.jac(x)  # type: ignore[misc]
            return 2 * (constraint.fun(x) - constraint_center) * grad  # type: ignore[operator]

        import scipy.optimize as sp_opt

        res = sp_opt.minimize(
            violation,
            u0,
            jac=violation_gradient,
            method="L-BFGS-B",
            bounds=bounds,
        )

        if not self._is_feasible(res.x, constraint):
            raise ValueError("Starting point cannot be made feasible.")

        return res.x

    def min_width_solve(
        self,
        maxiter: int,
    ) -> NDArray[np.floating]:
        # Use fair solution as starting point
        u0 = self.fair_solve()

        import scipy.optimize as sp_opt

        tol = 1e-2

        constraint = sp_opt.NonlinearConstraint(
            fun=self._min_width_constraint,  # type: ignore[arg-type]
            jac=self._min_width_constraint_gradient,
            lb=self.significance_level / 2 - tol,
            ub=self.significance_level / 2,
            keep_feasible=True,
        )

        bounds = sp_opt.Bounds(
            lb=0,
            ub=MAX_CRITICAL_VALUE,
        )

        u0_feasible = self.make_feasible(u0, constraint, bounds)  # type: ignore[arg-type]

        method_to_options = {
            "COBYLA": {"maxiter": maxiter, "rhobeg": np.min(u0) / 500},
            "trust-constr": {"maxiter": maxiter, "initial_tr_radius": np.min(u0) / 100},
        }

        method = "trust-constr"  # or "SLSQP", "trust-constr", etc.

        res = sp_opt.minimize(
            fun=self._min_width_objective,  # type: ignore[call-overload]
            x0=u0_feasible,
            jac=self._min_width_objective_gradient,
            constraints=constraint,
            method=method,
            options=method_to_options.get(method, None),
            bounds=bounds,
        )
        if not self._is_feasible(res.x, constraint):
            raise ValueError("Solution point is not feasible.")

        return res.x

    def _min_width_objective(self, u: NDArray[np.floating]) -> float:
        return np.dot(u**2, self.covariance_diag_integrals)

    def _min_width_objective_gradient(
        self, u: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        return 2 * u * self.covariance_diag_integrals

    def _min_width_constraint(self, u: NDArray[np.floating]) -> float:
        scalings_dot_roughness = np.dot(self._scaling(u), self.roughness_integrals)  # type: ignore[arg-type]
        return self._cdf(-u[0]) + scalings_dot_roughness

    def _min_width_constraint_gradient(
        self, u: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        grad_scaling = self._scaling_gradient(u)  # type: ignore[arg-type]
        grad_cdf = -self._cdf_gradient(-u[0])
        gradient = grad_scaling * self.roughness_integrals
        gradient[0] += grad_cdf
        return gradient

    # Abstract methods
    # ==================================================================================

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
