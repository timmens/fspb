from typing import Callable
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, replace
from fspb.bands.fair_algorithm import (
    calculate_piecewise_integrals,
    fair_critical_value_selection,
)
from fspb.types import DistributionType, parse_enum_type, BandType
import jax.numpy as jnp
from jax import Array, grad, jit
from jax.scipy.stats import norm as jax_norm
from scipy.stats import t as scipy_t
import optimagic as om
from optimagic.optimization.algorithm import Algorithm as OptimagicAlgorithm


def min_width_critical_value_selection(
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
    method: str = "brentq",
    n_cores: int = 1,
    raise_on_error: bool = True,
) -> NDArray[np.floating]:
    distribution_type = parse_enum_type(distribution_type, DistributionType)

    roughness_integrals = calculate_piecewise_integrals(
        interval_cutoffs, values=roughness, time_grid=time_grid
    )
    interval_lengths = interval_cutoffs[1:] - interval_cutoffs[:-1]

    # Get start parameters from fair critical value selection
    start_params = fair_critical_value_selection(
        significance_level=significance_level,
        interval_cutoffs=interval_cutoffs,
        time_grid=time_grid,
        roughness=roughness,
        distribution_type=distribution_type,
        degrees_of_freedom=degrees_of_freedom,
        method=method,
        n_cores=n_cores,
        raise_on_error=raise_on_error,
    )

    if band_type == BandType.CONFIDENCE:
        penalty = 0.1
        tol = 0.05
    elif band_type == BandType.PREDICTION:
        penalty = 100
        tol = 0.01

    if distribution_type == DistributionType.GAUSSIAN:
        # Convert to JAX arrays for GaussianAlgorithm
        interval_cutoffs_jax = jnp.array(interval_cutoffs)
        roughness_integrals_jax = jnp.array(roughness_integrals)
        interval_lengths_jax = jnp.array(interval_lengths)
        time_grid_jax = jnp.array(time_grid)
        sd_diag_jax = jnp.array(sd_diag)

        algo = GaussianAlgorithm(
            significance_level=significance_level,
            interval_cutoffs=interval_cutoffs_jax,
            roughness_integrals=roughness_integrals_jax,
            interval_lengths=interval_lengths_jax,
            time_grid=time_grid_jax,
            sd_diag=sd_diag_jax,
            norm_order=norm_order,
            n_samples=n_samples,
            band_type=band_type,
            start_params=start_params,
            penalty=penalty,
        )
    elif distribution_type == DistributionType.STUDENT_T:
        algo = StudentTAlgorithm(
            significance_level=significance_level,
            interval_cutoffs=interval_cutoffs,
            roughness_integrals=roughness_integrals,
            interval_lengths=interval_lengths,
            time_grid=time_grid,
            sd_diag=sd_diag,
            norm_order=norm_order,
            n_samples=n_samples,
            band_type=band_type,
            start_params=start_params,
            penalty=penalty,
            degrees_of_freedom=degrees_of_freedom,
        )
    else:
        msg = f"Unsupported distribution type: {distribution_type}"
        raise ValueError(msg)

    for _ in range(20):
        res = algo.solve(algorithm=om.algos.scipy_lbfgsb(stopping_maxfun=1_000))

        constraint_value = algo._constraint(res.params)
        print(f"Iteration with penalty {penalty}: {constraint_value}")
        success = constraint_value <= significance_level / 2 + tol
        if success:
            break
        else:
            penalty *= 1.5
            start_params = res.params
            algo = algo._replace(
                start_params=start_params,
                penalty=penalty,
            )

    return res.params


@dataclass
class GaussianAlgorithm:
    significance_level: float
    interval_cutoffs: Array
    roughness_integrals: Array
    interval_lengths: Array
    time_grid: Array
    sd_diag: Array
    norm_order: float
    n_samples: int
    band_type: BandType
    start_params: Array | None = None
    penalty: float = 1.0

    def __post_init__(self) -> None:
        # self.objective = allow_numpy_input(jit(self._objective))
        # self.objective_gradient = allow_numpy_input(jit(grad(self._objective)))
        # self.constraint = allow_numpy_input(jit(self._constraint))
        # self.constraint_gradient = allow_numpy_input(jit(grad(self._constraint)))
        self.sd_diag_interval_integral = calculate_piecewise_integrals(
            self.interval_cutoffs, values=self.sd_diag, time_grid=self.time_grid
        )
        self.penalized_objective = allow_numpy_input(jit(self._penalized_objective))
        self.penalized_objective_gradient = allow_numpy_input(
            jit(grad(self._penalized_objective))
        )
        if self.band_type == BandType.CONFIDENCE:
            self.sample_size_factor = 1 / self.n_samples
        elif self.band_type == BandType.PREDICTION:
            self.sample_size_factor = 1.0
        else:
            raise ValueError(f"Unknown band type: {self.band_type}")

    def solve(self, algorithm: OptimagicAlgorithm) -> om.OptimizeResult:
        return _solve(self, algorithm)

    def _penalized_objective(self, u: Array) -> float:
        return (
            self._objective(u)
            + self.penalty * (self._constraint(u) - self.significance_level / 2) ** 2
        )

    def _objective(self, u: Array) -> float:
        return jnp.sum(u**2 * self.sd_diag_interval_integral) * self.sample_size_factor

    def _constraint(self, u: Array) -> float:
        scalings_dot_roughness = jnp.dot(self._scaling(u), self.roughness_integrals)
        return self._cdf(-u[0]) + scalings_dot_roughness  # type: ignore[return-value]

    def _cdf(self, u: Array) -> Array:
        return jax_norm.cdf(u)

    def _scaling(self, u: Array) -> Array:
        return jax_norm.pdf(u) * jnp.sqrt(2 * jnp.pi)

    def _replace(self, **kwargs) -> "GaussianAlgorithm":
        return replace(self, **kwargs)


@dataclass
class StudentTAlgorithm:
    significance_level: float
    interval_cutoffs: NDArray[np.floating]
    roughness_integrals: NDArray[np.floating]
    interval_lengths: NDArray[np.floating]
    time_grid: NDArray[np.floating]
    sd_diag: NDArray[np.floating]
    norm_order: float
    n_samples: int
    band_type: BandType
    start_params: NDArray[np.floating] | None = None
    penalty: float = 1.0
    degrees_of_freedom: float | None = None

    def __post_init__(self) -> None:
        if self.degrees_of_freedom is None:
            msg = "degrees_of_freedom must be provided for Student-t distribution"
            raise ValueError(msg)
        self.sd_diag_interval_integral = calculate_piecewise_integrals(
            self.interval_cutoffs, values=self.sd_diag, time_grid=self.time_grid
        )
        if self.band_type == BandType.CONFIDENCE:
            self.sample_size_factor = 1 / self.n_samples
        elif self.band_type == BandType.PREDICTION:
            self.sample_size_factor = 1.0
        else:
            raise ValueError(f"Unknown band type: {self.band_type}")

    def solve(self, algorithm: OptimagicAlgorithm) -> om.OptimizeResult:
        return _solve(self, algorithm)

    def penalized_objective(self, u: NDArray[np.floating]) -> float:
        return (
            self._objective(u)
            + self.penalty * (self._constraint(u) - self.significance_level / 2) ** 2
        )

    def penalized_objective_gradient(
        self, u: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        return self._objective_gradient(u) + 2 * self.penalty * (
            self._constraint(u) - self.significance_level / 2
        ) * self._constraint_gradient(u)

    def _objective(self, u: NDArray[np.floating]) -> float:
        return np.sum(u**2 * self.sd_diag_interval_integral) * self.sample_size_factor

    def _objective_gradient(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        return 2 * u * self.sd_diag_interval_integral * self.sample_size_factor

    def _constraint(self, u: NDArray[np.floating]) -> float:
        scalings_dot_roughness = np.dot(self._scaling(u), self.roughness_integrals)
        return self._cdf(-u[0]) + scalings_dot_roughness

    def _constraint_gradient(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        grad_scaling = self._scaling_gradient(u)
        grad_cdf = -self._cdf_gradient(-u[0])
        gradient = np.zeros_like(u)
        gradient[0] = grad_cdf
        gradient += grad_scaling * self.roughness_integrals
        return gradient

    def _cdf(self, x: float) -> float:
        return scipy_t.cdf(x, df=self.degrees_of_freedom)

    def _cdf_gradient(self, x: float) -> float:
        return scipy_t.pdf(x, df=self.degrees_of_freedom)

    def _scaling(self, u: float) -> float:
        v = self.degrees_of_freedom
        return (1 + u**2 / v) ** (-v / 2) / (2 * np.pi)

    def _scaling_gradient(self, u: float) -> float:
        v = self.degrees_of_freedom
        return (-u / (2 * np.pi)) * (1 + u**2 / v) ** (-v / 2 - 1)

    def _replace(self, **kwargs) -> "StudentTAlgorithm":
        return replace(self, **kwargs)


def _solve(
    algo: GaussianAlgorithm | StudentTAlgorithm, algorithm: OptimagicAlgorithm
) -> om.OptimizeResult:
    if algo.start_params is not None:
        u0 = algo.start_params
    else:
        u0 = np.repeat(0.5, len(algo.interval_cutoffs) - 1)

    bounds = om.Bounds(lower=np.zeros_like(u0))

    return om.minimize(
        fun=algo.penalized_objective,
        jac=algo.penalized_objective_gradient,
        params=u0,
        bounds=bounds,
        algorithm=algorithm,
    )


def allow_numpy_input(
    func: Callable[[Array], Array],
) -> Callable[[NDArray[np.floating]], float | NDArray[np.floating]]:
    def wrapper(u: NDArray[np.floating]) -> float | NDArray[np.floating]:
        u_jax = jnp.array(u)
        value = func(u_jax)
        if isinstance(value, Array) and value.ndim > 0:
            return np.array(value)
        else:
            return float(value)

    return wrapper
