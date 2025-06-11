from typing import Callable
import numpy as np
from functools import partial
from numpy.typing import NDArray
from dataclasses import dataclass
from fspb.bands.fair_algorithm import (
    calculate_piecewise_integrals,
    fair_critical_value_selection,
)
from fspb.types import DistributionType, parse_enum_type
from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import Array, grad, jit
from jax.scipy.stats import norm
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
    degrees_of_freedom: float | None = None,
    norm_order: float = 2,
    *,
    method: str = "brentq",
    n_cores: int = 1,
    raise_on_error: bool = True,
) -> NDArray[np.floating]:
    distribution_type = parse_enum_type(distribution_type, DistributionType)

    if distribution_type == DistributionType.STUDENT_T:
        raise NotImplementedError("Student-t distribution not implemented")

    roughness_integrals = calculate_piecewise_integrals(
        interval_cutoffs, values=roughness, time_grid=time_grid
    )
    interval_lengths = interval_cutoffs[1:] - interval_cutoffs[:-1]

    # Convert to JAX arrays
    interval_cutoffs_jax = jnp.array(interval_cutoffs)
    roughness_integrals_jax = jnp.array(roughness_integrals)
    interval_lengths_jax = jnp.array(interval_lengths)
    time_grid_jax = jnp.array(time_grid)
    sd_diag_jax = jnp.array(sd_diag)

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

    tol = 0.05
    penalty = 0.5

    for _ in range(5):
        algo: Algorithm = GaussianAlgorithm(
            significance_level=significance_level,
            interval_cutoffs=interval_cutoffs_jax,
            roughness_integrals=roughness_integrals_jax,
            interval_lengths=interval_lengths_jax,
            time_grid=time_grid_jax,
            sd_diag=sd_diag_jax,
            norm_order=norm_order,
            n_samples=n_samples,
            start_params=start_params,
            penalty=penalty,
        )

        res = algo.solve(algorithm=om.algos.scipy_lbfgsb(stopping_maxfun=1_000))

        constraint_value = algo._constraint(res.params)
        print(f"Iteration with penalty {penalty}: {constraint_value}")
        success = constraint_value <= significance_level / 2 + tol
        if success:
            break
        else:
            penalty *= 1.5
            start_params = res.params

    return res.params


@dataclass
class Algorithm(ABC):
    significance_level: float
    interval_cutoffs: Array
    roughness_integrals: Array
    interval_lengths: Array
    time_grid: Array
    sd_diag: Array
    norm_order: float
    n_samples: int
    start_params: Array | None = None
    penalty: float = 1.0

    def __post_init__(self) -> None:
        self.objective = allow_numpy_input(jit(self._objective))
        self.objective_gradient = allow_numpy_input(jit(grad(self._objective)))
        self.constraint = allow_numpy_input(jit(self._constraint))
        self.constraint_gradient = allow_numpy_input(jit(grad(self._constraint)))
        self.sd_diag_interval_integral = calculate_piecewise_integrals(
            self.interval_cutoffs, values=self.sd_diag, time_grid=self.time_grid
        )
        self.penalized_objective = allow_numpy_input(
            jit(partial(self._penalized_objective, penalty=self.penalty))
        )
        self.penalized_objective_gradient = allow_numpy_input(
            jit(grad(partial(self._penalized_objective, penalty=self.penalty)))
        )

    def solve(self, algorithm: OptimagicAlgorithm) -> om.OptimizeResult:
        if self.start_params is not None:
            u0 = self.start_params
        else:
            u0 = np.repeat(0.5, len(self.interval_cutoffs) - 1)

        bounds = om.Bounds(lower=np.zeros_like(u0))

        return om.minimize(
            fun=self.penalized_objective,
            jac=self.penalized_objective_gradient,
            params=u0,
            bounds=bounds,
            algorithm=algorithm,
        )

    def _penalized_objective(self, u: Array, penalty: float) -> float:
        return (
            self._objective(u)
            + penalty * (self._constraint(u) - self.significance_level / 100) ** 2
        )

    def _objective(self, u: Array) -> float:
        return jnp.sum(u**2 * self.sd_diag_interval_integral) / self.n_samples

    def _constraint(self, u: Array) -> float:
        scalings_dot_roughness = jnp.dot(self._scaling(u), self.roughness_integrals)
        return self._cdf(-u[0]) + scalings_dot_roughness  # type: ignore[return-value]

    @abstractmethod
    def _cdf(self, u: Array) -> Array:
        pass

    @abstractmethod
    def _scaling(self, u: Array) -> Array:
        pass


@dataclass
class GaussianAlgorithm(Algorithm):
    def _cdf(self, u: Array) -> Array:
        return norm.cdf(u)

    def _scaling(self, u: Array) -> Array:
        return norm.pdf(u) * jnp.sqrt(2 * jnp.pi)


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
