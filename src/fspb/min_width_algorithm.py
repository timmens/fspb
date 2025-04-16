from typing import Callable
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from fspb.fair_algorithm import calculate_piecewise_integrals, DistributionType
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
    degrees_of_freedom: float | None = None,
    norm_order: float = 2,
    *,
    method: str = "brentq",
    n_cores: int = 1,
    raise_on_error: bool = True,
) -> NDArray[np.floating]:
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

    algo: Algorithm = GaussianAlgorithm(
        significance_level=significance_level,
        interval_cutoffs=interval_cutoffs_jax,
        roughness_integrals=roughness_integrals_jax,
        interval_lengths=interval_lengths_jax,
        time_grid=time_grid_jax,
        sd_diag=sd_diag_jax,
        norm_order=norm_order,
    )

    algorithms = [
        om.algos.ipopt(stopping_maxiter=100),
        om.algos.nlopt_slsqp(stopping_maxfun=100),
        om.algos.scipy_trust_constr(stopping_maxiter=100),
    ]

    results: list[om.OptimizeResult] = []

    for algorithm in algorithms:
        results.append(algo.solve(algorithm))

    fun_values = [res.fun for res in results]
    best_algo_idx = np.argmin(fun_values)

    return results[best_algo_idx].params


@dataclass
class Algorithm(ABC):
    significance_level: float
    interval_cutoffs: Array
    roughness_integrals: Array
    interval_lengths: Array
    time_grid: Array
    sd_diag: Array
    norm_order: float

    def __post_init__(self) -> None:
        self.objective = allow_numpy_input(jit(self._objective))
        self.objective_gradient = allow_numpy_input(jit(grad(self._objective)))
        self.constraint = allow_numpy_input(jit(self._constraint))
        self.constraint_gradient = allow_numpy_input(jit(grad(self._constraint)))

    def solve(self, algorithm: OptimagicAlgorithm) -> om.OptimizeResult:
        u0 = np.repeat(0.1, len(self.interval_cutoffs) - 1)

        bounds = om.Bounds(lower=np.zeros_like(u0))

        constraints = om.NonlinearConstraint(
            selector=lambda u: u,
            func=self.constraint,
            derivative=self.constraint_gradient,
            lower_bound=self.significance_level / 2,
            tol=0.005,
        )

        return om.minimize(
            fun=self.objective,
            params=u0,
            bounds=bounds,
            constraints=constraints,
            algorithm=algorithm,
        )

    def _objective(self, u: Array) -> float:
        u_on_grid = _interval_constants_on_grid(
            u, self.interval_cutoffs, self.time_grid
        )
        return jnp.linalg.norm(self.sd_diag * u_on_grid, ord=self.norm_order)

    def _constraint(self, u: Array) -> float:
        scalings_dot_roughness = jnp.dot(self._scaling(u), self.roughness_integrals)
        return jnp.sum(self._cdf(u)) + scalings_dot_roughness  # type: ignore[return-value]

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
        return norm.pdf(u)


def _interval_constants_on_grid(
    constants: Array,
    interval_cutoffs: Array,
    time_grid: Array,
) -> Array:
    idx = (
        jnp.searchsorted(
            interval_cutoffs, time_grid, side="right", method="compare_all"
        )
        - 1
    )
    return constants[idx]


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
