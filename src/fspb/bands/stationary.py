from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Tuple

from scipy.linalg import toeplitz, cholesky
from scipy.stats import linregress
from fspb.bands.covariance import _calculate_error_covariance


# ==========
# Utilities
# ==========


def symmetrize(A: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return (A + A.T)/2 for numerical symmetry."""
    return 0.5 * (A + A.T)


def cov_to_corr(C: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    """
    Convert a covariance matrix C to a correlation matrix K.
    Ensures symmetry and unit diagonal.
    """
    C = symmetrize(np.asarray(C, float))
    d = np.sqrt(np.clip(np.diag(C), eps, None))
    K = (C / d).T / d  # K = D^{-1} C D^{-1}
    np.fill_diagonal(K, 1.0)
    return symmetrize(K)


def frob_norm_sq(A: NDArray[np.floating]) -> float:
    """Squared Frobenius norm ||A||_F^2."""
    return float(np.sum(A * A))


# ==============================
# Stage 1: Variance constancy test
# ==============================


def variance_constancy_test(
    C: NDArray[np.floating],
    grid: Optional[NDArray[np.floating]] = None,
) -> float:
    """
    Test H0: Var(X(t)) is constant over t (necessary for weak stationarity).
    Simple and robust: regress diag(C) on t and test slope == 0.

    Returns
    -------
    p_value : float
        Two-sided p-value for slope == 0 (smaller => evidence of non-constant variance).
    """
    C = np.asarray(C, float)
    m = C.shape[0]
    t = np.linspace(0.0, 1.0, m) if grid is None else np.asarray(grid, float)
    v = np.diag(C)

    # SciPy's linregress gives slope p-value directly (assumes iid noise; fine as a screen)
    lr = linregress(t, v)
    return lr.pvalue  # small p => reject constancy (non-stationary)


# =========================================
# Stage 2: Toeplitz (lag-only) fit + bootstrap
# =========================================


def fit_stationary_correlation(
    K: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Given correlation matrix K on an equally spaced grid, estimate the stationary
    correlation g(d) by averaging along anti-diagonals (constant |i-j|), and
    return the Toeplitz matrix built from g.

    Returns
    -------
    g : (m,) array
        Estimated correlation by lag (g[0]=1, g[1], ..., g[m-1]).
    K_toeplitz : (m,m) array
        Toeplitz matrix with entries g[|i-j|].
    """
    K = symmetrize(np.asarray(K, float))
    m = K.shape[0]

    # Compute absolute lags for all entries at once
    idx = np.arange(m)
    lags = np.abs(idx[:, None] - idx[None, :])  # shape (m, m)

    # Average K by lag using bincount (fast & clear)
    flat_lags = lags.ravel()
    flat_vals = K.ravel()
    counts = np.bincount(flat_lags, minlength=m)
    sums = np.bincount(flat_lags, weights=flat_vals, minlength=m)
    g = sums / np.maximum(counts, 1)

    # Build Toeplitz correlation from g
    K_toeplitz = toeplitz(g)
    # Enforce exact unit diagonal
    np.fill_diagonal(K_toeplitz, 1.0)
    return g, K_toeplitz


def sample_corr_from_stationary_g(
    g: NDArray[np.floating],
    n_paths: int,
    rng: Optional[np.random.Generator] = None,
    jitter: float = 1e-10,
) -> NDArray[np.floating]:
    """
    Simulate a sample correlation matrix by drawing n_paths iid trajectories
    from N(0, Toeplitz(g)) and computing the sample correlation.

    Parameters
    ----------
    g : (m,) array
        Stationary correlation by lag.
    n_paths : int
        Number of independent trajectories used to form the sample correlation.
        (Choose to reflect how your empirical covariance was formed.)
    rng : np.random.Generator, optional
        Random generator for reproducibility.
    jitter : float
        Small diagonal inflation to stabilize Cholesky if needed.

    Returns
    -------
    K_hat : (m,m) array
        Sample correlation matrix estimated from the simulated data.
    """
    rng = np.random.default_rng() if rng is None else rng
    m = g.shape[0]
    K0 = toeplitz(g)

    # Cholesky with gentle jitter (increases if needed)
    J = jitter
    for _ in range(6):
        try:
            L = cholesky(K0 + J * np.eye(m), lower=True)
            break
        except np.linalg.LinAlgError:
            J *= 10.0
    else:
        # Last-resort: eigen clip (rare on well-behaved g)
        w, V = np.linalg.eigh(symmetrize(K0))
        w = np.clip(w, 1e-12, None)
        L = V @ np.diag(np.sqrt(w))

    # Draw n_paths vectors with covariance K0
    Z = rng.standard_normal(size=(n_paths, m))
    X = Z @ L.T  # shape (n_paths, m)

    # Sample covariance & convert to correlation
    X -= X.mean(axis=0, keepdims=True)
    S = (X.T @ X) / max(n_paths - 1, 1)
    K_hat = cov_to_corr(S)
    return K_hat


def toeplitz_lack_of_fit_stat(
    K: NDArray[np.floating], K_toeplitz: NDArray[np.floating]
) -> float:
    """Squared Frobenius distance between the observed K and its Toeplitz fit."""
    return frob_norm_sq(np.asarray(K, float) - np.asarray(K_toeplitz, float))


@dataclass
class StationarityTestResult:
    decision: str  # "stationary" or "non-stationary"
    p_var: float  # Stage 1 p-value (variance constancy)
    p_toeplitz: Optional[float]  # Stage 2 bootstrap p-value (None if skipped)
    T_obs: Optional[float]  # Observed lack-of-fit stat (Stage 2)
    T_boot: Optional[
        NDArray[np.floating]
    ]  # Bootstrap stats (Stage 2), can be large -> optional


def stationarity_test_from_residuals(
    residuals: NDArray[np.floating],
    *,
    alpha: float = 0.05,
    B: int = 499,
    n_paths_boot: int = 400,
    rng: Optional[np.random.Generator] = None,
) -> StationarityTestResult:
    """
    Two-stage, easy-to-interpret stationarity check.

    Stage 1 (necessary condition): test whether Var(X(t)) is constant.
    - If rejected, declare non-stationary (stop early).

    Stage 2 (goodness-of-fit under correlation stationarity): test whether K is Toeplitz
    using a simple Frobenius lack-of-fit and a parametric bootstrap under the fitted
    stationary model K_toeplitz.

    Parameters
    ----------
    C : (m,m) covariance matrix on an equally spaced grid
    alpha : significance level
    B : number of bootstrap replicates for Stage 2
    n_paths_boot : number of trajectories per bootstrap replicate (proxy for how C was formed)
    rng : random generator for reproducibility

    Returns
    -------
    StationarityTestResult
    """
    C = _calculate_error_covariance(
        residuals,
        n_parameters=2,
    )

    rng = np.random.default_rng() if rng is None else rng

    # --------
    # Stage 1
    # --------
    p_var = variance_constancy_test(C)
    if p_var < alpha:
        return StationarityTestResult(
            decision="non-stationary",
            p_var=p_var,
            p_toeplitz=None,
            T_obs=None,
            T_boot=None,
        )

    # --------
    # Stage 2
    # --------
    K = cov_to_corr(C)
    g_hat, K_toep = fit_stationary_correlation(K)
    T_obs = toeplitz_lack_of_fit_stat(K, K_toep)

    # Bootstrap under H0: stationary with correlation g_hat
    T_boot = np.empty(B, dtype=float)
    for b in range(B):
        Kb = sample_corr_from_stationary_g(g_hat, n_paths=n_paths_boot, rng=rng)
        _, K0b = fit_stationary_correlation(Kb)
        T_boot[b] = toeplitz_lack_of_fit_stat(Kb, K0b)

    # One-sided p-value: large distances => reject stationarity
    p_toeplitz = (1.0 + np.sum(T_boot >= T_obs)) / (1.0 + B)
    decision = "stationary" if p_toeplitz >= alpha else "non-stationary"

    return StationarityTestResult(
        decision=decision,
        p_var=p_var,
        p_toeplitz=p_toeplitz,
        T_obs=T_obs,
        T_boot=T_boot,
    )
