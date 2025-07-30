import pandas as pd
import numpy as np
from pathlib import Path
from pytask import task

from fspb.config import SRC, BLD_APPLICATION


# ======================================================================================
# Clean predictor data
# ======================================================================================

# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------

for amputee in (False, True):
    fname = {False: "x_train", True: "x_pred"}[amputee]

    @task()
    def task_create_x_train(
        x_data_path: Path = SRC / "application" / "CovariatesReduced.csv",
        produces: Path = BLD_APPLICATION / f"{fname}.pickle",
        amputee: bool = amputee,
    ) -> None:
        df_x = pd.read_csv(x_data_path, delimiter=",")
        x_arr = _create_predictor_array(df_x, amputee=amputee, n_time_points=101)
        pd.to_pickle(x_arr, produces)

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def _create_predictor_array(
    df_x: pd.DataFrame, amputee: bool, n_time_points: int
) -> np.ndarray:
    clean_df = _clean_predictors(df_x)
    dropped_df = clean_df.loc[clean_df["amputee"] == amputee].drop(columns=["amputee"])
    dropped_df.insert(0, "intercept", 1)
    x_arr = dropped_df.to_numpy(dtype=np.float64)
    return x_arr[:, :, np.newaxis].repeat(n_time_points, axis=2)


def _clean_predictors(df_x: pd.DataFrame) -> pd.DataFrame:
    clean = pd.DataFrame(index=df_x.index)
    clean["mass"] = df_x["Mass"].astype(pd.Float64Dtype())
    clean["age"] = df_x["Age"].astype(pd.Int64Dtype())
    clean["height"] = df_x["Height"].astype(pd.Float64Dtype())
    clean["amputee"] = (df_x.index >= len(df_x) - 7).astype(bool)
    clean["sex"] = _clean_sex(df_x["Sex"], amputee=clean["amputee"])
    clean["t_push_v"] = df_x["TPush_V"].astype(pd.Float64Dtype())
    return clean


def _clean_sex(sex: pd.Series, amputee: pd.Series) -> pd.Series:
    """Clean sex column.

    1. Sex of amputees is set to 'male'
    2. Converts 'female' to 0 and 'male' to 1.

    """
    cleaned = sex.astype(pd.CategoricalDtype()).cat.rename_categories(
        {-1: "female", 1: "male"}
    )
    # There was confusion on the amputee sprinters sex, as the data
    # shows that two of them are female, but in Willwacher et al. (2016), they are
    # described to all be male. When Dominik asked Steffen, he was confident this
    # was just a typo in the data.
    cleaned.loc[amputee] = "male"
    return cleaned.cat.codes


# ======================================================================================
# Clean outcome data
# ======================================================================================

for amputee in (False, True):
    fname = {False: "y_train", True: "y_pred"}[amputee]

    @task()
    def task_create_y_train(
        y_data_path: Path = SRC / "application" / "FRONT_V.csv",
        produces: Path = BLD_APPLICATION / f"{fname}.pickle",
        amputee: bool = amputee,
    ) -> None:
        y_arr = pd.read_csv(y_data_path, delimiter=";", header=None).T.to_numpy()
        updated = _create_outcome_array(
            y_arr,
            amputee=amputee,
            n_time_points=101,
            interval=(70, 90),  # Python index, corresponds to 0.7 to 0.9 in the grid
            target_idx=80,  # Align all peaks at 0.8
        )
        pd.to_pickle(updated, produces)


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def _create_outcome_array(
    y_arr: np.ndarray,
    amputee: bool,
    n_time_points: int,
    interval: tuple[int, int],
    target_idx: int,
) -> np.ndarray:
    if amputee:
        _y_arr = y_arr[-7:, :]
    else:
        _y_arr = y_arr[:-7, :]

    return _align_and_truncate_functional_outcomes(
        _y_arr,
        interval=interval,
        n_interp_points=n_time_points,
        target_idx=target_idx,
    )


def _align_and_truncate_functional_outcomes(
    y: np.ndarray,
    interval: tuple[int, int],
    n_interp_points: int,
    target_idx: int,
) -> np.ndarray:
    """Align and truncate trajectories.

    Aligns all curves so their max in the interval occurs at target_idx,
    truncates after the aligned max, and interpolates to n_interp_points.

    Args:
        y: Raw functional data, shape (n_samples, n_time_points).
        interval: (start_idx, end_idx) for searching the max.
        n_interp_points: Number of points for interpolation.
        target_idx: Index to align all peaks to.

    Returns:
        Array of shape (n_samples, n_interp_points) with aligned, truncated, and interpolated curves.
    """
    peak_indices = _find_peak_indices(y, interval)
    y_aligned = _align_curves_at_peak(y, peak_indices, target_idx)
    y_trunc = _truncate_after_peak(y_aligned, target_idx)
    y_interp = _interpolate_curves(y_trunc, n_interp_points)
    return y_interp


def _find_peak_indices(y: np.ndarray, interval: tuple[int, int]) -> np.ndarray:
    """Find the index of the maximum value within a given interval for each curve.

    Args:
        y: Array of shape (n_samples, n_time_points).
        interval: (start_idx, end_idx) inclusive.

    Returns:
        Array of shape (n_samples,) with the index of the max in the interval for each sample.
    """
    int_start, int_end = interval
    interval_indices = np.arange(int_start, int_end + 1)
    max_in_interval = np.argmax(y[:, interval_indices], axis=1)
    return interval_indices[max_in_interval]


def _align_curves_at_peak(
    y: np.ndarray, peak_indices: np.ndarray, target_idx: int
) -> np.ndarray:
    """
    Shift each curve so its peak is at the target index.

    Args:
        y: Array of shape (n_samples, n_time_points).
        peak_indices: Array of shape (n_samples,) with peak indices for each curve.
        target_idx: Index to align all peaks to.

    Returns:
        Shifted array of same shape as y.
    """
    n_samples, n_time_points = y.shape
    shifted = np.full_like(y, np.nan)
    for i in range(n_samples):
        shift = target_idx - peak_indices[i]
        if shift >= 0:
            shifted[i, shift:] = y[i, : n_time_points - shift]
        else:
            shifted[i, :shift] = y[i, -shift:]
    return shifted


def _truncate_after_peak(y_aligned: np.ndarray, peak_idx: int) -> np.ndarray:
    """Truncate each curve after the aligned peak (keep up to and including peak).

    Args:
        y_aligned: Array of shape (n_samples, n_time_points), aligned so all peaks at peak_idx.
        peak_idx: Index where all peaks are aligned.

    Returns:
        Truncated array of shape (n_samples, peak_idx+1).
    """
    return y_aligned[:, : peak_idx + 1]


def _interpolate_curves(y: np.ndarray, n_points: int) -> np.ndarray:
    """Linearly interpolate each curve to n_points.

    Args:
        y: Array of shape (n_samples, n_time_points).
        n_points: Number of points to interpolate to.

    Returns:
        Array of shape (n_samples, n_points).
    """
    n_samples, n_time_points = y.shape
    x_old = np.linspace(0, 1, n_time_points)
    x_new = np.linspace(0, 1, n_points)
    y_interp = np.empty((n_samples, n_points))
    for i in range(n_samples):
        y_interp[i] = np.interp(x_new, x_old, y[i])
    return y_interp
