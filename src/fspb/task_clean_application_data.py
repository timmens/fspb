import pandas as pd
import numpy as np
from pathlib import Path
from pytask import task
from scipy.interpolate import interp1d

from fspb.config import SRC, BLD_APPLICATION, USE_CLEANED_DATA


# ======================================================================================
# Clean predictor data
# ======================================================================================


# --------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------

for amputee in (False, True):
    fname = {False: "x_train", True: "x_pred"}[amputee]

    if USE_CLEANED_DATA:
        x_data_path = SRC / "application" / "covariates.csv"
    else:
        x_data_path = SRC / "application" / "CovariatesReduced.csv"

    @task()
    def task_create_x_train(
        x_data_path: Path = x_data_path,
        produces: Path = BLD_APPLICATION / f"{fname}.pickle",
        amputee: bool = amputee,
    ) -> None:
        if USE_CLEANED_DATA:
            clean_df = pd.read_csv(x_data_path, delimiter=",", index_col="id")
            # Sex column needs to be stored as codes
            clean_df["sex"] = _clean_sex(clean_df["sex"], amputee=clean_df["amputee"])
        else:
            df_x = pd.read_csv(x_data_path, delimiter=",")
            clean_df = _clean_predictors(df_x)
        x_arr = _create_predictor_array(clean_df, amputee=amputee, n_time_points=101)
        pd.to_pickle(x_arr, produces)  # type: ignore[attr-defined]

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def _create_predictor_array(
    clean_df: pd.DataFrame, amputee: bool, n_time_points: int
) -> np.ndarray:
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

    Works on cleaned and uncleaned sex column.

    1. Sex of amputees is set to 'male'
    2. Converts 'female' to 0 and 'male' to 1.

    """
    cleaned = sex.astype(pd.CategoricalDtype())
    if set(cleaned.cat.categories) == {-1, 1}:
        cleaned = cleaned.cat.rename_categories({-1: "female", 1: "male"})
    # There was confusion on the amputee sprinters sex, as the data
    # shows that two of them are female, but in Willwacher et al. (2016), they are
    # described to all be male. When Dominik Liebl asked Steffen Willwacher, he was
    # confident this was just a typo in the data.
    cleaned.loc[amputee] = "male"  # type: ignore[call-overload]
    return cleaned.cat.codes


# ======================================================================================
# Clean outcome data
# ======================================================================================

for amputee in (False, True):
    fname = {False: "y_train", True: "y_pred"}[amputee]

    if USE_CLEANED_DATA:
        y_data_path = SRC / "application" / "outcomes.csv"
    else:
        y_data_path = SRC / "application" / "FRONT_V.csv"

    @task()
    def task_create_y_train(
        y_data_path: Path = y_data_path,
        produces: Path = BLD_APPLICATION / f"{fname}.pickle",
        amputee: bool = amputee,
    ) -> None:
        if USE_CLEANED_DATA:
            y_arr = pd.read_csv(y_data_path, delimiter=",", index_col="id").to_numpy()
        else:
            y_arr = pd.read_csv(y_data_path, delimiter=";", header=None).T.to_numpy()
        updated = _create_outcome_array(
            y_arr,
            amputee=amputee,
            n_time_points=101,
            interval=(70, 90),  # Python index, corresponds to 0.7 to 0.9 in the grid
            target_idx=80,  # Align all peaks at 0.8
        )
        pd.to_pickle(updated, produces)  # type: ignore[attr-defined]


# ======================================================================================
# Find nearest-neighbor for each amputee
# ======================================================================================


def task_find_and_store_nearest_neighbor_trajectories(
    x_train_path: Path = BLD_APPLICATION / "x_train.pickle",
    x_new_path: Path = BLD_APPLICATION / "x_pred.pickle",
    y_train_path: Path = BLD_APPLICATION / "y_train.pickle",
    produces: Path = BLD_APPLICATION / "y_nearest_neighbors.pickle",
) -> None:
    x_train = pd.read_pickle(x_train_path)
    x_new = pd.read_pickle(x_new_path)
    y_train = pd.read_pickle(y_train_path)
    nearest_neighbors = _find_nearest_neighbors_trajectories(
        x_train=x_train,
        x_new=x_new,
        y_train=y_train,
    )
    pd.to_pickle(nearest_neighbors, produces)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def _find_nearest_neighbors_trajectories(
    x_train: np.ndarray,
    x_new: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    y_nearest_neighbors_list = []
    for amputee_index in range(len(x_new)):
        idx = _find_nearest_neighbor(x_new[amputee_index], x_train)
        y_nearest_neighbors_list.append(y_train[idx])
    return np.array(y_nearest_neighbors_list)


def _find_nearest_neighbor(
    x1: np.ndarray,
    x2: np.ndarray,
) -> int:
    """Find the index of the nearest neighbor in x2 for each sample in x1."""
    distances = np.linalg.norm(x1.reshape(1, -1) - x2.reshape(len(x2), -1), axis=1)
    return int(np.argmin(distances))


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
    """Shift each curve so its peak is at the target index.

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
    time_grid_old = np.linspace(0, 1, n_time_points)
    time_grid_new = np.linspace(0, 1, n_points)
    y_interp = np.empty((n_samples, n_points))
    for i in range(n_samples):
        # Step 1: Find the valid (non-nan) indices
        valid = ~np.isnan(y[i])
        # Step 2: Create the interpolator with extrapolation
        interp_func = interp1d(
            time_grid_old[valid],
            y[i][valid],
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )
        # Step 3: Interpolate/extrapolate over the new time grid
        y_interp[i] = interp_func(time_grid_new)

    return y_interp
