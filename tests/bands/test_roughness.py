import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from fspb.bands.roughness import (
    _cov_to_corr,
    calculate_roughness_on_grid,
    _calculate_squared_roughness_on_grid_numerical_diff,
    _calculate_squared_roughness_on_grid_splines,
)
from sklearn.gaussian_process.kernels import RBF


def _generate_cov_and_time_grid(length_scale, num):
    time_grid = np.linspace(0, 1, num=num, dtype=np.float64)
    rbf_kernel = RBF(length_scale=length_scale)
    cov = rbf_kernel(time_grid.reshape(-1, 1))
    return cov, time_grid


@pytest.mark.parametrize("length_scale", [1 / np.sqrt(2), 1, np.sqrt(2)])
@pytest.mark.parametrize("num", [50, 100, 150])
def test_calculate_roughness_on_grid(length_scale, num):
    cov, time_grid = _generate_cov_and_time_grid(length_scale, num)

    got = calculate_roughness_on_grid(
        cov=cov,
        time_grid=time_grid,
    )
    expected = 1 / length_scale  # can be derived from structure of RBF
    aaae(expected, got, decimal=3)


@pytest.mark.parametrize(
    "calculator",
    (
        _calculate_squared_roughness_on_grid_numerical_diff,
        _calculate_squared_roughness_on_grid_splines,
    ),
)
def test_calculate_roughness_on_grid_implementations(calculator):
    cov, time_grid = _generate_cov_and_time_grid(length_scale=1, num=100)
    got = calculator(cov, time_grid)
    aaae(got, 1, decimal=3)


def test_cov_to_corr():
    cov = np.array([[1, 0, 0.5], [0, 2, 0], [0.5, 0, 3]])
    expected = np.array([[1, 0, 0.5 / np.sqrt(3)], [0, 1, 0], [0.5 / np.sqrt(3), 0, 1]])
    got = _cov_to_corr(cov)
    aaae(expected, got)
