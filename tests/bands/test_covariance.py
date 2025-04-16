from fspb.bands.covariance import (
    _calculate_covariance_confidence_band,
    _calculate_sigma_x_inv,
    _calculate_error_covariance,
    _multiply_x_new_sigma_x_inv_x_newT,
)
import pytest
import numpy as np
from numpy.testing import assert_array_equal


@pytest.fixture
def data():
    x = np.stack([np.eye(3), 2 * np.eye(3)], axis=2)
    x_new = 3 * np.eye(3)[:, :2]
    residuals = np.array(
        [
            [-1, -2],
            [0, 0],
            [1, 2],
        ]
    )
    return x, x_new, residuals


def test_calculate_sigma_inv(data):
    x, _, _ = data
    result = _calculate_sigma_x_inv(x)
    expected_shape = (2, 2, 3, 3)
    expected = np.empty(expected_shape)
    for s in (0, 1):
        for t in (0, 1):
            if s == t == 0:
                arr = np.eye(3)
            elif s == t == 1:
                arr = np.eye(3) / 4
            else:
                arr = np.eye(3) / 2
            expected[s, t] = 3 * arr  # 3 is the number of samples

    assert result.shape == expected_shape
    assert_array_equal(result, expected)


def test_calculate_homoskedastic_error(data):
    _, _, residuals = data
    expected_shape = (2, 2)
    expected = np.array(
        [
            [1, 2],
            [2, 4],
        ]
    )
    got = _calculate_error_covariance(residuals)
    assert got.shape == expected_shape
    assert_array_equal(got, expected)


def test_multiply_x_new_sigma_x_inv_x_newT(data):
    x, x_new, _ = data
    sigma_x_inv = _calculate_sigma_x_inv(x)
    expected = np.array(
        [
            [27, 0],
            [0, 6.75],
        ]
    )
    got = _multiply_x_new_sigma_x_inv_x_newT(x_new, sigma_x_inv)
    assert got.shape == expected.shape
    assert_array_equal(got, expected)


def test_calculate_covariance_on_grid(data):
    x, x_new, residuals = data
    expected = np.array(
        [
            [2 * 27 / 2, 0],
            [0, 8 * 6.75 / 2],
        ]
    )
    got = _calculate_covariance_confidence_band(residuals, x=x, x_new=x_new)
    assert got.shape == expected.shape
    assert_array_equal(got, expected)
