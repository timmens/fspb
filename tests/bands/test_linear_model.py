from fspb.bands.linear_model import ConcurrentLinearModel
import pytest
import numpy as np


@pytest.fixture()
def linear_data():
    """Fixture for linear data."""
    rng = np.random.default_rng(0)
    n_samples = 1_000
    n_time_points = 100
    slope = rng.normal(size=(n_samples, n_time_points))
    intercept = np.ones_like(slope)
    x = np.stack([intercept, slope], axis=1)
    y = 3 + 2 * x[:, 1, :] + rng.normal(scale=0.1, size=(n_samples, n_time_points))
    return x, y


def test_fit(linear_data):
    x, y = linear_data
    model = ConcurrentLinearModel()
    model.fit(x, y)
    assert model.x_shape == x.shape
    assert np.allclose(model.coefs[0], 3, atol=0.01)
    assert np.allclose(model.coefs[1], 2, atol=0.01)
    assert model.is_fitted


def test_is_fitted():
    assert not ConcurrentLinearModel().is_fitted
    assert ConcurrentLinearModel(
        coefs=np.zeros((2, 1)),
        x_shape=(0, 2, 1),
    ).is_fitted
    assert not ConcurrentLinearModel(
        coefs=np.zeros((2, 1)),
        x_shape=tuple(),
    ).is_fitted
    assert not ConcurrentLinearModel(
        coefs=np.array([0, np.nan]), x_shape=(1, 2)
    ).is_fitted
    assert not ConcurrentLinearModel(
        coefs=np.array([np.nan, 1]), x_shape=(1, 2)
    ).is_fitted


def test_predict_2d():
    time_grid = np.linspace(0, 1, 3)
    model = ConcurrentLinearModel(
        coefs=np.stack(
            [
                np.zeros_like(time_grid),
                np.arange(len(time_grid), dtype=np.float64),
            ],
            axis=1,
        ).T,
        x_shape=(0, 2, len(time_grid)),
    )

    x_new = np.ones((2, len(time_grid)))
    got = model.predict(x_new)
    expected = np.arange(len(time_grid))
    np.testing.assert_array_equal(got, expected)


def test_predict_3d():
    time_grid = np.linspace(0, 1, 3)
    model = ConcurrentLinearModel(
        coefs=np.stack(
            [
                np.zeros_like(time_grid),
                np.arange(len(time_grid), dtype=np.float64),
            ],
            axis=1,
        ).T,
        x_shape=(0, 2, len(time_grid)),
    )

    x_new_1 = np.ones((2, len(time_grid)))
    x_new_2 = 2 * np.ones((2, len(time_grid)))
    x_new = np.stack([x_new_1, x_new_2], axis=0)

    got = model.predict(x_new)
    expected = np.stack(
        [np.arange(len(time_grid)), 2 * np.arange(len(time_grid))], axis=0
    )
    np.testing.assert_array_equal(got, expected)
