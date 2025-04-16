import numpy as np

from fspb.simulation.model_simulation import (
    CovarianceType,
    simulate_from_model,
    _slope_function,
    _simulate_predictor,
    _simulate_binary_covariate,
    _predictor_function,
    _simulate_error,
    _matern_covariance,
    generate_time_grid,
    SimulationData,
)


def test_generate_time_grid():
    n_points = 10
    t = generate_time_grid(n_points)
    assert len(t) == n_points
    assert t[0] == 0
    assert t[-1] == 1


def test_slope_function():
    t = np.array([0.0, 0.5, 1.0])
    slopes = _slope_function(t)
    assert slopes.shape == (3,)


def test_simulate_binary_covariate():
    rng = np.random.default_rng(0)
    n_samples = 5
    b = _simulate_binary_covariate(n_samples, rng)
    assert b.shape == (n_samples,)
    assert set(b) <= {0, 1}


def test_predictor_function():
    t = np.linspace(0, 1, 5)
    binary_covariate = np.array([0, 1])
    x = _predictor_function(t, binary_covariate)
    # Expect shape = (2, 5)
    assert x.shape == (2, 5)


def test_simulate_predictor():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 5)
    n_samples = 3
    x = _simulate_predictor(t, n_samples, rng)
    # Expect shape = (n_samples, 2, n_points)
    assert x.shape == (3, 2, 5)


def test_matern_covariance():
    t = np.linspace(0, 1, 4)
    cov_matrix = _matern_covariance(
        t, CovarianceType.STATIONARY, length_scale=1.0, sigma=1.0
    )
    # Expect shape = (4, 4)
    assert cov_matrix.shape == (4, 4)
    # Matrix should be symmetric
    assert np.allclose(cov_matrix, cov_matrix.T)


def test_simulate_error():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 5)
    n_samples = 3
    e = _simulate_error(
        n_samples, t, 3, CovarianceType.STATIONARY, rng, length_scale=1.0
    )
    # Expect shape = (n_samples, n_points)
    assert e.shape == (3, 5)


def test_simulate_from_model():
    rng = np.random.default_rng(0)
    n_samples = 5
    t = generate_time_grid(10)
    sim_data = simulate_from_model(
        n_samples=n_samples,
        time_grid=t,
        dof=4,
        covariance_type=CovarianceType.STATIONARY,
        length_scale=1.0,
        rng=rng,
    )
    assert isinstance(sim_data, SimulationData)
    assert sim_data.y.shape == (n_samples, t.size)
    assert sim_data.x.shape == (n_samples, 2, t.size)
    assert np.allclose(sim_data.time_grid, t)
    assert sim_data.model is not None
