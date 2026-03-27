"""Tests for advanced regime detection integrations."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_has_pomegranate = importlib.util.find_spec("pomegranate") is not None
_has_filterpy = importlib.util.find_spec("filterpy") is not None
_has_river = importlib.util.find_spec("river") is not None
_has_jax = importlib.util.find_spec("jax") is not None
_has_pykalman = importlib.util.find_spec("pykalman") is not None
try:
    from dynamax.linear_gaussian_ssm import LinearGaussianSSM  # noqa: F401
    _has_dynamax = True
except Exception:
    _has_dynamax = False


def _make_regime_data(n: int = 300, seed: int = 42) -> pd.Series:
    """Generate synthetic data with two regimes."""
    rng = np.random.default_rng(seed)
    low_vol = rng.normal(0.001, 0.01, n // 2)
    high_vol = rng.normal(-0.001, 0.03, n - n // 2)
    return pd.Series(np.concatenate([low_vol, high_vol]), name="returns")


class TestPomegranateHMM:
    @pytest.mark.skipif(not _has_pomegranate, reason="pomegranate not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.regimes.integrations import pomegranate_hmm

        data = _make_regime_data()
        result = pomegranate_hmm(data, n_states=2)
        assert "states" in result
        assert "means" in result
        assert "model" in result
        assert "n_states" in result

    @pytest.mark.skipif(not _has_pomegranate, reason="pomegranate not installed")
    def test_states_length(self) -> None:
        from wraquant.regimes.integrations import pomegranate_hmm

        data = _make_regime_data()
        result = pomegranate_hmm(data, n_states=2)
        assert len(result["states"]) == len(data)

    @pytest.mark.skipif(not _has_pomegranate, reason="pomegranate not installed")
    def test_n_states_matches(self) -> None:
        from wraquant.regimes.integrations import pomegranate_hmm

        data = _make_regime_data()
        result = pomegranate_hmm(data, n_states=3)
        assert result["n_states"] == 3
        assert len(result["means"]) == 3

    @pytest.mark.skipif(not _has_pomegranate, reason="pomegranate not installed")
    def test_states_are_integers(self) -> None:
        from wraquant.regimes.integrations import pomegranate_hmm

        data = _make_regime_data()
        result = pomegranate_hmm(data, n_states=2)
        assert result["states"].dtype in (np.int32, np.int64)


class TestFilterpyKalman:
    @pytest.mark.skipif(not _has_filterpy, reason="filterpy not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.regimes.integrations import filterpy_kalman

        observations = np.random.default_rng(42).normal(0, 1, 100)
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[1.0]])
        result = filterpy_kalman(observations, F=F, H=H, Q=Q, R=R)
        assert "filtered_states" in result
        assert "filtered_covariances" in result
        assert "log_likelihood" in result
        assert "residuals" in result

    @pytest.mark.skipif(not _has_filterpy, reason="filterpy not installed")
    def test_filtered_states_shape(self) -> None:
        from wraquant.regimes.integrations import filterpy_kalman

        n = 80
        observations = np.random.default_rng(42).normal(0, 1, n)
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[1.0]])
        result = filterpy_kalman(observations, F=F, H=H, Q=Q, R=R)
        assert result["filtered_states"].shape == (n, 1)
        assert result["filtered_covariances"].shape == (n, 1, 1)

    @pytest.mark.skipif(not _has_filterpy, reason="filterpy not installed")
    def test_multivariate(self) -> None:
        from wraquant.regimes.integrations import filterpy_kalman

        rng = np.random.default_rng(42)
        n = 50
        observations = rng.normal(0, 1, (n, 2))
        F = np.eye(2)
        H = np.eye(2)
        Q = 0.01 * np.eye(2)
        R = np.eye(2)
        result = filterpy_kalman(observations, F=F, H=H, Q=Q, R=R)
        assert result["filtered_states"].shape == (n, 2)

    @pytest.mark.skipif(not _has_filterpy, reason="filterpy not installed")
    def test_with_initial_state(self) -> None:
        from wraquant.regimes.integrations import filterpy_kalman

        observations = np.random.default_rng(42).normal(5, 1, 50)
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[1.0]])
        x0 = np.array([5.0])
        P0 = np.array([[1.0]])
        result = filterpy_kalman(observations, F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
        # With correct initial state, should track closely
        assert result["filtered_states"].shape == (50, 1)


class TestRiverDriftDetector:
    @pytest.mark.skipif(not _has_river, reason="river not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.regimes.integrations import river_drift_detector

        rng = np.random.default_rng(42)
        stream = np.concatenate([
            rng.normal(0, 1, 200),
            rng.normal(5, 1, 200),
        ])
        result = river_drift_detector(stream, method="adwin")
        assert "drift_indices" in result
        assert "n_drifts" in result
        assert "method" in result
        assert result["method"] == "adwin"

    @pytest.mark.skipif(not _has_river, reason="river not installed")
    def test_detects_drift_in_shifted_data(self) -> None:
        from wraquant.regimes.integrations import river_drift_detector

        rng = np.random.default_rng(42)
        stream = np.concatenate([
            rng.normal(0, 1, 500),
            rng.normal(10, 1, 500),
        ])
        result = river_drift_detector(stream, method="adwin")
        assert result["n_drifts"] > 0

    @pytest.mark.skipif(not _has_river, reason="river not installed")
    def test_unknown_method_raises(self) -> None:
        from wraquant.regimes.integrations import river_drift_detector

        with pytest.raises(ValueError, match="Unknown method"):
            river_drift_detector([1.0, 2.0], method="nonexistent")

    @pytest.mark.skipif(not _has_river, reason="river not installed")
    def test_page_hinkley_method(self) -> None:
        from wraquant.regimes.integrations import river_drift_detector

        rng = np.random.default_rng(42)
        stream = rng.normal(0, 1, 100)
        result = river_drift_detector(stream, method="page_hinkley")
        assert result["method"] == "page_hinkley"
        assert isinstance(result["drift_indices"], list)

    @pytest.mark.skipif(not _has_river, reason="river not installed")
    def test_eddm_method(self) -> None:
        from wraquant.regimes.integrations import river_drift_detector

        rng = np.random.default_rng(42)
        stream = np.concatenate([
            rng.normal(0, 1, 200),
            rng.normal(5, 1, 200),
        ])
        result = river_drift_detector(stream, method="eddm")
        assert result["method"] == "eddm"
        assert isinstance(result["drift_indices"], list)


# ---------------------------------------------------------------------------
# dynamax Linear Gaussian SSM
# ---------------------------------------------------------------------------


class TestDynamaxLGSSM:
    @pytest.mark.skipif(
        not (_has_dynamax and _has_jax), reason="dynamax/jax not installed"
    )
    def test_returns_expected_keys(self) -> None:
        from wraquant.regimes.integrations import dynamax_lgssm

        rng = np.random.default_rng(42)
        observations = rng.normal(0, 1, (100, 1))
        result = dynamax_lgssm(observations, state_dim=2, n_iters=10)
        assert "filtered_means" in result
        assert "filtered_covs" in result
        assert "smoothed_means" in result
        assert "params" in result
        assert "log_likelihoods" in result

    @pytest.mark.skipif(
        not (_has_dynamax and _has_jax), reason="dynamax/jax not installed"
    )
    def test_filtered_means_shape(self) -> None:
        from wraquant.regimes.integrations import dynamax_lgssm

        T = 80
        state_dim = 3
        rng = np.random.default_rng(42)
        observations = rng.normal(0, 1, (T, 2))
        result = dynamax_lgssm(
            observations, state_dim=state_dim, emission_dim=2, n_iters=5
        )
        assert result["filtered_means"].shape[0] == T
        assert result["filtered_means"].shape[1] == state_dim

    @pytest.mark.skipif(
        not (_has_dynamax and _has_jax), reason="dynamax/jax not installed"
    )
    def test_univariate_observations(self) -> None:
        from wraquant.regimes.integrations import dynamax_lgssm

        rng = np.random.default_rng(42)
        observations = rng.normal(0, 1, 50)
        result = dynamax_lgssm(observations, state_dim=2, n_iters=5)
        assert result["filtered_means"].shape[0] == 50


# ---------------------------------------------------------------------------
# pykalman Kalman filter with EM learning
# ---------------------------------------------------------------------------


class TestPykalmanFilter:
    @pytest.mark.skipif(not _has_pykalman, reason="pykalman not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.regimes.integrations import pykalman_filter

        rng = np.random.default_rng(42)
        observations = np.cumsum(rng.normal(0, 1, 100))
        result = pykalman_filter(observations, n_em_iter=5)
        assert "filtered_means" in result
        assert "filtered_covs" in result
        assert "smoothed_means" in result
        assert "smoothed_covs" in result
        assert "learned_params" in result
        assert "log_likelihood" in result

    @pytest.mark.skipif(not _has_pykalman, reason="pykalman not installed")
    def test_filtered_means_shape(self) -> None:
        from wraquant.regimes.integrations import pykalman_filter

        n = 80
        rng = np.random.default_rng(42)
        observations = rng.normal(0, 1, n)
        result = pykalman_filter(observations, n_em_iter=3)
        assert result["filtered_means"].shape[0] == n
        assert result["smoothed_means"].shape[0] == n

    @pytest.mark.skipif(not _has_pykalman, reason="pykalman not installed")
    def test_multivariate_observations(self) -> None:
        from wraquant.regimes.integrations import pykalman_filter

        rng = np.random.default_rng(42)
        observations = rng.normal(0, 1, (60, 2))
        result = pykalman_filter(observations, n_em_iter=3)
        assert result["filtered_means"].shape[0] == 60
        assert result["filtered_means"].shape[1] == 2

    @pytest.mark.skipif(not _has_pykalman, reason="pykalman not installed")
    def test_with_initial_transition_matrix(self) -> None:
        from wraquant.regimes.integrations import pykalman_filter

        rng = np.random.default_rng(42)
        observations = np.cumsum(rng.normal(0, 1, 100))
        F = np.array([[1.0]])
        result = pykalman_filter(observations, transition_matrices=F, n_em_iter=5)
        assert result["learned_params"]["transition"].shape == (1, 1)
        assert isinstance(result["log_likelihood"], float)

    @pytest.mark.skipif(not _has_pykalman, reason="pykalman not installed")
    def test_learned_params_keys(self) -> None:
        from wraquant.regimes.integrations import pykalman_filter

        rng = np.random.default_rng(42)
        observations = rng.normal(0, 1, 50)
        result = pykalman_filter(observations, n_em_iter=3)
        params = result["learned_params"]
        assert "transition" in params
        assert "observation" in params
        assert "transition_cov" in params
        assert "observation_cov" in params
