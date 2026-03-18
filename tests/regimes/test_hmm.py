"""Tests for HMM, Markov-switching, and GMM regime detection.

Tests use synthetic 2-regime data (low vol + high vol) to verify
that the models can recover known regime structure.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_has_hmmlearn = importlib.util.find_spec("hmmlearn") is not None
_has_statsmodels = importlib.util.find_spec("statsmodels") is not None
_has_sklearn = importlib.util.find_spec("sklearn") is not None


# ---------------------------------------------------------------------------
# Fixtures: synthetic 2-regime data
# ---------------------------------------------------------------------------


def _make_two_regime_series(
    n_low: int = 300,
    n_high: int = 200,
    seed: int = 42,
) -> tuple[pd.Series, np.ndarray]:
    """Generate synthetic 2-regime return series.

    Regime 0 (low vol): mean=+0.001, std=0.008
    Regime 1 (high vol): mean=-0.002, std=0.025

    Returns the series and the ground-truth state array.
    """
    rng = np.random.default_rng(seed)
    low_vol = rng.normal(0.001, 0.008, n_low)
    high_vol = rng.normal(-0.002, 0.025, n_high)

    returns = np.concatenate([low_vol, high_vol])
    true_states = np.concatenate([
        np.zeros(n_low, dtype=int),
        np.ones(n_high, dtype=int),
    ])

    dates = pd.bdate_range("2020-01-01", periods=len(returns))
    series = pd.Series(returns, index=dates, name="returns")
    return series, true_states


def _make_alternating_regime_series(
    n_segments: int = 6,
    segment_length: int = 80,
    seed: int = 123,
) -> tuple[pd.Series, np.ndarray]:
    """Generate alternating regime data for transition testing."""
    rng = np.random.default_rng(seed)
    parts = []
    states = []
    for i in range(n_segments):
        if i % 2 == 0:
            parts.append(rng.normal(0.001, 0.008, segment_length))
            states.append(np.zeros(segment_length, dtype=int))
        else:
            parts.append(rng.normal(-0.002, 0.025, segment_length))
            states.append(np.ones(segment_length, dtype=int))

    returns = np.concatenate(parts)
    true_states = np.concatenate(states)
    series = pd.Series(returns, name="returns")
    return series, true_states


# ---------------------------------------------------------------------------
# Tests: fit_gaussian_hmm
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_hmmlearn, reason="hmmlearn not installed")
class TestFitGaussianHMM:
    def test_output_keys(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        expected_keys = {
            "states", "state_probs", "transition_matrix", "means",
            "covariances", "startprob", "log_likelihood", "aic", "bic",
            "n_states", "steady_state", "avg_duration", "model", "index",
        }
        assert expected_keys.issubset(result.keys())

    def test_states_shape(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        assert result["states"].shape == (len(returns),)
        assert set(np.unique(result["states"])).issubset({0, 1})

    def test_state_probs_shape_and_sum(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        assert result["state_probs"].shape == (len(returns), 2)
        row_sums = result["state_probs"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        row_sums = result["transition_matrix"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_transition_matrix_non_negative(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        assert np.all(result["transition_matrix"] >= 0)

    def test_separates_regimes(self) -> None:
        """HMM should separate low-vol and high-vol regimes."""
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, true_states = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=5)

        # State 0 should be low-vol, state 1 should be high-vol
        # (we sort by variance)
        assert result["covariances"][0] < result["covariances"][1]

        # Check that the model mostly agrees with true states
        predicted = result["states"]
        # Count agreement (allowing for label alignment)
        agreement = np.mean(predicted == true_states)
        flipped_agreement = np.mean(predicted == (1 - true_states))
        accuracy = max(agreement, flipped_agreement)
        assert accuracy > 0.70, f"Regime separation accuracy {accuracy:.2f} too low"

    def test_steady_state_sums_to_one(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        np.testing.assert_allclose(
            result["steady_state"].sum(), 1.0, atol=1e-6
        )

    def test_avg_duration_positive(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        assert np.all(result["avg_duration"] > 0)

    def test_aic_bic_finite(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])

    def test_numpy_input(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns.values, n_states=2, n_init=3)

        assert result["states"].shape == (len(returns),)
        assert result["index"] is None

    def test_preserves_pandas_index(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        assert result["index"] is not None
        assert len(result["index"]) == len(returns)

    def test_three_states(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=3, n_init=3)

        assert result["n_states"] == 3
        assert result["transition_matrix"].shape == (3, 3)
        assert result["means"].shape == (3,)
        assert result["covariances"].shape == (3,)
        assert result["state_probs"].shape[1] == 3

    def test_covariance_types(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        for cov_type in ["full", "diag", "spherical", "tied"]:
            result = fit_gaussian_hmm(
                returns, n_states=2, n_init=2, covariance_type=cov_type
            )
            assert result["states"].shape == (len(returns),)

    def test_startprob_sums_to_one(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        np.testing.assert_allclose(
            result["startprob"].sum(), 1.0, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Tests: fit_hmm (legacy)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_hmmlearn, reason="hmmlearn not installed")
class TestFitHmmLegacy:
    def test_returns_model(self) -> None:
        from wraquant.regimes.hmm import fit_hmm

        returns, _ = _make_two_regime_series()
        model = fit_hmm(returns, n_states=2)

        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


# ---------------------------------------------------------------------------
# Tests: predict_regime
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_hmmlearn, reason="hmmlearn not installed")
class TestPredictRegime:
    def test_predict_from_dict(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm, predict_regime

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        pred = predict_regime(result, returns)
        assert "states" in pred
        assert "state_probs" in pred
        assert pred["states"].shape == (len(returns),)

    def test_predict_from_model(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm, predict_regime

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        pred = predict_regime(result["model"], returns)
        assert pred["states"].shape == (len(returns),)

    def test_predict_preserves_index(self) -> None:
        from wraquant.regimes.hmm import fit_gaussian_hmm, predict_regime

        returns, _ = _make_two_regime_series()
        result = fit_gaussian_hmm(returns, n_states=2, n_init=3)

        pred = predict_regime(result, returns)
        assert pred["index"] is not None
        assert pred["index"].equals(returns.index)


# ---------------------------------------------------------------------------
# Tests: fit_ms_regression
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_statsmodels, reason="statsmodels not installed")
class TestFitMsRegression:
    def test_output_keys(self) -> None:
        from wraquant.regimes.hmm import fit_ms_regression

        returns, _ = _make_two_regime_series()
        result = fit_ms_regression(returns, k_regimes=2)

        expected_keys = {
            "smoothed_probs", "filtered_probs", "states",
            "transition_matrix", "regime_params", "log_likelihood",
            "aic", "bic", "summary", "model_result",
        }
        assert expected_keys.issubset(result.keys())

    def test_smoothed_probs_shape(self) -> None:
        from wraquant.regimes.hmm import fit_ms_regression

        returns, _ = _make_two_regime_series()
        result = fit_ms_regression(returns, k_regimes=2)

        assert result["smoothed_probs"].shape == (len(returns), 2)

    def test_smoothed_probs_sum_to_one(self) -> None:
        from wraquant.regimes.hmm import fit_ms_regression

        returns, _ = _make_two_regime_series()
        result = fit_ms_regression(returns, k_regimes=2)

        row_sums = result["smoothed_probs"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        from wraquant.regimes.hmm import fit_ms_regression

        returns, _ = _make_two_regime_series()
        result = fit_ms_regression(returns, k_regimes=2)

        transmat = result["transition_matrix"]
        # statsmodels transition matrices may have minor numerical
        # deviations from perfect row-stochasticity
        row_sums = transmat.sum(axis=-1).flatten()
        np.testing.assert_allclose(row_sums, 1.0, atol=5e-3)

    def test_regime_params_present(self) -> None:
        from wraquant.regimes.hmm import fit_ms_regression

        returns, _ = _make_two_regime_series()
        result = fit_ms_regression(returns, k_regimes=2)

        assert "mean_0" in result["regime_params"]
        assert "mean_1" in result["regime_params"]
        assert "sigma2_0" in result["regime_params"]
        assert "sigma2_1" in result["regime_params"]

    def test_switching_variance_distinguishes_regimes(self) -> None:
        from wraquant.regimes.hmm import fit_ms_regression

        returns, _ = _make_two_regime_series()
        result = fit_ms_regression(returns, k_regimes=2, switching_variance=True)

        # The two regime variances should be meaningfully different
        s0 = result["regime_params"]["sigma2_0"]
        s1 = result["regime_params"]["sigma2_1"]
        assert s0 != pytest.approx(s1, rel=0.5)

    def test_aic_bic_finite(self) -> None:
        from wraquant.regimes.hmm import fit_ms_regression

        returns, _ = _make_two_regime_series()
        result = fit_ms_regression(returns, k_regimes=2)

        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])


# ---------------------------------------------------------------------------
# Tests: regime_statistics (from hmm.py)
# ---------------------------------------------------------------------------


class TestRegimeStatisticsHMM:
    def test_output_columns(self) -> None:
        from wraquant.regimes.hmm import regime_statistics

        returns, true_states = _make_two_regime_series()
        stats = regime_statistics(returns, true_states)

        expected_cols = {
            "mean", "std", "sharpe", "sortino_ratio", "min", "max",
            "skewness", "kurtosis", "max_drawdown", "VaR_95", "CVaR_95",
            "n_observations", "pct_time", "avg_duration",
        }
        assert expected_cols == set(stats.columns)

    def test_n_observations_sum(self) -> None:
        from wraquant.regimes.hmm import regime_statistics

        returns, true_states = _make_two_regime_series()
        stats = regime_statistics(returns, true_states)

        assert stats["n_observations"].sum() == len(returns)

    def test_pct_time_sums_to_one(self) -> None:
        from wraquant.regimes.hmm import regime_statistics

        returns, true_states = _make_two_regime_series()
        stats = regime_statistics(returns, true_states)

        np.testing.assert_allclose(stats["pct_time"].sum(), 1.0, atol=1e-10)

    def test_low_vol_regime_has_higher_sharpe(self) -> None:
        from wraquant.regimes.hmm import regime_statistics

        returns, true_states = _make_two_regime_series()
        stats = regime_statistics(returns, true_states)

        # Regime 0 (low vol) should have higher Sharpe than regime 1
        assert stats.loc[0, "sharpe"] > stats.loc[1, "sharpe"]

    def test_high_vol_regime_has_higher_std(self) -> None:
        from wraquant.regimes.hmm import regime_statistics

        returns, true_states = _make_two_regime_series()
        stats = regime_statistics(returns, true_states)

        assert stats.loc[1, "std"] > stats.loc[0, "std"]

    def test_avg_duration_positive(self) -> None:
        from wraquant.regimes.hmm import regime_statistics

        returns, true_states = _make_two_regime_series()
        stats = regime_statistics(returns, true_states)

        assert all(stats["avg_duration"] > 0)

    def test_length_mismatch_raises(self) -> None:
        from wraquant.regimes.hmm import regime_statistics

        with pytest.raises(ValueError, match="same length"):
            regime_statistics(np.ones(10), np.zeros(5, dtype=int))


# ---------------------------------------------------------------------------
# Tests: regime_transition_analysis
# ---------------------------------------------------------------------------


class TestRegimeTransitionAnalysis:
    def test_empirical_transition_matrix(self) -> None:
        from wraquant.regimes.hmm import regime_transition_analysis

        _, true_states = _make_alternating_regime_series()
        analysis = regime_transition_analysis(true_states)

        emp = analysis["empirical_transition_matrix"]
        assert emp.shape == (2, 2)
        # Rows should sum to 1
        row_sums = emp.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_with_model_transition_matrix(self) -> None:
        from wraquant.regimes.hmm import regime_transition_analysis

        _, true_states = _make_alternating_regime_series()
        model_transmat = np.array([[0.95, 0.05], [0.10, 0.90]])
        analysis = regime_transition_analysis(true_states, model_transmat)

        np.testing.assert_array_equal(
            analysis["transition_matrix"], model_transmat
        )

    def test_avg_duration_from_transition_matrix(self) -> None:
        from wraquant.regimes.hmm import regime_transition_analysis

        states = np.array([0, 0, 0, 1, 1, 0, 0, 1])
        transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
        analysis = regime_transition_analysis(states, transmat)

        # Expected duration = 1/(1-p_ii)
        np.testing.assert_allclose(analysis["avg_duration"][0], 10.0)
        np.testing.assert_allclose(analysis["avg_duration"][1], 5.0)

    def test_steady_state_sums_to_one(self) -> None:
        from wraquant.regimes.hmm import regime_transition_analysis

        states = np.array([0, 0, 1, 1, 0, 1])
        analysis = regime_transition_analysis(states)

        np.testing.assert_allclose(
            analysis["steady_state"].sum(), 1.0, atol=1e-6
        )

    def test_regime_durations_list(self) -> None:
        from wraquant.regimes.hmm import regime_transition_analysis

        states = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
        analysis = regime_transition_analysis(states)

        # Regime 0: durations [3, 2], Regime 1: durations [2, 3]
        assert analysis["regime_durations"][0] == [3, 2]
        assert analysis["regime_durations"][1] == [2, 3]

    def test_regime_counts(self) -> None:
        from wraquant.regimes.hmm import regime_transition_analysis

        states = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
        analysis = regime_transition_analysis(states)

        assert analysis["regime_counts"][0] == 2
        assert analysis["regime_counts"][1] == 2

    def test_expected_return_time_positive(self) -> None:
        from wraquant.regimes.hmm import regime_transition_analysis

        _, true_states = _make_alternating_regime_series()
        analysis = regime_transition_analysis(true_states)

        assert np.all(analysis["expected_return_time"] > 0)


# ---------------------------------------------------------------------------
# Tests: gaussian_mixture_regimes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_sklearn, reason="sklearn not installed")
class TestGaussianMixtureRegimes:
    def test_output_keys(self) -> None:
        from wraquant.regimes.hmm import gaussian_mixture_regimes

        returns, _ = _make_two_regime_series()
        result = gaussian_mixture_regimes(returns, n_components=2)

        expected_keys = {
            "states", "state_probs", "means", "covariances",
            "weights", "aic", "bic", "model", "index",
        }
        assert expected_keys.issubset(result.keys())

    def test_states_shape(self) -> None:
        from wraquant.regimes.hmm import gaussian_mixture_regimes

        returns, _ = _make_two_regime_series()
        result = gaussian_mixture_regimes(returns, n_components=2)

        assert result["states"].shape == (len(returns),)

    def test_state_probs_sum_to_one(self) -> None:
        from wraquant.regimes.hmm import gaussian_mixture_regimes

        returns, _ = _make_two_regime_series()
        result = gaussian_mixture_regimes(returns, n_components=2)

        row_sums = result["state_probs"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_weights_sum_to_one(self) -> None:
        from wraquant.regimes.hmm import gaussian_mixture_regimes

        returns, _ = _make_two_regime_series()
        result = gaussian_mixture_regimes(returns, n_components=2)

        np.testing.assert_allclose(result["weights"].sum(), 1.0, atol=1e-6)

    def test_ordered_by_variance(self) -> None:
        from wraquant.regimes.hmm import gaussian_mixture_regimes

        returns, _ = _make_two_regime_series()
        result = gaussian_mixture_regimes(returns, n_components=2)

        # Component 0 should have lower variance than component 1
        assert result["covariances"][0] < result["covariances"][1]

    def test_separates_regimes(self) -> None:
        from wraquant.regimes.hmm import gaussian_mixture_regimes

        returns, true_states = _make_two_regime_series()
        result = gaussian_mixture_regimes(returns, n_components=2)

        predicted = result["states"]
        agreement = np.mean(predicted == true_states)
        flipped_agreement = np.mean(predicted == (1 - true_states))
        accuracy = max(agreement, flipped_agreement)
        assert accuracy > 0.65, f"GMM accuracy {accuracy:.2f} too low"

    def test_aic_bic_finite(self) -> None:
        from wraquant.regimes.hmm import gaussian_mixture_regimes

        returns, _ = _make_two_regime_series()
        result = gaussian_mixture_regimes(returns, n_components=2)

        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])

    def test_preserves_index(self) -> None:
        from wraquant.regimes.hmm import gaussian_mixture_regimes

        returns, _ = _make_two_regime_series()
        result = gaussian_mixture_regimes(returns, n_components=2)

        assert result["index"] is not None
        assert result["index"].equals(returns.index)

    def test_numpy_input(self) -> None:
        from wraquant.regimes.hmm import gaussian_mixture_regimes

        returns, _ = _make_two_regime_series()
        result = gaussian_mixture_regimes(returns.values, n_components=2)

        assert result["index"] is None
        assert result["states"].shape == (len(returns),)


# ---------------------------------------------------------------------------
# Tests: regime_aware_portfolio
# ---------------------------------------------------------------------------


class TestRegimeAwarePortfolio:
    def test_blended_weights(self) -> None:
        from wraquant.regimes.hmm import regime_aware_portfolio

        probs = np.array([0.6, 0.4])
        weights = np.array([[0.8, 0.2], [0.3, 0.7]])
        blended = regime_aware_portfolio(probs, weights)

        expected = np.array([0.6 * 0.8 + 0.4 * 0.3, 0.6 * 0.2 + 0.4 * 0.7])
        np.testing.assert_allclose(blended, expected)

    def test_single_regime(self) -> None:
        from wraquant.regimes.hmm import regime_aware_portfolio

        probs = np.array([1.0, 0.0])
        weights = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
        blended = regime_aware_portfolio(probs, weights)

        np.testing.assert_allclose(blended, weights[0])

    def test_equal_probability(self) -> None:
        from wraquant.regimes.hmm import regime_aware_portfolio

        probs = np.array([0.5, 0.5])
        weights = np.array([[1.0, 0.0], [0.0, 1.0]])
        blended = regime_aware_portfolio(probs, weights)

        np.testing.assert_allclose(blended, [0.5, 0.5])

    def test_shape_mismatch_raises(self) -> None:
        from wraquant.regimes.hmm import regime_aware_portfolio

        probs = np.array([0.5, 0.5])
        weights = np.array([[1.0, 0.0]])  # Only 1 regime
        with pytest.raises(ValueError, match="must match"):
            regime_aware_portfolio(probs, weights)

    def test_1d_probs_required(self) -> None:
        from wraquant.regimes.hmm import regime_aware_portfolio

        probs = np.array([[0.5, 0.5]])  # 2-D
        weights = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="1-D"):
            regime_aware_portfolio(probs, weights)


# ---------------------------------------------------------------------------
# Tests: rolling_regime_probability (light test due to computational cost)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_hmmlearn, reason="hmmlearn not installed")
class TestRollingRegimeProbability:
    def test_output_shape(self) -> None:
        from wraquant.regimes.hmm import rolling_regime_probability

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 100))

        result = rolling_regime_probability(
            returns, n_states=2, min_window=50, n_init=2
        )

        assert result.shape == (100, 2)
        assert list(result.columns) == ["prob_0", "prob_1"]

    def test_early_rows_are_nan(self) -> None:
        from wraquant.regimes.hmm import rolling_regime_probability

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 100))

        result = rolling_regime_probability(
            returns, n_states=2, min_window=50, n_init=2
        )

        # First 50 rows should be NaN
        assert result.iloc[:50].isna().all().all()

    def test_later_rows_sum_to_one(self) -> None:
        from wraquant.regimes.hmm import rolling_regime_probability

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 100))

        result = rolling_regime_probability(
            returns, n_states=2, min_window=50, n_init=2
        )

        # Non-NaN rows should sum to ~1
        valid = result.dropna()
        if len(valid) > 0:
            row_sums = valid.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Helpers for new tests
# ---------------------------------------------------------------------------


def _make_multivariate_regime_data(
    n_low: int = 300,
    n_high: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic 2-regime multi-asset return data."""
    rng = np.random.default_rng(seed)
    cov_low = np.array([[0.0001, 0.00002], [0.00002, 0.00015]])
    cov_high = np.array([[0.0009, 0.0006], [0.0006, 0.0008]])
    low = rng.multivariate_normal([0.001, 0.0008], cov_low, n_low)
    high = rng.multivariate_normal([-0.001, -0.0005], cov_high, n_high)
    return pd.DataFrame(
        np.vstack([low, high]),
        columns=["SPY", "QQQ"],
    )


# ---------------------------------------------------------------------------
# Tests: fit_ms_autoregression
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_statsmodels, reason="statsmodels not installed")
class TestFitMsAutoregression:
    def test_basic_fit(self) -> None:
        from wraquant.regimes.hmm import fit_ms_autoregression

        returns, _ = _make_two_regime_series()
        result = fit_ms_autoregression(returns, k_regimes=2, order=1)

        expected_keys = {
            "smoothed_probs", "filtered_probs", "states",
            "transition_matrix", "expected_durations", "regime_params",
            "aic", "bic", "summary", "model_result",
        }
        assert expected_keys.issubset(result.keys())

    def test_smoothed_probs_shape(self) -> None:
        from wraquant.regimes.hmm import fit_ms_autoregression

        returns, _ = _make_two_regime_series()
        order = 1
        result = fit_ms_autoregression(returns, k_regimes=2, order=order)

        # MarkovAutoregression drops `order` initial observations
        T_eff = len(returns) - order
        assert result["smoothed_probs"].shape == (T_eff, 2)

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        from wraquant.regimes.hmm import fit_ms_autoregression

        returns, _ = _make_two_regime_series()
        result = fit_ms_autoregression(returns, k_regimes=2, order=1)

        row_sums = result["transition_matrix"].sum(axis=-1).flatten()
        np.testing.assert_allclose(row_sums, 1.0, atol=5e-3)

    def test_switching_variance(self) -> None:
        from wraquant.regimes.hmm import fit_ms_autoregression

        returns, _ = _make_two_regime_series()
        result = fit_ms_autoregression(
            returns, k_regimes=2, order=1, switching_variance=True
        )

        assert "sigma2_0" in result["regime_params"]
        assert "sigma2_1" in result["regime_params"]
        s0 = result["regime_params"]["sigma2_0"]
        s1 = result["regime_params"]["sigma2_1"]
        assert s0 != pytest.approx(s1, rel=0.5)

    def test_expected_durations_positive(self) -> None:
        from wraquant.regimes.hmm import fit_ms_autoregression

        returns, _ = _make_two_regime_series()
        result = fit_ms_autoregression(returns, k_regimes=2, order=1)

        assert np.all(result["expected_durations"] > 0)

    def test_aic_bic_finite(self) -> None:
        from wraquant.regimes.hmm import fit_ms_autoregression

        returns, _ = _make_two_regime_series()
        result = fit_ms_autoregression(returns, k_regimes=2, order=1)

        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])


# ---------------------------------------------------------------------------
# Tests: select_n_states
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_hmmlearn, reason="hmmlearn not installed")
class TestSelectNStates:
    def test_returns_optimal_between_2_and_max(self) -> None:
        from wraquant.regimes.hmm import select_n_states

        returns, _ = _make_two_regime_series()
        result = select_n_states(returns, max_states=5, n_init=3)

        assert 2 <= result["optimal_n_states"] <= 5

    def test_scores_dict_has_all_keys(self) -> None:
        from wraquant.regimes.hmm import select_n_states

        returns, _ = _make_two_regime_series()
        result = select_n_states(returns, max_states=5, n_init=3)

        for n in range(2, 6):
            assert n in result["scores"]

    def test_best_model_is_dict(self) -> None:
        from wraquant.regimes.hmm import select_n_states

        returns, _ = _make_two_regime_series()
        result = select_n_states(returns, max_states=4, n_init=3)

        assert isinstance(result["best_model"], dict)
        assert "states" in result["best_model"]
        assert "transition_matrix" in result["best_model"]

    def test_optimal_matches_best_bic(self) -> None:
        from wraquant.regimes.hmm import select_n_states

        returns, _ = _make_two_regime_series()
        result = select_n_states(returns, max_states=4, n_init=3)

        optimal = result["optimal_n_states"]
        best_bic = result["scores"][optimal]
        for n, bic_val in result["scores"].items():
            assert bic_val >= best_bic - 1e-6


# ---------------------------------------------------------------------------
# Tests: fit_multivariate_hmm
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_hmmlearn, reason="hmmlearn not installed")
class TestFitMultivariateHMM:
    def test_correct_shapes(self) -> None:
        from wraquant.regimes.hmm import fit_multivariate_hmm

        df = _make_multivariate_regime_data()
        result = fit_multivariate_hmm(df, n_states=2, n_init=5)

        T, d = df.shape
        assert result["states"].shape == (T,)
        assert result["state_probs"].shape == (T, 2)
        assert result["means"].shape == (2, d)
        assert result["covariances"].shape == (2, d, d)
        assert result["transition_matrix"].shape == (2, 2)

    def test_per_regime_correlations_exist(self) -> None:
        from wraquant.regimes.hmm import fit_multivariate_hmm

        df = _make_multivariate_regime_data()
        result = fit_multivariate_hmm(df, n_states=2, n_init=5)

        assert 0 in result["per_regime_correlations"]
        assert 1 in result["per_regime_correlations"]

        corr_0 = result["per_regime_correlations"][0]
        corr_1 = result["per_regime_correlations"][1]
        assert corr_0.shape == (2, 2)
        assert corr_1.shape == (2, 2)

        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(corr_0), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.diag(corr_1), 1.0, atol=1e-6)

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        from wraquant.regimes.hmm import fit_multivariate_hmm

        df = _make_multivariate_regime_data()
        result = fit_multivariate_hmm(df, n_states=2, n_init=5)

        row_sums = result["transition_matrix"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_preserves_columns(self) -> None:
        from wraquant.regimes.hmm import fit_multivariate_hmm

        df = _make_multivariate_regime_data()
        result = fit_multivariate_hmm(df, n_states=2, n_init=5)

        assert result["columns"] == ["SPY", "QQQ"]

    def test_aic_bic_finite(self) -> None:
        from wraquant.regimes.hmm import fit_multivariate_hmm

        df = _make_multivariate_regime_data()
        result = fit_multivariate_hmm(df, n_states=2, n_init=5)

        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])


# ---------------------------------------------------------------------------
# Tests: regime_conditional_moments
# ---------------------------------------------------------------------------


class TestRegimeConditionalMoments:
    def test_mean_cov_shapes_match_n_assets(self) -> None:
        from wraquant.regimes.hmm import regime_conditional_moments

        df = _make_multivariate_regime_data()
        T, d = df.shape
        # Use ground-truth states
        states = np.concatenate([
            np.zeros(300, dtype=int),
            np.ones(200, dtype=int),
        ])

        moments = regime_conditional_moments(df, states)

        assert 0 in moments
        assert 1 in moments
        for k in [0, 1]:
            assert moments[k]["mean"].shape == (d,)
            assert moments[k]["cov"].shape == (d, d)
            assert moments[k]["corr"].shape == (d, d)

    def test_correlation_diagonal_is_one(self) -> None:
        from wraquant.regimes.hmm import regime_conditional_moments

        df = _make_multivariate_regime_data()
        states = np.concatenate([
            np.zeros(300, dtype=int),
            np.ones(200, dtype=int),
        ])

        moments = regime_conditional_moments(df, states)

        for k in [0, 1]:
            np.testing.assert_allclose(
                np.diag(moments[k]["corr"]), 1.0, atol=1e-6
            )

    def test_length_mismatch_raises(self) -> None:
        from wraquant.regimes.hmm import regime_conditional_moments

        df = _make_multivariate_regime_data()
        with pytest.raises(ValueError, match="same length"):
            regime_conditional_moments(df, np.zeros(10, dtype=int))

    def test_type_error_on_non_dataframe(self) -> None:
        from wraquant.regimes.hmm import regime_conditional_moments

        with pytest.raises(TypeError):
            regime_conditional_moments(
                np.ones((10, 2)), np.zeros(10, dtype=int)
            )


# ---------------------------------------------------------------------------
# Tests: enhanced regime_statistics (new columns)
# ---------------------------------------------------------------------------


class TestRegimeStatisticsEnhanced:
    def test_new_columns_present(self) -> None:
        from wraquant.regimes.hmm import regime_statistics

        returns, true_states = _make_two_regime_series()
        stats = regime_statistics(returns, true_states)

        for col in ["max_drawdown", "sortino_ratio", "VaR_95", "CVaR_95"]:
            assert col in stats.columns, f"Missing column: {col}"

    def test_max_drawdown_is_non_positive(self) -> None:
        from wraquant.regimes.hmm import regime_statistics

        returns, true_states = _make_two_regime_series()
        stats = regime_statistics(returns, true_states)

        assert all(stats["max_drawdown"] <= 0)

    def test_var_95_is_negative_for_high_vol(self) -> None:
        from wraquant.regimes.hmm import regime_statistics

        returns, true_states = _make_two_regime_series()
        stats = regime_statistics(returns, true_states)

        # High-vol regime (1) should have a more negative VaR
        assert stats.loc[1, "VaR_95"] < stats.loc[0, "VaR_95"]

    def test_cvar_95_at_most_var_95(self) -> None:
        """CVaR (expected shortfall) should be <= VaR."""
        from wraquant.regimes.hmm import regime_statistics

        returns, true_states = _make_two_regime_series()
        stats = regime_statistics(returns, true_states)

        for regime in stats.index:
            assert stats.loc[regime, "CVaR_95"] <= stats.loc[regime, "VaR_95"] + 1e-12

    def test_sortino_ratio_finite(self) -> None:
        from wraquant.regimes.hmm import regime_statistics

        returns, true_states = _make_two_regime_series()
        stats = regime_statistics(returns, true_states)

        assert all(np.isfinite(stats["sortino_ratio"]))
