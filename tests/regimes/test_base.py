"""Tests for regime detection abstractions (base.py).

Tests cover RegimeResult construction, properties, detect_regimes dispatch
for multiple methods, regime_report, and multivariate handling.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_has_hmmlearn = importlib.util.find_spec("hmmlearn") is not None
_has_sklearn = importlib.util.find_spec("sklearn") is not None
_has_statsmodels = importlib.util.find_spec("statsmodels") is not None
_has_scipy = importlib.util.find_spec("scipy") is not None


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


def _make_multivariate_regime_data(
    n_low: int = 300,
    n_high: int = 200,
    n_assets: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate multivariate 2-regime return data."""
    rng = np.random.default_rng(seed)
    low_vol = rng.normal(0.001, 0.008, (n_low, n_assets))
    high_vol = rng.normal(-0.002, 0.025, (n_high, n_assets))

    returns = np.vstack([low_vol, high_vol])
    true_states = np.concatenate([
        np.zeros(n_low, dtype=int),
        np.ones(n_high, dtype=int),
    ])

    dates = pd.bdate_range("2020-01-01", periods=len(returns))
    df = pd.DataFrame(returns, index=dates, columns=["A", "B", "C"])
    return df, true_states


# ---------------------------------------------------------------------------
# Tests: RegimeResult construction and properties
# ---------------------------------------------------------------------------


class TestRegimeResult:
    """Tests for the RegimeResult dataclass."""

    def _make_result(self, n: int = 500, k: int = 2) -> "RegimeResult":
        from wraquant.regimes.base import RegimeResult

        rng = np.random.default_rng(99)
        states = np.concatenate([np.zeros(300, dtype=int), np.ones(200, dtype=int)])
        probabilities = np.zeros((n, k))
        probabilities[np.arange(n), states] = 0.9
        probabilities[np.arange(n), 1 - states] = 0.1

        transmat = np.array([[0.95, 0.05], [0.10, 0.90]])
        means = np.array([0.001, -0.002])
        covariances = np.array([0.008**2, 0.025**2])
        statistics = pd.DataFrame(
            {"mean": means, "std": np.sqrt(covariances)},
            index=pd.Index([0, 1], name="regime"),
        )

        return RegimeResult(
            states=states,
            probabilities=probabilities,
            transition_matrix=transmat,
            n_regimes=k,
            means=means,
            covariances=covariances,
            statistics=statistics,
            method="test",
        )

    def test_current_regime(self) -> None:
        result = self._make_result()
        # Last 200 observations are regime 1
        assert result.current_regime == 1

    def test_current_probabilities(self) -> None:
        result = self._make_result()
        probs = result.current_probabilities
        assert probs.shape == (2,)
        assert probs[1] == pytest.approx(0.9)

    def test_expected_durations(self) -> None:
        result = self._make_result()
        durations = result.expected_durations
        # 1 / (1 - 0.95) = 20, 1 / (1 - 0.90) = 10
        np.testing.assert_allclose(durations[0], 20.0)
        np.testing.assert_allclose(durations[1], 10.0)

    def test_steady_state(self) -> None:
        result = self._make_result()
        ss = result.steady_state
        assert ss.shape == (2,)
        np.testing.assert_allclose(ss.sum(), 1.0, atol=1e-6)
        # For transmat [[0.95, 0.05], [0.10, 0.90]]
        # steady state should be [2/3, 1/3]
        np.testing.assert_allclose(ss[0], 2.0 / 3.0, atol=1e-6)
        np.testing.assert_allclose(ss[1], 1.0 / 3.0, atol=1e-6)

    def test_metadata_default(self) -> None:
        result = self._make_result()
        assert result.metadata == {}

    def test_model_default_none(self) -> None:
        result = self._make_result()
        assert result.model is None


# ---------------------------------------------------------------------------
# Tests: detect_regimes with method="hmm"
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_hmmlearn, reason="hmmlearn not installed")
class TestDetectRegimesHMM:
    def test_returns_regime_result(self) -> None:
        from wraquant.regimes.base import RegimeResult, detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)

        assert isinstance(result, RegimeResult)

    def test_states_shape(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)

        assert result.states.shape == (len(returns),)
        assert set(np.unique(result.states)).issubset({0, 1})

    def test_probabilities_shape_and_sum(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)

        assert result.probabilities.shape == (len(returns), 2)
        row_sums = result.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_transition_matrix_valid(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)

        assert result.transition_matrix.shape == (2, 2)
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
        assert np.all(result.transition_matrix >= 0)

    def test_statistics_is_dataframe(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)

        assert isinstance(result.statistics, pd.DataFrame)
        assert len(result.statistics) == 2

    def test_method_field(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)

        assert result.method == "hmm"

    def test_model_is_not_none(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)

        assert result.model is not None

    def test_properties_work(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)

        assert result.current_regime in {0, 1}
        assert result.current_probabilities.shape == (2,)
        assert result.expected_durations.shape == (2,)
        assert np.all(result.expected_durations > 0)
        np.testing.assert_allclose(result.steady_state.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: detect_regimes with method="gmm"
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_sklearn, reason="sklearn not installed")
class TestDetectRegimesGMM:
    def test_returns_regime_result(self) -> None:
        from wraquant.regimes.base import RegimeResult, detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="gmm", n_regimes=2)

        assert isinstance(result, RegimeResult)

    def test_method_field(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="gmm", n_regimes=2)

        assert result.method == "gmm"

    def test_states_and_probs(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="gmm", n_regimes=2)

        assert result.states.shape == (len(returns),)
        assert result.probabilities.shape == (len(returns), 2)
        row_sums = result.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_transition_matrix_empirical(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="gmm", n_regimes=2)

        # GMM builds an empirical transition matrix
        assert result.transition_matrix.shape == (2, 2)
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_properties_work(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="gmm", n_regimes=2)

        assert result.current_regime in {0, 1}
        assert result.expected_durations.shape == (2,)
        np.testing.assert_allclose(result.steady_state.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: detect_regimes with method="changepoint"
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_scipy, reason="scipy not installed")
class TestDetectRegimesChangepoint:
    def test_returns_regime_result(self) -> None:
        from wraquant.regimes.base import RegimeResult, detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="changepoint", n_regimes=2)

        assert isinstance(result, RegimeResult)

    def test_method_field(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="changepoint", n_regimes=2)

        assert result.method == "changepoint"

    def test_states_shape(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="changepoint", n_regimes=2)

        assert result.states.shape == (len(returns),)

    def test_probabilities_shape(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="changepoint", n_regimes=2)

        assert result.probabilities.shape[0] == len(returns)
        assert result.probabilities.shape[1] == 2

    def test_metadata_has_changepoints(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="changepoint", n_regimes=2)

        assert "changepoints" in result.metadata
        assert "run_lengths" in result.metadata

    def test_transition_matrix_valid(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="changepoint", n_regimes=2)

        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: detect_regimes with method="kmeans"
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_sklearn, reason="sklearn not installed")
class TestDetectRegimesKMeans:
    def test_returns_regime_result(self) -> None:
        from wraquant.regimes.base import RegimeResult, detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="kmeans", n_regimes=2)

        assert isinstance(result, RegimeResult)

    def test_method_field(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="kmeans", n_regimes=2)

        assert result.method == "kmeans"

    def test_states_shape(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="kmeans", n_regimes=2)

        assert result.states.shape == (len(returns),)
        assert set(np.unique(result.states)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests: detect_regimes invalid method
# ---------------------------------------------------------------------------


class TestDetectRegimesInvalid:
    def test_unknown_method_raises(self) -> None:
        from wraquant.regimes.base import detect_regimes

        returns, _ = _make_two_regime_series()
        with pytest.raises(ValueError, match="Unknown method"):
            detect_regimes(returns, method="nonexistent")


# ---------------------------------------------------------------------------
# Tests: regime_report
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_hmmlearn, reason="hmmlearn not installed")
class TestRegimeReport:
    def test_returns_expected_keys(self) -> None:
        from wraquant.regimes.base import detect_regimes, regime_report

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)
        report = regime_report(returns, result)

        expected_keys = {
            "summary",
            "current_regime",
            "transition_analysis",
            "regime_history",
            "allocation_suggestion",
            "risk_assessment",
        }
        assert expected_keys == set(report.keys())

    def test_summary_is_dataframe(self) -> None:
        from wraquant.regimes.base import detect_regimes, regime_report

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)
        report = regime_report(returns, result)

        assert isinstance(report["summary"], pd.DataFrame)

    def test_current_regime_info(self) -> None:
        from wraquant.regimes.base import detect_regimes, regime_report

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)
        report = regime_report(returns, result)

        cur = report["current_regime"]
        assert "label" in cur
        assert "probability" in cur
        assert "duration_so_far" in cur
        assert cur["duration_so_far"] >= 1

    def test_transition_analysis_keys(self) -> None:
        from wraquant.regimes.base import detect_regimes, regime_report

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)
        report = regime_report(returns, result)

        trans = report["transition_analysis"]
        assert "expected_durations" in trans
        assert "steady_state" in trans
        assert "visit_counts" in trans

    def test_regime_history_shape(self) -> None:
        from wraquant.regimes.base import detect_regimes, regime_report

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)
        report = regime_report(returns, result)

        hist = report["regime_history"]
        assert isinstance(hist, pd.DataFrame)
        assert "regime" in hist.columns
        assert "prob_0" in hist.columns
        assert "prob_1" in hist.columns

    def test_allocation_suggestion(self) -> None:
        from wraquant.regimes.base import detect_regimes, regime_report

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)
        report = regime_report(returns, result)

        alloc = report["allocation_suggestion"]
        assert "suggestion" in alloc
        assert alloc["suggestion"] in {"risk-on", "risk-off", "neutral", "unknown"}

    def test_risk_assessment_is_dataframe(self) -> None:
        from wraquant.regimes.base import detect_regimes, regime_report

        returns, _ = _make_two_regime_series()
        result = detect_regimes(returns, method="hmm", n_regimes=2, n_init=3)
        report = regime_report(returns, result)

        risk = report["risk_assessment"]
        assert isinstance(risk, pd.DataFrame)
        assert "VaR_95" in risk.columns
        assert "CVaR_95" in risk.columns
        assert "max_drawdown" in risk.columns


# ---------------------------------------------------------------------------
# Tests: RegimeResult with multivariate data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_hmmlearn, reason="hmmlearn not installed")
class TestRegimeResultMultivariate:
    def test_detect_regimes_with_dataframe(self) -> None:
        """detect_regimes should handle DataFrame input (uses first column)."""
        from wraquant.regimes.base import RegimeResult, detect_regimes

        df, _ = _make_multivariate_regime_data()
        result = detect_regimes(df, method="hmm", n_regimes=2, n_init=3)

        assert isinstance(result, RegimeResult)
        assert result.states.shape == (len(df),)

    def test_multivariate_properties(self) -> None:
        from wraquant.regimes.base import detect_regimes

        df, _ = _make_multivariate_regime_data()
        result = detect_regimes(df, method="hmm", n_regimes=2, n_init=3)

        assert result.current_regime in {0, 1}
        assert result.current_probabilities.shape == (2,)
        assert result.expected_durations.shape == (2,)

    def test_regime_report_with_dataframe(self) -> None:
        from wraquant.regimes.base import detect_regimes, regime_report

        df, _ = _make_multivariate_regime_data()
        result = detect_regimes(df, method="hmm", n_regimes=2, n_init=3)
        report = regime_report(df, result)

        assert "summary" in report
        assert "risk_assessment" in report


# ---------------------------------------------------------------------------
# Tests: RegimeDetector ABC
# ---------------------------------------------------------------------------


class TestRegimeDetector:
    def test_cannot_instantiate_abc(self) -> None:
        from wraquant.regimes.base import RegimeDetector

        with pytest.raises(TypeError):
            RegimeDetector()

    def test_subclass_must_implement_methods(self) -> None:
        from wraquant.regimes.base import RegimeDetector

        class IncompleteDetector(RegimeDetector):
            def fit(self, returns):
                return self

        with pytest.raises(TypeError):
            IncompleteDetector()

    def test_complete_subclass_works(self) -> None:
        from wraquant.regimes.base import RegimeDetector, RegimeResult

        class DummyDetector(RegimeDetector):
            def fit(self, returns):
                self._n = len(returns) if hasattr(returns, "__len__") else 100
                return self

            def predict(self, returns):
                n = len(returns) if hasattr(returns, "__len__") else 100
                return np.zeros(n, dtype=int)

            def predict_proba(self, returns):
                n = len(returns) if hasattr(returns, "__len__") else 100
                probs = np.zeros((n, 2))
                probs[:, 0] = 1.0
                return probs

            def to_result(self):
                return RegimeResult(
                    states=np.zeros(self._n, dtype=int),
                    probabilities=np.column_stack([
                        np.ones(self._n),
                        np.zeros(self._n),
                    ]),
                    transition_matrix=np.eye(2),
                    n_regimes=2,
                    means=np.array([0.0, 0.0]),
                    covariances=np.array([0.01, 0.02]),
                    statistics=pd.DataFrame(
                        {"mean": [0.0, 0.0]},
                        index=pd.Index([0, 1], name="regime"),
                    ),
                    method="dummy",
                )

        det = DummyDetector()
        det.fit(np.zeros(50))
        states = det.predict(np.zeros(50))
        assert states.shape == (50,)

        result = det.to_result()
        assert isinstance(result, RegimeResult)
        assert result.method == "dummy"
