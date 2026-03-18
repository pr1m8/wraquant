"""Tests for regime quality assessment (scoring.py).

Verifies that stability, separation, and predictability scores are
bounded in [0, 1], and that compare_regime_methods returns a DataFrame.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_has_hmmlearn = importlib.util.find_spec("hmmlearn") is not None
_has_sklearn = importlib.util.find_spec("sklearn") is not None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_stable_states(n: int = 500, seed: int = 42) -> np.ndarray:
    """Two long-duration regimes."""
    return np.array([0] * (n // 2) + [1] * (n // 2), dtype=int)


def _make_unstable_states(n: int = 500, seed: int = 42) -> np.ndarray:
    """Rapidly alternating regimes."""
    return np.array([i % 2 for i in range(n)], dtype=int)


def _make_returns_with_states(
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns and states with clearly separated distributions."""
    rng = np.random.default_rng(seed)
    r0 = rng.normal(0.01, 0.005, 250)
    r1 = rng.normal(-0.02, 0.03, 250)
    returns = np.concatenate([r0, r1])
    states = np.array([0] * 250 + [1] * 250, dtype=int)
    return returns, states


# ---------------------------------------------------------------------------
# Tests: regime_stability_score
# ---------------------------------------------------------------------------


class TestRegimeStabilityScore:
    def test_score_bounded(self) -> None:
        from wraquant.regimes.scoring import regime_stability_score

        states = _make_stable_states()
        result = regime_stability_score(states)
        assert 0.0 <= result["stability_score"] <= 1.0

    def test_stable_scores_high(self) -> None:
        from wraquant.regimes.scoring import regime_stability_score

        states = _make_stable_states()
        result = regime_stability_score(states)
        assert result["stability_score"] > 0.7

    def test_unstable_scores_low(self) -> None:
        from wraquant.regimes.scoring import regime_stability_score

        states = _make_unstable_states()
        result = regime_stability_score(states)
        assert result["stability_score"] < 0.5

    def test_stable_higher_than_unstable(self) -> None:
        from wraquant.regimes.scoring import regime_stability_score

        stable = regime_stability_score(_make_stable_states())
        unstable = regime_stability_score(_make_unstable_states())
        assert stable["stability_score"] > unstable["stability_score"]

    def test_avg_duration_positive(self) -> None:
        from wraquant.regimes.scoring import regime_stability_score

        result = regime_stability_score(_make_stable_states())
        assert result["avg_duration"] > 0

    def test_transition_frequency_bounded(self) -> None:
        from wraquant.regimes.scoring import regime_stability_score

        result = regime_stability_score(_make_stable_states())
        assert 0.0 <= result["transition_frequency"] <= 1.0

    def test_per_regime_duration_keys(self) -> None:
        from wraquant.regimes.scoring import regime_stability_score

        result = regime_stability_score(_make_stable_states())
        assert 0 in result["per_regime_duration"]
        assert 1 in result["per_regime_duration"]

    def test_with_explicit_transition_matrix(self) -> None:
        from wraquant.regimes.scoring import regime_stability_score

        states = _make_stable_states()
        tm = np.array([[0.99, 0.01], [0.02, 0.98]])
        result = regime_stability_score(states, transition_matrix=tm)
        assert result["stability_score"] > 0.7

    def test_single_regime(self) -> None:
        from wraquant.regimes.scoring import regime_stability_score

        states = np.zeros(100, dtype=int)
        result = regime_stability_score(states)
        # Single regime = max stability
        assert result["stability_score"] >= 0.9
        assert result["n_transitions"] == 0


# ---------------------------------------------------------------------------
# Tests: regime_separation_score
# ---------------------------------------------------------------------------


class TestRegimeSeparationScore:
    def test_score_bounded(self) -> None:
        from wraquant.regimes.scoring import regime_separation_score

        returns, states = _make_returns_with_states()
        result = regime_separation_score(returns, states)
        assert 0.0 <= result["separation_score"] <= 1.0

    def test_well_separated_scores_high(self) -> None:
        from wraquant.regimes.scoring import regime_separation_score

        returns, states = _make_returns_with_states()
        result = regime_separation_score(returns, states)
        assert result["separation_score"] > 0.5

    def test_identical_distributions_score_low(self) -> None:
        from wraquant.regimes.scoring import regime_separation_score

        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, 500)
        states = np.array([0] * 250 + [1] * 250, dtype=int)
        result = regime_separation_score(returns, states)
        assert result["separation_score"] < 0.5

    def test_bhattacharyya_positive(self) -> None:
        from wraquant.regimes.scoring import regime_separation_score

        returns, states = _make_returns_with_states()
        result = regime_separation_score(returns, states)
        for pair, dist in result["pairwise_bhattacharyya"].items():
            assert dist >= 0.0

    def test_overlap_bounded(self) -> None:
        from wraquant.regimes.scoring import regime_separation_score

        returns, states = _make_returns_with_states()
        result = regime_separation_score(returns, states)
        for pair, ov in result["pairwise_overlap"].items():
            assert 0.0 <= ov <= 1.0

    def test_per_regime_stats(self) -> None:
        from wraquant.regimes.scoring import regime_separation_score

        returns, states = _make_returns_with_states()
        result = regime_separation_score(returns, states)
        assert 0 in result["per_regime_stats"]
        assert 1 in result["per_regime_stats"]
        for k in [0, 1]:
            assert "mean" in result["per_regime_stats"][k]
            assert "std" in result["per_regime_stats"][k]

    def test_length_mismatch_raises(self) -> None:
        from wraquant.regimes.scoring import regime_separation_score

        with pytest.raises(ValueError, match="same length"):
            regime_separation_score(np.ones(10), np.zeros(5, dtype=int))


# ---------------------------------------------------------------------------
# Tests: regime_predictability
# ---------------------------------------------------------------------------


class TestRegimePredictability:
    def test_score_bounded(self) -> None:
        from wraquant.regimes.scoring import regime_predictability

        states = _make_stable_states()
        result = regime_predictability(states)
        assert 0.0 <= result["predictability_score"] <= 1.0

    def test_accuracy_bounded(self) -> None:
        from wraquant.regimes.scoring import regime_predictability

        states = _make_stable_states()
        result = regime_predictability(states)
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["baseline_accuracy"] <= 1.0

    def test_high_persistence_high_accuracy(self) -> None:
        from wraquant.regimes.scoring import regime_predictability

        # Highly persistent chain
        rng = np.random.default_rng(0)
        states = np.zeros(1000, dtype=int)
        for t in range(1, 1000):
            if states[t - 1] == 0:
                states[t] = 0 if rng.random() < 0.98 else 1
            else:
                states[t] = 1 if rng.random() < 0.95 else 0

        result = regime_predictability(states)
        assert result["accuracy"] > 0.8

    def test_transition_matrix_shape(self) -> None:
        from wraquant.regimes.scoring import regime_predictability

        states = _make_stable_states()
        result = regime_predictability(states)
        tm = result["transition_matrix_train"]
        assert tm.shape[0] == tm.shape[1]
        # Rows should sum to 1
        np.testing.assert_allclose(tm.sum(axis=1), 1.0, atol=1e-6)

    def test_with_explicit_transition_matrix(self) -> None:
        from wraquant.regimes.scoring import regime_predictability

        states = _make_stable_states()
        tm = np.array([[0.95, 0.05], [0.10, 0.90]])
        result = regime_predictability(states, transition_matrix=tm)
        assert 0.0 <= result["predictability_score"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: compare_regime_methods
# ---------------------------------------------------------------------------


class TestCompareRegimeMethods:
    def test_returns_dataframe(self) -> None:
        from wraquant.regimes.scoring import compare_regime_methods

        returns, states = _make_returns_with_states()

        # Provide simple custom methods to avoid optional dep issues
        def method_a(r, **kw):
            s = np.array([0] * (len(np.asarray(r).flatten()) // 2) +
                         [1] * (len(np.asarray(r).flatten()) - len(np.asarray(r).flatten()) // 2),
                         dtype=int)
            return (s, None)

        def method_b(r, **kw):
            s = np.array([i % 2 for i in range(len(np.asarray(r).flatten()))],
                         dtype=int)
            return (s, None)

        result = compare_regime_methods(
            returns, methods={"stable_split": method_a, "alternating": method_b},
        )

        assert isinstance(result, pd.DataFrame)
        assert "stability" in result.columns
        assert "separation" in result.columns
        assert "predictability" in result.columns
        assert "composite" in result.columns

    def test_composite_bounded(self) -> None:
        from wraquant.regimes.scoring import compare_regime_methods

        returns, _ = _make_returns_with_states()

        def simple_method(r, **kw):
            n = len(np.asarray(r).flatten())
            return (np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=int), None)

        result = compare_regime_methods(
            returns, methods={"simple": simple_method},
        )

        assert 0.0 <= result.loc["simple", "composite"] <= 1.0

    def test_handles_method_failure_gracefully(self) -> None:
        from wraquant.regimes.scoring import compare_regime_methods

        returns, _ = _make_returns_with_states()

        def failing_method(r, **kw):
            raise RuntimeError("test failure")

        def ok_method(r, **kw):
            n = len(np.asarray(r).flatten())
            return (np.zeros(n, dtype=int), None)

        result = compare_regime_methods(
            returns,
            methods={"failing": failing_method, "ok": ok_method},
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert np.isnan(result.loc["failing", "composite"])
        assert np.isfinite(result.loc["ok", "composite"])

    def test_multiple_methods_ranked(self) -> None:
        from wraquant.regimes.scoring import compare_regime_methods

        returns, _ = _make_returns_with_states()

        def stable(r, **kw):
            n = len(np.asarray(r).flatten())
            return (np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=int), None)

        def noisy(r, **kw):
            n = len(np.asarray(r).flatten())
            return (np.array([i % 2 for i in range(n)], dtype=int), None)

        result = compare_regime_methods(
            returns, methods={"stable": stable, "noisy": noisy},
        )

        # Stable should have higher stability
        assert result.loc["stable", "stability"] > result.loc["noisy", "stability"]
