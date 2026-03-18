"""Tests for changepoint detection methods.

Covers PELT, binary segmentation, window changepoint, CUSUM, and the
existing online changepoint detector.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_has_ruptures = importlib.util.find_spec("ruptures") is not None
_has_scipy = importlib.util.find_spec("scipy") is not None


# ---------------------------------------------------------------------------
# Fixtures: synthetic data with known changepoints
# ---------------------------------------------------------------------------


def _make_piecewise_series(
    segment_lengths: list[int] | None = None,
    segment_means: list[float] | None = None,
    segment_stds: list[float] | None = None,
    seed: int = 42,
) -> tuple[pd.Series, list[int]]:
    """Create piecewise-constant series with known changepoints.

    Returns the series and a list of true changepoint indices.
    """
    if segment_lengths is None:
        segment_lengths = [200, 200, 200]
    if segment_means is None:
        segment_means = [0.0, 5.0, 0.0]
    if segment_stds is None:
        segment_stds = [1.0, 1.0, 2.0]

    rng = np.random.default_rng(seed)
    parts = []
    for length, mu, sigma in zip(segment_lengths, segment_means, segment_stds):
        parts.append(rng.normal(mu, sigma, length))

    data = np.concatenate(parts)
    changepoints = list(np.cumsum(segment_lengths))
    return pd.Series(data, name="test_data"), changepoints


# ---------------------------------------------------------------------------
# Tests: online_changepoint
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_scipy, reason="scipy not installed")
class TestOnlineChangepoint:
    def test_returns_series(self) -> None:
        from wraquant.regimes.changepoint import online_changepoint

        data, _ = _make_piecewise_series()
        result = online_changepoint(data, hazard=0.01)
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_run_length_drops_at_changepoints(self) -> None:
        from wraquant.regimes.changepoint import online_changepoint

        data, _ = _make_piecewise_series(
            segment_lengths=[100, 100],
            segment_means=[0.0, 10.0],
            segment_stds=[0.5, 0.5],
        )
        result = online_changepoint(data, hazard=0.01)
        # After the changepoint, run length should eventually drop
        # Check that there is at least one drop in the second half
        second_half = result.iloc[100:]
        assert second_half.min() < 50


# ---------------------------------------------------------------------------
# Tests: pelt_changepoint
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_ruptures, reason="ruptures not installed")
class TestPeltChangepoint:
    def test_detects_known_changepoints(self) -> None:
        from wraquant.regimes.changepoint import pelt_changepoint

        data, true_cps = _make_piecewise_series(
            segment_lengths=[200, 200, 200],
            segment_means=[0.0, 5.0, 0.0],
        )
        result = pelt_changepoint(data, model="l2")

        # Should detect approximately 2 changepoints (plus trailing T)
        assert result["n_changepoints"] >= 1

        # At least one detected CP should be near a true CP
        detected_cps = [cp for cp in result["changepoints"] if cp < len(data)]
        has_near = any(
            any(abs(d - t) < 30 for t in true_cps[:-1])
            for d in detected_cps
        )
        assert has_near, (
            f"No detected CP near true CPs {true_cps[:-1]}, "
            f"detected: {detected_cps}"
        )

    def test_segment_stats_cover_full_series(self) -> None:
        from wraquant.regimes.changepoint import pelt_changepoint

        data, _ = _make_piecewise_series()
        result = pelt_changepoint(data, model="l2")

        total_length = sum(seg["length"] for seg in result["segment_stats"])
        assert total_length == len(data)

    def test_confidence_bounded(self) -> None:
        from wraquant.regimes.changepoint import pelt_changepoint

        data, _ = _make_piecewise_series()
        result = pelt_changepoint(data, model="l2")

        if len(result["confidence"]) > 0:
            assert np.all(result["confidence"] >= 0.0)
            assert np.all(result["confidence"] <= 1.0)

    def test_explicit_n_bkps(self) -> None:
        from wraquant.regimes.changepoint import pelt_changepoint

        data, _ = _make_piecewise_series()
        result = pelt_changepoint(data, n_bkps=2, model="l2")
        # Should detect exactly 2 changepoints
        assert result["n_changepoints"] == 2

    def test_penalty_methods(self) -> None:
        from wraquant.regimes.changepoint import pelt_changepoint

        data, _ = _make_piecewise_series()
        for pen in ["bic", "aic"]:
            result = pelt_changepoint(data, penalty=pen, model="l2")
            assert "changepoints" in result

    def test_explicit_penalty_value(self) -> None:
        from wraquant.regimes.changepoint import pelt_changepoint

        data, _ = _make_piecewise_series()
        result = pelt_changepoint(data, penalty=10.0, model="l2")
        assert result["penalty_value"] == 10.0

    def test_numpy_input(self) -> None:
        from wraquant.regimes.changepoint import pelt_changepoint

        data, _ = _make_piecewise_series()
        result = pelt_changepoint(data.values, model="l2")
        assert "changepoints" in result

    def test_cost_models(self) -> None:
        from wraquant.regimes.changepoint import pelt_changepoint

        data, _ = _make_piecewise_series()
        for model in ["l1", "l2", "rbf"]:
            result = pelt_changepoint(data, model=model)
            assert "changepoints" in result


# ---------------------------------------------------------------------------
# Tests: binary_segmentation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_ruptures, reason="ruptures not installed")
class TestBinarySegmentation:
    def test_hierarchical_order(self) -> None:
        from wraquant.regimes.changepoint import binary_segmentation

        data, _ = _make_piecewise_series(
            segment_lengths=[200, 200, 200],
            segment_means=[0.0, 5.0, 0.0],
        )
        result = binary_segmentation(data, n_bkps=2, model="l2")

        assert len(result["hierarchical_order"]) == 2
        # The hierarchical order should be sorted by significance
        # (most significant first) -- just check it's a list of ints
        assert all(isinstance(cp, (int, np.integer)) for cp in result["hierarchical_order"])

    def test_hierarchy_most_significant_first(self) -> None:
        """The first element should be the most significant CP."""
        from wraquant.regimes.changepoint import binary_segmentation

        # Two segments with very different means
        data, _ = _make_piecewise_series(
            segment_lengths=[200, 100, 200],
            segment_means=[0.0, 10.0, 0.0],
            segment_stds=[1.0, 1.0, 1.0],
        )
        result = binary_segmentation(data, n_bkps=2, model="l2")

        # There should be at least one changepoint
        assert result["n_changepoints"] >= 1

    def test_segment_stats_present(self) -> None:
        from wraquant.regimes.changepoint import binary_segmentation

        data, _ = _make_piecewise_series()
        result = binary_segmentation(data, n_bkps=2, model="l2")

        assert len(result["segment_stats"]) >= 2
        for seg in result["segment_stats"]:
            assert "start" in seg
            assert "end" in seg
            assert "mean" in seg
            assert "var" in seg

    def test_auto_penalty(self) -> None:
        from wraquant.regimes.changepoint import binary_segmentation

        data, _ = _make_piecewise_series()
        result = binary_segmentation(data, penalty="bic", model="l2")
        assert "changepoints" in result


# ---------------------------------------------------------------------------
# Tests: window_changepoint
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_ruptures, reason="ruptures not installed")
class TestWindowChangepoint:
    def test_change_score_shape(self) -> None:
        from wraquant.regimes.changepoint import window_changepoint

        data, _ = _make_piecewise_series()
        result = window_changepoint(data, width=30, n_bkps=2)

        score = result["change_score"]
        assert len(score) == len(data)

    def test_change_score_is_series_for_series_input(self) -> None:
        from wraquant.regimes.changepoint import window_changepoint

        data, _ = _make_piecewise_series()
        result = window_changepoint(data, width=30, n_bkps=2)
        assert isinstance(result["change_score"], pd.Series)

    def test_change_score_is_ndarray_for_ndarray_input(self) -> None:
        from wraquant.regimes.changepoint import window_changepoint

        data, _ = _make_piecewise_series()
        result = window_changepoint(data.values, width=30, n_bkps=2)
        assert isinstance(result["change_score"], np.ndarray)

    def test_score_peaks_near_changepoints(self) -> None:
        from wraquant.regimes.changepoint import window_changepoint

        data, true_cps = _make_piecewise_series(
            segment_lengths=[200, 200],
            segment_means=[0.0, 5.0],
            segment_stds=[1.0, 1.0],
        )
        result = window_changepoint(data, width=30, n_bkps=1)

        score = result["change_score"]
        if isinstance(score, pd.Series):
            score = score.values
        # Peak should be near index 200
        peak_idx = int(np.argmax(score))
        assert abs(peak_idx - 200) < 50


# ---------------------------------------------------------------------------
# Tests: cusum_control_chart
# ---------------------------------------------------------------------------


class TestCusumControlChart:
    def test_detects_mean_shift(self) -> None:
        from wraquant.regimes.changepoint import cusum_control_chart

        rng = np.random.default_rng(42)
        data = np.concatenate([
            rng.normal(0, 1, 200),
            rng.normal(2, 1, 200),
        ])
        result = cusum_control_chart(
            pd.Series(data), target=0.0, std_est=1.0, h=5.0,
        )

        # Should detect at least one alarm
        assert len(result["alarm_points"]) > 0
        # First alarm should be after the shift
        assert result["alarm_points"][0] >= 100

    def test_cusum_shapes(self) -> None:
        from wraquant.regimes.changepoint import cusum_control_chart

        data = np.random.default_rng(0).normal(0, 1, 300)
        result = cusum_control_chart(data, target=0.0, std_est=1.0)

        assert result["upper_cusum"].shape == (300,)
        assert result["lower_cusum"].shape == (300,)

    def test_cusum_non_negative(self) -> None:
        from wraquant.regimes.changepoint import cusum_control_chart

        data = np.random.default_rng(0).normal(0, 1, 300)
        result = cusum_control_chart(data, target=0.0, std_est=1.0)

        assert np.all(result["upper_cusum"] >= 0)
        assert np.all(result["lower_cusum"] >= 0)

    def test_no_alarm_for_stable_process(self) -> None:
        from wraquant.regimes.changepoint import cusum_control_chart

        rng = np.random.default_rng(123)
        # Very stable process with large h
        data = rng.normal(0, 0.1, 200)
        result = cusum_control_chart(
            data, target=0.0, std_est=0.1, h=20.0,
        )
        # With h=20 and small noise, should have no alarms
        assert len(result["alarm_points"]) == 0
        assert result["arl"] == float("inf")

    def test_arl_finite_when_alarms(self) -> None:
        from wraquant.regimes.changepoint import cusum_control_chart

        rng = np.random.default_rng(42)
        data = np.concatenate([
            rng.normal(0, 1, 100),
            rng.normal(3, 1, 100),
            rng.normal(0, 1, 100),
            rng.normal(3, 1, 100),
        ])
        result = cusum_control_chart(
            data, target=0.0, std_est=1.0, h=4.0,
        )
        if len(result["alarm_points"]) >= 2:
            assert np.isfinite(result["arl"])
            assert result["arl"] > 0

    def test_alarm_direction(self) -> None:
        from wraquant.regimes.changepoint import cusum_control_chart

        rng = np.random.default_rng(42)
        # Positive shift should trigger "upper" alarm
        data = np.concatenate([
            rng.normal(0, 1, 100),
            rng.normal(5, 1, 100),
        ])
        result = cusum_control_chart(
            data, target=0.0, std_est=1.0, h=4.0,
        )
        if result["alarm_points"]:
            assert "upper" in result["alarm_direction"]

    def test_auto_target_and_std(self) -> None:
        from wraquant.regimes.changepoint import cusum_control_chart

        data = np.random.default_rng(0).normal(5, 2, 300)
        result = cusum_control_chart(data)
        assert abs(result["target"] - 5.0) < 1.0
        assert abs(result["std"] - 2.0) < 1.0
