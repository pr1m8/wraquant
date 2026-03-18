"""Tests for time series anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ts.anomaly import grubbs_test_ts, prophet_anomaly

# isolation_forest_ts requires sklearn (ml extra) -- test conditionally
try:
    from wraquant.ts.anomaly import isolation_forest_ts

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def _make_series_with_anomalies(
    n: int = 500, seed: int = 42, anomaly_positions: list[int] | None = None,
) -> pd.Series:
    """Create a normal series with injected anomalies."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    if anomaly_positions is None:
        anomaly_positions = [n // 5, n * 3 // 5]
    for pos in anomaly_positions:
        if pos < n:
            x[pos] = 20.0 * (1 if rng.random() > 0.5 else -1)
    return pd.Series(x)


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
class TestIsolationForestTS:
    def test_detects_injected_anomalies(self) -> None:
        data = _make_series_with_anomalies(n=500, anomaly_positions=[100, 300])
        result = isolation_forest_ts(data, window=20, contamination=0.05)
        assert result["n_anomalies"] > 0

    def test_anomaly_labels_shape(self) -> None:
        data = _make_series_with_anomalies(n=500)
        result = isolation_forest_ts(data, window=20)
        # Labels should exist for all non-NaN points after rolling window
        assert len(result["anomaly_labels"]) > 0
        assert set(result["anomaly_labels"].unique()).issubset({-1, 1})

    def test_scores_are_series(self) -> None:
        data = _make_series_with_anomalies(n=500)
        result = isolation_forest_ts(data, window=20)
        assert isinstance(result["anomaly_scores"], pd.Series)


# ---------------------------------------------------------------------------
# Forecast-Based Anomaly (Prophet-style)
# ---------------------------------------------------------------------------


class TestProphetAnomaly:
    def test_detects_large_deviations(self) -> None:
        data = _make_series_with_anomalies(n=300, anomaly_positions=[50, 200])
        result = prophet_anomaly(data, k=3.0)
        assert result["n_anomalies"] >= 2

    def test_output_keys(self) -> None:
        data = _make_series_with_anomalies(n=200)
        result = prophet_anomaly(data, k=3.0)
        assert "forecast" in result
        assert "residuals" in result
        assert "anomaly_mask" in result
        assert "upper_bound" in result
        assert "lower_bound" in result
        assert "sigma" in result

    def test_bounds_contain_forecast(self) -> None:
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(0, 1, 200))
        result = prophet_anomaly(data, k=3.0)
        # Upper bound should be above lower bound everywhere
        assert (result["upper_bound"] >= result["lower_bound"]).all()


# ---------------------------------------------------------------------------
# Rolling Grubbs Test
# ---------------------------------------------------------------------------


class TestGrubbsTestTS:
    def test_detects_injected_outlier(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        x[75] = 20.0
        data = pd.Series(x)
        result = grubbs_test_ts(data, window=50)
        assert 75 in result["outlier_indices"]

    def test_output_keys(self) -> None:
        data = _make_series_with_anomalies(n=200)
        result = grubbs_test_ts(data, window=50)
        assert "outlier_mask" in result
        assert "test_statistics" in result
        assert "outlier_indices" in result
        assert "n_outliers" in result

    def test_no_false_positives_on_clean_data(self) -> None:
        """Clean normal data should have very few outliers."""
        rng = np.random.default_rng(42)
        clean = pd.Series(rng.normal(0, 1, 500))
        result = grubbs_test_ts(clean, window=50, significance=0.01)
        # Allow a small number of false positives but not many
        assert result["n_outliers"] < 25  # < 5% of 500
