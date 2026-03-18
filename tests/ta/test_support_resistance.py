"""Tests for wraquant.ta.support_resistance module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.support_resistance import (
    find_support_resistance,
    fractal_levels,
    price_clustering,
    round_number_levels,
    supply_demand_zones,
    trendline_detection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv() -> dict[str, pd.Series]:
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, size=n).astype(float)
    return {
        "open": pd.Series(open_, name="open"),
        "high": pd.Series(high, name="high"),
        "low": pd.Series(low, name="low"),
        "close": pd.Series(close, name="close"),
        "volume": pd.Series(volume, name="volume"),
    }


# ---------------------------------------------------------------------------
# Find Support / Resistance
# ---------------------------------------------------------------------------


class TestFindSupportResistance:
    def test_output_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = find_support_resistance(ohlcv["high"], ohlcv["low"])
        assert set(result.keys()) == {"support", "resistance"}

    def test_output_types(self, ohlcv: dict[str, pd.Series]) -> None:
        result = find_support_resistance(ohlcv["high"], ohlcv["low"])
        assert isinstance(result["support"], list)
        assert isinstance(result["resistance"], list)

    def test_levels_sorted(self, ohlcv: dict[str, pd.Series]) -> None:
        result = find_support_resistance(ohlcv["high"], ohlcv["low"])
        assert result["support"] == sorted(result["support"])
        assert result["resistance"] == sorted(result["resistance"])

    def test_max_levels(self, ohlcv: dict[str, pd.Series]) -> None:
        result = find_support_resistance(ohlcv["high"], ohlcv["low"], num_levels=3)
        assert len(result["support"]) <= 3
        assert len(result["resistance"]) <= 3

    def test_type_error(self) -> None:
        with pytest.raises(TypeError):
            find_support_resistance([1], [2])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Price Clustering
# ---------------------------------------------------------------------------


class TestPriceClustering:
    def test_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = price_clustering(ohlcv["close"])
        assert isinstance(result, np.ndarray)

    def test_num_levels(self, ohlcv: dict[str, pd.Series]) -> None:
        result = price_clustering(ohlcv["close"], num_levels=3)
        assert len(result) == 3

    def test_sorted(self, ohlcv: dict[str, pd.Series]) -> None:
        result = price_clustering(ohlcv["close"], num_levels=5)
        assert list(result) == sorted(result)

    def test_empty_series(self) -> None:
        result = price_clustering(pd.Series([], dtype=float))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Fractal Levels
# ---------------------------------------------------------------------------


class TestFractalLevels:
    def test_output_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = fractal_levels(ohlcv["high"], ohlcv["low"])
        assert set(result.keys()) == {"up_fractals", "down_fractals"}

    def test_output_boolean(self, ohlcv: dict[str, pd.Series]) -> None:
        result = fractal_levels(ohlcv["high"], ohlcv["low"])
        assert result["up_fractals"].dtype == bool
        assert result["down_fractals"].dtype == bool

    def test_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = fractal_levels(ohlcv["high"], ohlcv["low"])
        assert len(result["up_fractals"]) == len(ohlcv["high"])
        assert len(result["down_fractals"]) == len(ohlcv["low"])

    def test_edges_false(self, ohlcv: dict[str, pd.Series]) -> None:
        """Fractals cannot occur at the edges."""
        period = 2
        result = fractal_levels(ohlcv["high"], ohlcv["low"], period=period)
        assert not result["up_fractals"].iloc[:period].any()
        assert not result["down_fractals"].iloc[:period].any()

    def test_has_fractals(self, ohlcv: dict[str, pd.Series]) -> None:
        """With 200 bars of random data, there should be some fractals."""
        result = fractal_levels(ohlcv["high"], ohlcv["low"])
        assert result["up_fractals"].sum() > 0
        assert result["down_fractals"].sum() > 0

    def test_type_error(self) -> None:
        with pytest.raises(TypeError):
            fractal_levels([1], [2])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round Number Levels
# ---------------------------------------------------------------------------


class TestRoundNumberLevels:
    def test_sorted(self) -> None:
        result = round_number_levels(105.3, num_levels=3, step=10.0)
        assert result == sorted(result)

    def test_positive_only(self) -> None:
        result = round_number_levels(15.0, num_levels=5, step=10.0)
        assert all(lvl > 0 for lvl in result)

    def test_auto_step(self) -> None:
        result = round_number_levels(105.3, num_levels=3)
        assert len(result) > 0

    def test_invalid_price(self) -> None:
        with pytest.raises(ValueError, match="current_price"):
            round_number_levels(-10.0)

    def test_custom_step(self) -> None:
        result = round_number_levels(100.0, num_levels=2, step=25.0)
        # Should include 50, 75, 100, 125, 150
        assert 100.0 in result


# ---------------------------------------------------------------------------
# Supply / Demand Zones
# ---------------------------------------------------------------------------


class TestSupplyDemandZones:
    def test_output_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = supply_demand_zones(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert set(result.keys()) == {"demand", "supply"}

    def test_output_types(self, ohlcv: dict[str, pd.Series]) -> None:
        result = supply_demand_zones(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert isinstance(result["demand"], list)
        assert isinstance(result["supply"], list)

    def test_zone_structure(self, ohlcv: dict[str, pd.Series]) -> None:
        result = supply_demand_zones(
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        for zone_list in [result["demand"], result["supply"]]:
            for zone in zone_list:
                assert "zone_low" in zone
                assert "zone_high" in zone
                assert "index" in zone
                assert zone["zone_high"] >= zone["zone_low"]

    def test_type_error(self) -> None:
        with pytest.raises(TypeError):
            supply_demand_zones([1], [2], [3], [4])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Trendline Detection
# ---------------------------------------------------------------------------


class TestTrendlineDetection:
    def test_output_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = trendline_detection(ohlcv["high"], ohlcv["low"])
        assert set(result.keys()) == {"resistance_lines", "support_lines"}

    def test_output_types(self, ohlcv: dict[str, pd.Series]) -> None:
        result = trendline_detection(ohlcv["high"], ohlcv["low"])
        assert isinstance(result["resistance_lines"], list)
        assert isinstance(result["support_lines"], list)

    def test_line_structure(self, ohlcv: dict[str, pd.Series]) -> None:
        result = trendline_detection(ohlcv["high"], ohlcv["low"])
        for line_list in [result["resistance_lines"], result["support_lines"]]:
            for line in line_list:
                assert "slope" in line
                assert "intercept" in line
                assert "num_touches" in line

    def test_type_error(self) -> None:
        with pytest.raises(TypeError):
            trendline_detection([1], [2])  # type: ignore[arg-type]
