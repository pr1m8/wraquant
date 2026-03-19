"""Tests for wraquant.ta.exotic module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.exotic import (
    choppiness_index,
    connors_tps,
    directional_movement_index,
    efficiency_ratio,
    elder_thermometer,
    ergodic_oscillator,
    gopalakrishnan_range,
    kairi,
    market_facilitation_index,
    polarized_fractal_efficiency,
    pretty_good_oscillator,
    price_zone_oscillator,
    random_walk_index,
    relative_momentum_index,
    trend_intensity_index,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def close_series() -> pd.Series:
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    return pd.Series(prices, name="close")


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
# Choppiness Index
# ---------------------------------------------------------------------------


class TestChoppinessIndex:
    def test_ci_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = choppiness_index(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(result) == len(ohlcv["close"])

    def test_ci_has_valid_values(self, ohlcv: dict[str, pd.Series]) -> None:
        result = choppiness_index(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert result.notna().sum() > 0

    def test_ci_name(self, ohlcv: dict[str, pd.Series]) -> None:
        result = choppiness_index(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert result.name == "choppiness_index"


# ---------------------------------------------------------------------------
# Random Walk Index
# ---------------------------------------------------------------------------


class TestRandomWalkIndex:
    def test_rwi_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = random_walk_index(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert set(result.keys()) == {"rwi_high", "rwi_low"}

    def test_rwi_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = random_walk_index(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert len(result["rwi_high"]) == len(ohlcv["close"])
        assert len(result["rwi_low"]) == len(ohlcv["close"])

    def test_rwi_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        result = random_walk_index(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        valid_h = result["rwi_high"].dropna()
        valid_l = result["rwi_low"].dropna()
        assert (valid_h >= -1e-10).all()
        assert (valid_l >= -1e-10).all()


# ---------------------------------------------------------------------------
# Polarized Fractal Efficiency
# ---------------------------------------------------------------------------


class TestPolarizedFractalEfficiency:
    def test_pfe_length(self, close_series: pd.Series) -> None:
        result = polarized_fractal_efficiency(close_series)
        assert len(result) == len(close_series)

    def test_pfe_has_valid_values(self, close_series: pd.Series) -> None:
        result = polarized_fractal_efficiency(close_series)
        assert result.notna().sum() > 0

    def test_pfe_name(self, close_series: pd.Series) -> None:
        result = polarized_fractal_efficiency(close_series)
        assert result.name == "pfe"


# ---------------------------------------------------------------------------
# Price Zone Oscillator
# ---------------------------------------------------------------------------


class TestPriceZoneOscillator:
    def test_pzo_length(self, close_series: pd.Series) -> None:
        result = price_zone_oscillator(close_series)
        assert len(result) == len(close_series)

    def test_pzo_has_valid_values(self, close_series: pd.Series) -> None:
        result = price_zone_oscillator(close_series)
        assert result.notna().sum() > 0

    def test_pzo_name(self, close_series: pd.Series) -> None:
        result = price_zone_oscillator(close_series)
        assert result.name == "pzo"


# ---------------------------------------------------------------------------
# Ergodic Oscillator
# ---------------------------------------------------------------------------


class TestErgodicOscillator:
    def test_ergodic_keys(self, close_series: pd.Series) -> None:
        result = ergodic_oscillator(close_series)
        assert set(result.keys()) == {"ergodic", "signal", "histogram"}

    def test_ergodic_length(self, close_series: pd.Series) -> None:
        result = ergodic_oscillator(close_series)
        for key, series in result.items():
            assert len(series) == len(close_series), f"{key} length mismatch"

    def test_ergodic_histogram_is_diff(self, close_series: pd.Series) -> None:
        """Histogram should be ergodic minus signal."""
        result = ergodic_oscillator(close_series)
        expected_hist = result["ergodic"] - result["signal"]
        valid = expected_hist.dropna()
        actual = result["histogram"].loc[valid.index]
        pd.testing.assert_series_equal(
            actual.rename(None), valid.rename(None), atol=1e-10
        )


# ---------------------------------------------------------------------------
# Elder Thermometer
# ---------------------------------------------------------------------------


class TestElderThermometer:
    def test_thermo_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = elder_thermometer(ohlcv["high"], ohlcv["low"])
        assert len(result) == len(ohlcv["close"])

    def test_thermo_name(self, ohlcv: dict[str, pd.Series]) -> None:
        result = elder_thermometer(ohlcv["high"], ohlcv["low"])
        assert result.name == "elder_thermometer"

    def test_thermo_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        """Thermometer values should be non-negative after smoothing."""
        result = elder_thermometer(ohlcv["high"], ohlcv["low"])
        valid = result.dropna()
        assert (valid >= -1e-10).all()


# ---------------------------------------------------------------------------
# Market Facilitation Index
# ---------------------------------------------------------------------------


class TestMarketFacilitationIndex:
    def test_mfi_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = market_facilitation_index(
            ohlcv["high"], ohlcv["low"], ohlcv["volume"]
        )
        assert len(result) == len(ohlcv["close"])

    def test_mfi_name(self, ohlcv: dict[str, pd.Series]) -> None:
        result = market_facilitation_index(
            ohlcv["high"], ohlcv["low"], ohlcv["volume"]
        )
        assert result.name == "mfi_bw"

    def test_mfi_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        """Since high >= low and volume > 0, MFI should be non-negative."""
        result = market_facilitation_index(
            ohlcv["high"], ohlcv["low"], ohlcv["volume"]
        )
        valid = result.dropna()
        assert (valid >= -1e-10).all()


# ---------------------------------------------------------------------------
# Efficiency Ratio
# ---------------------------------------------------------------------------


class TestEfficiencyRatio:
    def test_er_length(self, close_series: pd.Series) -> None:
        result = efficiency_ratio(close_series, period=10)
        assert len(result) == len(close_series)

    def test_er_bounded(self, close_series: pd.Series) -> None:
        result = efficiency_ratio(close_series, period=10)
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 1.0 + 1e-10).all()

    def test_er_name(self, close_series: pd.Series) -> None:
        result = efficiency_ratio(close_series)
        assert result.name == "efficiency_ratio"

    def test_er_on_straight_line(self) -> None:
        """A perfectly linear series should have ER = 1."""
        data = pd.Series(np.arange(1.0, 51.0))
        result = efficiency_ratio(data, period=10)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Trend Intensity Index
# ---------------------------------------------------------------------------


class TestTrendIntensityIndex:
    def test_tii_length(self, close_series: pd.Series) -> None:
        result = trend_intensity_index(close_series, period=30)
        assert len(result) == len(close_series)

    def test_tii_bounded(self, close_series: pd.Series) -> None:
        """TII should be in [-100, 100]."""
        result = trend_intensity_index(close_series, period=30)
        valid = result.dropna()
        assert (valid >= -100.0 - 1e-10).all()
        assert (valid <= 100.0 + 1e-10).all()

    def test_tii_name(self, close_series: pd.Series) -> None:
        result = trend_intensity_index(close_series)
        assert result.name == "tii"


# ---------------------------------------------------------------------------
# Directional Movement Index
# ---------------------------------------------------------------------------


class TestDirectionalMovementIndex:
    def test_dmi_keys(self, ohlcv: dict[str, pd.Series]) -> None:
        result = directional_movement_index(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert set(result.keys()) == {"plus_di", "minus_di"}

    def test_dmi_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = directional_movement_index(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert len(result["plus_di"]) == len(ohlcv["close"])
        assert len(result["minus_di"]) == len(ohlcv["close"])

    def test_dmi_non_negative(self, ohlcv: dict[str, pd.Series]) -> None:
        """DI values should be non-negative."""
        result = directional_movement_index(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        for key in ("plus_di", "minus_di"):
            valid = result[key].dropna()
            assert (valid >= -1e-10).all(), f"{key} has negative values"


# ---------------------------------------------------------------------------
# KAIRI
# ---------------------------------------------------------------------------


class TestKairi:
    def test_kairi_length(self, close_series: pd.Series) -> None:
        result = kairi(close_series, period=14)
        assert len(result) == len(close_series)

    def test_kairi_zero_at_sma(self) -> None:
        """If data is constant, KAIRI should be 0."""
        data = pd.Series([5.0] * 20)
        result = kairi(data, period=10)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)

    def test_kairi_name(self, close_series: pd.Series) -> None:
        result = kairi(close_series)
        assert result.name == "kairi"


# ---------------------------------------------------------------------------
# Gopalakrishnan Range
# ---------------------------------------------------------------------------


class TestGopalakrishnanRange:
    def test_gapo_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = gopalakrishnan_range(ohlcv["high"], ohlcv["low"])
        assert len(result) == len(ohlcv["close"])

    def test_gapo_has_valid_values(self, ohlcv: dict[str, pd.Series]) -> None:
        result = gopalakrishnan_range(ohlcv["high"], ohlcv["low"])
        assert result.notna().sum() > 0

    def test_gapo_name(self, ohlcv: dict[str, pd.Series]) -> None:
        result = gopalakrishnan_range(ohlcv["high"], ohlcv["low"])
        assert result.name == "gapo"


# ---------------------------------------------------------------------------
# Pretty Good Oscillator
# ---------------------------------------------------------------------------


class TestPrettyGoodOscillator:
    def test_pgo_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pretty_good_oscillator(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert len(result) == len(ohlcv["close"])

    def test_pgo_has_valid_values(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pretty_good_oscillator(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert result.notna().sum() > 0

    def test_pgo_name(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pretty_good_oscillator(
            ohlcv["high"], ohlcv["low"], ohlcv["close"]
        )
        assert result.name == "pgo"


# ---------------------------------------------------------------------------
# Connors TPS
# ---------------------------------------------------------------------------


class TestConnorsTPS:
    def test_tps_length(self, close_series: pd.Series) -> None:
        result = connors_tps(close_series)
        assert len(result) == len(close_series)

    def test_tps_has_valid_values(self, close_series: pd.Series) -> None:
        result = connors_tps(close_series)
        assert result.notna().sum() > 0

    def test_tps_name(self, close_series: pd.Series) -> None:
        result = connors_tps(close_series)
        assert result.name == "connors_tps"


# ---------------------------------------------------------------------------
# Relative Momentum Index
# ---------------------------------------------------------------------------


class TestRelativeMomentumIndex:
    def test_rmi_length(self, close_series: pd.Series) -> None:
        result = relative_momentum_index(close_series)
        assert len(result) == len(close_series)

    def test_rmi_bounded(self, close_series: pd.Series) -> None:
        """RMI should be in [0, 100] (where defined)."""
        result = relative_momentum_index(close_series)
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 100.0 + 1e-10).all()

    def test_rmi_name(self, close_series: pd.Series) -> None:
        result = relative_momentum_index(close_series)
        assert result.name == "rmi"

    def test_rmi_different_momentum_periods(self, close_series: pd.Series) -> None:
        """Different momentum periods should yield different results."""
        r1 = relative_momentum_index(close_series, momentum_period=1)
        r2 = relative_momentum_index(close_series, momentum_period=8)
        assert not r1.equals(r2)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_accepts_list_input(self) -> None:
        # Lists are now auto-coerced to pd.Series
        result = choppiness_index(
            list(range(10, 40)), list(range(1, 31)), list(range(5, 35))
        )
        assert isinstance(result, pd.Series)

    def test_invalid_period_raises(self) -> None:
        data = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            kairi(data, period=0)
