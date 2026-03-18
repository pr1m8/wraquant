"""Tests for wraquant.ta.volume module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.volume import (
    accumulation_distribution_oscillator,
    ad_line,
    adosc,
    cmf,
    elder_force,
    eom,
    force_index,
    klinger,
    mfi,
    nvi,
    obv,
    positive_volume_index,
    pvt,
    pvi,
    taker_buy_ratio,
    volume_profile,
    volume_roc,
    vpt,
    vpt_smoothed,
    vwma,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv() -> dict[str, pd.Series]:
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 10000, size=n).astype(float)
    return {
        "high": pd.Series(high, name="high"),
        "low": pd.Series(low, name="low"),
        "close": pd.Series(close, name="close"),
        "volume": pd.Series(volume, name="volume"),
    }


# ---------------------------------------------------------------------------
# OBV
# ---------------------------------------------------------------------------


class TestOBV:
    def test_direction_with_known_data(self) -> None:
        """OBV should increase on up days, decrease on down days."""
        close = pd.Series([10, 11, 10, 12, 11.0])
        volume = pd.Series([100, 200, 150, 300, 250.0])
        result = obv(close, volume)
        # Day 0: 0 (baseline)
        # Day 1: up → +200
        # Day 2: down → 200 - 150 = 50
        # Day 3: up → 50 + 300 = 350
        # Day 4: down → 350 - 250 = 100
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 200.0
        assert result.iloc[2] == 50.0
        assert result.iloc[3] == 350.0
        assert result.iloc[4] == 100.0

    def test_obv_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = obv(ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# CMF
# ---------------------------------------------------------------------------


class TestCMF:
    def test_bounds(self, ohlcv: dict[str, pd.Series]) -> None:
        """CMF should be in [-1, 1]."""
        result = cmf(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        valid = result.dropna()
        assert (valid >= -1 - 1e-10).all()
        assert (valid <= 1 + 1e-10).all()

    def test_cmf_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = cmf(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# MFI
# ---------------------------------------------------------------------------


class TestMFI:
    def test_bounds(self, ohlcv: dict[str, pd.Series]) -> None:
        """MFI should be in [0, 100]."""
        result = mfi(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        valid = result.dropna()
        assert (valid >= -1e-10).all()
        assert (valid <= 100 + 1e-10).all()

    def test_mfi_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = mfi(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# AD Line / ADOSC
# ---------------------------------------------------------------------------


class TestADLine:
    def test_ad_line_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = ad_line(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])


class TestADOSC:
    def test_adosc_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = adosc(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# Other Volume Indicators
# ---------------------------------------------------------------------------


class TestEOM:
    def test_eom_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = eom(ohlcv["high"], ohlcv["low"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])


class TestForceIndex:
    def test_force_index_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = force_index(ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])


class TestNVIPVI:
    def test_nvi_starts_at_1000(self, ohlcv: dict[str, pd.Series]) -> None:
        result = nvi(ohlcv["close"], ohlcv["volume"])
        assert result.iloc[0] == 1000.0

    def test_pvi_starts_at_1000(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pvi(ohlcv["close"], ohlcv["volume"])
        assert result.iloc[0] == 1000.0


class TestVPT:
    def test_vpt_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = vpt(ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# VWMA
# ---------------------------------------------------------------------------


class TestVWMA:
    def test_vwma_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = vwma(ohlcv["close"], ohlcv["volume"], period=5)
        assert len(result) == len(ohlcv["close"])

    def test_vwma_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = vwma(ohlcv["close"], ohlcv["volume"])
        assert isinstance(result, pd.Series)
        assert result.name == "vwma"

    def test_vwma_equals_sma_for_uniform_volume(self) -> None:
        """With uniform volume, VWMA should equal simple SMA."""
        close = pd.Series([10, 20, 30, 40, 50.0])
        volume = pd.Series([1, 1, 1, 1, 1.0])
        result = vwma(close, volume, period=3)
        sma = close.rolling(3).mean()
        pd.testing.assert_series_equal(
            result.dropna().reset_index(drop=True),
            sma.dropna().reset_index(drop=True),
            check_names=False,
        )

    def test_vwma_hand_computed(self) -> None:
        close = pd.Series([10, 20, 30.0])
        volume = pd.Series([1, 2, 3.0])
        result = vwma(close, volume, period=3)
        # (10*1 + 20*2 + 30*3) / (1+2+3) = (10+40+90)/6 = 140/6
        expected = (10 * 1 + 20 * 2 + 30 * 3) / (1 + 2 + 3)
        assert abs(result.iloc[2] - expected) < 1e-10

    def test_vwma_invalid_period(self) -> None:
        close = pd.Series([1, 2, 3.0])
        volume = pd.Series([10, 20, 30.0])
        with pytest.raises(ValueError):
            vwma(close, volume, period=0)

    def test_vwma_invalid_input(self) -> None:
        with pytest.raises(TypeError):
            vwma([1, 2, 3], pd.Series([10, 20, 30.0]))


# ---------------------------------------------------------------------------
# PVT
# ---------------------------------------------------------------------------


class TestPVT:
    def test_pvt_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pvt(ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])

    def test_pvt_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = pvt(ohlcv["close"], ohlcv["volume"])
        assert isinstance(result, pd.Series)
        assert result.name == "pvt"

    def test_pvt_hand_computed(self) -> None:
        close = pd.Series([100, 110, 105.0])
        volume = pd.Series([1000, 2000, 1500.0])
        result = pvt(close, volume)
        # pct_change: NaN, 0.1, -0.04545...
        # volume * pct: NaN, 200, -68.18...
        # cumsum: NaN, 200, 131.818...
        pct1 = (110 - 100) / 100.0
        pct2 = (105 - 110) / 110.0
        expected_1 = 2000 * pct1
        expected_2 = expected_1 + 1500 * pct2
        assert abs(result.iloc[1] - expected_1) < 1e-10
        assert abs(result.iloc[2] - expected_2) < 1e-10

    def test_pvt_matches_vpt(self, ohlcv: dict[str, pd.Series]) -> None:
        """PVT and VPT should produce identical values (just different names)."""
        result_pvt = pvt(ohlcv["close"], ohlcv["volume"])
        result_vpt = vpt(ohlcv["close"], ohlcv["volume"])
        pd.testing.assert_series_equal(
            result_pvt, result_vpt, check_names=False
        )


# ---------------------------------------------------------------------------
# VPT Smoothed
# ---------------------------------------------------------------------------


class TestVPTSmoothed:
    def test_vpt_smoothed_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = vpt_smoothed(ohlcv["close"], ohlcv["volume"], period=5)
        assert len(result) == len(ohlcv["close"])

    def test_vpt_smoothed_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = vpt_smoothed(ohlcv["close"], ohlcv["volume"])
        assert isinstance(result, pd.Series)
        assert result.name == "vpt_smoothed"

    def test_vpt_smoothed_less_volatile(self, ohlcv: dict[str, pd.Series]) -> None:
        """Smoothed VPT should have smaller standard deviation than raw VPT."""
        raw = vpt(ohlcv["close"], ohlcv["volume"])
        smoothed = vpt_smoothed(ohlcv["close"], ohlcv["volume"], period=10)
        raw_valid = raw.dropna()
        smooth_valid = smoothed.dropna()
        # Compare std on the diff to account for trending nature
        assert smooth_valid.diff().std() < raw_valid.diff().std()

    def test_vpt_smoothed_invalid_period(self) -> None:
        close = pd.Series([1, 2, 3.0])
        volume = pd.Series([10, 20, 30.0])
        with pytest.raises(ValueError):
            vpt_smoothed(close, volume, period=0)


# ---------------------------------------------------------------------------
# Klinger Volume Oscillator
# ---------------------------------------------------------------------------


class TestKlinger:
    def test_klinger_output_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = klinger(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert isinstance(result, dict)
        assert "kvo" in result
        assert "signal" in result

    def test_klinger_output_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = klinger(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert len(result["kvo"]) == len(ohlcv["close"])
        assert len(result["signal"]) == len(ohlcv["close"])

    def test_klinger_series_names(self, ohlcv: dict[str, pd.Series]) -> None:
        result = klinger(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert result["kvo"].name == "kvo"
        assert result["signal"].name == "kvo_signal"

    def test_klinger_invalid_period(self, ohlcv: dict[str, pd.Series]) -> None:
        with pytest.raises(ValueError):
            klinger(
                ohlcv["high"],
                ohlcv["low"],
                ohlcv["close"],
                ohlcv["volume"],
                fast=0,
            )

    def test_klinger_type_validation(self) -> None:
        with pytest.raises(TypeError):
            klinger([1], pd.Series([2]), pd.Series([3]), pd.Series([4]))


# ---------------------------------------------------------------------------
# Taker Buy Ratio
# ---------------------------------------------------------------------------


class TestTakerBuyRatio:
    def test_taker_buy_ratio_length(self) -> None:
        buy = pd.Series([50, 60, 40.0])
        total = pd.Series([100, 100, 100.0])
        result = taker_buy_ratio(buy, total)
        assert len(result) == 3

    def test_taker_buy_ratio_type(self) -> None:
        buy = pd.Series([50.0])
        total = pd.Series([100.0])
        result = taker_buy_ratio(buy, total)
        assert isinstance(result, pd.Series)
        assert result.name == "taker_buy_ratio"

    def test_taker_buy_ratio_hand_computed(self) -> None:
        buy = pd.Series([50, 60, 40.0])
        total = pd.Series([100, 100, 100.0])
        result = taker_buy_ratio(buy, total)
        assert result.iloc[0] == pytest.approx(0.5)
        assert result.iloc[1] == pytest.approx(0.6)
        assert result.iloc[2] == pytest.approx(0.4)

    def test_taker_buy_ratio_zero_total(self) -> None:
        """Division by zero should produce NaN, not error."""
        buy = pd.Series([50, 0.0])
        total = pd.Series([0, 100.0])
        result = taker_buy_ratio(buy, total)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(0.0)

    def test_taker_buy_ratio_invalid_input(self) -> None:
        with pytest.raises(TypeError):
            taker_buy_ratio([50], pd.Series([100.0]))


# ---------------------------------------------------------------------------
# Elder's Force Index
# ---------------------------------------------------------------------------


class TestElderForce:
    def test_elder_force_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = elder_force(ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])

    def test_elder_force_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = elder_force(ohlcv["close"], ohlcv["volume"])
        assert isinstance(result, pd.Series)
        assert result.name == "elder_force"

    def test_elder_force_default_period(self, ohlcv: dict[str, pd.Series]) -> None:
        """Default period for elder_force should be 2."""
        result_default = elder_force(ohlcv["close"], ohlcv["volume"])
        result_p2 = elder_force(ohlcv["close"], ohlcv["volume"], period=2)
        pd.testing.assert_series_equal(result_default, result_p2)

    def test_elder_force_hand_computed(self) -> None:
        close = pd.Series([10, 12, 11.0])
        volume = pd.Series([100, 200, 150.0])
        result = elder_force(close, volume, period=2)
        # raw = diff(close) * volume: NaN, 2*200=400, -1*150=-150
        # EMA(span=2, adjust=False): alpha = 2/(2+1) = 2/3
        # EMA[1] = 400 (first valid)
        # EMA[2] = 2/3 * (-150) + 1/3 * 400 = -100 + 133.33 = 33.33
        assert abs(result.iloc[2] - 33.333333) < 0.01

    def test_elder_force_invalid_period(self) -> None:
        close = pd.Series([1, 2.0])
        volume = pd.Series([10, 20.0])
        with pytest.raises(ValueError):
            elder_force(close, volume, period=0)


# ---------------------------------------------------------------------------
# Volume Profile
# ---------------------------------------------------------------------------


class TestVolumeProfile:
    def test_volume_profile_output_type(self) -> None:
        close = pd.Series([10, 11, 12, 11, 10.0])
        volume = pd.Series([100, 200, 300, 200, 100.0])
        result = volume_profile(close, volume, bins=3)
        assert isinstance(result, dict)
        assert "price_bins" in result
        assert "volume" in result

    def test_volume_profile_bin_count(self) -> None:
        close = pd.Series([10, 11, 12, 11, 10.0])
        volume = pd.Series([100, 200, 300, 200, 100.0])
        result = volume_profile(close, volume, bins=5)
        assert len(result["price_bins"]) == 5
        assert len(result["volume"]) == 5

    def test_volume_profile_total_volume(self) -> None:
        """Total volume across bins should equal sum of input volume."""
        close = pd.Series([10, 11, 12, 11, 10.0])
        volume = pd.Series([100, 200, 300, 200, 100.0])
        result = volume_profile(close, volume, bins=3)
        assert result["volume"].sum() == pytest.approx(volume.sum())

    def test_volume_profile_single_bin(self) -> None:
        close = pd.Series([10, 11, 12.0])
        volume = pd.Series([100, 200, 300.0])
        result = volume_profile(close, volume, bins=1)
        assert len(result["volume"]) == 1
        assert result["volume"].iloc[0] == pytest.approx(600.0)

    def test_volume_profile_identical_prices(self) -> None:
        """All prices are the same — should produce a single-element result."""
        close = pd.Series([10, 10, 10.0])
        volume = pd.Series([100, 200, 300.0])
        result = volume_profile(close, volume, bins=5)
        assert result["volume"].sum() == pytest.approx(600.0)

    def test_volume_profile_invalid_bins(self) -> None:
        close = pd.Series([10, 11.0])
        volume = pd.Series([100, 200.0])
        with pytest.raises(ValueError):
            volume_profile(close, volume, bins=0)

    def test_volume_profile_series_names(self) -> None:
        close = pd.Series([10, 11, 12.0])
        volume = pd.Series([100, 200, 300.0])
        result = volume_profile(close, volume, bins=2)
        assert result["price_bins"].name == "price_bins"
        assert result["volume"].name == "volume"


# ---------------------------------------------------------------------------
# Accumulation/Distribution Oscillator
# ---------------------------------------------------------------------------


class TestAccumulationDistributionOscillator:
    def test_ad_oscillator_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = accumulation_distribution_oscillator(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert len(result) == len(ohlcv["close"])

    def test_ad_oscillator_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = accumulation_distribution_oscillator(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        assert isinstance(result, pd.Series)
        assert result.name == "ad_oscillator"

    def test_ad_oscillator_matches_adosc_values(
        self, ohlcv: dict[str, pd.Series]
    ) -> None:
        """Should produce the same values as adosc (Chaikin Oscillator)."""
        result_ad = accumulation_distribution_oscillator(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        result_ch = adosc(
            ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        )
        pd.testing.assert_series_equal(result_ad, result_ch, check_names=False)

    def test_ad_oscillator_custom_periods(self, ohlcv: dict[str, pd.Series]) -> None:
        result = accumulation_distribution_oscillator(
            ohlcv["high"],
            ohlcv["low"],
            ohlcv["close"],
            ohlcv["volume"],
            fast=5,
            slow=20,
        )
        assert len(result) == len(ohlcv["close"])


# ---------------------------------------------------------------------------
# Volume ROC
# ---------------------------------------------------------------------------


class TestVolumeROC:
    def test_volume_roc_length(self, ohlcv: dict[str, pd.Series]) -> None:
        result = volume_roc(ohlcv["volume"])
        assert len(result) == len(ohlcv["volume"])

    def test_volume_roc_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = volume_roc(ohlcv["volume"])
        assert isinstance(result, pd.Series)
        assert result.name == "volume_roc"

    def test_volume_roc_hand_computed(self) -> None:
        volume = pd.Series([100, 120, 150, 130.0])
        result = volume_roc(volume, period=2)
        # index 2: (150 - 100) / 100 * 100 = 50.0
        # index 3: (130 - 120) / 120 * 100 = 8.333...
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(50.0)
        assert result.iloc[3] == pytest.approx(100.0 * (130 - 120) / 120)

    def test_volume_roc_invalid_period(self) -> None:
        volume = pd.Series([100, 200.0])
        with pytest.raises(ValueError):
            volume_roc(volume, period=0)

    def test_volume_roc_all_same_volume(self) -> None:
        """Constant volume should produce 0% ROC."""
        volume = pd.Series([100.0] * 10)
        result = volume_roc(volume, period=3)
        valid = result.dropna()
        assert (valid == 0.0).all()


# ---------------------------------------------------------------------------
# Positive Volume Index (descriptive alias)
# ---------------------------------------------------------------------------


class TestPositiveVolumeIndex:
    def test_positive_volume_index_starts_at_1000(self) -> None:
        close = pd.Series([10, 11, 10, 12, 11.0])
        volume = pd.Series([100, 200, 150, 300, 100.0])
        result = positive_volume_index(close, volume)
        assert result.iloc[0] == 1000.0

    def test_positive_volume_index_type(self, ohlcv: dict[str, pd.Series]) -> None:
        result = positive_volume_index(ohlcv["close"], ohlcv["volume"])
        assert isinstance(result, pd.Series)
        assert result.name == "positive_volume_index"

    def test_positive_volume_index_matches_pvi_values(
        self, ohlcv: dict[str, pd.Series]
    ) -> None:
        """Should produce the same values as pvi (just different name)."""
        result_pvi_alias = positive_volume_index(ohlcv["close"], ohlcv["volume"])
        result_pvi = pvi(ohlcv["close"], ohlcv["volume"])
        pd.testing.assert_series_equal(
            result_pvi_alias, result_pvi, check_names=False
        )

    def test_positive_volume_index_length(
        self, ohlcv: dict[str, pd.Series]
    ) -> None:
        result = positive_volume_index(ohlcv["close"], ohlcv["volume"])
        assert len(result) == len(ohlcv["close"])

    def test_positive_volume_index_only_moves_on_up_volume(self) -> None:
        """PVI should only change on days when volume increases."""
        close = pd.Series([10, 11, 10, 12, 11.0])
        volume = pd.Series([100, 200, 150, 300, 100.0])
        result = positive_volume_index(close, volume)
        # Day 1: vol up (200 > 100) → changes
        # Day 2: vol down (150 < 200) → no change
        # Day 3: vol up (300 > 150) → changes
        # Day 4: vol down (100 < 300) → no change
        assert result.iloc[2] == result.iloc[1]  # no change on down-vol day
        assert result.iloc[4] == result.iloc[3]  # no change on down-vol day
