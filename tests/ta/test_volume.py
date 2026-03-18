"""Tests for wraquant.ta.volume module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.ta.volume import (
    ad_line,
    adosc,
    cmf,
    eom,
    force_index,
    mfi,
    nvi,
    obv,
    pvi,
    vpt,
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
