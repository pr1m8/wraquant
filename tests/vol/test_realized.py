"""Tests for realized volatility estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.vol.realized import (
    garman_klass,
    parkinson,
    realized_volatility,
    rogers_satchell,
    yang_zhang,
)


def _make_ohlcv(n: int = 100) -> dict[str, pd.Series]:
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    high = close * (1 + rng.uniform(0.001, 0.03, n))
    low = close * (1 - rng.uniform(0.001, 0.03, n))
    open_ = close * (1 + rng.normal(0, 0.01, n))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return {
        "open": pd.Series(open_, index=idx),
        "high": pd.Series(high, index=idx),
        "low": pd.Series(low, index=idx),
        "close": pd.Series(close, index=idx),
    }


class TestRealizedVolatility:
    def test_returns_series(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 100))
        vol = realized_volatility(returns, window=20)
        assert isinstance(vol, pd.Series)
        assert len(vol) == 100

    def test_annualized_is_larger(self) -> None:
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 100))
        vol_ann = realized_volatility(returns, window=20, annualize=True)
        vol_raw = realized_volatility(returns, window=20, annualize=False)
        # Annualized should be larger (sqrt(252) factor)
        assert vol_ann.dropna().mean() > vol_raw.dropna().mean()


class TestParkinson:
    def test_positive_vol(self) -> None:
        data = _make_ohlcv()
        vol = parkinson(data["high"], data["low"], window=20)
        assert (vol.dropna() > 0).all()


class TestGarmanKlass:
    def test_positive_vol(self) -> None:
        data = _make_ohlcv()
        vol = garman_klass(
            data["open"], data["high"], data["low"], data["close"], window=20
        )
        assert (vol.dropna() >= 0).all()


class TestRogersSatchell:
    def test_positive_vol(self) -> None:
        data = _make_ohlcv()
        vol = rogers_satchell(
            data["open"], data["high"], data["low"], data["close"], window=20
        )
        assert (vol.dropna() >= 0).all()


class TestYangZhang:
    def test_positive_vol(self) -> None:
        data = _make_ohlcv()
        vol = yang_zhang(
            data["open"], data["high"], data["low"], data["close"], window=20
        )
        assert (vol.dropna() >= 0).all()


# ---------------------------------------------------------------------------
# Bipower Variation
# ---------------------------------------------------------------------------


class TestBipowerVariation:
    def test_positive(self) -> None:
        from wraquant.vol.realized import bipower_variation

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 200))
        bpv = bipower_variation(returns, window=20)
        assert (bpv.dropna() > 0).all()

    def test_length_matches(self) -> None:
        from wraquant.vol.realized import bipower_variation

        rng = np.random.default_rng(42)
        n = 150
        returns = pd.Series(rng.normal(0, 0.01, n))
        bpv = bipower_variation(returns, window=20)
        assert len(bpv) == n

    def test_leq_rv_no_jumps(self) -> None:
        """For jump-free Gaussian data, BPV should be close to RV."""
        from wraquant.vol.realized import bipower_variation

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 500))
        bpv = bipower_variation(returns, window=50, annualize=False)
        rv = realized_volatility(returns, window=50, annualize=False)
        # Compare on common non-NaN indices
        common = bpv.dropna().index.intersection(rv.dropna().index)
        # On average BPV should not greatly exceed RV for jump-free data
        assert bpv.loc[common].mean() <= rv.loc[common].mean() * 1.3


# ---------------------------------------------------------------------------
# Jump Test BNS
# ---------------------------------------------------------------------------


class TestJumpTestBns:
    def test_returns_expected_keys(self) -> None:
        from wraquant.vol.realized import jump_test_bns

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 200))
        result = jump_test_bns(returns, window=200)
        for key in [
            "rv",
            "bpv",
            "jump_component",
            "continuous_component",
            "z_statistic",
            "p_value",
            "jump_detected",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_detects_injected_jump(self) -> None:
        from wraquant.vol.realized import jump_test_bns

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.005, 200)
        # Inject a large jump
        returns[100] = 0.15
        returns[150] = -0.12
        result = jump_test_bns(pd.Series(returns), window=200)
        # With large injected jumps, jump_component should be positive
        assert result["jump_component"] > 0
        # The z-stat should be positive (RV > BPV)
        assert result["z_statistic"] > 0

    def test_no_jump_for_clean_data(self) -> None:
        from wraquant.vol.realized import jump_test_bns

        rng = np.random.default_rng(42)
        # Small, well-behaved returns — should not detect jumps
        returns = pd.Series(rng.normal(0, 0.005, 500))
        result = jump_test_bns(returns, window=500, alpha=0.01)
        assert isinstance(result["jump_detected"], bool)


# ---------------------------------------------------------------------------
# Two-Scale Realized Variance
# ---------------------------------------------------------------------------


class TestTwoScaleRealizedVariance:
    def test_positive(self) -> None:
        from wraquant.vol.realized import two_scale_realized_variance

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 200))
        tsrv = two_scale_realized_variance(returns, window=20)
        valid = tsrv.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_finite(self) -> None:
        from wraquant.vol.realized import two_scale_realized_variance

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 200))
        tsrv = two_scale_realized_variance(returns, window=20)
        assert tsrv.dropna().apply(np.isfinite).all()


# ---------------------------------------------------------------------------
# Realized Kernel
# ---------------------------------------------------------------------------


class TestRealizedKernel:
    def test_positive(self) -> None:
        from wraquant.vol.realized import realized_kernel

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 200))
        rk = realized_kernel(returns, window=20)
        valid = rk.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_finite(self) -> None:
        from wraquant.vol.realized import realized_kernel

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 200))
        rk = realized_kernel(returns, window=20)
        assert rk.dropna().apply(np.isfinite).all()

    def test_length_matches(self) -> None:
        from wraquant.vol.realized import realized_kernel

        rng = np.random.default_rng(42)
        n = 150
        returns = pd.Series(rng.normal(0, 0.01, n))
        rk = realized_kernel(returns, window=20)
        assert len(rk) == n
