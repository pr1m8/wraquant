"""Tests for descriptive statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.stats.descriptive import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    max_drawdown,
    omega_ratio,
    summary_stats,
)


def _make_returns(n: int = 252, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0.0005, 0.02, size=n), index=dates, name="returns")


def _make_prices(n: int = 252, seed: int = 42) -> pd.Series:
    returns = _make_returns(n, seed)
    prices = 100 * (1 + returns).cumprod()
    return prices.rename("price")


class TestSummaryStats:
    def test_keys(self) -> None:
        result = summary_stats(_make_returns())
        expected_keys = {"mean", "std", "skew", "kurtosis", "min", "max", "count"}
        assert set(result.keys()) == expected_keys

    def test_count_matches_length(self) -> None:
        ret = _make_returns(100)
        result = summary_stats(ret)
        assert result["count"] == 100

    def test_mean_is_float(self) -> None:
        result = summary_stats(_make_returns())
        assert isinstance(result["mean"], float)


class TestAnnualizedReturn:
    def test_positive_drift(self) -> None:
        # With positive mean returns, annualized should be positive
        rng = np.random.default_rng(42)
        ret = pd.Series(rng.normal(0.001, 0.01, size=252))
        ann = annualized_return(ret)
        assert ann > 0

    def test_scaling(self) -> None:
        ret = _make_returns()
        ann_252 = annualized_return(ret, periods_per_year=252)
        ann_12 = annualized_return(ret, periods_per_year=12)
        # Annual returns should differ for different scaling
        assert ann_252 != ann_12


class TestAnnualizedVolatility:
    def test_positive(self) -> None:
        vol = annualized_volatility(_make_returns())
        assert vol > 0

    def test_scales_with_sqrt(self) -> None:
        ret = _make_returns()
        vol_252 = annualized_volatility(ret, periods_per_year=252)
        vol_1 = annualized_volatility(ret, periods_per_year=1)
        assert vol_252 > vol_1


class TestMaxDrawdown:
    def test_negative(self) -> None:
        prices = _make_prices()
        mdd = max_drawdown(prices)
        assert mdd <= 0

    def test_flat_prices_zero_drawdown(self) -> None:
        flat = pd.Series([100.0] * 50)
        assert max_drawdown(flat) == 0.0

    def test_monotone_up_zero_drawdown(self) -> None:
        up = pd.Series(range(1, 51), dtype=float)
        assert max_drawdown(up) == 0.0


class TestCalmarRatio:
    def test_returns_float(self) -> None:
        ret = _make_returns()
        ratio = calmar_ratio(ret)
        assert isinstance(ratio, float)


class TestOmegaRatio:
    def test_all_positive_returns(self) -> None:
        ret = pd.Series([0.01, 0.02, 0.03, 0.01])
        assert omega_ratio(ret) == float("inf")

    def test_mixed_returns(self) -> None:
        ret = pd.Series([0.01, -0.01, 0.02, -0.005])
        ratio = omega_ratio(ret)
        assert ratio > 0
        assert np.isfinite(ratio)


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------


class TestRollingSharpe:
    def test_finite_values(self) -> None:
        from wraquant.stats.descriptive import rolling_sharpe

        ret = _make_returns()
        rs = rolling_sharpe(ret, window=60)
        finite = rs.dropna()
        assert len(finite) > 0
        assert np.all(np.isfinite(finite.values))

    def test_length(self) -> None:
        from wraquant.stats.descriptive import rolling_sharpe

        ret = _make_returns()
        rs = rolling_sharpe(ret, window=60)
        assert len(rs) == len(ret)

    def test_nan_before_window(self) -> None:
        from wraquant.stats.descriptive import rolling_sharpe

        ret = _make_returns()
        rs = rolling_sharpe(ret, window=60)
        # First 59 values should be NaN
        assert rs.iloc[:59].isna().all()


# ---------------------------------------------------------------------------
# Rolling drawdown
# ---------------------------------------------------------------------------


class TestRollingDrawdown:
    def test_non_positive(self) -> None:
        from wraquant.stats.descriptive import rolling_drawdown

        ret = _make_returns()
        rd = rolling_drawdown(ret, window=60)
        assert (rd.dropna() <= 0).all()

    def test_length(self) -> None:
        from wraquant.stats.descriptive import rolling_drawdown

        ret = _make_returns()
        rd = rolling_drawdown(ret, window=60)
        assert len(rd) == len(ret)


# ---------------------------------------------------------------------------
# Return attribution
# ---------------------------------------------------------------------------


class TestReturnAttribution:
    def test_sums_to_excess(self) -> None:
        from wraquant.stats.descriptive import return_attribution

        pw = pd.Series({"Tech": 0.4, "Fin": 0.3, "Health": 0.3})
        bw = pd.Series({"Tech": 0.3, "Fin": 0.4, "Health": 0.3})
        pr = pd.Series({"Tech": 0.05, "Fin": 0.02, "Health": 0.03})
        br = pd.Series({"Tech": 0.04, "Fin": 0.03, "Health": 0.03})

        result = return_attribution(pw, bw, pr, br)

        # Components should sum to total excess
        component_sum = result["allocation"] + result["selection"] + result["interaction"]
        np.testing.assert_allclose(component_sum, result["total_excess"], atol=1e-10)

        # Total excess should match portfolio return minus benchmark return
        port_ret = float(pw @ pr)
        bench_ret = float(bw @ br)
        np.testing.assert_allclose(result["total_excess"], port_ret - bench_ret, atol=1e-10)

    def test_keys(self) -> None:
        from wraquant.stats.descriptive import return_attribution

        pw = pd.Series({"A": 0.5, "B": 0.5})
        bw = pd.Series({"A": 0.5, "B": 0.5})
        pr = pd.Series({"A": 0.01, "B": 0.02})
        br = pd.Series({"A": 0.01, "B": 0.02})

        result = return_attribution(pw, bw, pr, br)
        assert set(result.keys()) == {"allocation", "selection", "interaction", "total_excess", "detail"}


# ---------------------------------------------------------------------------
# Risk contribution
# ---------------------------------------------------------------------------


class TestRiskContribution:
    def test_sums_to_portfolio_std(self) -> None:
        from wraquant.stats.descriptive import risk_contribution

        w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        cov = pd.DataFrame(
            np.diag([0.04, 0.09, 0.01]),
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        rc = risk_contribution(w, cov)
        port_std = np.sqrt(float(w.values @ cov.values @ w.values))
        np.testing.assert_allclose(rc.sum(), port_std, atol=1e-10)

    def test_returns_series(self) -> None:
        from wraquant.stats.descriptive import risk_contribution

        w = pd.Series({"A": 0.6, "B": 0.4})
        cov = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        rc = risk_contribution(w, cov)
        assert isinstance(rc, pd.Series)
        assert list(rc.index) == ["A", "B"]
