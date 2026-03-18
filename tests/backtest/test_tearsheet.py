"""Tests for tearsheet / reporting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.backtest.tearsheet import (
    drawdown_table,
    generate_tearsheet,
    monthly_returns_table,
    rolling_metrics_table,
    trade_analysis,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _daily_returns(n: int = 504, seed: int = 42) -> pd.Series:
    """Two years of synthetic daily returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(rng.normal(0.0004, 0.012, size=n), index=dates, name="strat")


def _benchmark_returns(n: int = 504, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(rng.normal(0.0003, 0.01, size=n), index=dates, name="bench")


def _make_trades(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pnl = rng.normal(50, 200, size=n)
    return pd.DataFrame({"pnl": pnl})


# ------------------------------------------------------------------
# generate_tearsheet
# ------------------------------------------------------------------

class TestGenerateTearsheet:
    def test_basic_keys(self) -> None:
        ts = generate_tearsheet(_daily_returns())
        expected_keys = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "skewness",
            "kurtosis",
            "var_95",
            "cvar_95",
            "win_rate",
            "profit_factor",
            "n_periods",
        }
        assert expected_keys.issubset(ts.keys())

    def test_max_drawdown_is_negative(self) -> None:
        ts = generate_tearsheet(_daily_returns())
        assert ts["max_drawdown"] <= 0

    def test_with_benchmark(self) -> None:
        ts = generate_tearsheet(
            _daily_returns(), benchmark=_benchmark_returns()
        )
        assert "beta" in ts
        assert "alpha" in ts
        assert "tracking_error" in ts
        assert "information_ratio" in ts
        assert "up_capture" in ts
        assert "down_capture" in ts

    def test_positive_sharpe_for_positive_drift(self) -> None:
        rets = pd.Series(
            np.full(252, 0.001),
            index=pd.bdate_range("2022-01-03", periods=252),
        )
        ts = generate_tearsheet(rets)
        assert ts["sharpe_ratio"] > 0

    def test_empty_returns(self) -> None:
        ts = generate_tearsheet(pd.Series(dtype=float))
        assert ts["n_periods"] == 0


# ------------------------------------------------------------------
# monthly_returns_table
# ------------------------------------------------------------------

class TestMonthlyReturnsTable:
    def test_shape(self) -> None:
        rets = _daily_returns()
        tbl = monthly_returns_table(rets)
        assert isinstance(tbl, pd.DataFrame)
        # At least one year, columns should be months
        assert tbl.shape[1] <= 12

    def test_empty(self) -> None:
        tbl = monthly_returns_table(pd.Series(dtype=float))
        assert isinstance(tbl, pd.DataFrame)
        assert tbl.empty

    def test_values_are_finite(self) -> None:
        tbl = monthly_returns_table(_daily_returns())
        assert tbl.notna().all().all()


# ------------------------------------------------------------------
# drawdown_table
# ------------------------------------------------------------------

class TestDrawdownTable:
    def test_returns_dataframe(self) -> None:
        dt = drawdown_table(_daily_returns())
        assert isinstance(dt, pd.DataFrame)
        assert "depth" in dt.columns

    def test_top_n_limit(self) -> None:
        dt = drawdown_table(_daily_returns(), top_n=3)
        assert len(dt) <= 3

    def test_depths_are_negative(self) -> None:
        dt = drawdown_table(_daily_returns())
        if len(dt) > 0:
            assert (dt["depth"] < 0).all()


# ------------------------------------------------------------------
# rolling_metrics_table
# ------------------------------------------------------------------

class TestRollingMetricsTable:
    def test_columns(self) -> None:
        rm = rolling_metrics_table(_daily_returns(), windows=[21, 63])
        assert isinstance(rm, pd.DataFrame)
        assert isinstance(rm.columns, pd.MultiIndex)
        # Check that expected window/metric combinations exist
        cols = rm.columns.tolist()
        assert (21, "rolling_return") in cols
        assert (63, "rolling_sharpe") in cols

    def test_default_windows(self) -> None:
        rm = rolling_metrics_table(_daily_returns())
        windows_in_result = set(rm.columns.get_level_values(0))
        assert windows_in_result == {21, 63, 126, 252}


# ------------------------------------------------------------------
# trade_analysis
# ------------------------------------------------------------------

class TestTradeAnalysis:
    def test_basic(self) -> None:
        ta = trade_analysis(_make_trades())
        assert 0 <= ta["win_rate"] <= 1
        assert ta["n_trades"] == 50

    def test_all_winners(self) -> None:
        trades = pd.DataFrame({"pnl": [100, 200, 50]})
        ta = trade_analysis(trades)
        assert ta["win_rate"] == 1.0
        assert ta["profit_factor"] == float("inf")

    def test_all_losers(self) -> None:
        trades = pd.DataFrame({"pnl": [-100, -200, -50]})
        ta = trade_analysis(trades)
        assert ta["win_rate"] == 0.0
        assert ta["avg_win"] == 0.0

    def test_empty_trades(self) -> None:
        trades = pd.DataFrame({"pnl": pd.Series(dtype=float)})
        ta = trade_analysis(trades)
        assert ta["n_trades"] == 0.0

    def test_missing_pnl_column(self) -> None:
        trades = pd.DataFrame({"profit": [1, 2, 3]})
        with pytest.raises(ValueError, match="pnl"):
            trade_analysis(trades)

    def test_expectancy(self) -> None:
        trades = pd.DataFrame({"pnl": [100, -50, 100, -50, 100]})
        ta = trade_analysis(trades)
        # win_rate=0.6, avg_win=100, avg_loss=-50
        expected_expectancy = 0.6 * 100 + 0.4 * (-50)
        assert ta["expectancy"] == pytest.approx(expected_expectancy, abs=1e-10)


# ------------------------------------------------------------------
# comprehensive_tearsheet
# ------------------------------------------------------------------


from wraquant.backtest.tearsheet import comprehensive_tearsheet, strategy_comparison


class TestComprehensiveTearsheet:
    def test_returns_expected_top_level_keys(self) -> None:
        ts = comprehensive_tearsheet(_daily_returns())
        expected_keys = {
            "summary",
            "extended_metrics",
            "monthly_returns",
            "yearly_returns",
            "drawdown_analysis",
            "rolling_metrics",
        }
        assert expected_keys.issubset(ts.keys())

    def test_summary_has_core_metrics(self) -> None:
        ts = comprehensive_tearsheet(_daily_returns())
        summary = ts["summary"]
        assert "sharpe_ratio" in summary
        assert "max_drawdown" in summary
        assert "total_return" in summary

    def test_extended_metrics_present(self) -> None:
        ts = comprehensive_tearsheet(_daily_returns())
        ext = ts["extended_metrics"]
        for key in [
            "omega_ratio",
            "burke_ratio",
            "ulcer_performance_index",
            "kappa_2",
            "tail_ratio",
            "rachev_ratio",
            "gain_to_pain_ratio",
            "recovery_factor",
            "system_quality_number",
        ]:
            assert key in ext, f"Missing extended metric: {key}"

    def test_monthly_returns_is_dataframe(self) -> None:
        ts = comprehensive_tearsheet(_daily_returns())
        assert isinstance(ts["monthly_returns"], pd.DataFrame)

    def test_yearly_returns_populated(self) -> None:
        ts = comprehensive_tearsheet(_daily_returns())
        assert len(ts["yearly_returns"]) > 0

    def test_drawdown_analysis_is_dataframe(self) -> None:
        ts = comprehensive_tearsheet(_daily_returns())
        assert isinstance(ts["drawdown_analysis"], pd.DataFrame)

    def test_rolling_metrics_is_dataframe(self) -> None:
        ts = comprehensive_tearsheet(_daily_returns())
        assert isinstance(ts["rolling_metrics"], pd.DataFrame)

    def test_with_trades(self) -> None:
        ts = comprehensive_tearsheet(_daily_returns(), trades_df=_make_trades())
        assert "trade_analysis" in ts
        assert "win_rate" in ts["trade_analysis"]
        assert "best_5_trades" in ts["trade_analysis"]
        assert "worst_5_trades" in ts["trade_analysis"]

    def test_with_regime_states(self) -> None:
        rets = _daily_returns()
        regimes = pd.Series(
            np.where(np.arange(len(rets)) % 3 == 0, "bull", "bear"),
            index=rets.index,
        )
        ts = comprehensive_tearsheet(rets, regime_states=regimes)
        assert "regime_performance" in ts
        assert "bull" in ts["regime_performance"]
        assert "bear" in ts["regime_performance"]
        # Each regime should have core stats
        for regime_data in ts["regime_performance"].values():
            assert "annualized_return" in regime_data
            assert "sharpe" in regime_data

    def test_with_benchmark(self) -> None:
        ts = comprehensive_tearsheet(
            _daily_returns(), benchmark=_benchmark_returns()
        )
        assert "beta" in ts["summary"]
        assert "alpha" in ts["summary"]


# ------------------------------------------------------------------
# strategy_comparison
# ------------------------------------------------------------------

class TestStrategyComparison:
    def test_returns_dataframe(self) -> None:
        strats = {
            "strat_a": _daily_returns(seed=42),
            "strat_b": _daily_returns(seed=99),
        }
        comp = strategy_comparison(strats)
        assert isinstance(comp, pd.DataFrame)

    def test_columns_match_strategy_names(self) -> None:
        strats = {
            "momentum": _daily_returns(seed=1),
            "mean_rev": _daily_returns(seed=2),
        }
        comp = strategy_comparison(strats)
        assert "momentum" in comp.columns
        assert "mean_rev" in comp.columns

    def test_has_extended_metrics(self) -> None:
        strats = {"s1": _daily_returns(seed=10)}
        comp = strategy_comparison(strats)
        assert "omega_ratio" in comp.index
        assert "tail_ratio" in comp.index
        assert "system_quality_number" in comp.index

    def test_shape_matches_strategies(self) -> None:
        strats = {f"s{i}": _daily_returns(seed=i) for i in range(5)}
        comp = strategy_comparison(strats)
        assert comp.shape[1] == 5
