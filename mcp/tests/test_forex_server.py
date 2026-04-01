"""Tests for forex MCP server tools.

Tests carry_analysis, cross_rate, fx_risk, pip_calculator via
underlying wraquant.forex functions with synthetic data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add mcp source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


@pytest.fixture
def ctx(tmp_path):
    """Create an AnalysisContext with a temporary workspace."""
    context = AnalysisContext(workspace_dir=tmp_path / "test_workspace")
    yield context
    context.close()


@pytest.fixture
def prices_df():
    """Create synthetic price data."""
    np.random.seed(42)
    n = 252
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.3,
            "high": close + abs(np.random.randn(n) * 0.5),
            "low": close - abs(np.random.randn(n) * 0.5),
            "close": close,
            "volume": np.random.randint(100_000, 1_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def returns_df(prices_df):
    """Create returns from prices."""
    returns = prices_df["close"].pct_change().dropna()
    return returns.to_frame(name="returns")


class TestForexServer:
    """Test forex MCP tool functions via underlying wraquant.forex."""

    def test_carry_analysis(self):
        """carry_analysis with interest rates builds a carry portfolio."""
        from wraquant.forex.carry import carry_portfolio

        rates = {"USD": 0.05, "EUR": 0.04, "JPY": 0.001, "AUD": 0.045, "CHF": 0.015, "NZD": 0.04}

        result = carry_portfolio(rates_dict=rates)

        output = _sanitize_for_json({
            "tool": "carry_analysis",
            "n_currencies": len(rates),
            "rates": rates,
            **result,
        })

        assert output["tool"] == "carry_analysis"
        assert output["n_currencies"] == 6
        assert "weights" in output
        assert "expected_carry" in output
        assert "long_currencies" in output
        assert "short_currencies" in output
        assert isinstance(output["weights"], dict)
        assert isinstance(output["expected_carry"], float)
        assert output["expected_carry"] > 0  # long high-yield, short low-yield
        assert len(output["long_currencies"]) == 3
        assert len(output["short_currencies"]) == 3

    def test_cross_rate_multiply(self):
        """cross_rate with multiply computes EUR/JPY from EUR/USD * USD/JPY."""
        from wraquant.forex.pairs import cross_rate as _cross_rate

        rate_a = 1.10  # EUR/USD
        rate_b = 150.0  # USD/JPY

        result = _cross_rate(pair1_rate=rate_a, pair2_rate=rate_b, method="multiply")

        output = _sanitize_for_json({
            "tool": "cross_rate",
            "rate_a": rate_a,
            "rate_b": rate_b,
            "cross_type": "multiply",
            "cross_rate": float(result),
        })

        assert output["tool"] == "cross_rate"
        assert isinstance(output["cross_rate"], float)
        assert output["cross_rate"] == pytest.approx(165.0)  # 1.10 * 150.0
        assert output["rate_a"] == 1.10
        assert output["rate_b"] == 150.0
        assert output["cross_type"] == "multiply"

    def test_fx_risk(self):
        """fx_risk computes FX portfolio risk with currency exposure."""
        from wraquant.forex.risk import fx_portfolio_risk

        positions = {"EUR_stock": 100_000, "GBP_bond": 50_000}
        exchange_rates = {"EUR": 1.10, "GBP": 1.27}

        result = fx_portfolio_risk(
            positions=positions,
            exchange_rates=exchange_rates,
            base_currency="USD",
        )

        output = _sanitize_for_json({
            "tool": "fx_risk",
            "base_currency": "USD",
            "n_positions": len(positions),
            **result,
        })

        assert output["tool"] == "fx_risk"
        assert output["base_currency"] == "USD"
        assert output["n_positions"] == 2
        assert "total_value_base" in output
        assert "positions_base" in output
        assert "currency_exposure" in output
        assert isinstance(output["total_value_base"], float)
        assert output["total_value_base"] > 0
        assert isinstance(output["positions_base"], dict)
        assert isinstance(output["currency_exposure"], dict)

    def test_pip_calculator(self):
        """pip_calculator computes pips, pip value, and P&L."""
        from wraquant.forex.analysis import pip_value, pips
        from wraquant.forex.pairs import CurrencyPair

        entry = 1.1000
        exit_price = 1.1050
        pair = "EURUSD"
        lot_size = 100_000

        is_jpy = "JPY" in pair.upper()
        cp = CurrencyPair(pair[:3].upper(), pair[3:6].upper())

        price_change = exit_price - entry
        pip_count = pips(price_change, pair=cp, is_jpy=is_jpy)
        pv = pip_value(pair=cp, lot_size_units=lot_size, is_jpy=is_jpy)
        pnl = float(pip_count) * pv

        output = _sanitize_for_json({
            "tool": "pip_calculator",
            "pair": pair.upper(),
            "entry": entry,
            "exit": exit_price,
            "lot_size": lot_size,
            "pips": float(pip_count),
            "pip_value": float(pv),
            "pnl": pnl,
            "direction": "long" if price_change > 0 else "short",
        })

        assert output["tool"] == "pip_calculator"
        assert output["pair"] == "EURUSD"
        assert isinstance(output["pips"], float)
        assert output["pips"] == pytest.approx(50.0)  # 0.0050 / 0.0001
        assert isinstance(output["pip_value"], float)
        assert output["pip_value"] == pytest.approx(10.0)  # 0.0001 * 100000
        assert isinstance(output["pnl"], float)
        assert output["pnl"] == pytest.approx(500.0)  # 50 pips * $10/pip
        assert output["direction"] == "long"
