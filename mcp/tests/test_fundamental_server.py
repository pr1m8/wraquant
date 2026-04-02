"""Tests for fundamental analysis MCP server tools.

Tests fundamental_ratios, piotroski_score, dcf_valuation via
underlying wraquant.fundamental functions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add mcp source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wraquant_mcp.context import _sanitize_for_json


class TestFundamentalServer:
    """Test fundamental MCP tool functions via underlying wraquant.fundamental."""

    def test_fundamental_ratios(self):
        """fundamental_ratios computes P/E, P/B, ROE, D/E."""
        from wraquant.fundamental.ratios import (
            debt_to_equity,
            pb_ratio,
            pe_ratio,
            roe,
        )

        price = 150.0
        earnings = 7.5
        book_value = 50.0
        net_income = 1_000_000.0
        equity = 5_000_000.0
        debt = 2_000_000.0

        pe = pe_ratio(price, earnings)
        pb = pb_ratio(price, book_value)
        r = roe(net_income, equity)
        de = debt_to_equity(debt, equity)
        earnings_yield = 1.0 / pe if pe != 0.0 else 0.0

        output = _sanitize_for_json({
            "tool": "fundamental_ratios",
            "pe_ratio": pe,
            "pb_ratio": pb,
            "roe": r,
            "debt_to_equity": de,
            "earnings_yield": earnings_yield,
        })

        assert output["tool"] == "fundamental_ratios"
        assert isinstance(output["pe_ratio"], float)
        assert output["pe_ratio"] == pytest.approx(20.0)  # 150 / 7.5
        assert isinstance(output["pb_ratio"], float)
        assert output["pb_ratio"] == pytest.approx(3.0)  # 150 / 50
        assert isinstance(output["roe"], float)
        assert output["roe"] == pytest.approx(0.2)  # 1M / 5M
        assert isinstance(output["debt_to_equity"], float)
        assert output["debt_to_equity"] == pytest.approx(0.4)  # 2M / 5M
        assert isinstance(output["earnings_yield"], float)
        assert output["earnings_yield"] == pytest.approx(0.05)  # 1 / 20

    def test_piotroski_score(self):
        """piotroski_score returns an integer 0-9."""
        from wraquant.fundamental.valuation import piotroski_f_score

        # Construct financially strong company: all 9 tests should pass
        financials = {
            "net_income": 1e6,
            "prev_net_income": 8e5,
            "operating_cash_flow": 1.2e6,
            "total_assets": 5e6,
            "prev_total_assets": 4.8e6,
            "long_term_debt": 1e6,
            "prev_long_term_debt": 1.1e6,
            "current_ratio": 1.5,
            "prev_current_ratio": 1.3,
            "shares_outstanding": 1e6,
            "prev_shares_outstanding": 1e6,
            "gross_margin": 0.4,
            "prev_gross_margin": 0.38,
            "asset_turnover": 0.8,
            "prev_asset_turnover": 0.75,
        }

        score = piotroski_f_score(financials)

        # Interpret
        if score >= 8:
            interpretation = "strong"
        elif score >= 5:
            interpretation = "neutral"
        else:
            interpretation = "weak"

        output = _sanitize_for_json({
            "tool": "piotroski_score",
            "score": score,
            "max_score": 9,
            "interpretation": interpretation,
        })

        assert output["tool"] == "piotroski_score"
        assert isinstance(output["score"], int)
        assert 0 <= output["score"] <= 9
        assert output["score"] == 9  # all tests pass for strong financials
        assert output["max_score"] == 9
        assert output["interpretation"] == "strong"

    def test_dcf_valuation(self):
        """dcf_valuation returns intrinsic value with cash flow components."""
        from wraquant.fundamental.valuation import dcf_valuation as _dcf

        cash_flows = [100.0, 110.0, 121.0, 133.1]
        discount_rate = 0.10
        terminal_growth = 0.02

        result = _dcf(
            cash_flows=cash_flows,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
        )

        tv_pct = (
            result["pv_terminal"] / result["present_value"] * 100
            if result["present_value"] != 0.0
            else 0.0
        )

        output = _sanitize_for_json({
            "tool": "dcf_valuation",
            "discount_rate": discount_rate,
            "terminal_growth": terminal_growth,
            "n_periods": len(cash_flows),
            "terminal_value_pct": tv_pct,
            **result,
        })

        assert output["tool"] == "dcf_valuation"
        assert isinstance(output["present_value"], float)
        assert output["present_value"] > 0
        assert isinstance(output["pv_cash_flows"], float)
        assert output["pv_cash_flows"] > 0
        assert isinstance(output["pv_terminal"], float)
        assert output["pv_terminal"] > 0
        assert isinstance(output["terminal_value"], float)
        assert output["terminal_value"] > 0
        assert output["n_periods"] == 4
        assert output["discount_rate"] == 0.10
        assert output["terminal_growth"] == 0.02
        assert isinstance(output["terminal_value_pct"], float)
        assert 0 < output["terminal_value_pct"] < 100
        # Terminal value should dominate for short projection periods
        assert output["pv_terminal"] > output["pv_cash_flows"]
        # Intrinsic value = PV(cash flows) + PV(terminal)
        assert output["present_value"] == pytest.approx(
            output["pv_cash_flows"] + output["pv_terminal"], rel=1e-6,
        )
