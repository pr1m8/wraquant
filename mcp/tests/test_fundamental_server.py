"""Tests for fundamental analysis MCP server tools.

Tests piotroski_score via underlying wraquant.fundamental functions.
Other fundamental tools require FMP API key (tested in tests/fundamental/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add mcp source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wraquant_mcp.context import _sanitize_for_json


class TestFundamentalServer:
    """Test fundamental MCP tool functions via underlying wraquant.fundamental."""

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

        if score >= 8:
            interpretation = "strong"
        elif score >= 5:
            interpretation = "neutral"
        else:
            interpretation = "weak"

        output = _sanitize_for_json(
            {
                "tool": "piotroski_score",
                "score": score,
                "max_score": 9,
                "interpretation": interpretation,
            }
        )

        assert output["tool"] == "piotroski_score"
        assert isinstance(output["score"], int)
        assert 0 <= output["score"] <= 9
        assert output["score"] == 9
        assert output["max_score"] == 9
        assert output["interpretation"] == "strong"

    def test_piotroski_weak_company(self):
        """piotroski_score returns low score for distressed company."""
        from wraquant.fundamental.valuation import piotroski_f_score

        financials = {
            "net_income": -1e6,
            "prev_net_income": -5e5,
            "operating_cash_flow": -2e6,
            "total_assets": 5e6,
            "prev_total_assets": 5.5e6,
            "long_term_debt": 3e6,
            "prev_long_term_debt": 2.5e6,
            "current_ratio": 0.8,
            "prev_current_ratio": 1.0,
            "shares_outstanding": 1.2e6,
            "prev_shares_outstanding": 1e6,
            "gross_margin": 0.25,
            "prev_gross_margin": 0.30,
            "asset_turnover": 0.6,
            "prev_asset_turnover": 0.7,
        }

        score = piotroski_f_score(financials)
        assert score <= 2

    def test_sanitize_fundamental_output(self):
        """Verify _sanitize_for_json handles fundamental output dicts."""
        output = _sanitize_for_json(
            {
                "tool": "financial_ratios",
                "symbol": "AAPL",
                "roe": 0.1567,
                "pe_ratio": 28.5,
                "debt_to_equity": 1.23,
            }
        )

        assert output["tool"] == "financial_ratios"
        assert isinstance(output["roe"], float)
        assert isinstance(output["pe_ratio"], float)
