"""Fundamental analysis MCP tools.

Tools: fundamental_ratios, piotroski_score, dcf_valuation,
quality_screen, altman_z.
"""

from __future__ import annotations

import json
from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_fundamental_tools(mcp, ctx: AnalysisContext) -> None:
    """Register fundamental analysis tools on the MCP server."""

    @mcp.tool()
    def fundamental_ratios(
        price: float,
        earnings: float,
        book_value: float,
        net_income: float,
        equity: float,
        debt: float,
    ) -> dict[str, Any]:
        """Compute all standard fundamental ratios for a single stock.

        Returns P/E, P/B, ROE, D/E, and derived metrics in one call.

        Parameters:
            price: Current share price.
            earnings: Earnings per share (TTM).
            book_value: Book value per share.
            net_income: Net income (annual or TTM).
            equity: Total shareholders' equity.
            debt: Total debt (short-term + long-term).
        """
        try:
            from wraquant.fundamental.ratios import (
                debt_to_equity,
                pb_ratio,
                pe_ratio,
                roe,
            )

            pe = pe_ratio(price, earnings)
            pb = pb_ratio(price, book_value)
            r = roe(net_income, equity)
            de = debt_to_equity(debt, equity)

            # Derived: earnings yield = 1 / P/E
            earnings_yield = 1.0 / pe if pe != 0.0 else 0.0

            return _sanitize_for_json(
                {
                    "tool": "fundamental_ratios",
                    "pe_ratio": pe,
                    "pb_ratio": pb,
                    "roe": r,
                    "debt_to_equity": de,
                    "earnings_yield": earnings_yield,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "fundamental_ratios"}

    @mcp.tool()
    def piotroski_score(
        financials_json: str,
    ) -> dict[str, Any]:
        """Compute the Piotroski F-Score (0--9) for financial health.

        The F-Score is a composite of nine binary tests evaluating
        profitability, leverage/liquidity, and operating efficiency.
        Scores of 8--9 indicate financial strength; 0--2 indicate
        distress.

        Parameters:
            financials_json: JSON string with financial statement data.
                Required keys: net_income, prev_net_income,
                operating_cash_flow, total_assets, prev_total_assets,
                long_term_debt, prev_long_term_debt, current_ratio,
                prev_current_ratio, shares_outstanding,
                prev_shares_outstanding, gross_margin, prev_gross_margin,
                asset_turnover, prev_asset_turnover.
        """
        try:
            from wraquant.fundamental.valuation import piotroski_f_score

            financials = json.loads(financials_json)
            score = piotroski_f_score(financials)

            # Interpret the score
            if score >= 8:
                interpretation = "strong"
            elif score >= 5:
                interpretation = "neutral"
            else:
                interpretation = "weak"

            return _sanitize_for_json(
                {
                    "tool": "piotroski_score",
                    "score": score,
                    "max_score": 9,
                    "interpretation": interpretation,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "piotroski_score"}

    @mcp.tool()
    def dcf_valuation(
        cash_flows_json: str,
        discount_rate: float,
        terminal_growth: float = 0.02,
    ) -> dict[str, Any]:
        """Estimate intrinsic value using discounted cash flow analysis.

        Projects future cash flows and discounts them to present value
        using the Gordon growth model for terminal value.

        Parameters:
            cash_flows_json: JSON array of projected free cash flows
                (e.g., '[100, 110, 121, 133]').
            discount_rate: WACC or required return (e.g., 0.10 = 10%).
            terminal_growth: Perpetual growth rate (e.g., 0.02 = 2%).
                Must be less than discount_rate.
        """
        try:
            from wraquant.fundamental.valuation import dcf_valuation as _dcf

            cash_flows = json.loads(cash_flows_json)

            result = _dcf(
                cash_flows=cash_flows,
                discount_rate=discount_rate,
                terminal_growth=terminal_growth,
            )

            # Terminal value as percentage of total
            tv_pct = (
                result["pv_terminal"] / result["present_value"] * 100
                if result["present_value"] != 0.0
                else 0.0
            )

            return _sanitize_for_json(
                {
                    "tool": "dcf_valuation",
                    "discount_rate": discount_rate,
                    "terminal_growth": terminal_growth,
                    "n_periods": len(cash_flows),
                    "terminal_value_pct": tv_pct,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "dcf_valuation"}

    @mcp.tool()
    def quality_screen(
        dataset: str,
        metrics_json: str | None = None,
    ) -> dict[str, Any]:
        """Rank stocks by a composite quality score.

        Computes percentile ranks on multiple fundamental metrics and
        averages them into a single quality score.  Higher scores
        indicate higher quality.

        Parameters:
            dataset: Dataset with fundamental columns (e.g., roe,
                operating_margin, current_ratio).  Each row is a stock.
            metrics_json: Optional JSON array of column names to use
                (e.g., '["roe", "operating_margin"]').  If None, defaults
                to ["roe", "operating_margin", "current_ratio"].
        """
        try:
            from wraquant.fundamental.valuation import quality_screen as _screen

            df = ctx.get_dataset(dataset)

            metrics = json.loads(metrics_json) if metrics_json is not None else None

            result_df = _screen(df, metrics=metrics)

            # Store the screened result
            result_name = f"{dataset}_quality"
            stored = ctx.store_dataset(
                result_name,
                result_df,
                source_op="quality_screen",
                parent=dataset,
            )

            # Top 5 and bottom 5
            top = result_df.head(5)
            bottom = result_df.tail(5)

            return _sanitize_for_json(
                {
                    "tool": "quality_screen",
                    "dataset": dataset,
                    "n_stocks": len(result_df),
                    "metrics_used": metrics
                    or ["roe", "operating_margin", "current_ratio"],
                    "top_5": top[["quality_score", "quality_rank"]].to_dict(
                        orient="index"
                    ),
                    "bottom_5": bottom[["quality_score", "quality_rank"]].to_dict(
                        orient="index"
                    ),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "quality_screen"}

    @mcp.tool()
    def altman_z(
        working_capital: float,
        total_assets: float,
        retained_earnings: float,
        ebit: float,
        market_equity: float,
        total_liabilities: float,
        sales: float,
    ) -> dict[str, Any]:
        """Compute the Altman Z-Score for bankruptcy prediction.

        The Z-Score combines five financial ratios into a single
        discriminant score.  Z > 2.99 indicates safety; Z < 1.81
        indicates distress; values in between are the gray zone.

        Parameters:
            working_capital: Current assets minus current liabilities.
            total_assets: Total assets.
            retained_earnings: Retained earnings.
            ebit: Earnings before interest and taxes.
            market_equity: Market value of equity (market cap).
            total_liabilities: Total liabilities.
            sales: Total sales / revenue.
        """
        try:
            from wraquant.risk.credit import altman_z_score

            z = altman_z_score(
                working_capital=working_capital,
                retained_earnings=retained_earnings,
                ebit=ebit,
                market_cap=market_equity,
                total_liabilities=total_liabilities,
                total_assets=total_assets,
                sales=sales,
            )

            # Interpret
            if isinstance(z, dict):
                score = z.get("z_score", z.get("score", 0.0))
                result = z
            else:
                score = float(z)
                result = {"z_score": score}

            if score > 2.99:
                zone = "safe"
            elif score > 1.81:
                zone = "gray"
            else:
                zone = "distress"

            return _sanitize_for_json(
                {
                    "tool": "altman_z",
                    "z_score": score,
                    "zone": zone,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "altman_z"}
