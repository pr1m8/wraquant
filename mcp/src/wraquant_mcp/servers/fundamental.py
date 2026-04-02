"""Fundamental analysis MCP tools.

Tools: company_profile, financial_ratios, income_analysis, balance_sheet_analysis,
cash_flow_analysis, dcf_valuation, relative_valuation, financial_health,
piotroski_score, altman_z, earnings_quality, dupont_analysis, graham_number,
stock_screener.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_fundamental_tools(mcp, ctx: AnalysisContext) -> None:
    """Register fundamental analysis tools on the MCP server."""

    @mcp.tool()
    def company_profile(symbol: str) -> dict[str, Any]:
        """Get comprehensive company profile from FMP.

        Returns sector, industry, market cap, description, CEO, employees,
        and key financial metrics for the given ticker.

        Parameters:
            symbol: Stock ticker (e.g., 'AAPL').
        """
        try:
            from wraquant.data.providers.fmp import FMPClient

            client = FMPClient()
            profile = client.company_profile(symbol)

            return _sanitize_for_json(
                {
                    "tool": "company_profile",
                    "symbol": symbol,
                    **profile,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "company_profile"}

    @mcp.tool()
    def financial_ratios(
        symbol: str,
        period: str = "annual",
    ) -> dict[str, Any]:
        """Compute comprehensive financial ratios for a stock.

        Returns profitability, liquidity, leverage, efficiency, valuation,
        and growth ratios using live FMP data.

        Parameters:
            symbol: Stock ticker.
            period: 'annual' or 'quarter'.
        """
        try:
            from wraquant.fundamental.ratios import comprehensive_ratios

            result = comprehensive_ratios(symbol)

            return _sanitize_for_json(
                {
                    "tool": "financial_ratios",
                    "symbol": symbol,
                    "period": period,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "financial_ratios"}

    @mcp.tool()
    def income_analysis(
        symbol: str,
        period: str = "annual",
    ) -> dict[str, Any]:
        """Analyze income statement trends: revenue growth, margins, profitability.

        Returns multi-year trends in revenue, operating income, net income,
        and key margins with growth rates.

        Parameters:
            symbol: Stock ticker.
            period: 'annual' or 'quarter'.
        """
        try:
            from wraquant.fundamental.financials import income_analysis as _income

            result = _income(symbol, period=period)

            return _sanitize_for_json(
                {
                    "tool": "income_analysis",
                    "symbol": symbol,
                    "period": period,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "income_analysis"}

    @mcp.tool()
    def balance_sheet_analysis(
        symbol: str,
        period: str = "annual",
    ) -> dict[str, Any]:
        """Analyze balance sheet composition and leverage trends.

        Returns asset/liability breakdown, working capital, leverage ratios,
        and year-over-year changes.

        Parameters:
            symbol: Stock ticker.
            period: 'annual' or 'quarter'.
        """
        try:
            from wraquant.fundamental.financials import balance_sheet_analysis as _bs

            result = _bs(symbol, period=period)

            return _sanitize_for_json(
                {
                    "tool": "balance_sheet_analysis",
                    "symbol": symbol,
                    "period": period,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "balance_sheet_analysis"}

    @mcp.tool()
    def cash_flow_analysis(
        symbol: str,
        period: str = "annual",
    ) -> dict[str, Any]:
        """Analyze cash flow statement: FCF, cash conversion, capex trends.

        Returns free cash flow, operating cash flow quality, capex intensity,
        and cash conversion cycle metrics.

        Parameters:
            symbol: Stock ticker.
            period: 'annual' or 'quarter'.
        """
        try:
            from wraquant.fundamental.financials import cash_flow_analysis as _cf

            result = _cf(symbol, period=period)

            return _sanitize_for_json(
                {
                    "tool": "cash_flow_analysis",
                    "symbol": symbol,
                    "period": period,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "cash_flow_analysis"}

    @mcp.tool()
    def dcf_valuation(
        symbol: str,
        discount_rate: float = 0.10,
        terminal_growth: float = 0.025,
    ) -> dict[str, Any]:
        """Estimate intrinsic value using discounted cash flow analysis.

        Uses FMP financial data to project cash flows and discount to
        present value. Returns intrinsic value per share and margin of safety.

        Parameters:
            symbol: Stock ticker.
            discount_rate: WACC or required return (default 10%).
            terminal_growth: Perpetual growth rate (default 2.5%).
        """
        try:
            from wraquant.fundamental.valuation import dcf_valuation as _dcf

            result = _dcf(
                symbol, discount_rate=discount_rate, terminal_growth=terminal_growth
            )

            return _sanitize_for_json(
                {
                    "tool": "dcf_valuation",
                    "symbol": symbol,
                    "discount_rate": discount_rate,
                    "terminal_growth": terminal_growth,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "dcf_valuation"}

    @mcp.tool()
    def relative_valuation(
        symbol: str,
        peers_json: str | None = None,
    ) -> dict[str, Any]:
        """Compare valuation multiples against peer companies.

        Returns P/E, P/B, EV/EBITDA, P/S for the target and peers,
        with percentile ranking and implied fair values.

        Parameters:
            symbol: Stock ticker.
            peers_json: Optional JSON list of peer tickers.
                If None, uses stocks in same sector.
        """
        try:
            import json as _json

            from wraquant.fundamental.valuation import relative_valuation as _rel

            peers = _json.loads(peers_json) if peers_json else None
            result = _rel(symbol, peers=peers)

            return _sanitize_for_json(
                {
                    "tool": "relative_valuation",
                    "symbol": symbol,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "relative_valuation"}

    @mcp.tool()
    def financial_health(
        symbol: str,
    ) -> dict[str, Any]:
        """Compute a composite financial health score (0-100).

        Combines profitability, leverage, liquidity, efficiency, and
        growth metrics into a single score with letter grade.

        Parameters:
            symbol: Stock ticker.
        """
        try:
            from wraquant.fundamental.financials import financial_health_score

            result = financial_health_score(symbol)

            return _sanitize_for_json(
                {
                    "tool": "financial_health",
                    "symbol": symbol,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "financial_health"}

    @mcp.tool()
    def piotroski_score(symbol: str) -> dict[str, Any]:
        """Compute the Piotroski F-Score (0-9) for financial strength.

        Nine binary tests evaluating profitability, leverage, and
        operating efficiency. Score >= 7 is strong; <= 2 is weak.

        Parameters:
            symbol: Stock ticker.
        """
        try:
            from wraquant.data.providers.fmp import FMPClient

            client = FMPClient()
            score_data = client.score(symbol)

            score = (
                score_data.get("piotroskiScore", 0)
                if isinstance(score_data, dict)
                else 0
            )
            interpretation = (
                "strong" if score >= 7 else "neutral" if score >= 4 else "weak"
            )

            return _sanitize_for_json(
                {
                    "tool": "piotroski_score",
                    "symbol": symbol,
                    "score": score,
                    "max_score": 9,
                    "interpretation": interpretation,
                    **(score_data if isinstance(score_data, dict) else {}),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "piotroski_score"}

    @mcp.tool()
    def altman_z(symbol: str) -> dict[str, Any]:
        """Compute the Altman Z-Score for bankruptcy prediction.

        Z > 2.99 = safe, 1.81-2.99 = gray zone, < 1.81 = distress.
        Uses live financial data from FMP.

        Parameters:
            symbol: Stock ticker.
        """
        try:
            from wraquant.data.providers.fmp import FMPClient

            client = FMPClient()
            score_data = client.score(symbol)

            z = (
                score_data.get("altmanZScore", 0.0)
                if isinstance(score_data, dict)
                else 0.0
            )
            zone = "safe" if z > 2.99 else "gray" if z > 1.81 else "distress"

            return _sanitize_for_json(
                {
                    "tool": "altman_z",
                    "symbol": symbol,
                    "z_score": z,
                    "zone": zone,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "altman_z"}

    @mcp.tool()
    def earnings_quality(symbol: str) -> dict[str, Any]:
        """Assess earnings quality: accruals, cash conversion, persistence.

        High quality = earnings backed by cash flow, low accruals,
        consistent over time.

        Parameters:
            symbol: Stock ticker.
        """
        try:
            from wraquant.fundamental.financials import earnings_quality as _eq

            result = _eq(symbol)

            return _sanitize_for_json(
                {
                    "tool": "earnings_quality",
                    "symbol": symbol,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "earnings_quality"}

    @mcp.tool()
    def dupont_analysis(symbol: str) -> dict[str, Any]:
        """Decompose ROE using 3-way and 5-way DuPont analysis.

        3-way: ROE = Profit Margin x Asset Turnover x Equity Multiplier
        5-way: Adds tax burden and interest burden components.

        Parameters:
            symbol: Stock ticker.
        """
        try:
            from wraquant.fundamental.ratios import dupont_decomposition

            result = dupont_decomposition(symbol)

            return _sanitize_for_json(
                {
                    "tool": "dupont_analysis",
                    "symbol": symbol,
                    **result,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "dupont_analysis"}

    @mcp.tool()
    def graham_number(symbol: str) -> dict[str, Any]:
        """Compute the Benjamin Graham intrinsic value number.

        Graham Number = sqrt(22.5 x EPS x BVPS). If the stock trades
        below this value, it may be undervalued by Graham's criteria.

        Parameters:
            symbol: Stock ticker.
        """
        try:
            from wraquant.fundamental.valuation import graham_number as _graham

            result = _graham(symbol)

            return _sanitize_for_json(
                {
                    "tool": "graham_number",
                    "symbol": symbol,
                    **(
                        result
                        if isinstance(result, dict)
                        else {"graham_number": result}
                    ),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "graham_number"}

    @mcp.tool()
    def stock_screener(
        criteria_json: str,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Screen stocks by fundamental criteria using FMP screener.

        Pass criteria as JSON with min/max values for any metric:
        PE, ROE, market_cap, dividend_yield, revenue_growth, etc.

        Parameters:
            criteria_json: JSON dict of screening criteria, e.g.
                '{"min_roe": 0.15, "max_pe": 25, "min_dividend_yield": 0.02}'
            top_n: Maximum number of results (default 20).
        """
        try:
            import json as _json

            from wraquant.fundamental.screening import custom_screen

            criteria = _json.loads(criteria_json)
            result_df = custom_screen(criteria)

            if len(result_df) > top_n:
                result_df = result_df.head(top_n)

            stored = ctx.store_dataset(
                "screener_results",
                result_df,
                source_op="stock_screener",
            )

            return _sanitize_for_json(
                {
                    "tool": "stock_screener",
                    "criteria": criteria,
                    "n_results": len(result_df),
                    "columns": list(result_df.columns),
                    "top_results": result_df.head(10).to_dict(orient="records"),
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "stock_screener"}
