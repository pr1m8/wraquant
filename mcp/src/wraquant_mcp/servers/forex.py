"""Forex analysis MCP tools.

Tools: carry_analysis, cross_rate, fx_risk.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_forex_tools(mcp, ctx: AnalysisContext) -> None:
    """Register forex tools on the MCP server."""

    @mcp.tool()
    def carry_analysis(
        rates: dict[str, float],
        pairs: list[list[str]] | None = None,
    ) -> dict[str, Any]:
        """Rank currency pairs by carry attractiveness.

        Computes interest rate differentials for all pairs and ranks
        them by carry opportunity. Higher differential = more carry
        income (but potentially more crash risk).

        Parameters:
            rates: Dict mapping currency code to annual interest rate
                (e.g., {"USD": 0.05, "JPY": 0.001, "AUD": 0.04}).
            pairs: Optional list of [base, quote] pairs to evaluate.
                If None, evaluates all combinations.
        """
        from wraquant.forex.carry import carry_attractiveness

        pair_tuples = [tuple(p) for p in pairs] if pairs else None
        result = carry_attractiveness(rates, pairs=pair_tuples)

        return _sanitize_for_json({
            "tool": "carry_analysis",
            "n_pairs": len(result),
            "top_carry": result.head(5).to_dict(orient="records"),
            "bottom_carry": result.tail(3).to_dict(orient="records"),
        })

    @mcp.tool()
    def cross_rate(
        rate_a: float,
        rate_b: float,
        method: str = "divide",
    ) -> dict[str, Any]:
        """Calculate a cross rate from two currency pairs.

        Derives the exchange rate for a pair not directly quoted by
        combining two pairs sharing a common currency.

        Parameters:
            rate_a: Rate for first pair.
            rate_b: Rate for second pair.
            method: 'divide' (rate_a / rate_b) or 'multiply'
                (rate_a * rate_b).
        """
        from wraquant.forex.pairs import cross_rate as _cross

        result = _cross(rate_a, rate_b, method=method)

        return _sanitize_for_json({
            "tool": "cross_rate",
            "rate_a": rate_a,
            "rate_b": rate_b,
            "method": method,
            "cross_rate": float(result),
        })

    @mcp.tool()
    def fx_risk(
        positions: dict[str, float],
        exchange_rates: dict[str, float],
        base_currency: str = "USD",
        returns_dataset: str | None = None,
        fx_returns_dataset: str | None = None,
    ) -> dict[str, Any]:
        """Compute FX-adjusted portfolio risk.

        Converts positions to base currency and computes currency
        exposure. Optionally includes FX-adjusted volatility if
        asset and FX return datasets are provided.

        Parameters:
            positions: Dict mapping asset names to position values
                in local currency.
            exchange_rates: Dict mapping currency codes to base
                currency values (e.g., {"USD": 1.0, "JPY": 0.0067}).
            base_currency: Reporting currency (default "USD").
            returns_dataset: Optional dataset with asset returns.
            fx_returns_dataset: Optional dataset with FX returns.
        """
        from wraquant.forex.risk import fx_portfolio_risk

        returns = None
        fx_returns = None

        if returns_dataset is not None:
            returns = ctx.get_dataset(returns_dataset)
        if fx_returns_dataset is not None:
            fx_returns = ctx.get_dataset(fx_returns_dataset)

        result = fx_portfolio_risk(
            positions=positions,
            exchange_rates=exchange_rates,
            base_currency=base_currency,
            returns=returns,
            fx_returns=fx_returns,
        )

        return _sanitize_for_json({
            "tool": "fx_risk",
            "base_currency": base_currency,
            "result": result,
        })
