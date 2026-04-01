"""Forex analysis MCP tools.

Tools: carry_analysis, cross_rate, fx_risk, pip_calculator,
session_info, currency_strength.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_forex_tools(mcp, ctx: AnalysisContext) -> None:
    """Register forex-specific tools on the MCP server."""

    @mcp.tool()
    def carry_analysis(
        rates_json: str,
    ) -> dict[str, Any]:
        """Construct a carry trade portfolio from interest rates.

        Goes long the highest-yielding currencies and short the
        lowest-yielding currencies.

        Parameters:
            rates_json: JSON object mapping currency codes to interest
                rates, e.g. '{"USD": 0.05, "EUR": 0.04, "JPY": 0.001}'.
        """
        import json

        from wraquant.forex.carry import carry_portfolio

        rates = json.loads(rates_json)

        result = carry_portfolio(rates_dict=rates)

        return _sanitize_for_json({
            "tool": "carry_analysis",
            "n_currencies": len(rates),
            "rates": rates,
            **result,
        })

    @mcp.tool()
    def cross_rate(
        rate_a: float,
        rate_b: float,
        cross_type: str = "multiply",
    ) -> dict[str, Any]:
        """Calculate a cross exchange rate from two currency pairs.

        Parameters:
            rate_a: Exchange rate of pair A (e.g. EUR/USD = 1.10).
            rate_b: Exchange rate of pair B (e.g. USD/JPY = 150.0).
            cross_type: How the pairs share a common currency.
                'multiply' when common currency is in A's quote and
                B's base (e.g. EUR/USD * USD/JPY = EUR/JPY).
                'divide' when common currency is in both bases or
                both quotes.
        """
        from wraquant.forex.pairs import cross_rate as _cross_rate

        result = _cross_rate(
            pair1_rate=rate_a,
            pair2_rate=rate_b,
            method=cross_type,
        )

        return _sanitize_for_json({
            "tool": "cross_rate",
            "rate_a": rate_a,
            "rate_b": rate_b,
            "cross_type": cross_type,
            "cross_rate": float(result),
        })

    @mcp.tool()
    def fx_risk(
        positions_json: str,
        exchange_rates_json: str,
        base_currency: str = "USD",
    ) -> dict[str, Any]:
        """Compute FX portfolio risk including currency exposure.

        Parameters:
            positions_json: JSON object mapping currency codes to
                position sizes, e.g. '{"EUR": 100000, "GBP": 50000}'.
            exchange_rates_json: JSON object mapping currency codes to
                exchange rates vs base, e.g. '{"EUR": 1.10, "GBP": 1.27}'.
            base_currency: Base currency for risk aggregation.
        """
        import json

        from wraquant.forex.risk import fx_portfolio_risk

        positions = json.loads(positions_json)
        exchange_rates = json.loads(exchange_rates_json)

        result = fx_portfolio_risk(
            positions=positions,
            exchange_rates=exchange_rates,
            base_currency=base_currency,
        )

        return _sanitize_for_json({
            "tool": "fx_risk",
            "base_currency": base_currency,
            "n_positions": len(positions),
            **result,
        })

    @mcp.tool()
    def pip_calculator(
        entry: float,
        exit: float,
        pair: str,
        lot_size: float = 100_000,
    ) -> dict[str, Any]:
        """Calculate pip distance, pip value, and P&L for a forex trade.

        Parameters:
            entry: Entry price.
            exit: Exit price.
            pair: Currency pair string (e.g. 'EURUSD', 'USDJPY').
            lot_size: Position size in units of base currency.
        """
        from wraquant.forex.analysis import pip_value, pips
        from wraquant.forex.pairs import CurrencyPair

        is_jpy = "JPY" in pair.upper()
        cp = CurrencyPair(pair[:3].upper(), pair[3:6].upper())

        price_change = exit - entry
        pip_count = pips(price_change, pair=cp, is_jpy=is_jpy)
        pv = pip_value(pair=cp, lot_size_units=lot_size, is_jpy=is_jpy)
        pnl = float(pip_count) * pv

        return _sanitize_for_json({
            "tool": "pip_calculator",
            "pair": pair.upper(),
            "entry": entry,
            "exit": exit,
            "lot_size": lot_size,
            "pips": float(pip_count),
            "pip_value": float(pv),
            "pnl": pnl,
            "direction": "long" if price_change > 0 else "short" if price_change < 0 else "flat",
        })

    @mcp.tool()
    def session_info() -> dict[str, Any]:
        """Get current forex trading session and overlap information.

        Returns which sessions are currently active and the major
        session overlap windows (highest liquidity periods).
        """
        from wraquant.forex.session import (
            current_session,
            session_overlaps,
        )

        active = current_session()
        overlaps = session_overlaps()

        return _sanitize_for_json({
            "tool": "session_info",
            "active_sessions": [s.value for s in active],
            "n_active": len(active),
            "overlaps": [
                {
                    "session_a": s1.value,
                    "session_b": s2.value,
                    "start_utc": start.isoformat(),
                    "end_utc": end.isoformat(),
                }
                for s1, s2, start, end in overlaps
            ],
        })

    @mcp.tool()
    def currency_strength(
        pairs_dataset: str,
        window: int = 20,
    ) -> dict[str, Any]:
        """Compute relative strength scores for each currency.

        Columns in the dataset should be named as currency pairs
        (e.g. 'EURUSD', 'GBPUSD') containing exchange rate time series.

        Parameters:
            pairs_dataset: Dataset with currency pair columns.
            window: Rolling window for strength calculation.
        """
        import pandas as pd

        from wraquant.forex.pairs import currency_strength as _strength

        df = ctx.get_dataset(pairs_dataset)

        strength = _strength(df, window=window)

        strength_df = strength.to_frame("strength") if isinstance(strength, pd.Series) else strength
        stored = ctx.store_dataset(
            f"fx_strength_{pairs_dataset}", strength_df,
            source_op="currency_strength", parent=pairs_dataset,
        )

        strength_dict = (
            {str(k): float(v) for k, v in strength.items()}
            if isinstance(strength, pd.Series)
            else {}
        )

        return _sanitize_for_json({
            "tool": "currency_strength",
            "dataset": pairs_dataset,
            "window": window,
            "strength_scores": strength_dict,
            **stored,
        })
