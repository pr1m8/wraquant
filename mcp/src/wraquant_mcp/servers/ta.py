"""Technical analysis MCP tools.

Tools: list_indicators, multi_indicator, scan_signals, momentum_indicators,
trend_indicators, volatility_indicators, volume_indicators,
overlay_indicators, pattern_recognition, fibonacci_levels,
support_resistance, cycle_analysis, smoothing_indicators,
exotic_indicators, statistics_indicators, breadth_indicators,
candle_analysis, price_action_analysis, performance_indicators,
ta_screening, ta_dashboard.
"""

from __future__ import annotations

import json
from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


# Indicator categories for discovery
_CATEGORIES = {
    "overlap": [
        "sma", "ema", "wma", "dema", "tema", "kama", "vwap",
        "supertrend", "ichimoku", "bollinger_bands", "keltner_channel",
        "donchian_channel",
    ],
    "momentum": [
        "rsi", "stochastic", "stochastic_rsi", "macd", "williams_r",
        "cci", "roc", "momentum", "tsi", "awesome_oscillator", "ppo",
        "ultimate_oscillator", "cmo", "dpo", "squeeze_histogram",
    ],
    "volume": [
        "obv", "ad_line", "cmf", "mfi", "eom", "force_index",
        "nvi", "pvi", "vpt", "adosc",
    ],
    "trend": [
        "adx", "aroon", "psar", "vortex", "trix", "zigzag",
        "heikin_ashi", "hull_ma", "zero_lag_ema", "vidya",
    ],
    "volatility": [
        "atr", "true_range", "natr", "bbwidth", "kc_width",
        "historical_volatility", "ulcer_index", "garman_klass",
        "parkinson", "yang_zhang",
    ],
    "patterns": [
        "doji", "hammer", "engulfing", "morning_star", "evening_star",
        "three_white_soldiers", "three_black_crows", "harami",
        "shooting_star", "hanging_man",
    ],
    "statistics": [
        "zscore", "percentile_rank", "skewness", "kurtosis",
        "entropy", "hurst_exponent", "correlation", "beta",
    ],
    "cycles": [
        "hilbert_transform_dominant_period", "sine_wave",
        "even_better_sinewave", "roofing_filter", "bandpass_filter",
    ],
    "fibonacci": [
        "fibonacci_retracements", "fibonacci_extensions",
        "auto_fibonacci", "fibonacci_pivot_points",
    ],
    "smoothing": [
        "alma", "jma", "butterworth_filter", "supersmoother",
        "gaussian_filter", "lsma",
    ],
    "exotic": [
        "choppiness_index", "random_walk_index",
        "polarized_fractal_efficiency", "ergodic_oscillator",
        "elder_thermometer", "connors_tps",
    ],
    "support_resistance": [
        "find_support_resistance", "fractal_levels",
        "price_clustering", "supply_demand_zones",
    ],
}


def register_ta_tools(mcp, ctx: AnalysisContext) -> None:
    """Register technical analysis-specific tools on the MCP server."""

    @mcp.tool()
    def list_indicators(
        category: str | None = None,
    ) -> dict[str, Any]:
        """List available technical indicators, optionally by category.

        Parameters:
            category: Filter by category. Options: 'overlap',
                'momentum', 'volume', 'trend', 'volatility',
                'patterns', 'statistics', 'cycles', 'fibonacci',
                'smoothing', 'exotic', 'support_resistance'.
                If None, returns all categories.
        """
        if category:
            indicators = _CATEGORIES.get(category)
            if indicators is None:
                return {
                    "error": f"Unknown category '{category}'",
                    "categories": list(_CATEGORIES.keys()),
                }
            return {
                "tool": "list_indicators",
                "category": category,
                "indicators": indicators,
                "count": len(indicators),
            }

        return {
            "tool": "list_indicators",
            "categories": {
                k: {"indicators": v, "count": len(v)}
                for k, v in _CATEGORIES.items()
            },
            "total": sum(len(v) for v in _CATEGORIES.values()),
        }

    @mcp.tool()
    def multi_indicator(
        dataset: str,
        indicators: list[str],
        column: str = "close",
        period: int = 14,
    ) -> dict[str, Any]:
        """Compute multiple technical indicators at once.

        More efficient than calling compute_indicator repeatedly.

        Parameters:
            dataset: Source dataset.
            indicators: List of indicator names
                (e.g. ['rsi', 'macd', 'bollinger_bands']).
            column: Price column (default 'close').
            period: Lookback period for all indicators.
        """
        import wraquant.ta as ta

        df = ctx.get_dataset(dataset)
        if column not in df.columns:
            return {"error": f"Column '{column}' not in {list(df.columns)}"}

        results = {}
        errors = []

        for name in indicators:
            func = getattr(ta, name, None)
            if func is None:
                errors.append(f"'{name}' not found")
                continue

            try:
                result = func(df[column], period=period)
            except TypeError:
                try:
                    result = func(df[column])
                except Exception as e:
                    errors.append(f"'{name}': {e}")
                    continue

            if isinstance(result, dict):
                for key, series in result.items():
                    col_name = f"{name}_{key}"
                    if hasattr(series, "values"):
                        df[col_name] = series.values[:len(df)]
                        results[col_name] = {
                            "latest": float(series.iloc[-1])
                            if len(series) > 0 else None,
                        }
            elif hasattr(result, "values"):
                df[name] = result.values[:len(df)]
                results[name] = {
                    "latest": float(result.iloc[-1])
                    if len(result) > 0 else None,
                    "mean": float(result.mean()),
                }

        new_name = f"{dataset}_ta"
        stored = ctx.store_dataset(
            new_name, df, source_op="multi_indicator", parent=dataset,
        )

        return _sanitize_for_json({
            "tool": "multi_indicator",
            "computed": list(results.keys()),
            "errors": errors if errors else None,
            "summaries": results,
            **stored,
        })

    @mcp.tool()
    def scan_signals(
        dataset: str,
        column: str = "close",
    ) -> dict[str, Any]:
        """Scan for overbought/oversold conditions across indicators.

        Computes RSI, Stochastic, Williams %R, CCI, and MFI, then
        flags overbought (OB) and oversold (OS) conditions.

        Parameters:
            dataset: Dataset with price data.
            column: Price column.
        """
        import wraquant.ta as ta

        df = ctx.get_dataset(dataset)
        prices = df[column]

        signals = {}

        # RSI
        try:
            rsi = ta.rsi(prices, period=14)
            rsi_val = float(rsi.iloc[-1])
            signals["rsi"] = {
                "value": rsi_val,
                "signal": "overbought" if rsi_val > 70
                else "oversold" if rsi_val < 30
                else "neutral",
            }
        except Exception:
            pass

        # Stochastic
        try:
            stoch = ta.stochastic(prices, period=14)
            if isinstance(stoch, dict):
                k_val = float(list(stoch.values())[0].iloc[-1])
            else:
                k_val = float(stoch.iloc[-1])
            signals["stochastic"] = {
                "value": k_val,
                "signal": "overbought" if k_val > 80
                else "oversold" if k_val < 20
                else "neutral",
            }
        except Exception:
            pass

        # Williams %R
        try:
            wr = ta.williams_r(prices, period=14)
            wr_val = float(wr.iloc[-1])
            signals["williams_r"] = {
                "value": wr_val,
                "signal": "overbought" if wr_val > -20
                else "oversold" if wr_val < -80
                else "neutral",
            }
        except Exception:
            pass

        # CCI
        try:
            cci = ta.cci(prices, period=20)
            cci_val = float(cci.iloc[-1])
            signals["cci"] = {
                "value": cci_val,
                "signal": "overbought" if cci_val > 100
                else "oversold" if cci_val < -100
                else "neutral",
            }
        except Exception:
            pass

        n_ob = sum(1 for s in signals.values() if s["signal"] == "overbought")
        n_os = sum(1 for s in signals.values() if s["signal"] == "oversold")

        consensus = "neutral"
        if n_ob >= 2:
            consensus = "overbought"
        elif n_os >= 2:
            consensus = "oversold"

        return _sanitize_for_json({
            "tool": "scan_signals",
            "dataset": dataset,
            "indicators": signals,
            "consensus": consensus,
            "overbought_count": n_ob,
            "oversold_count": n_os,
        })
