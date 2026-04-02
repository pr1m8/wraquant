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
        "sma",
        "ema",
        "wma",
        "dema",
        "tema",
        "kama",
        "vwap",
        "supertrend",
        "ichimoku",
        "bollinger_bands",
        "keltner_channel",
        "donchian_channel",
    ],
    "momentum": [
        "rsi",
        "stochastic",
        "stochastic_rsi",
        "macd",
        "williams_r",
        "cci",
        "roc",
        "momentum",
        "tsi",
        "awesome_oscillator",
        "ppo",
        "ultimate_oscillator",
        "cmo",
        "dpo",
        "squeeze_histogram",
    ],
    "volume": [
        "obv",
        "ad_line",
        "cmf",
        "mfi",
        "eom",
        "force_index",
        "nvi",
        "pvi",
        "vpt",
        "adosc",
    ],
    "trend": [
        "adx",
        "aroon",
        "psar",
        "vortex",
        "trix",
        "zigzag",
        "heikin_ashi",
        "hull_ma",
        "zero_lag_ema",
        "vidya",
    ],
    "volatility": [
        "atr",
        "true_range",
        "natr",
        "bbwidth",
        "kc_width",
        "historical_volatility",
        "ulcer_index",
        "garman_klass",
        "parkinson",
        "yang_zhang",
    ],
    "patterns": [
        "doji",
        "hammer",
        "engulfing",
        "morning_star",
        "evening_star",
        "three_white_soldiers",
        "three_black_crows",
        "harami",
        "shooting_star",
        "hanging_man",
    ],
    "statistics": [
        "zscore",
        "percentile_rank",
        "skewness",
        "kurtosis",
        "entropy",
        "hurst_exponent",
        "correlation",
        "beta",
    ],
    "cycles": [
        "hilbert_transform_dominant_period",
        "sine_wave",
        "even_better_sinewave",
        "roofing_filter",
        "bandpass_filter",
    ],
    "fibonacci": [
        "fibonacci_retracements",
        "fibonacci_extensions",
        "auto_fibonacci",
        "fibonacci_pivot_points",
    ],
    "smoothing": [
        "alma",
        "jma",
        "butterworth_filter",
        "supersmoother",
        "gaussian_filter",
        "lsma",
    ],
    "exotic": [
        "choppiness_index",
        "random_walk_index",
        "polarized_fractal_efficiency",
        "ergodic_oscillator",
        "elder_thermometer",
        "connors_tps",
    ],
    "support_resistance": [
        "find_support_resistance",
        "fractal_levels",
        "price_clustering",
        "supply_demand_zones",
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
        try:
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
        except Exception as e:
            return {"error": str(e), "tool": "list_indicators"}

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
        try:
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
                            df[col_name] = series.values[: len(df)]
                            results[col_name] = {
                                "latest": (
                                    float(series.iloc[-1]) if len(series) > 0 else None
                                ),
                            }
                elif hasattr(result, "values"):
                    df[name] = result.values[: len(df)]
                    results[name] = {
                        "latest": float(result.iloc[-1]) if len(result) > 0 else None,
                        "mean": float(result.mean()),
                    }

            new_name = f"{dataset}_ta"
            stored = ctx.store_dataset(
                new_name,
                df,
                source_op="multi_indicator",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "multi_indicator",
                    "computed": list(results.keys()),
                    "errors": errors if errors else None,
                    "summaries": results,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "multi_indicator"}

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
        try:
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
                    "signal": (
                        "overbought"
                        if rsi_val > 70
                        else "oversold" if rsi_val < 30 else "neutral"
                    ),
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
                    "signal": (
                        "overbought"
                        if k_val > 80
                        else "oversold" if k_val < 20 else "neutral"
                    ),
                }
            except Exception:
                pass

            # Williams %R
            try:
                wr = ta.williams_r(prices, period=14)
                wr_val = float(wr.iloc[-1])
                signals["williams_r"] = {
                    "value": wr_val,
                    "signal": (
                        "overbought"
                        if wr_val > -20
                        else "oversold" if wr_val < -80 else "neutral"
                    ),
                }
            except Exception:
                pass

            # CCI
            try:
                cci = ta.cci(prices, period=20)
                cci_val = float(cci.iloc[-1])
                signals["cci"] = {
                    "value": cci_val,
                    "signal": (
                        "overbought"
                        if cci_val > 100
                        else "oversold" if cci_val < -100 else "neutral"
                    ),
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

            return _sanitize_for_json(
                {
                    "tool": "scan_signals",
                    "dataset": dataset,
                    "indicators": signals,
                    "consensus": consensus,
                    "overbought_count": n_ob,
                    "oversold_count": n_os,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "scan_signals"}

    # ------------------------------------------------------------------
    # Category-specific multi-indicator tools
    # ------------------------------------------------------------------

    @mcp.tool()
    def momentum_indicators(
        dataset: str,
        column: str = "close",
        period: int = 14,
    ) -> dict[str, Any]:
        """Compute key momentum indicators: RSI, MACD, Stochastic, ROC, Williams %R, CCI, MFI.

        Returns latest values for each and an overall momentum assessment
        (bullish/bearish/neutral based on consensus).

        Parameters:
            dataset: Dataset with price data.
            column: Price column.
            period: Lookback period (default 14).
        """
        try:
            import wraquant.ta as ta

            df = ctx.get_dataset(dataset)
            prices = df[column]
            results = {}

            indicators = {
                "rsi": lambda: ta.rsi(prices, period=period),
                "roc": lambda: ta.roc(prices, period=period),
                "williams_r": lambda: ta.williams_r(prices, period=period),
                "cci": lambda: ta.cci(prices, period=max(period, 20)),
                "cmo": lambda: ta.cmo(prices, period=period),
                "awesome_oscillator": lambda: ta.awesome_oscillator(prices),
            }

            for name, func in indicators.items():
                try:
                    result = func()
                    if isinstance(result, dict):
                        for k, v in result.items():
                            results[f"{name}_{k}"] = (
                                float(v.iloc[-1]) if len(v) > 0 else None
                            )
                            df[f"{name}_{k}"] = v.values[: len(df)]
                    elif hasattr(result, "iloc"):
                        results[name] = (
                            float(result.iloc[-1]) if len(result) > 0 else None
                        )
                        df[name] = result.values[: len(df)]
                except Exception:
                    pass

            # MACD (special — needs short/long/signal periods)
            try:
                macd_result = ta.macd(prices)
                if isinstance(macd_result, dict):
                    for k, v in macd_result.items():
                        results[f"macd_{k}"] = float(v.iloc[-1]) if len(v) > 0 else None
                        df[f"macd_{k}"] = v.values[: len(df)]
                elif hasattr(macd_result, "iloc"):
                    results["macd"] = float(macd_result.iloc[-1])
                    df["macd"] = macd_result.values[: len(df)]
            except Exception:
                pass

            # Momentum assessment
            bullish = 0
            bearish = 0
            rsi_val = results.get("rsi")
            if rsi_val is not None:
                if rsi_val > 50:
                    bullish += 1
                else:
                    bearish += 1
            roc_val = results.get("roc")
            if roc_val is not None:
                if roc_val > 0:
                    bullish += 1
                else:
                    bearish += 1
            wr_val = results.get("williams_r")
            if wr_val is not None:
                if wr_val > -50:
                    bullish += 1
                else:
                    bearish += 1

            assessment = (
                "bullish"
                if bullish > bearish
                else "bearish" if bearish > bullish else "neutral"
            )

            stored = ctx.store_dataset(
                f"{dataset}_momentum",
                df,
                source_op="momentum_indicators",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "momentum_indicators",
                    "indicators": results,
                    "assessment": assessment,
                    "bullish_count": bullish,
                    "bearish_count": bearish,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "momentum_indicators"}

    @mcp.tool()
    def trend_indicators(
        dataset: str,
        column: str = "close",
        period: int = 14,
    ) -> dict[str, Any]:
        """Compute trend indicators: ADX, Aroon, SuperTrend, PSAR, Ichimoku, TRIX.

        Returns latest values and a trend strength/direction assessment.

        Parameters:
            dataset: Dataset with price data.
            column: Price column.
            period: Lookback period (default 14).
        """
        try:
            import wraquant.ta as ta

            df = ctx.get_dataset(dataset)
            prices = df[column]
            results = {}

            indicators = {
                "adx": lambda: ta.adx(prices, period=period),
                "aroon": lambda: ta.aroon(prices, period=period),
                "trix": lambda: ta.trix(prices, period=period),
            }

            for name, func in indicators.items():
                try:
                    result = func()
                    if isinstance(result, dict):
                        for k, v in result.items():
                            results[f"{name}_{k}"] = (
                                float(v.iloc[-1]) if len(v) > 0 else None
                            )
                            df[f"{name}_{k}"] = v.values[: len(df)]
                    elif hasattr(result, "iloc"):
                        results[name] = (
                            float(result.iloc[-1]) if len(result) > 0 else None
                        )
                        df[name] = result.values[: len(df)]
                except Exception:
                    pass

            # PSAR
            try:
                psar = ta.psar(prices)
                if isinstance(psar, dict):
                    for k, v in psar.items():
                        results[f"psar_{k}"] = float(v.iloc[-1]) if len(v) > 0 else None
                elif hasattr(psar, "iloc"):
                    results["psar"] = float(psar.iloc[-1])
                    df["psar"] = psar.values[: len(df)]
            except Exception:
                pass

            # Trend assessment
            adx_val = results.get("adx") or results.get("adx_adx")
            trend_strength = "none"
            if adx_val is not None:
                if adx_val > 40:
                    trend_strength = "strong"
                elif adx_val > 25:
                    trend_strength = "moderate"
                elif adx_val > 15:
                    trend_strength = "weak"

            # Direction from price vs moving averages
            try:
                sma50 = ta.sma(prices, period=50)
                sma200 = ta.sma(prices, period=200)
                price_now = float(prices.iloc[-1])
                sma50_now = float(sma50.iloc[-1])
                sma200_now = float(sma200.iloc[-1])
                results["sma_50"] = sma50_now
                results["sma_200"] = sma200_now
                results["price"] = price_now

                if price_now > sma50_now > sma200_now:
                    trend_direction = "strong_uptrend"
                elif price_now > sma50_now:
                    trend_direction = "uptrend"
                elif price_now < sma50_now < sma200_now:
                    trend_direction = "strong_downtrend"
                elif price_now < sma50_now:
                    trend_direction = "downtrend"
                else:
                    trend_direction = "sideways"
            except Exception:
                trend_direction = "unknown"

            stored = ctx.store_dataset(
                f"{dataset}_trend",
                df,
                source_op="trend_indicators",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "trend_indicators",
                    "indicators": results,
                    "trend_strength": trend_strength,
                    "trend_direction": trend_direction,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "trend_indicators"}

    @mcp.tool()
    def volatility_indicators(
        dataset: str,
        column: str = "close",
        period: int = 14,
    ) -> dict[str, Any]:
        """Compute volatility indicators: ATR, Bollinger Bands, Keltner Channel, Donchian, BB Width.

        Returns latest values and volatility regime assessment (high/normal/low).

        Parameters:
            dataset: Dataset with price data.
            column: Price column.
            period: Lookback period (default 14).
        """
        try:
            import wraquant.ta as ta

            df = ctx.get_dataset(dataset)
            prices = df[column]
            results = {}

            # ATR
            try:
                atr = ta.atr(prices, period=period)
                results["atr"] = float(atr.iloc[-1]) if len(atr) > 0 else None
                results["atr_pct"] = (
                    results["atr"] / float(prices.iloc[-1]) * 100
                    if results["atr"]
                    else None
                )
                df["atr"] = atr.values[: len(df)]
            except Exception:
                pass

            # Bollinger Bands
            try:
                bb = ta.bollinger_bands(prices, period=20)
                if isinstance(bb, dict):
                    for k, v in bb.items():
                        results[f"bb_{k}"] = float(v.iloc[-1]) if len(v) > 0 else None
                        df[f"bb_{k}"] = v.values[: len(df)]
            except Exception:
                pass

            # BB Width
            try:
                bbw = ta.bbwidth(prices, period=20)
                results["bbwidth"] = float(bbw.iloc[-1]) if len(bbw) > 0 else None
                df["bbwidth"] = bbw.values[: len(df)]
            except Exception:
                pass

            # Keltner Channel
            try:
                kc = ta.keltner_channel(prices, period=20)
                if isinstance(kc, dict):
                    for k, v in kc.items():
                        results[f"kc_{k}"] = float(v.iloc[-1]) if len(v) > 0 else None
            except Exception:
                pass

            # Donchian Channel
            try:
                dc = ta.donchian_channel(prices, period=20)
                if isinstance(dc, dict):
                    for k, v in dc.items():
                        results[f"dc_{k}"] = float(v.iloc[-1]) if len(v) > 0 else None
            except Exception:
                pass

            # Historical vol
            try:
                hv = ta.historical_volatility(prices, period=21)
                results["hist_vol_21d"] = float(hv.iloc[-1]) if len(hv) > 0 else None
            except Exception:
                pass

            # Vol regime
            bbw = results.get("bbwidth")
            if bbw is not None:
                vol_regime = "high" if bbw > 0.1 else "low" if bbw < 0.03 else "normal"
            else:
                vol_regime = "unknown"

            stored = ctx.store_dataset(
                f"{dataset}_vol_ta",
                df,
                source_op="volatility_indicators",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "volatility_indicators",
                    "indicators": results,
                    "vol_regime": vol_regime,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "volatility_indicators"}

    @mcp.tool()
    def volume_indicators(
        dataset: str,
        price_col: str = "close",
        volume_col: str = "volume",
    ) -> dict[str, Any]:
        """Compute volume indicators: OBV, AD line, CMF, MFI, Force Index, VWAP.

        Returns latest values and volume confirmation assessment.

        Parameters:
            dataset: Dataset with price and volume data.
            price_col: Price column.
            volume_col: Volume column.
        """
        try:
            import wraquant.ta as ta

            df = ctx.get_dataset(dataset)
            prices = df[price_col]
            results = {}

            # OBV
            try:
                obv = ta.obv(prices, df[volume_col])
                results["obv"] = float(obv.iloc[-1]) if len(obv) > 0 else None
                # OBV trend: compare to 20-day ago
                if len(obv) > 20:
                    results["obv_trend"] = (
                        "rising" if obv.iloc[-1] > obv.iloc[-20] else "falling"
                    )
                df["obv"] = obv.values[: len(df)]
            except Exception:
                pass

            # AD Line
            try:
                ad = ta.ad_line(prices, df[volume_col])
                results["ad_line"] = float(ad.iloc[-1]) if len(ad) > 0 else None
                df["ad_line"] = ad.values[: len(df)]
            except Exception:
                pass

            # CMF
            try:
                cmf = ta.cmf(prices, df[volume_col], period=20)
                results["cmf"] = float(cmf.iloc[-1]) if len(cmf) > 0 else None
                results["cmf_signal"] = (
                    "bullish"
                    if results["cmf"] and results["cmf"] > 0.05
                    else (
                        "bearish"
                        if results["cmf"] and results["cmf"] < -0.05
                        else "neutral"
                    )
                )
                df["cmf"] = cmf.values[: len(df)]
            except Exception:
                pass

            # MFI
            try:
                mfi = ta.mfi(prices, df[volume_col], period=14)
                results["mfi"] = float(mfi.iloc[-1]) if len(mfi) > 0 else None
                results["mfi_signal"] = (
                    "overbought"
                    if results["mfi"] and results["mfi"] > 80
                    else (
                        "oversold"
                        if results["mfi"] and results["mfi"] < 20
                        else "neutral"
                    )
                )
                df["mfi"] = mfi.values[: len(df)]
            except Exception:
                pass

            # Force Index
            try:
                fi = ta.force_index(prices, df[volume_col])
                results["force_index"] = float(fi.iloc[-1]) if len(fi) > 0 else None
            except Exception:
                pass

            # Volume confirmation
            price_up = (
                float(prices.iloc[-1]) > float(prices.iloc[-5])
                if len(prices) > 5
                else None
            )
            obv_up = results.get("obv_trend") == "rising"
            if price_up is not None:
                if price_up and obv_up:
                    vol_confirmation = "confirmed_up"
                elif not price_up and not obv_up:
                    vol_confirmation = "confirmed_down"
                elif price_up and not obv_up:
                    vol_confirmation = "divergence_bearish"
                else:
                    vol_confirmation = "divergence_bullish"
            else:
                vol_confirmation = "unknown"

            stored = ctx.store_dataset(
                f"{dataset}_volume_ta",
                df,
                source_op="volume_indicators",
                parent=dataset,
            )

            return _sanitize_for_json(
                {
                    "tool": "volume_indicators",
                    "indicators": results,
                    "volume_confirmation": vol_confirmation,
                    **stored,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "volume_indicators"}

    @mcp.tool()
    def pattern_recognition(
        dataset: str,
        column: str = "close",
    ) -> dict[str, Any]:
        """Detect candlestick patterns: doji, hammer, engulfing, morning/evening star, etc.

        Returns detected patterns with their bullish/bearish signal and confidence.

        Parameters:
            dataset: Dataset with price data (needs OHLC columns ideally).
            column: Close price column.
        """
        try:
            import wraquant.ta as ta

            df = ctx.get_dataset(dataset)
            prices = df[column]
            detected = []

            # Pattern functions to check
            patterns = {
                "doji": ("neutral", lambda: ta.doji(prices)),
                "hammer": ("bullish", lambda: ta.hammer(prices)),
                "engulfing": ("varies", lambda: ta.engulfing(prices)),
                "morning_star": ("bullish", lambda: ta.morning_star(prices)),
                "evening_star": ("bearish", lambda: ta.evening_star(prices)),
                "shooting_star": ("bearish", lambda: ta.shooting_star(prices)),
                "hanging_man": ("bearish", lambda: ta.hanging_man(prices)),
                "harami": ("varies", lambda: ta.harami(prices)),
                "dark_cloud_cover": ("bearish", lambda: ta.dark_cloud_cover(prices)),
            }

            for name, (bias, func) in patterns.items():
                try:
                    result = func()
                    if hasattr(result, "iloc"):
                        last_val = result.iloc[-1]
                        if last_val != 0 and last_val is not None:
                            detected.append(
                                {
                                    "pattern": name,
                                    "bias": bias,
                                    "value": float(last_val),
                                    "bars_ago": 0,
                                }
                            )
                        # Check last 3 bars
                        for i in range(1, min(4, len(result))):
                            val = result.iloc[-1 - i]
                            if val != 0 and val is not None:
                                detected.append(
                                    {
                                        "pattern": name,
                                        "bias": bias,
                                        "value": float(val),
                                        "bars_ago": i,
                                    }
                                )
                except Exception:
                    pass

            bullish = sum(
                1 for p in detected if p["bias"] == "bullish" and p["bars_ago"] <= 1
            )
            bearish = sum(
                1 for p in detected if p["bias"] == "bearish" and p["bars_ago"] <= 1
            )

            return _sanitize_for_json(
                {
                    "tool": "pattern_recognition",
                    "dataset": dataset,
                    "detected": detected[:20],
                    "total_detected": len(detected),
                    "recent_bullish": bullish,
                    "recent_bearish": bearish,
                    "pattern_signal": (
                        "bullish"
                        if bullish > bearish
                        else "bearish" if bearish > bullish else "neutral"
                    ),
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "pattern_recognition"}

    @mcp.tool()
    def support_resistance(
        dataset: str,
        column: str = "close",
        n_levels: int = 5,
    ) -> dict[str, Any]:
        """Compute support and resistance levels from price data.

        Uses multiple methods: fractal levels, price clustering, and pivot points.

        Parameters:
            dataset: Dataset with price data.
            column: Price column.
            n_levels: Number of S/R levels to identify (default 5).
        """
        try:
            import numpy as np

            import wraquant.ta as ta

            df = ctx.get_dataset(dataset)
            prices = df[column].dropna()
            current_price = float(prices.iloc[-1])
            results = {"current_price": current_price}

            # Try wraquant S/R functions
            try:
                sr = ta.find_support_resistance(prices)
                if isinstance(sr, dict):
                    results["support_levels"] = [
                        float(x) for x in sr.get("support", [])[:n_levels]
                    ]
                    results["resistance_levels"] = [
                        float(x) for x in sr.get("resistance", [])[:n_levels]
                    ]
            except Exception:
                # Fallback: simple percentile-based levels
                support = []
                resistance = []
                for pct in [10, 20, 30]:
                    level = float(np.percentile(prices, pct))
                    if level < current_price:
                        support.append(level)
                for pct in [70, 80, 90]:
                    level = float(np.percentile(prices, pct))
                    if level > current_price:
                        resistance.append(level)
                results["support_levels"] = support[:n_levels]
                results["resistance_levels"] = resistance[:n_levels]

            # Fibonacci levels
            try:
                fib = ta.fibonacci_retracements(prices)
                if isinstance(fib, dict):
                    results["fibonacci"] = {
                        k: float(v)
                        for k, v in fib.items()
                        if isinstance(v, (int, float, np.floating))
                    }
            except Exception:
                # Manual fib
                high = float(prices.max())
                low = float(prices.min())
                diff = high - low
                results["fibonacci"] = {
                    "0.0": high,
                    "0.236": high - 0.236 * diff,
                    "0.382": high - 0.382 * diff,
                    "0.5": high - 0.5 * diff,
                    "0.618": high - 0.618 * diff,
                    "1.0": low,
                }

            # Pivot points
            if all(c in df.columns for c in ["high", "low", "close"]):
                h = float(df["high"].iloc[-1])
                lo = float(df["low"].iloc[-1])
                c = float(df["close"].iloc[-1])
                pivot = (h + lo + c) / 3
                results["pivot_points"] = {
                    "R2": pivot + (h - lo),
                    "R1": 2 * pivot - lo,
                    "P": pivot,
                    "S1": 2 * pivot - h,
                    "S2": pivot - (h - lo),
                }

            # Nearest levels
            all_supports = results.get("support_levels", [])
            all_resistances = results.get("resistance_levels", [])
            if all_supports:
                results["nearest_support"] = max(all_supports)
                results["support_distance_pct"] = (
                    (current_price - results["nearest_support"]) / current_price * 100
                )
            if all_resistances:
                results["nearest_resistance"] = min(all_resistances)
                results["resistance_distance_pct"] = (
                    (results["nearest_resistance"] - current_price)
                    / current_price
                    * 100
                )

            return _sanitize_for_json(
                {
                    "tool": "support_resistance",
                    "dataset": dataset,
                    **results,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "support_resistance"}

    @mcp.tool()
    def ta_summary(
        dataset: str,
        column: str = "close",
    ) -> dict[str, Any]:
        """Comprehensive TA summary: momentum, trend, volume, volatility, S/R levels.

        Computes the most important indicators from each category and provides
        an overall technical assessment (bullish/bearish/neutral).

        Parameters:
            dataset: Dataset with price data.
            column: Price column.
        """
        try:
            import wraquant.ta as ta

            df = ctx.get_dataset(dataset)
            prices = df[column]
            current_price = float(prices.iloc[-1])
            summary = {"current_price": current_price}
            scores = {"bullish": 0, "bearish": 0}

            # Momentum
            try:
                rsi_val = float(ta.rsi(prices, period=14).iloc[-1])
                summary["rsi_14"] = rsi_val
                if rsi_val > 50:
                    scores["bullish"] += 1
                else:
                    scores["bearish"] += 1
            except Exception:
                pass

            try:
                macd_result = ta.macd(prices)
                if isinstance(macd_result, dict):
                    vals = list(macd_result.values())
                    if len(vals) >= 2:
                        macd_line = float(vals[0].iloc[-1])
                        signal_line = float(vals[1].iloc[-1])
                        summary["macd"] = macd_line
                        summary["macd_signal"] = signal_line
                        if macd_line > signal_line:
                            scores["bullish"] += 1
                        else:
                            scores["bearish"] += 1
            except Exception:
                pass

            # Trend
            try:
                sma20 = float(ta.sma(prices, period=20).iloc[-1])
                sma50 = float(ta.sma(prices, period=50).iloc[-1])
                sma200 = float(ta.sma(prices, period=200).iloc[-1])
                summary["sma_20"] = sma20
                summary["sma_50"] = sma50
                summary["sma_200"] = sma200

                if current_price > sma20:
                    scores["bullish"] += 1
                else:
                    scores["bearish"] += 1
                if current_price > sma50:
                    scores["bullish"] += 1
                else:
                    scores["bearish"] += 1
                if sma50 > sma200:
                    scores["bullish"] += 1
                    summary["golden_cross"] = True
                else:
                    scores["bearish"] += 1
                    summary["death_cross"] = True
            except Exception:
                pass

            try:
                adx_result = ta.adx(prices, period=14)
                if isinstance(adx_result, dict):
                    adx_val = float(list(adx_result.values())[0].iloc[-1])
                else:
                    adx_val = float(adx_result.iloc[-1])
                summary["adx"] = adx_val
                summary["trend_strength"] = (
                    "strong" if adx_val > 40 else "moderate" if adx_val > 25 else "weak"
                )
            except Exception:
                pass

            # Volatility
            try:
                atr = ta.atr(prices, period=14)
                summary["atr_14"] = float(atr.iloc[-1])
                summary["atr_pct"] = summary["atr_14"] / current_price * 100
            except Exception:
                pass

            try:
                bb = ta.bollinger_bands(prices, period=20)
                if isinstance(bb, dict):
                    vals = list(bb.values())
                    if len(vals) >= 3:
                        upper = float(vals[0].iloc[-1])
                        float(vals[1].iloc[-1])
                        lower = float(vals[2].iloc[-1])
                        summary["bb_upper"] = upper
                        summary["bb_lower"] = lower
                        summary["bb_pct_b"] = (
                            (current_price - lower) / (upper - lower)
                            if upper != lower
                            else 0.5
                        )
            except Exception:
                pass

            # Overall assessment
            total = scores["bullish"] + scores["bearish"]
            if total > 0:
                bull_pct = scores["bullish"] / total * 100
                if bull_pct >= 65:
                    overall = "bullish"
                elif bull_pct <= 35:
                    overall = "bearish"
                else:
                    overall = "neutral"
            else:
                overall = "unknown"
                bull_pct = 50

            return _sanitize_for_json(
                {
                    "tool": "ta_summary",
                    "dataset": dataset,
                    "summary": summary,
                    "scores": scores,
                    "bullish_pct": bull_pct,
                    "overall_assessment": overall,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "ta_summary"}

    @mcp.tool()
    def ta_screening(
        dataset: str,
        column: str = "close",
    ) -> dict[str, Any]:
        """Screen multiple indicators for buy/sell signals.

        Generates a signal score from -1 (strong sell) to +1 (strong buy) based on
        consensus across RSI, MACD, Stochastic, ADX, moving average crossovers,
        and volume indicators.

        Parameters:
            dataset: Dataset with price data.
            column: Price column.
        """
        try:
            import wraquant.ta as ta

            df = ctx.get_dataset(dataset)
            prices = df[column]
            signals = []

            # RSI signal
            try:
                rsi = ta.rsi(prices, period=14)
                rsi_val = float(rsi.iloc[-1])
                if rsi_val < 30:
                    signals.append(("rsi_oversold", "buy", 1.0))
                elif rsi_val > 70:
                    signals.append(("rsi_overbought", "sell", -1.0))
                elif rsi_val < 45:
                    signals.append(("rsi_weak", "lean_buy", 0.3))
                elif rsi_val > 55:
                    signals.append(("rsi_strong", "lean_sell", -0.3))
            except Exception:
                pass

            # MACD crossover
            try:
                macd_result = ta.macd(prices)
                if isinstance(macd_result, dict):
                    vals = list(macd_result.values())
                    if len(vals) >= 2:
                        macd_line = vals[0]
                        signal_line = vals[1]
                        if len(macd_line) > 1:
                            curr_cross = float(macd_line.iloc[-1]) - float(
                                signal_line.iloc[-1]
                            )
                            prev_cross = float(macd_line.iloc[-2]) - float(
                                signal_line.iloc[-2]
                            )
                            if curr_cross > 0 and prev_cross <= 0:
                                signals.append(("macd_bullish_cross", "buy", 1.0))
                            elif curr_cross < 0 and prev_cross >= 0:
                                signals.append(("macd_bearish_cross", "sell", -1.0))
                            elif curr_cross > 0:
                                signals.append(("macd_above_signal", "lean_buy", 0.3))
                            else:
                                signals.append(("macd_below_signal", "lean_sell", -0.3))
            except Exception:
                pass

            # Moving average crossover
            try:
                sma20 = ta.sma(prices, period=20)
                sma50 = ta.sma(prices, period=50)
                if len(sma20) > 1 and len(sma50) > 1:
                    curr = float(sma20.iloc[-1]) - float(sma50.iloc[-1])
                    prev = float(sma20.iloc[-2]) - float(sma50.iloc[-2])
                    if curr > 0 and prev <= 0:
                        signals.append(("sma_golden_cross", "buy", 1.0))
                    elif curr < 0 and prev >= 0:
                        signals.append(("sma_death_cross", "sell", -1.0))
                    elif curr > 0:
                        signals.append(("sma_uptrend", "lean_buy", 0.2))
                    else:
                        signals.append(("sma_downtrend", "lean_sell", -0.2))
            except Exception:
                pass

            # Bollinger Band position
            try:
                bb = ta.bollinger_bands(prices, period=20)
                if isinstance(bb, dict):
                    vals = list(bb.values())
                    if len(vals) >= 3:
                        upper = float(vals[0].iloc[-1])
                        lower = float(vals[2].iloc[-1])
                        price = float(prices.iloc[-1])
                        if price < lower:
                            signals.append(("bb_below_lower", "buy", 0.7))
                        elif price > upper:
                            signals.append(("bb_above_upper", "sell", -0.7))
            except Exception:
                pass

            # Compute signal score
            if signals:
                score = sum(s[2] for s in signals) / len(signals)
            else:
                score = 0.0

            if score > 0.5:
                recommendation = "strong_buy"
            elif score > 0.2:
                recommendation = "buy"
            elif score > -0.2:
                recommendation = "hold"
            elif score > -0.5:
                recommendation = "sell"
            else:
                recommendation = "strong_sell"

            return _sanitize_for_json(
                {
                    "tool": "ta_screening",
                    "dataset": dataset,
                    "signals": [
                        {"name": s[0], "action": s[1], "score": s[2]} for s in signals
                    ],
                    "signal_count": len(signals),
                    "composite_score": score,
                    "recommendation": recommendation,
                }
            )
        except Exception as e:
            return {"error": str(e), "tool": "ta_screening"}
