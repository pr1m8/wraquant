"""TA Screener page -- apply technical analysis indicators to price data.

Upload OHLCV data, choose from wraquant's 265 TA indicators, and
visualize the results on interactive charts.  Supports multi-indicator
overlay and side-by-side comparison.
"""

from __future__ import annotations

# Indicator registry: maps display name -> (module_path, function_name, input_type)
# input_type: "close" = pd.Series, "ohlcv" = requires OHLCV columns,
#             "ohlc" = requires OHLC, "hilo" = high+low, "hlcv" = high+low+close+volume
_INDICATOR_REGISTRY: dict[str, tuple[str, str, str]] = {
    # Overlap
    "SMA": ("wraquant.ta.overlap", "sma", "close"),
    "EMA": ("wraquant.ta.overlap", "ema", "close"),
    "WMA": ("wraquant.ta.overlap", "wma", "close"),
    "DEMA": ("wraquant.ta.overlap", "dema", "close"),
    "TEMA": ("wraquant.ta.overlap", "tema", "close"),
    "KAMA": ("wraquant.ta.overlap", "kama", "close"),
    "Bollinger Bands": ("wraquant.ta.overlap", "bollinger_bands", "close"),
    # Momentum
    "RSI": ("wraquant.ta.momentum", "rsi", "close"),
    "MACD": ("wraquant.ta.momentum", "macd", "close"),
    "Stochastic": ("wraquant.ta.momentum", "stochastic", "hlc"),
    "Williams %R": ("wraquant.ta.momentum", "williams_r", "hlc"),
    "CCI": ("wraquant.ta.momentum", "cci", "hlc"),
    "ROC": ("wraquant.ta.momentum", "roc", "close"),
    "Momentum": ("wraquant.ta.momentum", "momentum", "close"),
    "TSI": ("wraquant.ta.momentum", "tsi", "close"),
    "Awesome Oscillator": ("wraquant.ta.momentum", "awesome_oscillator", "hl"),
    "CMO": ("wraquant.ta.momentum", "cmo", "close"),
    "Stochastic RSI": ("wraquant.ta.momentum", "stochastic_rsi", "close"),
    # Volume
    "OBV": ("wraquant.ta.volume", "obv", "cv"),
    "CMF": ("wraquant.ta.volume", "cmf", "hlcv"),
    "MFI": ("wraquant.ta.volume", "mfi", "hlcv"),
    "Force Index": ("wraquant.ta.volume", "force_index", "cv"),
    "EOM": ("wraquant.ta.volume", "eom", "hlv"),
    # Trend
    "ADX": ("wraquant.ta.trend", "adx", "hlc"),
    "Aroon": ("wraquant.ta.trend", "aroon", "hl"),
    "PSAR": ("wraquant.ta.trend", "psar", "hlc"),
    "Vortex": ("wraquant.ta.trend", "vortex", "hlc"),
    "TRIX": ("wraquant.ta.trend", "trix", "close"),
    # Volatility
    "ATR": ("wraquant.ta.volatility", "atr", "hlc"),
    "True Range": ("wraquant.ta.volatility", "true_range", "hlc"),
    "NATR": ("wraquant.ta.volatility", "natr", "hlc"),
    "Bollinger Width": ("wraquant.ta.volatility", "bbwidth", "close"),
    "Historical Volatility": ("wraquant.ta.volatility", "historical_volatility", "close"),
    "Ulcer Index": ("wraquant.ta.volatility", "ulcer_index", "close"),
    # Exotic
    "Choppiness Index": ("wraquant.ta.exotic", "choppiness_index", "hlc"),
    "Elder Thermometer": ("wraquant.ta.exotic", "elder_thermometer", "hl"),
}


def _call_indicator(
    name: str,
    df: "pd.DataFrame",  # noqa: F821
    period: int,
) -> "pd.Series | dict[str, pd.Series] | None":  # noqa: F821
    """Dynamically call a TA indicator function."""
    import importlib

    if name not in _INDICATOR_REGISTRY:
        return None

    module_path, func_name, input_type = _INDICATOR_REGISTRY[name]
    mod = importlib.import_module(module_path)
    func = getattr(mod, func_name)

    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    volume = df.get("volume")

    try:
        if input_type == "close":
            return func(close, period=period)
        elif input_type == "hlc":
            return func(high, low, close, period=period)
        elif input_type == "hl":
            return func(high, low, period=period)
        elif input_type == "cv":
            return func(close, volume, period=period) if volume is not None else None
        elif input_type == "hlcv":
            if volume is not None:
                return func(high, low, close, volume, period=period)
            return None
        elif input_type == "hlv":
            if volume is not None:
                return func(high, low, volume, period=period)
            return None
        else:
            return func(close, period=period)
    except TypeError:
        # Some indicators don't take a period argument -- retry without it
        try:
            if input_type == "close":
                return func(close)
            elif input_type == "hlc":
                return func(high, low, close)
            elif input_type == "hl":
                return func(high, low)
            elif input_type == "cv":
                return func(close, volume) if volume is not None else None
            elif input_type == "hlcv":
                if volume is not None:
                    return func(high, low, close, volume)
                return None
            elif input_type == "hlv":
                if volume is not None:
                    return func(high, low, volume)
                return None
            else:
                return func(close)
        except Exception:  # noqa: BLE001
            return None


def render() -> None:
    """Render the TA Screener page."""
    import streamlit as st

    st.header("TA Screener")

    upload = st.file_uploader(
        "Upload OHLCV CSV (columns: open, high, low, close, volume)",
        type=["csv"],
        key="ta_upload",
    )

    if upload is not None:
        import pandas as pd

        df = pd.read_csv(upload, index_col=0, parse_dates=True)

        # Normalise column names to lowercase
        df.columns = [c.lower().strip() for c in df.columns]

        if "close" not in df.columns:
            st.error("CSV must contain a 'close' column.")
            return

        st.write(
            f"Loaded **{len(df)}** bars with columns: "
            f"{', '.join(df.columns)}",
        )

        # Price chart
        st.subheader("Price Chart")
        st.line_chart(df["close"])

        # Indicator selection
        st.subheader("Apply Indicators")
        available = sorted(_INDICATOR_REGISTRY.keys())
        selected = st.multiselect(
            "Choose indicators",
            available,
            default=["RSI", "SMA"],
        )

        period = st.slider("Period", 2, 200, 14)

        if selected:
            for ind_name in selected:
                result = _call_indicator(ind_name, df, period)
                if result is None:
                    st.warning(
                        f"Could not compute {ind_name} "
                        f"(missing required columns or dependency).",
                    )
                elif isinstance(result, dict):
                    st.subheader(ind_name)
                    chart_df = pd.DataFrame(result)
                    st.line_chart(chart_df.dropna())
                elif isinstance(result, pd.Series):
                    st.subheader(ind_name)
                    st.line_chart(result.dropna())
                else:
                    st.subheader(ind_name)
                    st.write(result)
    else:
        st.info(
            "Upload an OHLCV CSV to screen with technical indicators. "
            "Columns should include at least 'close'. "
            "For full indicator support, include 'open', 'high', 'low', 'volume'.",
        )
        st.code(
            """\
# Available indicator categories:
# - Overlap: SMA, EMA, WMA, DEMA, TEMA, KAMA, Bollinger Bands
# - Momentum: RSI, MACD, Stochastic, Williams %R, CCI, ROC, TSI
# - Volume: OBV, CMF, MFI, Force Index, EOM
# - Trend: ADX, Aroon, PSAR, Vortex, TRIX
# - Volatility: ATR, True Range, NATR, Bollinger Width
# - Exotic: Choppiness Index, Elder Thermometer
#
# Example usage in code:
from wraquant.ta import rsi, macd, bollinger_bands
signals = rsi(prices, period=14)
""",
            language="python",
        )
