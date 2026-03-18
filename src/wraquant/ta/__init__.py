"""Technical analysis indicators for wraquant.

This package provides a comprehensive set of technical analysis indicators
organized into the following sub-modules:

- **overlap** — Moving averages and overlay studies (SMA, EMA, Bollinger
  Bands, Ichimoku, etc.)
- **momentum** — Oscillators measuring speed/magnitude of price moves
  (RSI, MACD, Stochastic, etc.)
- **volatility** — Measures of price dispersion (ATR, Bollinger Width,
  Historical Volatility, etc.)
- **volume** — Volume-based indicators (OBV, CMF, MFI, etc.)
- **trend** — Trend direction and strength indicators (ADX, Aroon, PSAR,
  etc.)
- **patterns** — Candlestick pattern recognition (Doji, Engulfing,
  Morning Star, etc.)
- **signals** — Signal generation utilities (crossover, crossunder,
  normalize, etc.)

All indicator functions accept ``pd.Series`` inputs and return either a
``pd.Series`` or a ``dict[str, pd.Series]`` for multi-output indicators.
"""

from __future__ import annotations

# ── Momentum Oscillators ─────────────────────────────────────────────────
from wraquant.ta.momentum import (
    awesome_oscillator,
    cci,
    cmo,
    dpo,
    macd,
    momentum,
    ppo,
    roc,
    rsi,
    stochastic,
    stochastic_rsi,
    tsi,
    ultimate_oscillator,
    williams_r,
)

# ── Overlap / Moving Averages ────────────────────────────────────────────
from wraquant.ta.overlap import (
    bollinger_bands,
    dema,
    donchian_channel,
    ema,
    ichimoku,
    kama,
    keltner_channel,
    sma,
    supertrend,
    tema,
    vwap,
    wma,
)

# ── Patterns ─────────────────────────────────────────────────────────────
from wraquant.ta.patterns import (
    doji,
    engulfing,
    evening_star,
    hammer,
    harami,
    marubozu,
    morning_star,
    spinning_top,
    three_black_crows,
    three_white_soldiers,
)

# ── Signals ──────────────────────────────────────────────────────────────
from wraquant.ta.signals import (
    above,
    below,
    crossover,
    crossunder,
    falling,
    highest,
    lowest,
    normalize,
    rising,
)

# ── Trend ────────────────────────────────────────────────────────────────
from wraquant.ta.trend import (
    adx,
    aroon,
    linear_regression,
    linear_regression_slope,
    psar,
    trix,
    vortex,
)

# ── Volatility ───────────────────────────────────────────────────────────
from wraquant.ta.volatility import (
    atr,
    bbwidth,
    chaikin_volatility,
    historical_volatility,
    kc_width,
    mass_index,
    natr,
    true_range,
)

# ── Volume ───────────────────────────────────────────────────────────────
from wraquant.ta.volume import (
    ad_line,
    adosc,
    cmf,
    eom,
    force_index,
    mfi,
    nvi,
    obv,
    pvi,
    vpt,
)

__all__ = [
    # Overlap
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
    # Momentum
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
    # Volatility
    "atr",
    "true_range",
    "natr",
    "bbwidth",
    "kc_width",
    "chaikin_volatility",
    "historical_volatility",
    "mass_index",
    # Volume
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
    # Trend
    "adx",
    "aroon",
    "psar",
    "vortex",
    "trix",
    "linear_regression",
    "linear_regression_slope",
    # Patterns
    "doji",
    "hammer",
    "engulfing",
    "morning_star",
    "evening_star",
    "three_white_soldiers",
    "three_black_crows",
    "harami",
    "spinning_top",
    "marubozu",
    # Signals
    "crossover",
    "crossunder",
    "above",
    "below",
    "rising",
    "falling",
    "highest",
    "lowest",
    "normalize",
]
