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
    abandoned_baby,
    belt_hold,
    dark_cloud_cover,
    doji,
    engulfing,
    evening_star,
    hammer,
    hanging_man,
    harami,
    inverted_hammer,
    kicking,
    marubozu,
    morning_star,
    piercing_pattern,
    shooting_star,
    spinning_top,
    three_black_crows,
    three_inside_down,
    three_inside_up,
    three_white_soldiers,
    tweezer_bottom,
    tweezer_top,
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
    fractal_adaptive_ma,
    guppy_mma,
    heikin_ashi,
    hull_ma,
    linear_regression,
    linear_regression_slope,
    mcginley_dynamic,
    psar,
    rainbow_ma,
    schaff_trend_cycle,
    tilson_t3,
    trix,
    vidya,
    vortex,
    zero_lag_ema,
    zigzag,
)

# ── Volatility ───────────────────────────────────────────────────────────
from wraquant.ta.volatility import (
    acceleration_bands,
    atr,
    bbwidth,
    chaikin_volatility,
    close_to_close,
    garman_klass,
    historical_volatility,
    kc_width,
    mass_index,
    natr,
    parkinson,
    relative_volatility_index,
    rogers_satchell,
    standard_deviation,
    true_range,
    ulcer_index,
    variance,
    yang_zhang,
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

# ── Statistics ────────────────────────────────────────────────────────
from wraquant.ta.statistics import (
    beta,
    correlation,
    entropy,
    hurst_exponent,
    information_coefficient,
    kurtosis,
    mean_deviation,
    median,
    percentile_rank,
    r_squared,
    skewness,
    zscore,
)

# ── Cycles ────────────────────────────────────────────────────────────
from wraquant.ta.cycles import (
    bandpass_filter,
    decycler,
    even_better_sinewave,
    hilbert_instantaneous_phase,
    hilbert_transform_dominant_period,
    hilbert_transform_trend_mode,
    roofing_filter,
    sine_wave,
)

# ── Custom / Advanced ────────────────────────────────────────────────
from wraquant.ta.custom import (
    adaptive_rsi,
    anchored_vwap,
    ehlers_fisher,
    linear_regression_channel,
    market_structure,
    pivot_points,
    relative_strength,
    squeeze_momentum,
    swing_points,
    volume_weighted_macd,
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
    "garman_klass",
    "parkinson",
    "rogers_satchell",
    "yang_zhang",
    "close_to_close",
    "ulcer_index",
    "relative_volatility_index",
    "acceleration_bands",
    "standard_deviation",
    "variance",
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
    "zigzag",
    "heikin_ashi",
    "mcginley_dynamic",
    "schaff_trend_cycle",
    "guppy_mma",
    "rainbow_ma",
    "hull_ma",
    "zero_lag_ema",
    "vidya",
    "tilson_t3",
    "fractal_adaptive_ma",
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
    "piercing_pattern",
    "dark_cloud_cover",
    "hanging_man",
    "inverted_hammer",
    "shooting_star",
    "tweezer_top",
    "tweezer_bottom",
    "three_inside_up",
    "three_inside_down",
    "abandoned_baby",
    "kicking",
    "belt_hold",
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
    # Statistics
    "zscore",
    "percentile_rank",
    "mean_deviation",
    "median",
    "skewness",
    "kurtosis",
    "entropy",
    "hurst_exponent",
    "correlation",
    "beta",
    "r_squared",
    "information_coefficient",
    # Cycles
    "hilbert_transform_dominant_period",
    "hilbert_transform_trend_mode",
    "hilbert_instantaneous_phase",
    "sine_wave",
    "even_better_sinewave",
    "roofing_filter",
    "decycler",
    "bandpass_filter",
    # Custom / Advanced
    "squeeze_momentum",
    "anchored_vwap",
    "linear_regression_channel",
    "pivot_points",
    "market_structure",
    "swing_points",
    "volume_weighted_macd",
    "ehlers_fisher",
    "adaptive_rsi",
    "relative_strength",
]
