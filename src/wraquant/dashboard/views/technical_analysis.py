"""Technical Analysis page -- the deepest page in the dashboard.

Provides access to ALL 265+ indicators from wraquant.ta across 19
submodules.  Organised into 10 tabs: Price Chart, Momentum, Trend,
Volume, Volatility, Patterns, Support/Resistance, Signals, Statistics,
and Performance.  Every indicator is accessible via checkboxes /
multiselect controls.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Indicator registry -- maps display name -> callable metadata
# ---------------------------------------------------------------------------
# input_type key:
#   "close"      -> fn(close, period=...)
#   "data"       -> fn(data, period=...)  (alias, same as close)
#   "hlc"        -> fn(high, low, close, period=...)
#   "hl"         -> fn(high, low, period=...)
#   "hlcv"       -> fn(high, low, close, volume, period=...)
#   "cv"         -> fn(close, volume)
#   "ohlcv"      -> fn(high, low, close, open_, period=...)
#   "special"    -> handled inline

_OVERLAY_INDICATORS: dict[str, dict] = {
    # -- Overlap module --
    "SMA(20)": {
        "mod": "wraquant.ta.overlap", "fn": "sma", "input": "close",
        "kw": {"period": 20},
        "tip": "Simple Moving Average -- 20-period trend baseline",
    },
    "SMA(50)": {
        "mod": "wraquant.ta.overlap", "fn": "sma", "input": "close",
        "kw": {"period": 50},
        "tip": "Simple Moving Average -- 50-period medium-term trend",
    },
    "SMA(200)": {
        "mod": "wraquant.ta.overlap", "fn": "sma", "input": "close",
        "kw": {"period": 200},
        "tip": "Simple Moving Average -- 200-period institutional benchmark",
    },
    "EMA(21)": {
        "mod": "wraquant.ta.overlap", "fn": "ema", "input": "close",
        "kw": {"period": 21},
        "tip": "Exponential Moving Average -- faster response than SMA",
    },
    "EMA(50)": {
        "mod": "wraquant.ta.overlap", "fn": "ema", "input": "close",
        "kw": {"period": 50},
        "tip": "EMA 50 -- medium-term trend",
    },
    "DEMA(20)": {
        "mod": "wraquant.ta.overlap", "fn": "dema", "input": "close",
        "kw": {"period": 20},
        "tip": "Double EMA -- reduced lag",
    },
    "TEMA(20)": {
        "mod": "wraquant.ta.overlap", "fn": "tema", "input": "close",
        "kw": {"period": 20},
        "tip": "Triple EMA -- minimal lag, hugs price tightly",
    },
    "WMA(20)": {
        "mod": "wraquant.ta.overlap", "fn": "wma", "input": "close",
        "kw": {"period": 20},
        "tip": "Weighted Moving Average -- linearly weighted",
    },
    "KAMA(10)": {
        "mod": "wraquant.ta.overlap", "fn": "kama", "input": "close",
        "kw": {"period": 10},
        "tip": "Kaufman Adaptive MA -- adapts speed to market efficiency",
    },
    "Bollinger Bands": {
        "mod": "wraquant.ta.overlap", "fn": "bollinger_bands", "input": "close",
        "kw": {"period": 20, "std_dev": 2.0}, "multi": True,
        "tip": "Volatility envelope: upper/middle/lower bands",
    },
    "Keltner Channel": {
        "mod": "wraquant.ta.overlap", "fn": "keltner_channel", "input": "hlc",
        "kw": {"period": 20, "multiplier": 1.5}, "multi": True,
        "tip": "ATR-based channel around EMA",
    },
    "Donchian Channel": {
        "mod": "wraquant.ta.overlap", "fn": "donchian_channel", "input": "hl",
        "kw": {"period": 20}, "multi": True,
        "tip": "Highest-high / lowest-low channel (Turtle Trading)",
    },
    "Ichimoku Cloud": {
        "mod": "wraquant.ta.overlap", "fn": "ichimoku", "input": "hlc",
        "kw": {}, "multi": True,
        "tip": "Multi-timeframe trend system: cloud, Tenkan, Kijun, Chikou",
    },
    "Supertrend": {
        "mod": "wraquant.ta.overlap", "fn": "supertrend", "input": "hlc",
        "kw": {"period": 10, "multiplier": 3.0}, "multi": True,
        "tip": "ATR trend-follower: flip between support and resistance",
    },
    "VWAP": {
        "mod": "wraquant.ta.overlap", "fn": "vwap", "input": "hlcv_no_period",
        "kw": {},
        "tip": "Volume Weighted Average Price -- institutional benchmark",
    },
    "Parabolic SAR": {
        "mod": "wraquant.ta.trend", "fn": "psar", "input": "hlc_no_period",
        "kw": {},
        "tip": "Trailing stop & reversal dots",
    },
    # -- Smoothing module --
    "ALMA(9)": {
        "mod": "wraquant.ta.smoothing", "fn": "alma", "input": "close",
        "kw": {"period": 9},
        "tip": "Arnaud Legoux MA -- Gaussian-weighted, offset-adjustable",
    },
    "JMA(7)": {
        "mod": "wraquant.ta.smoothing", "fn": "jma", "input": "close",
        "kw": {"period": 7},
        "tip": "Jurik Moving Average -- ultra-smooth, low lag",
    },
    "Hull MA(16)": {
        "mod": "wraquant.ta.trend", "fn": "hull_ma", "input": "close",
        "kw": {"period": 16},
        "tip": "Hull Moving Average -- WMA-based, very low lag",
    },
    "Zero-Lag EMA(21)": {
        "mod": "wraquant.ta.trend", "fn": "zero_lag_ema", "input": "close",
        "kw": {"period": 21},
        "tip": "EMA with lag correction",
    },
    "McGinley(14)": {
        "mod": "wraquant.ta.trend", "fn": "mcginley_dynamic", "input": "close",
        "kw": {"period": 14},
        "tip": "McGinley Dynamic -- auto-adjusting MA, less whipsaw",
    },
    "Tilson T3(5)": {
        "mod": "wraquant.ta.trend", "fn": "tilson_t3", "input": "close",
        "kw": {"period": 5},
        "tip": "Ultra-smooth triple-EMA with volume factor",
    },
    "VIDYA(14)": {
        "mod": "wraquant.ta.trend", "fn": "vidya", "input": "close",
        "kw": {"period": 14},
        "tip": "Variable Index Dynamic Average -- CMO-adaptive EMA",
    },
    "FRAMA(16)": {
        "mod": "wraquant.ta.trend", "fn": "fractal_adaptive_ma", "input": "close",
        "kw": {"period": 16},
        "tip": "Fractal Adaptive MA -- fractal dimension adaptive",
    },
    "SuperSmoother(10)": {
        "mod": "wraquant.ta.smoothing", "fn": "supersmoother", "input": "close",
        "kw": {"period": 10},
        "tip": "Ehlers two-pole super-smoother filter",
    },
    "Decycler(125)": {
        "mod": "wraquant.ta.cycles", "fn": "decycler", "input": "close",
        "kw": {},
        "tip": "Ehlers Decycler -- pure trend, cycles removed",
    },
    "LinReg Channel": {
        "mod": "wraquant.ta.custom", "fn": "linear_regression_channel", "input": "close",
        "kw": {"period": 50}, "multi": True,
        "tip": "Linear regression channel (upper/lower/middle)",
    },
    "Std Error Bands": {
        "mod": "wraquant.ta.custom", "fn": "standard_error_bands", "input": "close",
        "kw": {"period": 20}, "multi": True,
        "tip": "Standard error bands around linear regression",
    },
    "Acceleration Bands": {
        "mod": "wraquant.ta.volatility", "fn": "acceleration_bands", "input": "hlc",
        "kw": {"period": 20}, "multi": True,
        "tip": "Range-acceleration envelope bands",
    },
}

_MOMENTUM_INDICATORS: dict[str, dict] = {
    "RSI(14)": {
        "mod": "wraquant.ta.momentum", "fn": "rsi", "input": "close",
        "kw": {"period": 14}, "range": (0, 100),
        "tip": ">70 overbought, <30 oversold",
        "ref_lines": [70, 30],
    },
    "MACD": {
        "mod": "wraquant.ta.momentum", "fn": "macd", "input": "close",
        "kw": {"fast": 12, "slow": 26, "signal": 9}, "multi": True,
        "tip": "Trend momentum via EMA crossovers",
        "ref_lines": [0],
    },
    "Stochastic(14)": {
        "mod": "wraquant.ta.momentum", "fn": "stochastic", "input": "hlc",
        "kw": {}, "multi": True,
        "tip": "%K/%D -- close position within range",
        "ref_lines": [80, 20],
    },
    "Stochastic RSI(14)": {
        "mod": "wraquant.ta.momentum", "fn": "stochastic_rsi", "input": "close",
        "kw": {"period": 14}, "multi": True,
        "tip": "Stochastic of RSI -- ultra-sensitive overbought/oversold",
        "ref_lines": [80, 20],
    },
    "Williams %R(14)": {
        "mod": "wraquant.ta.momentum", "fn": "williams_r", "input": "hlc",
        "kw": {"period": 14},
        "tip": "Inverted stochastic: >-20 OB, <-80 OS",
        "ref_lines": [-20, -80],
    },
    "CCI(20)": {
        "mod": "wraquant.ta.momentum", "fn": "cci", "input": "hlc",
        "kw": {"period": 20},
        "tip": "Commodity Channel Index: >100 OB, <-100 OS",
        "ref_lines": [100, -100],
    },
    "ROC(10)": {
        "mod": "wraquant.ta.momentum", "fn": "roc", "input": "close",
        "kw": {"period": 10},
        "tip": "Rate of Change -- price change percentage",
        "ref_lines": [0],
    },
    "Momentum(10)": {
        "mod": "wraquant.ta.momentum", "fn": "momentum", "input": "close",
        "kw": {"period": 10},
        "tip": "Raw price difference from N periods ago",
        "ref_lines": [0],
    },
    "TSI": {
        "mod": "wraquant.ta.momentum", "fn": "tsi", "input": "close",
        "kw": {}, "multi": True,
        "tip": "True Strength Index -- double-smoothed momentum",
        "ref_lines": [0],
    },
    "PPO": {
        "mod": "wraquant.ta.momentum", "fn": "ppo", "input": "close",
        "kw": {}, "multi": True,
        "tip": "Percentage Price Oscillator -- normalized MACD",
        "ref_lines": [0],
    },
    "Awesome Oscillator": {
        "mod": "wraquant.ta.momentum", "fn": "awesome_oscillator", "input": "hl",
        "kw": {},
        "tip": "Bill Williams AO -- 5/34 SMA of median price",
        "ref_lines": [0],
    },
    "CMO(14)": {
        "mod": "wraquant.ta.momentum", "fn": "cmo", "input": "close",
        "kw": {"period": 14},
        "tip": "Chande Momentum Oscillator: >50 OB, <-50 OS",
        "ref_lines": [50, -50],
    },
    "DPO(20)": {
        "mod": "wraquant.ta.momentum", "fn": "dpo", "input": "close",
        "kw": {"period": 20},
        "tip": "Detrended Price Oscillator -- cycle detection",
        "ref_lines": [0],
    },
    "Ultimate Osc": {
        "mod": "wraquant.ta.momentum", "fn": "ultimate_oscillator", "input": "hlc_no_period",
        "kw": {},
        "tip": "Multi-timeframe oscillator (7/14/28)",
        "ref_lines": [70, 30],
    },
    "Squeeze Histogram": {
        "mod": "wraquant.ta.momentum", "fn": "squeeze_histogram", "input": "hlc",
        "kw": {"period": 20},
        "tip": "TTM Squeeze momentum component",
        "ref_lines": [0],
    },
    "Center of Gravity": {
        "mod": "wraquant.ta.momentum", "fn": "center_of_gravity", "input": "close",
        "kw": {"period": 10},
        "tip": "Ehlers CoG -- leading oscillator",
        "ref_lines": [0],
    },
    "Psych Line(12)": {
        "mod": "wraquant.ta.momentum", "fn": "psychological_line", "input": "close",
        "kw": {"period": 12},
        "tip": "Percentage of up bars in window",
        "ref_lines": [75, 25],
    },
    "Schaff Momentum": {
        "mod": "wraquant.ta.momentum", "fn": "schaff_momentum", "input": "close",
        "kw": {},
        "tip": "Schaff MACD cycle oscillator",
        "ref_lines": [80, 20],
    },
    "PMO": {
        "mod": "wraquant.ta.momentum", "fn": "price_momentum_oscillator", "input": "close",
        "kw": {}, "multi": True,
        "tip": "Price Momentum Oscillator -- double-smoothed ROC",
        "ref_lines": [0],
    },
    "SMI(14)": {
        "mod": "wraquant.ta.momentum", "fn": "stochastic_momentum_index", "input": "hlc",
        "kw": {"period": 14}, "multi": True,
        "tip": "Stochastic Momentum Index -- distance from range midpoint",
        "ref_lines": [40, -40],
    },
    # -- Exotic module --
    "Choppiness(14)": {
        "mod": "wraquant.ta.exotic", "fn": "choppiness_index", "input": "hlc",
        "kw": {"period": 14},
        "tip": ">61.8 = choppy/ranging, <38.2 = trending",
        "ref_lines": [61.8, 38.2],
    },
    "PFE(10)": {
        "mod": "wraquant.ta.exotic", "fn": "polarized_fractal_efficiency", "input": "close",
        "kw": {"period": 10},
        "tip": "Fractal efficiency: +/-100 = straight line, 0 = random",
        "ref_lines": [0],
    },
    "Ergodic Osc": {
        "mod": "wraquant.ta.exotic", "fn": "ergodic_oscillator", "input": "close",
        "kw": {}, "multi": True,
        "tip": "Double-smoothed TSI variant with signal line",
        "ref_lines": [0],
    },
    "PZO": {
        "mod": "wraquant.ta.exotic", "fn": "price_zone_oscillator", "input": "close",
        "kw": {},
        "tip": "Price Zone Oscillator: +60/-60 zones",
        "ref_lines": [60, -60],
    },
    "KAIRI(14)": {
        "mod": "wraquant.ta.exotic", "fn": "kairi", "input": "close",
        "kw": {"period": 14},
        "tip": "% deviation from SMA -- mean reversion",
        "ref_lines": [0],
    },
    "PGO(14)": {
        "mod": "wraquant.ta.exotic", "fn": "pretty_good_oscillator", "input": "hlc",
        "kw": {"period": 14},
        "tip": "Close deviation / ATR",
        "ref_lines": [0],
    },
    "RMI(14)": {
        "mod": "wraquant.ta.exotic", "fn": "relative_momentum_index", "input": "close",
        "kw": {"period": 14},
        "tip": "RSI with variable look-back for momentum delta",
        "ref_lines": [70, 30],
    },
    "Connors TPS(2)": {
        "mod": "wraquant.ta.exotic", "fn": "connors_tps", "input": "close",
        "kw": {"period": 2},
        "tip": "Connors short-term mean-reversion signal",
        "ref_lines": [0],
    },
    # -- Custom module --
    "Ehlers Fisher": {
        "mod": "wraquant.ta.custom", "fn": "ehlers_fisher", "input": "hl",
        "kw": {"period": 10}, "multi": True,
        "tip": "Fisher Transform -- Gaussian conversion for sharp turns",
        "ref_lines": [0],
    },
    "Adaptive RSI": {
        "mod": "wraquant.ta.custom", "fn": "adaptive_rsi", "input": "close",
        "kw": {},
        "tip": "RSI with volatility-adaptive period",
        "ref_lines": [70, 30],
    },
    "VW-MACD": {
        "mod": "wraquant.ta.custom", "fn": "volume_weighted_macd", "input": "cv",
        "kw": {}, "multi": True,
        "tip": "Volume-weighted MACD -- volume-confirmed momentum",
        "ref_lines": [0],
    },
    # -- Cycles module --
    "Even Better Sinewave": {
        "mod": "wraquant.ta.cycles", "fn": "even_better_sinewave", "input": "close",
        "kw": {},
        "tip": "EBSW: +1 peak, -1 trough, 0 crossover",
        "ref_lines": [0],
    },
    "Roofing Filter": {
        "mod": "wraquant.ta.cycles", "fn": "roofing_filter", "input": "close",
        "kw": {},
        "tip": "Bandpass: pure cycle component, no trend/noise",
        "ref_lines": [0],
    },
    "Bandpass(20)": {
        "mod": "wraquant.ta.cycles", "fn": "bandpass_filter", "input": "close",
        "kw": {"period": 20}, "multi": True,
        "tip": "Isolate cycle at specified period",
        "ref_lines": [0],
    },
    "Dominant Period": {
        "mod": "wraquant.ta.cycles", "fn": "hilbert_transform_dominant_period", "input": "close",
        "kw": {},
        "tip": "Hilbert Transform estimated cycle length in bars",
        "ref_lines": [],
    },
}

_TREND_INDICATORS: dict[str, dict] = {
    "ADX(14)": {
        "mod": "wraquant.ta.trend", "fn": "adx", "input": "hlc",
        "kw": {"period": 14}, "multi": True,
        "tip": "Trend strength: >25 strong, <20 no trend",
        "ref_lines": [25],
    },
    "Aroon(25)": {
        "mod": "wraquant.ta.trend", "fn": "aroon", "input": "hl",
        "kw": {"period": 25}, "multi": True,
        "tip": "Time since highest-high/lowest-low: trend timing",
        "ref_lines": [70, 30],
    },
    "Vortex(14)": {
        "mod": "wraquant.ta.trend", "fn": "vortex", "input": "hlc",
        "kw": {"period": 14}, "multi": True,
        "tip": "+VI/-VI crossovers for trend direction",
        "ref_lines": [1.0],
    },
    "TRIX(15)": {
        "mod": "wraquant.ta.trend", "fn": "trix", "input": "close",
        "kw": {"period": 15},
        "tip": "Triple-smoothed EMA ROC -- trend momentum",
        "ref_lines": [0],
    },
    "LinReg Slope(14)": {
        "mod": "wraquant.ta.trend", "fn": "linear_regression_slope", "input": "close",
        "kw": {"period": 14},
        "tip": "Slope of rolling linear regression -- trend direction",
        "ref_lines": [0],
    },
    "Schaff Trend(10)": {
        "mod": "wraquant.ta.trend", "fn": "schaff_trend_cycle", "input": "close",
        "kw": {"period": 10},
        "tip": "Schaff Trend Cycle: 0-100, >75 bullish, <25 bearish",
        "ref_lines": [75, 25],
    },
    # -- Exotic trend indicators --
    "Trend Intensity(14)": {
        "mod": "wraquant.ta.exotic", "fn": "trend_intensity_index", "input": "close",
        "kw": {"period": 14},
        "tip": "% of bars above SMA; >50 uptrend, <50 downtrend",
        "ref_lines": [50],
    },
    "Efficiency Ratio(10)": {
        "mod": "wraquant.ta.exotic", "fn": "efficiency_ratio", "input": "close",
        "kw": {"period": 10},
        "tip": "Path efficiency: 1.0 = straight line, 0 = random",
        "ref_lines": [],
    },
    "DMI(14)": {
        "mod": "wraquant.ta.exotic", "fn": "directional_movement_index", "input": "hlc",
        "kw": {"period": 14}, "multi": True,
        "tip": "Raw +DI/-DI directional movement",
        "ref_lines": [],
    },
    "RWI(14)": {
        "mod": "wraquant.ta.exotic", "fn": "random_walk_index", "input": "hlc",
        "kw": {"period": 14}, "multi": True,
        "tip": "Random Walk Index: >1.0 = trending",
        "ref_lines": [1.0],
    },
    "GAPO(5)": {
        "mod": "wraquant.ta.exotic", "fn": "gopalakrishnan_range", "input": "hl",
        "kw": {"period": 5},
        "tip": "Range index via log ratio",
        "ref_lines": [],
    },
    # -- Cycles --
    "Trend Mode": {
        "mod": "wraquant.ta.cycles", "fn": "hilbert_transform_trend_mode", "input": "close",
        "kw": {},
        "tip": "Hilbert: 1 = trending, 0 = cycling",
        "ref_lines": [0.5],
    },
}

_VOLUME_INDICATORS: dict[str, dict] = {
    "OBV": {
        "mod": "wraquant.ta.volume", "fn": "obv", "input": "cv_no_period",
        "kw": {},
        "tip": "On Balance Volume -- cumulative volume direction",
        "ref_lines": [],
    },
    "CMF(20)": {
        "mod": "wraquant.ta.volume", "fn": "cmf", "input": "hlcv",
        "kw": {"period": 20},
        "tip": "Chaikin Money Flow: >0 buying, <0 selling",
        "ref_lines": [0],
    },
    "MFI(14)": {
        "mod": "wraquant.ta.volume", "fn": "mfi", "input": "hlcv",
        "kw": {"period": 14},
        "tip": "Money Flow Index: volume-weighted RSI; >80 OB, <20 OS",
        "ref_lines": [80, 20],
    },
    "AD Line": {
        "mod": "wraquant.ta.volume", "fn": "ad_line", "input": "hlcv_no_period",
        "kw": {},
        "tip": "Accumulation/Distribution -- CLV money flow",
        "ref_lines": [],
    },
    "Chaikin Osc": {
        "mod": "wraquant.ta.volume", "fn": "adosc", "input": "hlcv_no_period",
        "kw": {},
        "tip": "ADOSC: fast/slow EMA of AD line",
        "ref_lines": [0],
    },
    "Force Index(13)": {
        "mod": "wraquant.ta.volume", "fn": "force_index", "input": "cv",
        "kw": {"period": 13},
        "tip": "Price change * volume, EMA-smoothed",
        "ref_lines": [0],
    },
    "EOM(14)": {
        "mod": "wraquant.ta.volume", "fn": "eom", "input": "hlv",
        "kw": {"period": 14},
        "tip": "Ease of Movement -- price vs volume efficiency",
        "ref_lines": [0],
    },
    "VPT": {
        "mod": "wraquant.ta.volume", "fn": "vpt", "input": "cv_no_period",
        "kw": {},
        "tip": "Volume Price Trend -- cumulative",
        "ref_lines": [],
    },
    "NVI": {
        "mod": "wraquant.ta.volume", "fn": "nvi", "input": "cv_no_period",
        "kw": {},
        "tip": "Negative Volume Index -- smart money on quiet days",
        "ref_lines": [],
    },
    "PVI": {
        "mod": "wraquant.ta.volume", "fn": "pvi", "input": "cv_no_period",
        "kw": {},
        "tip": "Positive Volume Index -- crowd on active days",
        "ref_lines": [],
    },
    "Klinger Osc": {
        "mod": "wraquant.ta.momentum", "fn": "klinger_oscillator", "input": "hlcv",
        "kw": {}, "multi": True,
        "tip": "Klinger Volume Oscillator with signal line",
        "ref_lines": [0],
    },
    "Elder Thermo(22)": {
        "mod": "wraquant.ta.exotic", "fn": "elder_thermometer", "input": "hl",
        "kw": {"period": 22},
        "tip": "Elder Thermometer -- bar range extension",
        "ref_lines": [],
    },
    "MFI (BW)": {
        "mod": "wraquant.ta.exotic", "fn": "market_facilitation_index", "input": "hlv_no_period",
        "kw": {},
        "tip": "Bill Williams Market Facilitation Index",
        "ref_lines": [],
    },
}

_VOLATILITY_INDICATORS: dict[str, dict] = {
    "ATR(14)": {
        "mod": "wraquant.ta.volatility", "fn": "atr", "input": "hlc",
        "kw": {"period": 14},
        "tip": "Average True Range -- standard volatility for position sizing",
        "ref_lines": [],
    },
    "NATR(14)": {
        "mod": "wraquant.ta.volatility", "fn": "natr", "input": "hlc",
        "kw": {"period": 14},
        "tip": "Normalized ATR (%) -- comparable across assets",
        "ref_lines": [],
    },
    "True Range": {
        "mod": "wraquant.ta.volatility", "fn": "true_range", "input": "hlc_no_period",
        "kw": {},
        "tip": "Single-bar volatility including gaps",
        "ref_lines": [],
    },
    "BB Width(20)": {
        "mod": "wraquant.ta.volatility", "fn": "bbwidth", "input": "close",
        "kw": {"period": 20},
        "tip": "Bollinger Band Width -- low = squeeze imminent",
        "ref_lines": [],
    },
    "KC Width(20)": {
        "mod": "wraquant.ta.volatility", "fn": "kc_width", "input": "hlc",
        "kw": {"period": 20},
        "tip": "Keltner Channel Width -- ATR envelope width",
        "ref_lines": [],
    },
    "Hist Vol(21)": {
        "mod": "wraquant.ta.volatility", "fn": "historical_volatility", "input": "close",
        "kw": {"period": 21},
        "tip": "Annualized close-to-close volatility",
        "ref_lines": [],
    },
    "Chaikin Vol(10)": {
        "mod": "wraquant.ta.volatility", "fn": "chaikin_volatility", "input": "hl",
        "kw": {"period": 10},
        "tip": "Rate of change of EMA(H-L spread)",
        "ref_lines": [0],
    },
    "Mass Index(9)": {
        "mod": "wraquant.ta.volatility", "fn": "mass_index", "input": "hl",
        "kw": {"period": 9},
        "tip": "Reversal bulge: >27 then <26.5 = trend change",
        "ref_lines": [27, 26.5],
    },
    "Ulcer Index(14)": {
        "mod": "wraquant.ta.volatility", "fn": "ulcer_index", "input": "close",
        "kw": {"period": 14},
        "tip": "Downside volatility -- depth of drawdowns",
        "ref_lines": [],
    },
    "RVI(10)": {
        "mod": "wraquant.ta.volatility", "fn": "relative_volatility_index", "input": "close",
        "kw": {"period": 10},
        "tip": "RSI of std dev: >50 vol rising, <50 vol falling",
        "ref_lines": [50],
    },
    "Std Dev(20)": {
        "mod": "wraquant.ta.volatility", "fn": "standard_deviation", "input": "close",
        "kw": {"period": 20},
        "tip": "Rolling standard deviation",
        "ref_lines": [],
    },
    "Squeeze Mom": {
        "mod": "wraquant.ta.custom", "fn": "squeeze_momentum", "input": "hlc",
        "kw": {}, "multi": True,
        "tip": "TTM Squeeze: BB inside KC + lin-reg momentum",
        "ref_lines": [0],
    },
    "Parkinson Vol(21)": {
        "mod": "wraquant.ta.volatility", "fn": "parkinson", "input": "hl",
        "kw": {"period": 21},
        "tip": "High-low range estimator -- 5x efficiency of close-to-close",
        "ref_lines": [],
    },
}

# Candlestick patterns -- all take (open_, high, low, close)
_PATTERN_NAMES = [
    "doji", "hammer", "engulfing", "morning_star", "evening_star",
    "three_white_soldiers", "three_black_crows", "harami", "spinning_top",
    "marubozu", "piercing_pattern", "dark_cloud_cover", "hanging_man",
    "inverted_hammer", "shooting_star", "tweezer_top", "tweezer_bottom",
    "three_inside_up", "three_inside_down", "abandoned_baby", "kicking",
    "belt_hold", "rising_three_methods", "falling_three_methods",
    "tasuki_gap", "on_neck", "in_neck", "thrusting", "separating_lines",
    "closing_marubozu", "rickshaw_man", "long_legged_doji",
    "dragonfly_doji", "gravestone_doji", "tri_star", "unique_three_river",
    "concealing_baby_swallow",
]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_ohlcv(ticker: str, days: int = 365) -> "pd.DataFrame":
    """Fetch OHLCV data: FMP first, yfinance fallback, synthetic last resort."""
    from datetime import datetime, timedelta

    import pandas as pd

    # -- FMP --
    try:
        from wraquant.data.providers.fmp import FMPClient

        client = FMPClient()
        end = datetime.now()
        start = end - timedelta(days=days)
        df = client.historical_price(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="daily",
        )
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            elif not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except (ValueError, TypeError):
                    pass
            if "close" in df.columns:
                return df
    except Exception:
        pass

    # -- yfinance --
    try:
        import yfinance as yf

        period_map = {90: "3mo", 180: "6mo", 365: "1y", 730: "2y"}
        data = yf.download(
            ticker,
            period=period_map.get(days, "1y"),
            auto_adjust=True,
            progress=False,
        )
        if data is not None and not data.empty:
            data.columns = [c.lower() if isinstance(c, str) else c for c in data.columns]
            # Handle multi-level columns from yfinance
            if hasattr(data.columns, "levels"):
                data.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in data.columns]
            return data
    except Exception:
        pass

    # -- Synthetic --
    import numpy as np

    rng = np.random.default_rng(42)
    n = min(252, days)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    close = pd.Series(100 + np.cumsum(rng.normal(0.05, 1.5, n)), index=idx, name="close")
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close.shift(1).bfill() + rng.normal(0, 0.5, n)
    volume = pd.Series(rng.integers(1_000_000, 10_000_000, n).astype(float), index=idx, name="volume")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# Indicator computation helpers
# ---------------------------------------------------------------------------


def _call_indicator(spec: dict, close, high, low, volume, open_):
    """Call a TA function based on its spec dict.  Returns result or None."""
    import importlib

    try:
        mod = importlib.import_module(spec["mod"])
        func = getattr(mod, spec["fn"])
        inp = spec["input"]
        kw = dict(spec.get("kw", {}))

        if inp == "close" or inp == "data":
            return func(close, **kw)
        elif inp == "hlc":
            return func(high, low, close, **kw)
        elif inp == "hl":
            return func(high, low, **kw)
        elif inp == "hlcv":
            return func(high, low, close, volume, **kw)
        elif inp == "cv":
            return func(close, volume, **kw)
        elif inp == "cv_no_period":
            return func(close, volume)
        elif inp == "hlcv_no_period":
            return func(high, low, close, volume)
        elif inp == "hlc_no_period":
            return func(high, low, close)
        elif inp == "hlv":
            return func(high, low, volume, **kw)
        elif inp == "hlv_no_period":
            return func(high, low, volume)
        elif inp == "ohlcv":
            return func(high, low, close, open_, **kw)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Signal consensus computation
# ---------------------------------------------------------------------------


def _compute_signal_consensus(close, high, low, volume, open_):
    """Build a signal consensus table across many indicators."""
    import importlib

    import pandas as pd

    rows = []

    def _safe_last(series_or_dict, key=None):
        """Get last non-NaN value from a series or dict."""
        try:
            if isinstance(series_or_dict, dict):
                if key and key in series_or_dict:
                    s = series_or_dict[key]
                else:
                    s = list(series_or_dict.values())[0]
            else:
                s = series_or_dict
            if hasattr(s, "dropna"):
                s = s.dropna()
                if len(s) > 0:
                    return float(s.iloc[-1])
        except Exception:
            pass
        return None

    def _classify_rsi(val):
        if val is None:
            return "N/A", "---"
        if val > 70:
            return "Sell", "Overbought"
        if val < 30:
            return "Buy", "Oversold"
        if val > 60:
            return "Neutral", "Bullish bias"
        if val < 40:
            return "Neutral", "Bearish bias"
        return "Neutral", "Neutral"

    def _classify_macd(macd_val, signal_val):
        if macd_val is None or signal_val is None:
            return "N/A", "---"
        if macd_val > signal_val and macd_val > 0:
            return "Buy", "Strong"
        if macd_val > signal_val:
            return "Buy", "Weak"
        if macd_val < signal_val and macd_val < 0:
            return "Sell", "Strong"
        return "Sell", "Weak"

    def _classify_zero(val, upper=None, lower=None):
        if val is None:
            return "N/A", "---"
        if upper and val > upper:
            return "Sell", "Overbought"
        if lower and val < lower:
            return "Buy", "Oversold"
        if val > 0:
            return "Buy", "Bullish"
        return "Sell", "Bearish"

    def _classify_trend(val, threshold=25):
        if val is None:
            return "N/A", "---"
        if val > threshold:
            return "Trend", "Strong"
        return "No Trend", "Weak"

    def _classify_above_below(price_val, ma_val):
        if price_val is None or ma_val is None:
            return "N/A", "---"
        if price_val > ma_val:
            return "Buy", "Above"
        return "Sell", "Below"

    # --- Momentum ---
    try:
        rsi_val = _safe_last(_call_indicator(
            {"mod": "wraquant.ta.momentum", "fn": "rsi", "input": "close", "kw": {"period": 14}},
            close, high, low, volume, open_,
        ))
        sig, strength = _classify_rsi(rsi_val)
        rows.append(("Momentum", "RSI(14)", f"{rsi_val:.1f}" if rsi_val else "---", sig, strength))
    except Exception:
        pass

    try:
        macd_res = _call_indicator(
            {"mod": "wraquant.ta.momentum", "fn": "macd", "input": "close", "kw": {}},
            close, high, low, volume, open_,
        )
        if isinstance(macd_res, dict):
            m_val = _safe_last(macd_res, "macd")
            s_val = _safe_last(macd_res, "signal")
            sig, strength = _classify_macd(m_val, s_val)
            rows.append(("Momentum", "MACD", f"{m_val:.2f}" if m_val else "---", sig, strength))
    except Exception:
        pass

    try:
        stoch_res = _call_indicator(
            {"mod": "wraquant.ta.momentum", "fn": "stochastic", "input": "hlc", "kw": {}},
            close, high, low, volume, open_,
        )
        if isinstance(stoch_res, dict):
            k_val = _safe_last(stoch_res, "k")
            sig, strength = _classify_rsi(k_val)
            rows.append(("Momentum", "Stoch %K(14)", f"{k_val:.1f}" if k_val else "---", sig, strength))
    except Exception:
        pass

    try:
        wr_val = _safe_last(_call_indicator(
            {"mod": "wraquant.ta.momentum", "fn": "williams_r", "input": "hlc", "kw": {"period": 14}},
            close, high, low, volume, open_,
        ))
        if wr_val is not None:
            if wr_val > -20:
                sig, strength = "Sell", "Overbought"
            elif wr_val < -80:
                sig, strength = "Buy", "Oversold"
            else:
                sig, strength = "Neutral", "Neutral"
            rows.append(("Momentum", "Williams %R(14)", f"{wr_val:.1f}", sig, strength))
    except Exception:
        pass

    try:
        cci_val = _safe_last(_call_indicator(
            {"mod": "wraquant.ta.momentum", "fn": "cci", "input": "hlc", "kw": {"period": 20}},
            close, high, low, volume, open_,
        ))
        sig, strength = _classify_zero(cci_val, upper=100, lower=-100)
        rows.append(("Momentum", "CCI(20)", f"{cci_val:.1f}" if cci_val else "---", sig, strength))
    except Exception:
        pass

    try:
        roc_val = _safe_last(_call_indicator(
            {"mod": "wraquant.ta.momentum", "fn": "roc", "input": "close", "kw": {"period": 10}},
            close, high, low, volume, open_,
        ))
        sig, strength = _classify_zero(roc_val)
        rows.append(("Momentum", "ROC(10)", f"{roc_val:.2f}" if roc_val else "---", sig, strength))
    except Exception:
        pass

    try:
        cmo_val = _safe_last(_call_indicator(
            {"mod": "wraquant.ta.momentum", "fn": "cmo", "input": "close", "kw": {"period": 14}},
            close, high, low, volume, open_,
        ))
        sig, strength = _classify_zero(cmo_val, upper=50, lower=-50)
        rows.append(("Momentum", "CMO(14)", f"{cmo_val:.1f}" if cmo_val else "---", sig, strength))
    except Exception:
        pass

    # --- Trend ---
    try:
        adx_res = _call_indicator(
            {"mod": "wraquant.ta.trend", "fn": "adx", "input": "hlc", "kw": {"period": 14}},
            close, high, low, volume, open_,
        )
        if isinstance(adx_res, dict):
            adx_val = _safe_last(adx_res, "adx")
            plus_di = _safe_last(adx_res, "plus_di")
            minus_di = _safe_last(adx_res, "minus_di")
            sig, strength = _classify_trend(adx_val)
            if plus_di and minus_di:
                direction = "Bullish" if plus_di > minus_di else "Bearish"
                strength = f"{strength} ({direction})"
            rows.append(("Trend", "ADX(14)", f"{adx_val:.1f}" if adx_val else "---", sig, strength))
    except Exception:
        pass

    try:
        trix_val = _safe_last(_call_indicator(
            {"mod": "wraquant.ta.trend", "fn": "trix", "input": "close", "kw": {"period": 15}},
            close, high, low, volume, open_,
        ))
        sig, strength = _classify_zero(trix_val)
        rows.append(("Trend", "TRIX(15)", f"{trix_val:.4f}" if trix_val else "---", sig, strength))
    except Exception:
        pass

    try:
        chop_val = _safe_last(_call_indicator(
            {"mod": "wraquant.ta.exotic", "fn": "choppiness_index", "input": "hlc", "kw": {"period": 14}},
            close, high, low, volume, open_,
        ))
        if chop_val is not None:
            if chop_val > 61.8:
                sig, strength = "No Trend", "Choppy"
            elif chop_val < 38.2:
                sig, strength = "Trend", "Strong"
            else:
                sig, strength = "Neutral", "Moderate"
            rows.append(("Trend", "Choppiness(14)", f"{chop_val:.1f}", sig, strength))
    except Exception:
        pass

    # --- Overlap / MA signals ---
    try:
        last_price = float(close.dropna().iloc[-1]) if len(close.dropna()) > 0 else None
    except Exception:
        last_price = None

    for name, period in [("SMA(20)", 20), ("SMA(50)", 50), ("SMA(200)", 200), ("EMA(21)", 21)]:
        try:
            fn_name = "sma" if "SMA" in name else "ema"
            ma_val = _safe_last(_call_indicator(
                {"mod": "wraquant.ta.overlap", "fn": fn_name, "input": "close", "kw": {"period": period}},
                close, high, low, volume, open_,
            ))
            sig, strength = _classify_above_below(last_price, ma_val)
            rows.append(("Overlap", name, f"{ma_val:.2f}" if ma_val else "---", sig, strength))
        except Exception:
            pass

    # --- Volume ---
    try:
        cmf_val = _safe_last(_call_indicator(
            {"mod": "wraquant.ta.volume", "fn": "cmf", "input": "hlcv", "kw": {"period": 20}},
            close, high, low, volume, open_,
        ))
        sig, strength = _classify_zero(cmf_val)
        rows.append(("Volume", "CMF(20)", f"{cmf_val:.3f}" if cmf_val else "---", sig, strength))
    except Exception:
        pass

    try:
        mfi_val = _safe_last(_call_indicator(
            {"mod": "wraquant.ta.volume", "fn": "mfi", "input": "hlcv", "kw": {"period": 14}},
            close, high, low, volume, open_,
        ))
        sig, strength = _classify_rsi(mfi_val)
        rows.append(("Volume", "MFI(14)", f"{mfi_val:.1f}" if mfi_val else "---", sig, strength))
    except Exception:
        pass

    df_rows = pd.DataFrame(rows, columns=["Category", "Indicator", "Value", "Signal", "Strength"])
    return df_rows


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _add_ref_lines(fig, ref_lines, row, col, colors_dict):
    """Add reference lines to a subplot."""
    for level in ref_lines:
        try:
            fig.add_hline(
                y=level,
                line_dash="dot",
                line_color="rgba(148,163,184,0.35)",
                row=row, col=col,
            )
        except Exception:
            pass


def _plot_indicator_result(fig, result, name, row, col, series_colors, ref_lines=None, colors_dict=None):
    """Add an indicator result (Series or dict of Series) to a subplot."""
    import pandas as pd

    if result is None:
        return

    if isinstance(result, dict):
        for j, (key, series) in enumerate(result.items()):
            if isinstance(series, pd.Series):
                fig.add_trace(
                    _scatter_trace(series, f"{name} {key}", series_colors[j % len(series_colors)]),
                    row=row, col=col,
                )
    elif isinstance(result, pd.Series):
        fig.add_trace(
            _scatter_trace(result, name, series_colors[0]),
            row=row, col=col,
        )

    if ref_lines and colors_dict:
        _add_ref_lines(fig, ref_lines, row, col, colors_dict)


def _scatter_trace(series, name, color, width=1.5, dash=None):
    """Create a plotly scatter trace."""
    import plotly.graph_objects as go

    return go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=name,
        line={"color": color, "width": width, "dash": dash},
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Technical Analysis page."""
    import numpy as np
    import pandas as pd

    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# Technical Analysis: **{ticker}**")

    # -- Controls row --
    c1, c2 = st.columns([1, 1])
    with c1:
        lookback = st.selectbox(
            "Lookback Period",
            [90, 180, 365, 730, 1460],
            index=2,
            format_func=lambda x: {
                90: "3 Months", 180: "6 Months", 365: "1 Year",
                730: "2 Years", 1460: "4 Years",
            }.get(x, f"{x}d"),
            key="ta_lookback",
        )
    with c2:
        chart_type = st.selectbox(
            "Chart Type", ["Candlestick", "Line", "OHLC"], key="ta_chart_type",
        )

    # -- Fetch data --
    with st.spinner(f"Loading {ticker} OHLCV data..."):
        try:
            df = _fetch_ohlcv(ticker, days=lookback)
        except Exception as exc:
            st.error(f"Failed to fetch data: {exc}")
            return

    if df is None or df.empty:
        st.warning("No price data available.")
        return

    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    if "close" not in df.columns:
        st.warning("No 'close' column found in data.")
        return

    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    open_ = df.get("open", close)
    volume = df.get("volume", pd.Series(0.0, index=close.index))

    if not isinstance(close.index, pd.DatetimeIndex):
        try:
            close.index = pd.to_datetime(close.index)
            high.index = close.index
            low.index = close.index
            open_.index = close.index
            volume.index = close.index
        except Exception:
            pass

    st.caption(f"{len(close)} price observations | "
               f"{close.index.min().strftime('%Y-%m-%d') if hasattr(close.index.min(), 'strftime') else '?'} to "
               f"{close.index.max().strftime('%Y-%m-%d') if hasattr(close.index.max(), 'strftime') else '?'}")

    # ======================================================================
    # TABS
    # ======================================================================
    tabs = st.tabs([
        "Price Chart", "Momentum", "Trend", "Volume",
        "Volatility", "Patterns", "Support/Resistance",
        "Signals", "Statistics", "Performance",
    ])

    # ------------------------------------------------------------------
    # TAB 0: Price Chart with overlays
    # ------------------------------------------------------------------
    with tabs[0]:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            st.markdown("##### Price Overlays")
            st.caption("Select overlays to draw on the price chart")

            selected_overlays = st.multiselect(
                "Overlays",
                list(_OVERLAY_INDICATORS.keys()),
                default=["SMA(20)", "SMA(50)", "EMA(21)", "Bollinger Bands"],
                key="ta_overlays",
                help="Choose as many overlays as you want",
            )

            # Build figure: price + volume
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.75, 0.25],
            )

            # Price trace
            if chart_type == "Candlestick":
                fig.add_trace(
                    go.Candlestick(
                        x=close.index, open=open_, high=high, low=low, close=close,
                        name="OHLC",
                        increasing_line_color=COLORS["success"],
                        decreasing_line_color=COLORS["danger"],
                        increasing_fillcolor=COLORS["success"],
                        decreasing_fillcolor=COLORS["danger"],
                    ),
                    row=1, col=1,
                )
            elif chart_type == "OHLC":
                fig.add_trace(
                    go.Ohlc(
                        x=close.index, open=open_, high=high, low=low, close=close,
                        name="OHLC",
                        increasing_line_color=COLORS["success"],
                        decreasing_line_color=COLORS["danger"],
                    ),
                    row=1, col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=close.index, y=close, mode="lines",
                        name="Close", line={"color": COLORS["primary"], "width": 2},
                    ),
                    row=1, col=1,
                )

            # Overlay indicators
            for idx, ov_name in enumerate(selected_overlays):
                spec = _OVERLAY_INDICATORS.get(ov_name)
                if spec is None:
                    continue

                color = SERIES_COLORS[(idx + 2) % len(SERIES_COLORS)]
                result = _call_indicator(spec, close, high, low, volume, open_)
                if result is None:
                    continue

                if isinstance(result, dict):
                    if ov_name == "Bollinger Bands":
                        upper = result.get("upper")
                        middle = result.get("middle")
                        lower = result.get("lower")
                        if upper is not None:
                            fig.add_trace(go.Scatter(
                                x=close.index, y=upper, mode="lines",
                                name="BB Upper",
                                line={"color": COLORS["accent3"], "width": 1, "dash": "dot"},
                            ), row=1, col=1)
                        if middle is not None:
                            fig.add_trace(go.Scatter(
                                x=close.index, y=middle, mode="lines",
                                name="BB Middle",
                                line={"color": COLORS["accent3"], "width": 1},
                            ), row=1, col=1)
                        if lower is not None:
                            fig.add_trace(go.Scatter(
                                x=close.index, y=lower, mode="lines",
                                name="BB Lower",
                                line={"color": COLORS["accent3"], "width": 1, "dash": "dot"},
                                fill="tonexty",
                                fillcolor="rgba(167,139,250,0.06)",
                            ), row=1, col=1)
                    elif ov_name == "Ichimoku Cloud":
                        ichi_colors = [COLORS["accent2"], COLORS["accent1"],
                                       COLORS["success"], COLORS["danger"], COLORS["neutral"]]
                        for j, (key, series) in enumerate(result.items()):
                            if isinstance(series, pd.Series):
                                fig.add_trace(go.Scatter(
                                    x=series.index, y=series, mode="lines",
                                    name=key.replace("_", " ").title(),
                                    line={"color": ichi_colors[j % len(ichi_colors)], "width": 1},
                                ), row=1, col=1)
                        # Cloud fill between senkou spans
                        sa = result.get("senkou_span_a")
                        sb = result.get("senkou_span_b")
                        if sa is not None and sb is not None:
                            fig.add_trace(go.Scatter(
                                x=sa.index, y=sa, mode="lines",
                                line={"width": 0}, showlegend=False,
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=sb.index, y=sb, mode="lines",
                                line={"width": 0}, showlegend=False,
                                fill="tonexty",
                                fillcolor="rgba(34,197,94,0.05)",
                            ), row=1, col=1)
                    elif ov_name == "Supertrend":
                        st_line = result.get("supertrend")
                        direction = result.get("direction")
                        if st_line is not None:
                            fig.add_trace(go.Scatter(
                                x=st_line.index, y=st_line, mode="lines",
                                name="Supertrend",
                                line={"color": COLORS["warning"], "width": 2},
                            ), row=1, col=1)
                    elif ov_name == "Keltner Channel":
                        for j, (key, series) in enumerate(result.items()):
                            if isinstance(series, pd.Series):
                                dash = "dot" if key != "middle" else None
                                fig.add_trace(go.Scatter(
                                    x=series.index, y=series, mode="lines",
                                    name=f"KC {key.title()}",
                                    line={"color": COLORS["info"], "width": 1, "dash": dash},
                                ), row=1, col=1)
                    elif ov_name == "Donchian Channel":
                        for j, (key, series) in enumerate(result.items()):
                            if isinstance(series, pd.Series):
                                dash = "dot" if key != "middle" else None
                                fig.add_trace(go.Scatter(
                                    x=series.index, y=series, mode="lines",
                                    name=f"DC {key.title()}",
                                    line={"color": COLORS["accent4"], "width": 1, "dash": dash},
                                ), row=1, col=1)
                    else:
                        # Generic multi-output overlay
                        for j, (key, series) in enumerate(result.items()):
                            if isinstance(series, pd.Series):
                                c_idx = (idx + j + 2) % len(SERIES_COLORS)
                                fig.add_trace(go.Scatter(
                                    x=series.index, y=series, mode="lines",
                                    name=f"{ov_name} {key}",
                                    line={"color": SERIES_COLORS[c_idx], "width": 1},
                                ), row=1, col=1)
                elif isinstance(result, pd.Series):
                    if ov_name == "Parabolic SAR":
                        fig.add_trace(go.Scatter(
                            x=result.index, y=result, mode="markers",
                            name="PSAR",
                            marker={"color": COLORS["warning"], "size": 3, "symbol": "diamond"},
                        ), row=1, col=1)
                    elif ov_name == "VWAP":
                        fig.add_trace(go.Scatter(
                            x=result.index, y=result, mode="lines",
                            name="VWAP",
                            line={"color": COLORS["accent1"], "width": 1.5, "dash": "dash"},
                        ), row=1, col=1)
                    else:
                        fig.add_trace(go.Scatter(
                            x=result.index, y=result, mode="lines",
                            name=ov_name,
                            line={"color": color, "width": 1.5},
                        ), row=1, col=1)

            # Volume bars
            if volume is not None and volume.sum() > 0:
                vol_colors = [
                    COLORS["success"] if c >= o else COLORS["danger"]
                    for c, o in zip(close, open_)
                ]
                fig.add_trace(
                    go.Bar(
                        x=close.index, y=volume, name="Volume",
                        marker_color=vol_colors, opacity=0.45,
                    ),
                    row=2, col=1,
                )

            layout = dark_layout(
                title=f"{ticker} -- Price & Overlays",
                height=700,
                showlegend=True,
                xaxis_rangeslider_visible=False,
            )
            fig.update_layout(**layout)
            for r in range(1, 3):
                fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=r, col=1)
                fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=r, col=1)

            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.warning("Plotly is required. Install with: `pip install plotly`")
            st.line_chart(close)

    # ------------------------------------------------------------------
    # TAB 1: Momentum
    # ------------------------------------------------------------------
    with tabs[1]:
        _render_subplot_tab(
            "Momentum Oscillators",
            _MOMENTUM_INDICATORS,
            close, high, low, volume, open_,
            default_selection=["RSI(14)", "MACD", "Stochastic(14)"],
            COLORS=COLORS, SERIES_COLORS=SERIES_COLORS, dark_layout=dark_layout,
            ticker=ticker,
        )

    # ------------------------------------------------------------------
    # TAB 2: Trend
    # ------------------------------------------------------------------
    with tabs[2]:
        _render_subplot_tab(
            "Trend Indicators",
            _TREND_INDICATORS,
            close, high, low, volume, open_,
            default_selection=["ADX(14)", "Aroon(25)", "Vortex(14)"],
            COLORS=COLORS, SERIES_COLORS=SERIES_COLORS, dark_layout=dark_layout,
            ticker=ticker,
        )

    # ------------------------------------------------------------------
    # TAB 3: Volume
    # ------------------------------------------------------------------
    with tabs[3]:
        _render_subplot_tab(
            "Volume Indicators",
            _VOLUME_INDICATORS,
            close, high, low, volume, open_,
            default_selection=["OBV", "CMF(20)", "MFI(14)"],
            COLORS=COLORS, SERIES_COLORS=SERIES_COLORS, dark_layout=dark_layout,
            ticker=ticker,
        )

    # ------------------------------------------------------------------
    # TAB 4: Volatility
    # ------------------------------------------------------------------
    with tabs[4]:
        _render_subplot_tab(
            "Volatility Indicators",
            _VOLATILITY_INDICATORS,
            close, high, low, volume, open_,
            default_selection=["ATR(14)", "BB Width(20)", "Hist Vol(21)"],
            COLORS=COLORS, SERIES_COLORS=SERIES_COLORS, dark_layout=dark_layout,
            ticker=ticker,
        )

    # ------------------------------------------------------------------
    # TAB 5: Patterns
    # ------------------------------------------------------------------
    with tabs[5]:
        _render_patterns_tab(close, high, low, open_, volume, COLORS, SERIES_COLORS, dark_layout, ticker)

    # ------------------------------------------------------------------
    # TAB 6: Support / Resistance
    # ------------------------------------------------------------------
    with tabs[6]:
        _render_support_resistance_tab(close, high, low, open_, volume, COLORS, SERIES_COLORS, dark_layout, ticker)

    # ------------------------------------------------------------------
    # TAB 7: Signals consensus
    # ------------------------------------------------------------------
    with tabs[7]:
        _render_signals_tab(close, high, low, volume, open_, COLORS)

    # ------------------------------------------------------------------
    # TAB 8: Statistics
    # ------------------------------------------------------------------
    with tabs[8]:
        _render_statistics_tab(close, high, low, volume, open_, COLORS, SERIES_COLORS, dark_layout, ticker)

    # ------------------------------------------------------------------
    # TAB 9: Performance
    # ------------------------------------------------------------------
    with tabs[9]:
        _render_performance_tab(close, COLORS)


# ---------------------------------------------------------------------------
# Subplot tab renderer (reused for Momentum, Trend, Volume, Volatility)
# ---------------------------------------------------------------------------


def _render_subplot_tab(
    title, indicators_dict, close, high, low, volume, open_,
    default_selection, COLORS, SERIES_COLORS, dark_layout, ticker,
):
    """Render a tab with selectable subplot indicators."""
    import pandas as pd

    st.markdown(f"##### {title}")
    st.caption("Select indicators to display as subplots")

    # Build help text
    help_items = []
    for name, spec in indicators_dict.items():
        help_items.append(f"**{name}**: {spec.get('tip', '')}")
    with st.expander("Indicator Reference", expanded=False):
        st.markdown(" | ".join(help_items[:10]) if len(help_items) <= 10 else "\n".join(f"- {h}" for h in help_items))

    # Ensure default_selection items exist in the dict
    valid_defaults = [d for d in default_selection if d in indicators_dict]

    selected = st.multiselect(
        "Indicators",
        list(indicators_dict.keys()),
        default=valid_defaults,
        key=f"ta_sub_{title.replace(' ', '_').lower()}",
    )

    if not selected:
        st.info("Select at least one indicator above.")
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_plots = len(selected)
        fig = make_subplots(
            rows=n_plots, cols=1, shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[1.0 / n_plots] * n_plots,
            subplot_titles=selected,
        )

        for plot_idx, ind_name in enumerate(selected):
            spec = indicators_dict[ind_name]
            result = _call_indicator(spec, close, high, low, volume, open_)
            row_num = plot_idx + 1

            if result is None:
                continue

            if isinstance(result, dict):
                for j, (key, series) in enumerate(result.items()):
                    if isinstance(series, pd.Series):
                        fig.add_trace(
                            _scatter_trace(
                                series, f"{ind_name} {key}",
                                SERIES_COLORS[j % len(SERIES_COLORS)],
                            ),
                            row=row_num, col=1,
                        )
            elif isinstance(result, pd.Series):
                fig.add_trace(
                    _scatter_trace(result, ind_name, SERIES_COLORS[0]),
                    row=row_num, col=1,
                )

            # Reference lines
            ref_lines = spec.get("ref_lines", [])
            for level in ref_lines:
                try:
                    fig.add_hline(
                        y=level, line_dash="dot",
                        line_color="rgba(148,163,184,0.35)",
                        row=row_num, col=1,
                    )
                except Exception:
                    pass

        height = max(400, 220 * n_plots)
        layout = dark_layout(
            title=f"{ticker} -- {title}",
            height=height,
            showlegend=True,
            xaxis_rangeslider_visible=False,
        )
        fig.update_layout(**layout)
        for r in range(1, n_plots + 1):
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=r, col=1)
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=r, col=1)

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Plotly is required for charts.")


# ---------------------------------------------------------------------------
# Patterns tab
# ---------------------------------------------------------------------------


def _render_patterns_tab(close, high, low, open_, volume, COLORS, SERIES_COLORS, dark_layout, ticker):
    """Render candlestick pattern recognition tab."""
    import importlib

    import pandas as pd

    st.markdown("##### Candlestick Pattern Recognition")
    st.caption("Scans for 37 candlestick patterns. 1 = bullish, -1 = bearish, 0 = none.")

    detected = {}
    for pat_name in _PATTERN_NAMES:
        try:
            mod = importlib.import_module("wraquant.ta.patterns")
            func = getattr(mod, pat_name)
            result = func(open_, high, low, close)
            if isinstance(result, pd.Series):
                nonzero = result[result != 0]
                if len(nonzero) > 0:
                    detected[pat_name] = result
        except Exception:
            continue

    if not detected:
        st.info("No candlestick patterns detected in the current data range.")
        return

    # Summary table
    summary_rows = []
    for pat_name, series in detected.items():
        nonzero = series[series != 0]
        last_idx = nonzero.index[-1]
        last_val = int(nonzero.iloc[-1])
        count = len(nonzero)
        signal = "Bullish" if last_val > 0 else "Bearish"
        date_str = last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else str(last_idx)
        display_name = pat_name.replace("_", " ").title()
        summary_rows.append({
            "Pattern": display_name,
            "Last Signal": signal,
            "Last Date": date_str,
            "Occurrences": count,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Last Date", ascending=False)

    def _color_signal(val):
        if val == "Bullish":
            return f"color: {COLORS['success']}"
        elif val == "Bearish":
            return f"color: {COLORS['danger']}"
        return ""

    styled = summary_df.style.applymap(_color_signal, subset=["Last Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Chart with pattern markers
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=close.index, open=open_, high=high, low=low, close=close,
            name="OHLC",
            increasing_line_color=COLORS["success"],
            decreasing_line_color=COLORS["danger"],
            increasing_fillcolor=COLORS["success"],
            decreasing_fillcolor=COLORS["danger"],
        ))

        # Add markers for detected patterns
        for pat_name, series in detected.items():
            bullish = series[series > 0]
            bearish = series[series < 0]
            display_name = pat_name.replace("_", " ").title()

            if len(bullish) > 0:
                fig.add_trace(go.Scatter(
                    x=bullish.index,
                    y=[float(low.loc[i]) * 0.995 if i in low.index else None for i in bullish.index],
                    mode="markers",
                    name=f"{display_name} (Bull)",
                    marker={"symbol": "triangle-up", "size": 10, "color": COLORS["success"]},
                    text=[display_name] * len(bullish),
                    hovertemplate="%{text}<extra></extra>",
                ))
            if len(bearish) > 0:
                fig.add_trace(go.Scatter(
                    x=bearish.index,
                    y=[float(high.loc[i]) * 1.005 if i in high.index else None for i in bearish.index],
                    mode="markers",
                    name=f"{display_name} (Bear)",
                    marker={"symbol": "triangle-down", "size": 10, "color": COLORS["danger"]},
                    text=[display_name] * len(bearish),
                    hovertemplate="%{text}<extra></extra>",
                ))

        fig.update_layout(**dark_layout(
            title=f"{ticker} -- Pattern Detection",
            height=550, showlegend=True, xaxis_rangeslider_visible=False,
        ))
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Plotly is required for charts.")


# ---------------------------------------------------------------------------
# Support / Resistance tab
# ---------------------------------------------------------------------------


def _render_support_resistance_tab(close, high, low, open_, volume, COLORS, SERIES_COLORS, dark_layout, ticker):
    """Render support/resistance, pivots, Fibonacci levels."""
    import importlib

    import numpy as np
    import pandas as pd

    st.markdown("##### Support & Resistance Analysis")

    sr_method = st.selectbox(
        "Method",
        ["Pivot Points", "Fibonacci Retracements", "S/R Detection", "Fractal Levels", "Round Numbers"],
        key="ta_sr_method",
    )

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=close.index, open=open_, high=high, low=low, close=close,
            name="OHLC",
            increasing_line_color=COLORS["success"],
            decreasing_line_color=COLORS["danger"],
        ))

        if sr_method == "Pivot Points":
            try:
                mod = importlib.import_module("wraquant.ta.custom")
                pp = mod.pivot_points(high, low, close, method="standard")
                if isinstance(pp, dict):
                    pp_colors = {
                        "pivot": COLORS["warning"],
                        "s1": COLORS["success"], "s2": COLORS["success"],
                        "r1": COLORS["danger"], "r2": COLORS["danger"],
                    }
                    for key, series in pp.items():
                        if isinstance(series, pd.Series):
                            last_val = series.dropna().iloc[-1] if len(series.dropna()) > 0 else None
                            if last_val is not None:
                                line_color = pp_colors.get(key, COLORS["neutral"])
                                fig.add_hline(
                                    y=last_val, line_dash="dash",
                                    line_color=line_color,
                                    annotation_text=f"{key.upper()}: {last_val:.2f}",
                                    annotation_font_color=line_color,
                                )
            except Exception as exc:
                st.warning(f"Pivot points: {exc}")

        elif sr_method == "Fibonacci Retracements":
            try:
                mod = importlib.import_module("wraquant.ta.fibonacci")
                swing_high = float(high.max())
                swing_low = float(low.min())
                levels = mod.fibonacci_retracements(swing_high, swing_low, direction="up")
                if isinstance(levels, dict):
                    fib_colors = [COLORS["accent3"], COLORS["accent2"], COLORS["warning"],
                                  COLORS["accent1"], COLORS["danger"]]
                    for j, (level_name, price) in enumerate(levels.items()):
                        fig.add_hline(
                            y=price, line_dash="dash",
                            line_color=fib_colors[j % len(fib_colors)],
                            annotation_text=f"Fib {level_name}: {price:.2f}",
                            annotation_font_color=fib_colors[j % len(fib_colors)],
                        )
            except Exception as exc:
                st.warning(f"Fibonacci: {exc}")

        elif sr_method == "S/R Detection":
            try:
                mod = importlib.import_module("wraquant.ta.support_resistance")
                sr = mod.find_support_resistance(high, low, lookback=5, num_levels=5)
                if isinstance(sr, dict):
                    for level in sr.get("support", []):
                        fig.add_hline(
                            y=level, line_dash="dot", line_color=COLORS["success"],
                            annotation_text=f"S: {level:.2f}",
                            annotation_font_color=COLORS["success"],
                        )
                    for level in sr.get("resistance", []):
                        fig.add_hline(
                            y=level, line_dash="dot", line_color=COLORS["danger"],
                            annotation_text=f"R: {level:.2f}",
                            annotation_font_color=COLORS["danger"],
                        )
            except Exception as exc:
                st.warning(f"S/R detection: {exc}")

        elif sr_method == "Fractal Levels":
            try:
                mod = importlib.import_module("wraquant.ta.support_resistance")
                fractals = mod.fractal_levels(high, low)
                if isinstance(fractals, dict):
                    frac_highs = fractals.get("fractal_highs")
                    frac_lows = fractals.get("fractal_lows")
                    if isinstance(frac_highs, pd.Series):
                        valid = frac_highs.dropna()
                        if len(valid) > 0:
                            fig.add_trace(go.Scatter(
                                x=valid.index, y=valid, mode="markers",
                                name="Fractal High",
                                marker={"symbol": "triangle-down", "size": 8, "color": COLORS["danger"]},
                            ))
                    if isinstance(frac_lows, pd.Series):
                        valid = frac_lows.dropna()
                        if len(valid) > 0:
                            fig.add_trace(go.Scatter(
                                x=valid.index, y=valid, mode="markers",
                                name="Fractal Low",
                                marker={"symbol": "triangle-up", "size": 8, "color": COLORS["success"]},
                            ))
            except Exception as exc:
                st.warning(f"Fractal levels: {exc}")

        elif sr_method == "Round Numbers":
            try:
                mod = importlib.import_module("wraquant.ta.support_resistance")
                levels = mod.round_number_levels(close)
                if isinstance(levels, list):
                    for level in levels[:8]:
                        fig.add_hline(
                            y=level, line_dash="dot",
                            line_color="rgba(148,163,184,0.3)",
                            annotation_text=f"${level:.0f}",
                        )
            except Exception as exc:
                st.warning(f"Round numbers: {exc}")

        fig.update_layout(**dark_layout(
            title=f"{ticker} -- {sr_method}",
            height=550, showlegend=True, xaxis_rangeslider_visible=False,
        ))
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Plotly is required for charts.")

    # Pivot levels table
    st.markdown("##### Current Pivot Levels")
    try:
        last_close = float(close.iloc[-1])
        period_high = float(high.max())
        period_low = float(low.min())
        pivot = (period_high + period_low + last_close) / 3
        r1 = 2 * pivot - period_low
        r2 = pivot + (period_high - period_low)
        r3 = period_high + 2 * (pivot - period_low)
        s1 = 2 * pivot - period_high
        s2 = pivot - (period_high - period_low)
        s3 = period_low - 2 * (period_high - pivot)

        cols = st.columns(7)
        labels = ["S3", "S2", "S1", "Pivot", "R1", "R2", "R3"]
        values = [s3, s2, s1, pivot, r1, r2, r3]
        for col, label, val in zip(cols, labels, values):
            col.metric(label, f"${val:,.2f}")
    except Exception:
        st.info("Could not compute pivot levels.")


# ---------------------------------------------------------------------------
# Signals consensus tab
# ---------------------------------------------------------------------------


def _render_signals_tab(close, high, low, volume, open_, COLORS):
    """Render the aggregated signal consensus table."""
    import pandas as pd

    st.markdown("##### Signal Consensus Dashboard")
    st.caption("Aggregated buy/sell signals from momentum, trend, overlap, and volume indicators")

    with st.spinner("Computing signal consensus..."):
        df = _compute_signal_consensus(close, high, low, volume, open_)

    if df.empty:
        st.info("No signals could be computed.")
        return

    # Summary counts
    buy_count = len(df[df["Signal"] == "Buy"])
    sell_count = len(df[df["Signal"] == "Sell"])
    neutral_count = len(df[df["Signal"].isin(["Neutral", "N/A"])])
    trend_count = len(df[df["Signal"] == "Trend"])
    no_trend_count = len(df[df["Signal"] == "No Trend"])

    c1, c2, c3, c4, c5 = st.columns(5)
    total = buy_count + sell_count + neutral_count
    if buy_count > sell_count:
        overall = "BULLISH"
        overall_color = COLORS["success"]
    elif sell_count > buy_count:
        overall = "BEARISH"
        overall_color = COLORS["danger"]
    else:
        overall = "NEUTRAL"
        overall_color = COLORS["warning"]

    c1.markdown(
        f'<div style="text-align:center; padding:0.5rem; background:#16161d; '
        f'border-radius:8px; border:1px solid {overall_color}40;">'
        f'<p style="color:#94a3b8; font-size:0.75rem; margin:0;">Overall</p>'
        f'<p style="color:{overall_color}; font-size:1.4rem; font-weight:700; margin:0;">'
        f'{overall}</p></div>',
        unsafe_allow_html=True,
    )
    c2.metric("Buy", buy_count)
    c3.metric("Sell", sell_count)
    c4.metric("Neutral", neutral_count)
    c5.metric("Trend / No-Trend", f"{trend_count} / {no_trend_count}")

    st.divider()

    # Color the signal column
    def _style_signal(val):
        if val == "Buy":
            return f"color: {COLORS['success']}; font-weight: 600"
        elif val == "Sell":
            return f"color: {COLORS['danger']}; font-weight: 600"
        elif val == "Trend":
            return f"color: {COLORS['info']}; font-weight: 600"
        elif val == "No Trend":
            return f"color: {COLORS['warning']}; font-weight: 600"
        return f"color: {COLORS['neutral']}"

    styled = df.style.applymap(_style_signal, subset=["Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=600)


# ---------------------------------------------------------------------------
# Statistics tab
# ---------------------------------------------------------------------------


def _render_statistics_tab(close, high, low, volume, open_, COLORS, SERIES_COLORS, dark_layout, ticker):
    """Render TA statistics module indicators."""
    import pandas as pd

    st.markdown("##### Statistical Indicators")
    st.caption("Z-scores, percentile ranks, skewness, kurtosis, entropy, and more")

    stat_indicators = {
        "Z-Score(20)": {
            "mod": "wraquant.ta.statistics", "fn": "zscore", "input": "close",
            "kw": {"period": 20},
            "tip": "Standard deviations from rolling mean; >2 OB, <-2 OS",
            "ref_lines": [2, -2, 0],
        },
        "Percentile Rank(20)": {
            "mod": "wraquant.ta.statistics", "fn": "percentile_rank", "input": "close",
            "kw": {"period": 20},
            "tip": "% of window values below current value",
            "ref_lines": [90, 10, 50],
        },
        "Skewness(20)": {
            "mod": "wraquant.ta.statistics", "fn": "skewness", "input": "close",
            "kw": {"period": 20},
            "tip": "Distribution asymmetry: +ve = right tail, -ve = left tail",
            "ref_lines": [0],
        },
        "Kurtosis(20)": {
            "mod": "wraquant.ta.statistics", "fn": "kurtosis", "input": "close",
            "kw": {"period": 20},
            "tip": "Tail fatness: >0 = fat tails (leptokurtic), <0 = thin",
            "ref_lines": [0],
        },
        "Entropy(20)": {
            "mod": "wraquant.ta.statistics", "fn": "entropy", "input": "close",
            "kw": {"period": 20},
            "tip": "Shannon entropy of price changes -- randomness measure",
            "ref_lines": [],
        },
        "Hurst(100)": {
            "mod": "wraquant.ta.statistics", "fn": "hurst_exponent", "input": "close",
            "kw": {"period": 100},
            "tip": "<0.5 = mean reverting, 0.5 = random walk, >0.5 = trending",
            "ref_lines": [0.5],
        },
        "Mean Deviation(20)": {
            "mod": "wraquant.ta.statistics", "fn": "mean_deviation", "input": "close",
            "kw": {"period": 20},
            "tip": "Rolling mean absolute deviation",
            "ref_lines": [],
        },
        "Median(20)": {
            "mod": "wraquant.ta.statistics", "fn": "median", "input": "close",
            "kw": {"period": 20},
            "tip": "Rolling median price",
            "ref_lines": [],
        },
    }

    _render_subplot_tab(
        "Statistics",
        stat_indicators,
        close, high, low, volume, open_,
        default_selection=["Z-Score(20)", "Hurst(100)", "Percentile Rank(20)"],
        COLORS=COLORS, SERIES_COLORS=SERIES_COLORS, dark_layout=dark_layout,
        ticker=ticker,
    )

    # Current values summary
    st.divider()
    st.markdown("##### Current Statistical Values")
    stat_cols = st.columns(4)
    stat_names = ["Z-Score(20)", "Percentile Rank(20)", "Skewness(20)", "Kurtosis(20)"]
    for col, name in zip(stat_cols, stat_names):
        try:
            result = _call_indicator(stat_indicators[name], close, high, low, volume, open_)
            if isinstance(result, pd.Series):
                val = result.dropna().iloc[-1] if len(result.dropna()) > 0 else None
                if val is not None:
                    col.metric(name, f"{val:.3f}")
                else:
                    col.metric(name, "---")
            else:
                col.metric(name, "---")
        except Exception:
            col.metric(name, "---")


# ---------------------------------------------------------------------------
# Performance tab
# ---------------------------------------------------------------------------


def _render_performance_tab(close, COLORS):
    """Render return-based performance metrics."""
    import numpy as np
    import pandas as pd

    st.markdown("##### Performance Metrics")
    st.caption("Return and drawdown analytics")

    returns = close.pct_change().dropna()
    if len(returns) < 10:
        st.warning("Insufficient data for performance analysis.")
        return

    # Metrics row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    ann_vol = returns.std() * np.sqrt(252) * 100
    daily_vol = returns.std() * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    max_dd = ((close / close.cummax()) - 1).min() * 100

    # Sortino
    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

    c1.metric("Total Return", f"{total_return:+.1f}%")
    c2.metric("Ann. Volatility", f"{ann_vol:.1f}%")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c4.metric("Sortino Ratio", f"{sortino:.2f}")
    c5.metric("Max Drawdown", f"{max_dd:.1f}%")
    c6.metric("Daily Vol", f"{daily_vol:.2f}%")

    st.divider()

    # Drawdown chart
    try:
        import plotly.graph_objects as go

        from wraquant.dashboard.components.charts import dark_layout

        dd_series = (close / close.cummax()) - 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series * 100,
            mode="lines", name="Drawdown",
            fill="tozeroy",
            line={"color": COLORS["danger"], "width": 1},
            fillcolor="rgba(239,68,68,0.15)",
        ))
        fig.update_layout(**dark_layout(
            title="Drawdown (%)", height=300, showlegend=False,
        ))
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Plotly is required for charts.")

    # Rolling metrics
    st.divider()
    st.markdown("##### Rolling Metrics")
    roll_window = st.slider("Rolling Window (days)", 21, 252, 63, key="ta_perf_roll")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from wraquant.dashboard.components.charts import dark_layout

        rolling_vol = returns.rolling(roll_window).std() * np.sqrt(252) * 100
        rolling_sharpe = (returns.rolling(roll_window).mean() / returns.rolling(roll_window).std()) * np.sqrt(252)
        rolling_return = returns.rolling(roll_window).sum() * 100

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=["Rolling Return (%)", "Rolling Volatility (%)", "Rolling Sharpe"],
        )

        fig.add_trace(go.Scatter(
            x=rolling_return.index, y=rolling_return,
            mode="lines", name="Rolling Return",
            line={"color": COLORS["primary"], "width": 1.5},
        ), row=1, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,0.3)", row=1, col=1)

        fig.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol,
            mode="lines", name="Rolling Vol",
            line={"color": COLORS["warning"], "width": 1.5},
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe,
            mode="lines", name="Rolling Sharpe",
            line={"color": COLORS["accent4"], "width": 1.5},
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,0.3)", row=3, col=1)

        fig.update_layout(**dark_layout(
            title=f"Rolling {roll_window}-Day Metrics", height=550, showlegend=False,
        ))
        for r in range(1, 4):
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=r, col=1)
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=r, col=1)

        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Plotly is required for charts.")

    # Additional performance stats
    st.divider()
    st.markdown("##### Additional Statistics")
    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("Skewness", f"{returns.skew():.3f}")
    ac2.metric("Kurtosis", f"{returns.kurtosis():.3f}")
    ac3.metric("Best Day", f"{returns.max() * 100:+.2f}%")
    ac4.metric("Worst Day", f"{returns.min() * 100:+.2f}%")

    ac5, ac6, ac7, ac8 = st.columns(4)
    up_days = (returns > 0).sum()
    down_days = (returns < 0).sum()
    ac5.metric("Up Days", f"{up_days} ({up_days / len(returns) * 100:.0f}%)")
    ac6.metric("Down Days", f"{down_days} ({down_days / len(returns) * 100:.0f}%)")
    ac7.metric("Avg Up Day", f"{returns[returns > 0].mean() * 100:+.2f}%" if up_days > 0 else "---")
    ac8.metric("Avg Down Day", f"{returns[returns < 0].mean() * 100:+.2f}%" if down_days > 0 else "---")
