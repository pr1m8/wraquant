"""Volume-based indicators.

This module provides indicators that incorporate volume data to measure
buying/selling pressure and trend confirmation. All functions accept
``pd.Series`` inputs and return ``pd.Series``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Re-export vwap from overlap (canonical location)
from wraquant.ta.overlap import vwap

__all__ = [
    "obv",
    "vwap",
    "ad_line",
    "cmf",
    "mfi",
    "eom",
    "force_index",
    "nvi",
    "pvi",
    "vpt",
    "adosc",
    "vwma",
    "pvt",
    "vpt_smoothed",
    "klinger",
    "taker_buy_ratio",
    "elder_force",
    "volume_profile",
    "accumulation_distribution_oscillator",
    "volume_roc",
    "positive_volume_index",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from wraquant.ta._validators import validate_period as _validate_period
from wraquant.ta._validators import validate_series as _validate_series


def _ema(data: pd.Series, period: int) -> pd.Series:
    """Internal EMA helper."""
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# On Balance Volume (OBV)
# ---------------------------------------------------------------------------


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume (OBV).

    OBV is a cumulative running total of volume. Volume is added on up-close
    days and subtracted on down-close days.

    Interpretation:
        - **Rising OBV**: Volume is flowing in (accumulation) -- bullish.
        - **Falling OBV**: Volume is flowing out (distribution) -- bearish.
        - **Divergence**: The most important signal. Price makes a new
          high but OBV does not = smart money is not confirming the
          move (bearish divergence). Price makes a new low but OBV
          does not = accumulation is occurring (bullish divergence).
        - **Breakout confirmation**: OBV breaking to a new high
          alongside price confirms the breakout is genuine.

    Trading rules:
        - Buy when OBV diverges bullishly from price (price falls,
          OBV holds or rises).
        - Sell when OBV diverges bearishly from price (price rises,
          OBV flattens or falls).
        - Use OBV trend (rising/falling) to confirm price trends.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        OBV values.
    """
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    direction = np.sign(close.diff())
    # First value has no diff — treat as 0
    direction.iloc[0] = 0
    result = (direction * volume).cumsum()
    result.name = "obv"
    return result


# ---------------------------------------------------------------------------
# Accumulation / Distribution Line
# ---------------------------------------------------------------------------


def ad_line(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Accumulation/Distribution Line (AD Line).

    Uses the Close Location Value (CLV) money flow multiplier.

    Interpretation:
        - **Rising AD line**: Accumulation -- buying pressure dominates.
          Volume is flowing into the asset.
        - **Falling AD line**: Distribution -- selling pressure dominates.
        - **Divergence**: Price makes new high but AD does not =
          distribution despite rising prices (bearish). Price makes
          new low but AD does not = accumulation (bullish).
        - Unlike OBV, the AD line considers where the close is within
          the high-low range, not just the direction of the close.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        Cumulative A/D line values.
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    hl_range = high - low
    clv = pd.Series(
        np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0),
        index=close.index,
    )
    money_flow_volume = clv * volume
    result = money_flow_volume.cumsum()
    result.name = "ad_line"
    return result


# ---------------------------------------------------------------------------
# Chaikin Money Flow (CMF)
# ---------------------------------------------------------------------------


def cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Chaikin Money Flow (CMF).

    Measures money flow volume over a rolling period, showing whether
    buying or selling pressure is dominant.

    Interpretation:
        - **Positive (> 0)**: Buying pressure (accumulation). The
          higher the value, the stronger the buying.
        - **Negative (< 0)**: Selling pressure (distribution).
        - **> +0.25**: Strong buying pressure.
        - **< -0.25**: Strong selling pressure.
        - **Zero-line crossover**: Shift from accumulation to
          distribution or vice versa.
        - **Divergence**: Price makes new high but CMF is falling =
          distribution.

    Trading rules:
        - Buy when CMF crosses above zero (accumulation starting).
        - Sell when CMF crosses below zero (distribution starting).
        - Confirm breakouts: price breaking resistance with positive
          CMF is more reliable.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    period : int, default 20
        Look-back period.

    Returns
    -------
    pd.Series
        CMF values in [-1, 1].
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")
    _validate_period(period)

    hl_range = high - low
    clv = pd.Series(
        np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0),
        index=close.index,
    )
    mf_volume = clv * volume
    result = (
        mf_volume.rolling(window=period, min_periods=period).sum()
        / volume.rolling(window=period, min_periods=period).sum()
    )
    result.name = "cmf"
    return result


# ---------------------------------------------------------------------------
# Money Flow Index (MFI)
# ---------------------------------------------------------------------------


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index (MFI).

    Often called the volume-weighted RSI. Incorporates both price and
    volume to identify overbought/oversold conditions.

    Interpretation:
        - **> 80**: Overbought -- high buying pressure, potential
          reversal.
        - **< 20**: Oversold -- high selling pressure, potential bounce.
        - **Divergence**: Price makes new high but MFI does not =
          weakening volume-confirmed momentum (bearish).
        - More reliable than RSI alone because it includes volume:
          a price move on heavy volume is more meaningful.

    Trading rules:
        - Buy when MFI crosses above 20 (leaving oversold).
        - Sell when MFI crosses below 80 (leaving overbought).
        - MFI divergence with price is a strong reversal signal.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    period : int, default 14
        Look-back period.

    Returns
    -------
    pd.Series
        MFI values in [0, 100].
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")
    _validate_period(period)

    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume

    tp_diff = typical_price.diff()
    pos_flow = pd.Series(np.where(tp_diff > 0, raw_money_flow, 0.0), index=close.index)
    neg_flow = pd.Series(np.where(tp_diff < 0, raw_money_flow, 0.0), index=close.index)

    pos_sum = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(window=period, min_periods=period).sum()

    money_ratio = pos_sum / neg_sum
    result = 100.0 - (100.0 / (1.0 + money_ratio))
    result.name = "mfi"
    return result


# ---------------------------------------------------------------------------
# Ease of Movement (EOM)
# ---------------------------------------------------------------------------


def eom(
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Ease of Movement (EMV / EOM).

    Relates price change to volume, showing how easily price is moving.

    Interpretation:
        - **Positive**: Price is advancing on relatively low volume
          (easy movement up) = bullish.
        - **Negative**: Price is declining on relatively low volume
          (easy movement down) = bearish.
        - **Near zero**: Price movement requires substantial volume
          = indecision or strong resistance/support.
        - **Zero-line crossover**: Shift from easy upward to easy
          downward movement or vice versa.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    volume : pd.Series
        Volume data.
    period : int, default 14
        Smoothing period.

    Returns
    -------
    pd.Series
        EOM values (smoothed).
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    volume = _validate_series(volume, "volume")
    _validate_period(period)

    distance = ((high + low) / 2.0) - ((high.shift(1) + low.shift(1)) / 2.0)
    box_ratio = (volume / 1e6) / (high - low)  # scale volume down
    raw_eom = distance / box_ratio
    result = raw_eom.rolling(window=period, min_periods=period).mean()
    result.name = "eom"
    return result


# ---------------------------------------------------------------------------
# Force Index
# ---------------------------------------------------------------------------


def force_index(
    close: pd.Series,
    volume: pd.Series,
    period: int = 13,
) -> pd.Series:
    """Force Index.

    ``Force = close.diff() * volume``, then smoothed with EMA.

    Combines price change and volume to measure the strength behind
    a move.

    Interpretation:
        - **Positive**: Buying force -- price is rising with volume.
        - **Negative**: Selling force -- price is falling with volume.
        - **Magnitude**: Larger values = stronger force behind the move.
        - **Zero-line crossover**: Shift from buying to selling force.
        - **Divergence**: Price makes new high but Force Index is
          declining = momentum weakening.

    Trading rules:
        - Buy when Force Index crosses above zero in an uptrend
          (pullback entry).
        - Sell when Force Index crosses below zero in a downtrend.
        - Use short period (2) for entries, longer period (13) for
          trend confirmation.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    period : int, default 13
        EMA smoothing period.

    Returns
    -------
    pd.Series
        Smoothed Force Index.
    """
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")
    _validate_period(period)

    raw_force = close.diff() * volume
    result = _ema(raw_force, period)
    result.name = "force_index"
    return result


# ---------------------------------------------------------------------------
# Negative Volume Index (NVI)
# ---------------------------------------------------------------------------


def nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Negative Volume Index (NVI).

    NVI focuses on days when volume decreases; the assumption is that
    smart money is active on low-volume days.

    Interpretation:
        - **Rising NVI**: Smart money is buying on quiet days.
        - **Falling NVI**: Smart money is selling on quiet days.
        - **NVI above its 255-day MA**: Bull market (historically
          correct ~96% of the time according to Norman Fosback).
        - **NVI below its 255-day MA**: Bear market.
        - Compare with PVI to distinguish smart vs. crowd behavior.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        NVI values (starts at 1000).
    """
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    pct_change = close.pct_change()
    vol_change = volume.diff()

    nvi_values = np.empty(len(close))
    nvi_values[0] = 1000.0

    pct_arr = pct_change.values
    vol_arr = vol_change.values

    for i in range(1, len(close)):
        if vol_arr[i] < 0:
            nvi_values[i] = nvi_values[i - 1] * (1.0 + pct_arr[i])
        else:
            nvi_values[i] = nvi_values[i - 1]

    result = pd.Series(nvi_values, index=close.index, name="nvi")
    return result


# ---------------------------------------------------------------------------
# Positive Volume Index (PVI)
# ---------------------------------------------------------------------------


def pvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Positive Volume Index (PVI).

    PVI focuses on days when volume increases; the assumption is that
    the crowd follows price on high-volume days.

    Interpretation:
        - **Rising PVI**: The crowd is buying on high-volume days.
        - **Falling PVI**: The crowd is selling on high-volume days.
        - **PVI below its 255-day MA**: Bearish sign (the crowd is
          pushing prices down on active days).
        - PVI tends to track what the public/retail traders are doing;
          NVI tracks institutional/smart money behavior.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        PVI values (starts at 1000).
    """
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    pct_change = close.pct_change()
    vol_change = volume.diff()

    pvi_values = np.empty(len(close))
    pvi_values[0] = 1000.0

    pct_arr = pct_change.values
    vol_arr = vol_change.values

    for i in range(1, len(close)):
        if vol_arr[i] > 0:
            pvi_values[i] = pvi_values[i - 1] * (1.0 + pct_arr[i])
        else:
            pvi_values[i] = pvi_values[i - 1]

    result = pd.Series(pvi_values, index=close.index, name="pvi")
    return result


# ---------------------------------------------------------------------------
# Volume Price Trend (VPT)
# ---------------------------------------------------------------------------


def vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Price Trend (VPT).

    ``VPT = cumsum(volume * pct_change(close))``

    Interpretation:
        - **Rising VPT**: Volume is confirming the price trend (bullish).
        - **Falling VPT**: Volume is working against the price trend.
        - **Divergence**: Price rises but VPT falls = distribution.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        VPT values.
    """
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    pct = close.pct_change()
    result = (volume * pct).cumsum()
    result.name = "vpt"
    return result


# ---------------------------------------------------------------------------
# Accumulation/Distribution Oscillator (Chaikin Oscillator)
# ---------------------------------------------------------------------------


def adosc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    """Accumulation/Distribution Oscillator (Chaikin Oscillator).

    ``ADOSC = EMA(AD, fast) - EMA(AD, slow)``

    Interpretation:
        - **Positive**: Short-term accumulation exceeds long-term
          = money is flowing in.
        - **Negative**: Short-term distribution exceeds long-term
          = money is flowing out.
        - **Zero-line crossover**: Buy when crossing above zero,
          sell when crossing below.
        - **Divergence**: Price makes new high but oscillator falls =
          distribution.

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    fast : int, default 3
        Fast EMA period.
    slow : int, default 10
        Slow EMA period.

    Returns
    -------
    pd.Series
        Chaikin Oscillator values.
    """
    ad = ad_line(high, low, close, volume)
    result = _ema(ad, fast) - _ema(ad, slow)
    result.name = "adosc"
    return result


# ---------------------------------------------------------------------------
# Volume Weighted Moving Average (VWMA)
# ---------------------------------------------------------------------------


def vwma(
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Volume Weighted Moving Average (VWMA).

    A simple moving average weighted by volume.  When volume is uniform this
    reduces to the standard SMA.

    ``VWMA = SUM(close * volume, period) / SUM(volume, period)``

    Interpretation:
        - **Price above VWMA**: Bullish -- volume-confirmed uptrend.
        - **Price below VWMA**: Bearish -- volume-confirmed downtrend.
        - **VWMA vs SMA**: When VWMA > SMA, heavy volume is occurring
          at higher prices (accumulation). When VWMA < SMA, heavy
          volume is at lower prices (distribution).
        - More responsive to volume spikes than a standard SMA.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    period : int, default 20
        Look-back period.

    Returns
    -------
    pd.Series
        VWMA values.

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([10, 11, 12, 13, 14.0])
    >>> volume = pd.Series([100, 100, 100, 100, 100.0])
    >>> vwma(close, volume, period=3)  # equals SMA when volume is uniform
    """
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")
    _validate_period(period)

    cv = close * volume
    result = (
        cv.rolling(window=period, min_periods=period).sum()
        / volume.rolling(window=period, min_periods=period).sum()
    )
    result.name = "vwma"
    return result


# ---------------------------------------------------------------------------
# Price Volume Trend (PVT)
# ---------------------------------------------------------------------------


def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Price Volume Trend (PVT).

    A cumulative indicator that adds a portion of volume proportional to
    the percentage change in close price.

    ``PVT = cumsum(pct_change(close) * volume)``

    This is functionally equivalent to :func:`vpt` but is provided as the
    commonly-used *PVT* alias.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        PVT values.

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([10, 11, 12.0])
    >>> volume = pd.Series([100, 200, 300.0])
    >>> pvt(close, volume)
    """
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")

    pct = close.pct_change()
    result = (volume * pct).cumsum()
    result.name = "pvt"
    return result


# ---------------------------------------------------------------------------
# Volume Price Trend — Smoothed (VPT Smoothed)
# ---------------------------------------------------------------------------


def vpt_smoothed(
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Volume Price Trend with EMA smoothing.

    Applies an EMA to the raw VPT line to reduce noise.

    ``VPT_SMOOTHED = EMA(cumsum(pct_change(close) * volume), period)``

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    period : int, default 14
        EMA smoothing period.

    Returns
    -------
    pd.Series
        Smoothed VPT values.

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([10, 11, 12, 13, 14.0])
    >>> volume = pd.Series([100, 200, 300, 400, 500.0])
    >>> vpt_smoothed(close, volume, period=3)
    """
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")
    _validate_period(period)

    raw_vpt = vpt(close, volume)
    result = _ema(raw_vpt, period)
    result.name = "vpt_smoothed"
    return result


# ---------------------------------------------------------------------------
# Klinger Volume Oscillator (KVO)
# ---------------------------------------------------------------------------


def klinger(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 34,
    slow: int = 55,
    signal: int = 13,
) -> dict[str, pd.Series]:
    """Klinger Volume Oscillator (KVO).

    Uses the relationship between price trend direction and volume to
    predict price reversals.

    Interpretation:
        - **KVO above zero**: Net volume flow is bullish (accumulation).
        - **KVO below zero**: Net volume flow is bearish (distribution).
        - **Signal line crossover**: KVO crossing above signal = buy;
          crossing below = sell.
        - **Divergence**: Price at new high but KVO declining = money
          flowing out despite rising prices (distribution).

    Trading rules:
        - Buy when KVO crosses above its signal line.
        - Sell when KVO crosses below its signal line.
        - Best for confirming price breakouts with volume support.

    ``trend = sign(hlc3 - hlc3.shift(1))``
    ``dm = high - low``
    ``cm = cumulative dm when trend unchanged, else dm``
    ``vf = volume * abs(2 * dm / cm - 1) * trend * 100``
    ``KVO = EMA(vf, fast) - EMA(vf, slow)``
    ``signal_line = EMA(KVO, signal)``

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    fast : int, default 34
        Fast EMA period.
    slow : int, default 55
        Slow EMA period.
    signal : int, default 13
        Signal line EMA period.

    Returns
    -------
    dict[str, pd.Series]
        ``{"kvo": <pd.Series>, "signal": <pd.Series>}``

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(0)
    >>> n = 100
    >>> close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
    >>> high = close + 0.5
    >>> low = close - 0.5
    >>> volume = pd.Series(np.random.randint(1000, 5000, n).astype(float))
    >>> result = klinger(high, low, close, volume)
    >>> result["kvo"].name
    'kvo'
    """
    high = _validate_series(high, "high")
    low = _validate_series(low, "low")
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")
    _validate_period(fast, "fast")
    _validate_period(slow, "slow")
    _validate_period(signal, "signal")

    hlc3 = (high + low + close) / 3.0
    trend = np.sign(hlc3 - hlc3.shift(1))
    trend.iloc[0] = 0

    dm = high - low
    dm_arr = dm.values.copy()
    trend_arr = trend.values

    # Cumulative dm resets when trend changes
    cm = np.empty(len(dm))
    cm[0] = dm_arr[0]
    for i in range(1, len(dm)):
        if trend_arr[i] == trend_arr[i - 1]:
            cm[i] = cm[i - 1] + dm_arr[i]
        else:
            cm[i] = dm_arr[i]

    cm_series = pd.Series(cm, index=close.index)

    # Volume Force
    ratio = pd.Series(
        np.where(cm_series != 0, 2.0 * dm / cm_series - 1.0, 0.0),
        index=close.index,
    )
    vf = volume * np.abs(ratio) * trend * 100.0

    kvo = _ema(vf, fast) - _ema(vf, slow)
    kvo.name = "kvo"
    sig = _ema(kvo, signal)
    sig.name = "kvo_signal"
    return {"kvo": kvo, "signal": sig}


# ---------------------------------------------------------------------------
# Taker Buy Ratio
# ---------------------------------------------------------------------------


def taker_buy_ratio(
    buy_volume: pd.Series,
    total_volume: pd.Series,
) -> pd.Series:
    """Taker Buy Ratio.

    A helper for exchange-level data that computes the fraction of volume
    coming from taker buys.

    ``ratio = buy_volume / total_volume``

    Values above 0.5 indicate net buying pressure; below 0.5 indicate net
    selling pressure.

    Parameters
    ----------
    buy_volume : pd.Series
        Taker buy volume.
    total_volume : pd.Series
        Total volume.

    Returns
    -------
    pd.Series
        Ratio in [0, 1] (NaN where total_volume is 0).

    Example
    -------
    >>> import pandas as pd
    >>> buy = pd.Series([50, 60, 40.0])
    >>> total = pd.Series([100, 100, 100.0])
    >>> taker_buy_ratio(buy, total)
    """
    buy_volume = _validate_series(buy_volume, "buy_volume")
    total_volume = _validate_series(total_volume, "total_volume")

    result = buy_volume / total_volume.replace(0, np.nan)
    result.name = "taker_buy_ratio"
    return result


# ---------------------------------------------------------------------------
# Elder's Force Index (EMA-smoothed variant)
# ---------------------------------------------------------------------------


def elder_force(
    close: pd.Series,
    volume: pd.Series,
    period: int = 2,
) -> pd.Series:
    """Elder's Force Index (EMA-smoothed variant).

    This is Elder's original formulation using a short EMA (default 2).
    For a longer-term variant, increase *period*.

    Interpretation:
        - **Positive**: Buying force (price rising with volume).
        - **Negative**: Selling force (price falling with volume).
        - **2-period**: Use for short-term entry timing within a trend.
          Buy dips to zero or slightly negative in an uptrend.
        - **13-period**: Use for trend direction confirmation.
        - **Divergence**: Price new high but Force declining = weakening.

    ``raw = close.diff() * volume``
    ``elder_force = EMA(raw, period)``

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    period : int, default 2
        EMA smoothing period.

    Returns
    -------
    pd.Series
        Smoothed Elder Force Index.

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([10, 11, 12, 11, 13.0])
    >>> volume = pd.Series([100, 200, 150, 300, 250.0])
    >>> elder_force(close, volume, period=2)
    """
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")
    _validate_period(period)

    raw = close.diff() * volume
    result = _ema(raw, period)
    result.name = "elder_force"
    return result


# ---------------------------------------------------------------------------
# Volume Profile
# ---------------------------------------------------------------------------


def volume_profile(
    close: pd.Series,
    volume: pd.Series,
    bins: int = 10,
) -> dict[str, pd.Series]:
    """Volume Profile (volume at price).

    Distributes volume into *bins* equally-spaced price buckets.  This is
    useful for identifying high-volume nodes (support/resistance) and
    low-volume nodes (price gaps).

    Interpretation:
        - **High-volume nodes (HVN)**: Price levels where significant
          trading occurred = strong support/resistance. Price tends to
          consolidate at these levels.
        - **Low-volume nodes (LVN)**: Price levels with little trading
          = price tends to move quickly through these zones.
        - **Point of control (POC)**: The price level with the highest
          volume = strongest S/R level.
        - **Value area**: The range containing ~70% of volume =
          fair value zone.

    Parameters
    ----------
    close : pd.Series
        Close prices (used to assign volume to price levels).
    volume : pd.Series
        Volume data.
    bins : int, default 10
        Number of price bins.

    Returns
    -------
    dict[str, pd.Series]
        ``{"price_bins": <pd.Series of bin labels>, "volume": <pd.Series of aggregated volume>}``

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([10, 11, 12, 11, 10.0])
    >>> volume = pd.Series([100, 200, 300, 200, 100.0])
    >>> vp = volume_profile(close, volume, bins=3)
    >>> len(vp["volume"])
    3
    """
    close = _validate_series(close, "close")
    volume = _validate_series(volume, "volume")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")

    price_min = close.min()
    price_max = close.max()

    # Handle the edge case where all prices are identical
    if price_min == price_max:
        bin_labels = pd.Series([price_min], name="price_bins")
        vol_agg = pd.Series([volume.sum()], name="volume")
        return {"price_bins": bin_labels, "volume": vol_agg}

    edges = np.linspace(price_min, price_max, bins + 1)
    bin_indices = np.digitize(close.values, edges) - 1
    # Clip so that the max value falls into the last bin
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    vol_by_bin = np.zeros(bins)
    for i, idx in enumerate(bin_indices):
        vol_by_bin[idx] += volume.values[i]

    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    return {
        "price_bins": pd.Series(bin_centers, name="price_bins"),
        "volume": pd.Series(vol_by_bin, name="volume"),
    }


# ---------------------------------------------------------------------------
# Accumulation/Distribution Oscillator (fast/slow EMA of AD line)
# ---------------------------------------------------------------------------


def accumulation_distribution_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    """Accumulation/Distribution Oscillator.

    Computes the difference between a fast and slow EMA of the
    Accumulation/Distribution line.  This is equivalent to the Chaikin
    Oscillator (:func:`adosc`) but provided under its full descriptive name.

    ``AD_OSC = EMA(AD, fast) - EMA(AD, slow)``

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.
    fast : int, default 3
        Fast EMA period.
    slow : int, default 10
        Slow EMA period.

    Returns
    -------
    pd.Series
        Oscillator values.

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(0)
    >>> n = 50
    >>> close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
    >>> high = close + 0.5
    >>> low = close - 0.5
    >>> volume = pd.Series(np.random.randint(1000, 5000, n).astype(float))
    >>> accumulation_distribution_oscillator(high, low, close, volume)
    """
    ad = ad_line(high, low, close, volume)
    result = _ema(ad, fast) - _ema(ad, slow)
    result.name = "ad_oscillator"
    return result


# ---------------------------------------------------------------------------
# Volume Rate of Change (Volume ROC)
# ---------------------------------------------------------------------------


def volume_roc(
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Volume Rate of Change (Volume ROC).

    Measures the percentage change in volume over a given period.

    Interpretation:
        - **Large positive**: Volume is surging compared to *period*
          bars ago = heightened interest, possible breakout.
        - **Large negative**: Volume is drying up = decreasing interest.
        - **Volume spike with price breakout**: Confirms the breakout.
        - **Volume spike without price movement**: Potential
          distribution or accumulation before a move.

    ``volume_roc = (volume - volume.shift(period)) / volume.shift(period) * 100``

    Parameters
    ----------
    volume : pd.Series
        Volume data.
    period : int, default 14
        Look-back period.

    Returns
    -------
    pd.Series
        Volume ROC as a percentage.

    Example
    -------
    >>> import pandas as pd
    >>> volume = pd.Series([100, 120, 110, 130, 140.0])
    >>> volume_roc(volume, period=2)
    """
    volume = _validate_series(volume, "volume")
    _validate_period(period)

    shifted = volume.shift(period)
    result = (volume - shifted) / shifted * 100.0
    result.name = "volume_roc"
    return result


# ---------------------------------------------------------------------------
# Positive Volume Index (descriptive alias)
# ---------------------------------------------------------------------------


def positive_volume_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Positive Volume Index (PVI).

    A descriptive-name alias for :func:`pvi`.  PVI focuses on days when
    volume increases relative to the prior day.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume data.

    Returns
    -------
    pd.Series
        PVI values (starts at 1000).

    Example
    -------
    >>> import pandas as pd
    >>> close = pd.Series([10, 11, 10, 12, 11.0])
    >>> volume = pd.Series([100, 200, 150, 300, 100.0])
    >>> positive_volume_index(close, volume).iloc[0]
    1000.0
    """
    result = pvi(close, volume)
    result.name = "positive_volume_index"
    return result
