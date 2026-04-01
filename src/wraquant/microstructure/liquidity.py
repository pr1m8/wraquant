"""Liquidity analytics for market microstructure.

Liquidity measures how easily an asset can be traded without
significantly moving its price. Illiquid assets carry a liquidity risk
premium and pose execution challenges. This module provides the
standard toolkit for measuring liquidity from trade and quote data.

Measures provided:

**Illiquidity / price impact**:
    - ``amihud_illiquidity``: the Amihud (2002) ratio -- average daily
      |return| / volume. Higher values indicate less liquid assets.
      The most widely used cross-sectional liquidity proxy because it
      only requires daily data.
    - ``kyle_lambda``: Kyle's lambda -- the permanent price impact
      coefficient estimated via rolling OLS of price changes on signed
      order flow. Higher lambda = more price impact per unit of volume.
    - ``price_impact``: per-trade permanent price impact.

**Spread estimators**:
    - ``roll_spread``: Roll (1984) implied spread from serial
      autocovariance of price changes. Requires only trade prices
      (no quote data needed).
    - ``effective_spread``: 2 * |trade_price - midpoint|. The
      standard measure of execution cost.
    - ``realized_spread``: spread earned by the liquidity provider
      after a delay, capturing adverse selection.

**Activity**:
    - ``turnover_ratio``: daily volume / shares outstanding. Measures
      trading activity relative to float.

How to choose:
    - **Cross-sectional liquidity ranking** (daily data only): use
      ``amihud_illiquidity``.
    - **Execution cost analysis** (trade + quote data): use
      ``effective_spread`` and ``realized_spread``.
    - **Price impact modeling**: use ``kyle_lambda`` for permanent
      impact; ``price_impact`` for per-trade measurement.
    - **No quote data available**: use ``roll_spread`` as a proxy
      for the bid-ask spread.

References:
    - Amihud (2002), "Illiquidity and Stock Returns"
    - Kyle (1985), "Continuous Auctions and Insider Trading"
    - Roll (1984), "A Simple Implicit Measure of the Effective
      Bid-Ask Spread"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from wraquant.core._coerce import coerce_series


def amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
    window: int | None = None,
) -> pd.Series | float:
    """Amihud (2002) illiquidity ratio: mean of |return| / dollar volume.

    A higher value indicates less liquid (more illiquid) markets.

    Parameters:
        returns: Asset return series.
        volume: Dollar volume series (price * shares traded).
        window: Rolling window size. If *None*, returns a single scalar
            average over the entire sample.

    Returns:
        Rolling Amihud illiquidity ratio (or a single float when
        *window* is *None*).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.randn(252) * 0.01)
        >>> volume = pd.Series(np.random.uniform(1e6, 5e6, 252))
        >>> illiq = amihud_illiquidity(returns, volume)
        >>> illiq > 0
        True

    See Also:
        kyle_lambda: Price impact coefficient (regression-based alternative).
        amihud_rolling: Rolling version with normalization.
    """
    returns = coerce_series(returns, "returns")
    volume = coerce_series(volume, "volume")
    ratio = np.abs(returns) / volume
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    if window is None:
        return float(np.nanmean(ratio))
    return ratio.rolling(window).mean()


def kyle_lambda(
    prices: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Kyle's lambda -- price impact coefficient via rolling OLS.

    Regresses price changes on signed order flow (volume) to estimate the
    permanent price impact per unit of volume.

    Parameters:
        prices: Price series.
        volume: Signed volume series (positive for buys, negative for sells).
        window: Rolling regression window.

    Returns:
        Rolling Kyle's lambda series.  Higher values indicate more
        price impact per unit of volume (less liquid).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
        >>> volume = pd.Series(np.random.randn(100) * 1000)
        >>> lam = kyle_lambda(prices, volume, window=20)
        >>> len(lam) == 100
        True

    See Also:
        amihud_illiquidity: Simpler illiquidity proxy (no signed volume needed).
        lambda_kyle_rolling: Kyle's lambda with confidence intervals.
    """
    prices = coerce_series(prices, "prices")
    volume = coerce_series(volume, "volume")
    delta_p = prices.diff()
    # Rolling OLS: lambda = cov(dp, v) / var(v)
    cov_pv = delta_p.rolling(window).cov(volume)
    var_v = volume.rolling(window).var()
    lam = cov_pv / var_v
    lam = lam.replace([np.inf, -np.inf], np.nan)
    lam.name = "kyle_lambda"
    return lam


def roll_spread(prices: pd.Series) -> float:
    """Roll (1984) implied bid-ask spread from serial covariance.

    Estimates the effective spread from the negative first-order
    autocovariance of price changes: spread = 2 * sqrt(-cov).

    Parameters:
        prices: Price series.

    Returns:
        Estimated implied spread. Returns *NaN* if the serial
        covariance is non-negative (model assumption violated).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> # Simulate trade prices with bid-ask bounce
        >>> mid = 100 + np.cumsum(np.random.randn(500) * 0.01)
        >>> bounce = np.random.choice([-0.05, 0.05], size=500)
        >>> prices = pd.Series(mid + bounce)
        >>> spread = roll_spread(prices)
        >>> spread > 0 or np.isnan(spread)  # positive spread or NaN
        True

    See Also:
        effective_spread: Direct spread from trade and quote data.
        corwin_schultz_spread: High-low spread estimator (OHLC data).
    """
    prices = coerce_series(prices, "prices")
    dp = prices.diff().dropna()
    cov = np.cov(dp.values[:-1], dp.values[1:])[0, 1]
    if cov >= 0:
        return np.nan
    return 2.0 * np.sqrt(-cov)


def effective_spread(
    trade_prices: pd.Series | NDArray[np.floating],
    midpoints: pd.Series | NDArray[np.floating],
) -> pd.Series | NDArray[np.floating]:
    """Effective bid-ask spread: 2 * |trade_price - midpoint|.

    Parameters:
        trade_prices: Executed trade prices.
        midpoints: Prevailing bid-ask midpoints at time of each trade.

    Returns:
        Per-trade effective spread, same type as the inputs.

    Example:
        >>> import pandas as pd, numpy as np
        >>> trades = pd.Series([100.05, 99.95, 100.03])
        >>> mids = pd.Series([100.0, 100.0, 100.0])
        >>> spreads = effective_spread(trades, mids)
        >>> float(spreads.iloc[0])
        0.1

    See Also:
        realized_spread: Post-trade spread (adverse selection component).
        roll_spread: Implied spread from price autocovariance.
    """
    trade_prices = coerce_series(trade_prices, "trade_prices")
    midpoints = coerce_series(midpoints, "midpoints")
    return 2.0 * np.abs(trade_prices - midpoints)


def realized_spread(
    trade_prices: pd.Series,
    midpoints: pd.Series,
    delay: int = 5,
) -> pd.Series:
    """Realized spread incorporating a post-trade midpoint delay.

    Measures the revenue to the liquidity provider:
    ``2 * direction * (trade_price - midpoint_{t+delay})``.

    Parameters:
        trade_prices: Executed trade prices.
        midpoints: Mid-quote series aligned to trades.
        delay: Number of observations to shift the midpoint forward.

    Returns:
        Per-trade realized spread series (NaN for the last *delay* rows).

    Example:
        >>> import pandas as pd, numpy as np
        >>> trades = pd.Series([100.05, 99.95, 100.03, 100.01, 99.98])
        >>> mids = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
        >>> rs = realized_spread(trades, mids, delay=2)
        >>> len(rs) == 5
        True

    See Also:
        effective_spread: Total execution cost (before adverse selection).
        spread_decomposition: Full Huang-Stoll decomposition.
    """
    trade_prices = coerce_series(trade_prices, "trade_prices")
    midpoints = coerce_series(midpoints, "midpoints")
    direction = np.sign(trade_prices - midpoints)
    future_mid = midpoints.shift(-delay)
    return 2.0 * direction * (trade_prices - future_mid)


def price_impact(
    trade_prices: pd.Series,
    volume: pd.Series,
    direction: pd.Series,
) -> pd.Series:
    """Permanent price impact per trade.

    Computed as ``direction * (midpoint_{t+1} - midpoint_t) / volume``,
    approximated here via successive trade prices.

    Parameters:
        trade_prices: Executed trade prices.
        volume: Volume for each trade.
        direction: Trade direction indicator (+1 buy, -1 sell).

    Returns:
        Per-trade permanent price impact series.

    Example:
        >>> import pandas as pd, numpy as np
        >>> trades = pd.Series([100.0, 100.05, 100.10, 100.08])
        >>> vol = pd.Series([1000, 2000, 1500, 1800])
        >>> direction = pd.Series([1, 1, -1, 1])
        >>> impact = price_impact(trades, vol, direction)
        >>> len(impact) == 4
        True

    See Also:
        kyle_lambda: Aggregate price impact coefficient.
        wraquant.microstructure.market_quality.price_impact_regression:
            Permanent vs. temporary impact decomposition.
    """
    trade_prices = coerce_series(trade_prices, "trade_prices")
    volume = coerce_series(volume, "volume")
    direction = coerce_series(direction, "direction")
    dp = trade_prices.diff().shift(-1)
    impact = direction * dp / volume
    impact = impact.replace([np.inf, -np.inf], np.nan)
    impact.name = "price_impact"
    return impact


def turnover_ratio(
    volume: pd.Series,
    shares_outstanding: pd.Series | float,
) -> pd.Series:
    """Turnover ratio: volume / shares outstanding.

    Parameters:
        volume: Daily trading volume.
        shares_outstanding: Total shares outstanding (scalar or series).

    Returns:
        Daily turnover ratio.  Higher values indicate more active trading.

    Example:
        >>> import pandas as pd
        >>> volume = pd.Series([1e6, 1.5e6, 0.8e6])
        >>> ratio = turnover_ratio(volume, shares_outstanding=100e6)
        >>> float(ratio.iloc[0])
        0.01

    See Also:
        amihud_illiquidity: Price-impact-based liquidity measure.
    """
    volume = coerce_series(volume, "volume")
    ratio = volume / shares_outstanding
    ratio.name = "turnover_ratio"
    return ratio


# ---------------------------------------------------------------------------
# Enhanced liquidity analytics
# ---------------------------------------------------------------------------


def corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    window: int = 1,
) -> pd.Series:
    """Corwin & Schultz (2012) high-low spread estimator.

    Estimates the effective bid-ask spread from consecutive daily high and
    low prices.  The key insight is that daily high prices are almost always
    buyer-initiated (at the ask) while daily lows are seller-initiated (at
    the bid).  The ratio of high-to-low therefore captures both volatility
    *and* the spread.  By comparing single-day and two-day high-low ranges
    the method disentangles the two components.

    **When to use**: When only daily OHLC data is available and you need a
    spread estimate.  More robust than the Roll (1984) estimator because it
    does not require negative serial covariance and performs better in the
    presence of stale prices.

    **Interpretation**: Output is in price units (same scale as the input).
    Values typically range from 0 (perfectly liquid) to several percent of
    price for illiquid stocks.  Negative estimates are floored at zero
    (model assumption violated, usually when volatility overwhelms spread).

    Parameters:
        high: Daily high prices.
        low: Daily low prices.
        window: Averaging window for the spread estimate.  ``window=1``
            returns the raw daily estimate.

    Returns:
        Estimated bid-ask spread series, floored at zero.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> close = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
        >>> high = close + np.abs(np.random.randn(100)) * 0.3
        >>> low = close - np.abs(np.random.randn(100)) * 0.3
        >>> spread = corwin_schultz_spread(high, low)
        >>> (spread >= 0).all()
        True

    References:
        Corwin, S. A. & Schultz, P. (2012). "A Simple Way to Estimate
        Bid-Ask Spreads from Daily High and Low Prices." *Journal of
        Finance*, 67(2), 719-760.

    See Also:
        roll_spread: Implied spread from trade prices only.
        effective_spread: Direct spread from trade and quote data.
    """
    high = coerce_series(high, "high")
    low = coerce_series(low, "low")
    # Natural log of high/low ratio, squared
    ln_hl = np.log(high / low)
    beta = ln_hl ** 2

    # Sum of beta over two consecutive days
    beta_sum = beta + beta.shift(1)

    # Two-day high-low range
    high_2d = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    low_2d = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = np.log(high_2d / low_2d) ** 2

    # Corwin-Schultz alpha and spread
    # alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
    #       - sqrt(gamma / (3 - 2*sqrt(2)))
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta_sum) - np.sqrt(beta_sum)) / k - np.sqrt(gamma / k)

    # S = 2 * (e^alpha - 1) / (1 + e^alpha)
    exp_alpha = np.exp(alpha)
    spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)

    # Floor at zero -- negative estimates are artefacts
    spread = spread.clip(lower=0.0)

    if window > 1:
        spread = spread.rolling(window).mean()

    spread.name = "corwin_schultz_spread"
    return spread


def closing_quoted_spread(
    bid_close: pd.Series,
    ask_close: pd.Series,
) -> pd.Series:
    """Quoted bid-ask spread at the market close.

    The closing spread is particularly relevant for investors who trade at or
    near the close (e.g., mutual fund NAV calculations, index rebalancing,
    MOC orders).  It also serves as a simple daily liquidity proxy when
    intraday data is unavailable.

    **When to use**: When analyzing execution costs for daily-frequency
    traders, evaluating end-of-day liquidity conditions, or constructing a
    daily spread time series from closing quote data.

    **Interpretation**: Narrower spreads indicate better end-of-day
    liquidity.  Spread widening at the close often precedes periods of
    higher volatility or information events (e.g., earnings releases).

    Parameters:
        bid_close: Best bid price at market close.
        ask_close: Best ask price at market close.

    Returns:
        Closing quoted spread series (ask - bid), in price units.

    Example:
        >>> import pandas as pd
        >>> bid = pd.Series([99.90, 99.85, 99.95])
        >>> ask = pd.Series([100.10, 100.15, 100.05])
        >>> spread = closing_quoted_spread(bid, ask)
        >>> float(spread.iloc[0])
        0.2

    References:
        Chordia, T., Roll, R. & Subrahmanyam, A. (2001). "Market Liquidity
        and Trading Activity." *Journal of Finance*, 56(2), 501-530.

    See Also:
        effective_spread: Execution-weighted spread measure.
        relative_spread: Spread normalized by midpoint.
    """
    bid_close = coerce_series(bid_close, "bid_close")
    ask_close = coerce_series(ask_close, "ask_close")
    spread = ask_close - bid_close
    spread.name = "closing_quoted_spread"
    return spread


def depth_imbalance(
    bid_depth: pd.Series | NDArray[np.floating],
    ask_depth: pd.Series | NDArray[np.floating],
) -> pd.Series | NDArray[np.floating]:
    """Order book depth imbalance.

    Computes ``(bid_depth - ask_depth) / (bid_depth + ask_depth)`` to
    measure the directional imbalance in resting limit order volume.

    **When to use**: For real-time assessment of supply-demand imbalance in
    the limit order book.  Commonly used as a short-horizon return predictor
    in high-frequency strategies.

    **Interpretation**:

    - **+1**: All depth is on the bid side (strong buying interest,
      bullish signal).
    - **-1**: All depth is on the ask side (strong selling interest,
      bearish signal).
    - **0**: Balanced book.

    Values persistently above +0.3 or below -0.3 often indicate directional
    pressure that leads to price movement in the direction of the deeper
    side.

    Parameters:
        bid_depth: Total volume at the best bid (or top-N bid levels).
        ask_depth: Total volume at the best ask (or top-N ask levels).

    Returns:
        Depth imbalance in [-1, 1].

    Example:
        >>> import pandas as pd
        >>> bid_depth = pd.Series([5000, 3000, 4000])
        >>> ask_depth = pd.Series([3000, 5000, 4000])
        >>> imb = depth_imbalance(bid_depth, ask_depth)
        >>> float(imb.iloc[0])  # more bids than asks -> positive
        0.25

    References:
        Cao, C., Hansch, O. & Wang, X. (2009). "The Information Content
        of an Open Limit-Order Book." *Journal of Futures Markets*, 29(1),
        16-41.

    See Also:
        wraquant.microstructure.toxicity.order_flow_imbalance:
            Volume-based imbalance measure.
        wraquant.microstructure.market_quality.depth: Total market depth.
    """
    is_series = isinstance(bid_depth, pd.Series)
    bid_series = coerce_series(bid_depth, "bid_depth")
    ask_series = coerce_series(ask_depth, "ask_depth")
    bid_arr = bid_series.to_numpy(dtype=np.float64)
    ask_arr = ask_series.to_numpy(dtype=np.float64)

    total = bid_arr + ask_arr
    imbalance = np.where(total > 0, (bid_arr - ask_arr) / total, 0.0)

    if is_series:
        return pd.Series(imbalance, index=bid_series.index, name="depth_imbalance")
    return imbalance


def lambda_kyle_rolling(
    prices: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """Rolling Kyle's lambda with confidence intervals.

    Extends :func:`kyle_lambda` by computing standard errors from the
    rolling OLS regression, yielding point estimates along with 95%
    confidence bounds.  This is essential for determining whether the
    estimated price impact is statistically significant at each point in
    time.

    **When to use**: When you need not just the *level* of price impact but
    also its *precision*.  Useful for detecting regime changes in market
    liquidity -- a significant widening of the confidence interval suggests
    structural uncertainty about the price impact coefficient.

    **Interpretation**: A positive lambda indicates that buy-initiated
    volume pushes prices up (and sell-initiated pushes down), consistent
    with the Kyle (1985) model.  Lambda values close to zero (or with
    confidence intervals spanning zero) suggest limited permanent price
    impact, i.e., a liquid market.

    Parameters:
        prices: Price series.
        volume: Signed volume series (positive for buys, negative for
            sells).
        window: Rolling regression window (must be >= 5).

    Returns:
        DataFrame with columns ``'lambda'``, ``'std_err'``,
        ``'ci_lower'``, ``'ci_upper'`` (95% confidence interval).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.1))
        >>> volume = pd.Series(np.random.randn(50) * 1000)
        >>> result = lambda_kyle_rolling(prices, volume, window=20)
        >>> list(result.columns)
        ['lambda', 'std_err', 'ci_lower', 'ci_upper']

    References:
        Kyle, A. S. (1985). "Continuous Auctions and Insider Trading."
        *Econometrica*, 53(6), 1315-1335.

    See Also:
        kyle_lambda: Simple point estimate without confidence intervals.
        amihud_rolling: Rolling Amihud illiquidity ratio.
    """
    prices = coerce_series(prices, "prices")
    volume = coerce_series(volume, "volume")
    delta_p = prices.diff()

    lam = pd.Series(np.nan, index=prices.index, name="lambda")
    se = pd.Series(np.nan, index=prices.index, name="std_err")

    for i in range(window, len(prices)):
        y = delta_p.iloc[i - window + 1 : i + 1].values
        x = volume.iloc[i - window + 1 : i + 1].values

        # Skip windows with NaN
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 5:
            continue

        y_clean = y[mask]
        x_clean = x[mask]

        n = len(y_clean)
        x_bar = np.mean(x_clean)
        var_x = np.sum((x_clean - x_bar) ** 2)

        if var_x < 1e-15:
            continue

        beta = np.sum((x_clean - x_bar) * (y_clean - np.mean(y_clean))) / var_x
        residuals = y_clean - (np.mean(y_clean) - beta * x_bar + beta * x_clean)
        s2 = np.sum(residuals ** 2) / max(n - 2, 1)
        std_err = np.sqrt(s2 / var_x)

        lam.iloc[i] = beta
        se.iloc[i] = std_err

    ci_lower = lam - 1.96 * se
    ci_upper = lam + 1.96 * se

    return pd.DataFrame(
        {"lambda": lam, "std_err": se, "ci_lower": ci_lower, "ci_upper": ci_upper},
        index=prices.index,
    )


def amihud_rolling(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 21,
    normalize: bool = True,
) -> pd.Series:
    """Rolling Amihud (2002) illiquidity ratio with proper normalization.

    Computes the Amihud ratio over a rolling window and optionally
    normalizes by the cross-sectional or time-series mean so that values
    are comparable across different assets and time periods.

    **When to use**: For tracking how an individual asset's liquidity
    evolves over time.  The normalization makes the measure comparable
    across assets with different price levels and trading volumes.

    **Interpretation**: Higher values indicate less liquidity (more price
    impact per unit of trading volume).  Sudden spikes often correspond
    to liquidity crises or market stress events.  Typical values for
    large-cap US stocks are 1e-11 to 1e-9 (unnormalized).

    Parameters:
        returns: Asset return series.
        volume: Dollar volume series (price * shares traded).
        window: Rolling window size (default 21 for ~1 month of trading
            days).
        normalize: If *True*, divide each rolling value by the full-sample
            mean so the time-series average is 1.0.

    Returns:
        Rolling Amihud illiquidity series.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.randn(100) * 0.01)
        >>> volume = pd.Series(np.random.uniform(1e6, 5e6, 100))
        >>> illiq = amihud_rolling(returns, volume, window=21)
        >>> illiq.name
        'amihud_rolling'

    References:
        Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section
        and Time-Series Effects." *Journal of Financial Markets*, 5(1),
        31-56.

    See Also:
        amihud_illiquidity: Static (full-sample) Amihud ratio.
        liquidity_commonality: How much liquidity co-moves with the market.
    """
    returns = coerce_series(returns, "returns")
    volume = coerce_series(volume, "volume")
    ratio = np.abs(returns) / volume
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    rolling = ratio.rolling(window).mean()

    if normalize:
        full_mean = np.nanmean(rolling)
        if full_mean > 0:
            rolling = rolling / full_mean

    rolling.name = "amihud_rolling"
    return rolling


def liquidity_commonality(
    asset_illiquidity: pd.Series,
    market_illiquidity: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Commonality in liquidity (Chordia, Roll & Subrahmanyam, 2000).

    Measures how much an individual asset's liquidity co-moves with
    market-wide liquidity.  The commonality coefficient is estimated via
    rolling regressions of changes in the asset's illiquidity measure on
    changes in the market-wide illiquidity measure.

    **When to use**: For assessing systematic liquidity risk.  Assets with
    high commonality become illiquid precisely when the entire market
    becomes illiquid -- an undesirable property that investors demand a
    premium for bearing.

    **Interpretation**: The output is the rolling R-squared from the
    regression.  Higher values (closer to 1) indicate stronger co-movement
    with market liquidity.  Values above 0.3 suggest meaningful systematic
    liquidity risk.  Most large-cap stocks show commonality R-squared of
    0.05-0.20.

    Parameters:
        asset_illiquidity: Individual asset's illiquidity measure (e.g.,
            Amihud ratio, effective spread) as a time series.
        market_illiquidity: Market-wide illiquidity aggregate (e.g.,
            equal-weighted average Amihud ratio across all stocks).
        window: Rolling regression window (default 60 for ~3 months).

    Returns:
        Rolling R-squared of the commonality regression.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> asset = pd.Series(np.random.randn(200).cumsum())
        >>> market = pd.Series(np.random.randn(200).cumsum())
        >>> r2 = liquidity_commonality(asset, market, window=60)
        >>> r2.name
        'liquidity_commonality'

    References:
        Chordia, T., Roll, R. & Subrahmanyam, A. (2000). "Commonality in
        Liquidity." *Journal of Financial Economics*, 56(1), 3-28.

    See Also:
        amihud_rolling: Generate the illiquidity input for this function.
    """
    asset_illiquidity = coerce_series(asset_illiquidity, "asset_illiquidity")
    market_illiquidity = coerce_series(market_illiquidity, "market_illiquidity")
    d_asset = asset_illiquidity.diff()
    d_market = market_illiquidity.diff()

    r_squared = pd.Series(np.nan, index=asset_illiquidity.index, name="liquidity_commonality")

    for i in range(window, len(d_asset)):
        y = d_asset.iloc[i - window + 1 : i + 1].values
        x = d_market.iloc[i - window + 1 : i + 1].values

        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 5:
            continue

        y_c = y[mask]
        x_c = x[mask]

        x_bar = np.mean(x_c)
        y_bar = np.mean(y_c)
        ss_xx = np.sum((x_c - x_bar) ** 2)
        ss_yy = np.sum((y_c - y_bar) ** 2)

        if ss_xx < 1e-15 or ss_yy < 1e-15:
            r_squared.iloc[i] = 0.0
            continue

        ss_xy = np.sum((x_c - x_bar) * (y_c - y_bar))
        r2 = (ss_xy ** 2) / (ss_xx * ss_yy)
        r_squared.iloc[i] = r2

    return r_squared


def spread_decomposition(
    trade_prices: pd.Series,
    bid: pd.Series,
    ask: pd.Series,
    direction: pd.Series,
    delay: int = 5,
) -> dict[str, float]:
    """Huang-Stoll (1997) three-way spread decomposition.

    Decomposes the effective spread into three economically distinct
    components:

    1. **Adverse selection**: compensation for trading against informed
       traders who possess private information.  This portion of the spread
       is a *permanent* price impact -- the midpoint moves against the
       liquidity provider after the trade.
    2. **Order processing**: compensation for the mechanical costs of
       market-making (exchange fees, technology, labor).
    3. **Inventory holding**: compensation for the risk of holding an
       unbalanced inventory.

    **When to use**: For understanding *why* spreads are wide.  If adverse
    selection dominates, the market has significant information asymmetry.
    If order processing dominates, the market is structurally costly.

    **Interpretation**:

    - Adverse selection fraction > 0.5 indicates a market dominated by
      informed trading (e.g., single-stock options, small-cap equities
      before earnings).
    - Order processing fraction > 0.5 indicates a market where mechanical
      costs dominate (e.g., bond markets, low-volatility large-cap
      equities).
    - Inventory fraction is typically the smallest component for equities
      but can be large for less liquid instruments.

    Parameters:
        trade_prices: Executed trade prices.
        bid: Best bid prices at time of each trade.
        ask: Best ask prices at time of each trade.
        direction: Trade direction indicator (+1 buy, -1 sell).
        delay: Number of observations to look ahead for measuring the
            permanent price impact (default 5).

    Returns:
        Dictionary with keys:

        - ``'adverse_selection'``: fraction of the spread due to
          information asymmetry.
        - ``'order_processing'``: fraction due to order handling costs.
        - ``'inventory_holding'``: fraction due to inventory risk.
        - ``'effective_spread_mean'``: average effective spread.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> n = 200
        >>> mid = 100 + np.cumsum(np.random.randn(n) * 0.01)
        >>> spread_half = 0.05
        >>> bid = pd.Series(mid - spread_half)
        >>> ask = pd.Series(mid + spread_half)
        >>> direction = pd.Series(np.random.choice([1, -1], n))
        >>> trades = pd.Series(np.where(direction > 0, ask, bid))
        >>> result = spread_decomposition(trades, bid, ask, direction)
        >>> 0 <= result['adverse_selection'] <= 1
        True

    References:
        Huang, R. D. & Stoll, H. R. (1997). "The Components of the
        Bid-Ask Spread: A General Approach." *Review of Financial Studies*,
        10(4), 995-1034.

    See Also:
        effective_spread: Total execution cost measure.
        realized_spread: Liquidity provider's revenue component.
    """
    trade_prices = coerce_series(trade_prices, "trade_prices")
    bid = coerce_series(bid, "bid")
    ask = coerce_series(ask, "ask")
    direction = coerce_series(direction, "direction")
    mid = (bid + ask) / 2.0

    # Effective half-spread per trade
    eff_half = direction * (trade_prices - mid)

    # Permanent component: midpoint revision in the direction of the trade
    mid_future = mid.shift(-delay)
    permanent = direction * (mid_future - mid)

    # Drop NaN rows at the end
    valid = ~(eff_half.isna() | permanent.isna())
    eff_valid = eff_half[valid]
    perm_valid = permanent[valid]

    mean_eff = float(np.nanmean(eff_valid))
    mean_perm = float(np.nanmean(perm_valid))

    if mean_eff <= 0:
        # Degenerate case
        return {
            "adverse_selection": float("nan"),
            "order_processing": float("nan"),
            "inventory_holding": float("nan"),
            "effective_spread_mean": float(mean_eff * 2.0),
        }

    # Adverse selection fraction
    adverse_frac = np.clip(mean_perm / mean_eff, 0.0, 1.0)

    # Realized spread = transitory component (order processing + inventory)
    transitory_frac = 1.0 - adverse_frac

    # Split transitory into order processing and inventory via serial
    # correlation of trade direction (proxy for inventory management)
    dir_arr = direction[valid].values.astype(np.float64)
    if len(dir_arr) > 1:
        autocorr = np.corrcoef(dir_arr[:-1], dir_arr[1:])[0, 1]
        if np.isnan(autocorr):
            autocorr = 0.0
        # Inventory fraction proportional to serial correlation of direction
        inventory_share = np.clip(abs(autocorr), 0.0, 1.0)
    else:
        inventory_share = 0.0

    inventory_frac = transitory_frac * inventory_share
    processing_frac = transitory_frac * (1.0 - inventory_share)

    return {
        "adverse_selection": float(adverse_frac),
        "order_processing": float(processing_frac),
        "inventory_holding": float(inventory_frac),
        "effective_spread_mean": float(mean_eff * 2.0),
    }
