"""Market quality metrics.

Provides bid-ask spread measures, market depth indicators, resilience
metrics, and variance ratio tests for assessing overall market quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def quoted_spread(
    bid: pd.Series | NDArray[np.floating],
    ask: pd.Series | NDArray[np.floating],
) -> pd.Series | NDArray[np.floating]:
    """Quoted bid-ask spread: ask - bid.

    Parameters:
        bid: Best bid prices.
        ask: Best ask prices.

    Returns:
        Absolute quoted spread.

    Example:
        >>> import numpy as np
        >>> bid = np.array([99.90, 99.85])
        >>> ask = np.array([100.10, 100.15])
        >>> quoted_spread(bid, ask)
        array([0.2, 0.3])

    See Also:
        relative_spread: Spread normalized by midpoint.
        wraquant.microstructure.liquidity.effective_spread:
            Execution-weighted spread.
    """
    return np.asarray(ask) - np.asarray(bid)


def relative_spread(
    bid: pd.Series | NDArray[np.floating],
    ask: pd.Series | NDArray[np.floating],
) -> pd.Series | NDArray[np.floating]:
    """Relative spread: (ask - bid) / midpoint.

    Parameters:
        bid: Best bid prices.
        ask: Best ask prices.

    Returns:
        Relative spread as a fraction of the midpoint.  Typical values
        are 0.001-0.01 for liquid large-cap stocks.

    Example:
        >>> import pandas as pd
        >>> bid = pd.Series([99.90, 99.85])
        >>> ask = pd.Series([100.10, 100.15])
        >>> rs = relative_spread(bid, ask)
        >>> float(rs.iloc[0])  # 0.20 / 100.0 = 0.002
        0.002

    See Also:
        quoted_spread: Absolute spread in price units.
    """
    bid_arr = np.asarray(bid, dtype=np.float64)
    ask_arr = np.asarray(ask, dtype=np.float64)
    mid = (bid_arr + ask_arr) / 2.0
    result = (ask_arr - bid_arr) / mid
    if isinstance(bid, pd.Series):
        return pd.Series(result, index=bid.index, name="relative_spread")
    return result


def depth(
    bid_volume: pd.DataFrame | NDArray[np.floating],
    ask_volume: pd.DataFrame | NDArray[np.floating],
    levels: int = 5,
) -> pd.Series | NDArray[np.floating]:
    """Market depth: total volume available at the top N price levels.

    Parameters:
        bid_volume: Volume at each bid level. Columns (or columns in the
            2-D array) represent successive price levels from best to worst.
        ask_volume: Volume at each ask level, same layout as *bid_volume*.
        levels: Number of price levels to include.

    Returns:
        Total depth (bid + ask) summed across the requested levels.

    Example:
        >>> import numpy as np
        >>> bid_vol = np.array([1000, 800, 500, 300, 200])
        >>> ask_vol = np.array([900, 700, 600, 400, 100])
        >>> depth(bid_vol, ask_vol, levels=3)
        4500.0

    See Also:
        wraquant.microstructure.liquidity.depth_imbalance:
            Directional imbalance between bid and ask depth.
    """
    bid_arr = np.asarray(bid_volume, dtype=np.float64)
    ask_arr = np.asarray(ask_volume, dtype=np.float64)

    # Handle 1-D vs 2-D
    if bid_arr.ndim == 1:
        bid_sum = np.sum(bid_arr[:levels])
        ask_sum = np.sum(ask_arr[:levels])
        return bid_sum + ask_sum

    bid_sum = np.sum(bid_arr[:, :levels], axis=1)
    ask_sum = np.sum(ask_arr[:, :levels], axis=1)
    total = bid_sum + ask_sum

    if isinstance(bid_volume, pd.DataFrame):
        return pd.Series(total, index=bid_volume.index, name="depth")
    return total


def resiliency(
    spreads: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Spread resiliency: how quickly the spread recovers after a shock.

    Measured as the negative autocorrelation of spread changes. A higher
    value indicates a more resilient market (spreads revert faster).

    Parameters:
        spreads: Time series of quoted or effective spreads.
        window: Rolling window for estimating autocorrelation of
            spread changes.

    Returns:
        Rolling resiliency measure.  Higher values indicate faster
        spread recovery (more resilient market).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> spreads = pd.Series(0.05 + np.random.randn(100) * 0.01)
        >>> res = resiliency(spreads, window=20)
        >>> res.name
        'resiliency'

    See Also:
        quoted_spread: Generate the spread input for this function.
        variance_ratio: Random walk efficiency test.
    """
    ds = spreads.diff()
    # Negative first-order autocorrelation of spread changes
    resilience = -ds.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan,
        raw=False,
    )
    resilience.name = "resiliency"
    return resilience


def variance_ratio(
    prices: pd.Series,
    short_period: int = 2,
    long_period: int = 10,
) -> dict[str, float]:
    """Lo-MacKinlay (1988) variance ratio test.

    Tests the random walk hypothesis by comparing the variance of
    *long_period* returns to *short_period* returns, scaled appropriately.
    Under a random walk, the ratio equals 1.

    Parameters:
        prices: Price series (levels, not returns).
        short_period: Short return horizon (default 2).
        long_period: Long return horizon (must be a multiple of
            *short_period* for a clean comparison, but this is not
            enforced).

    Returns:
        Dictionary with keys:

        - ``'vr'``: Variance ratio.
        - ``'z_stat'``: Asymptotic z-statistic under IID assumption.
        - ``'p_value'``: Two-sided p-value.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(500) * 0.01)))
        >>> result = variance_ratio(prices, short_period=2, long_period=10)
        >>> 'vr' in result and 'p_value' in result
        True

    See Also:
        market_efficiency_ratio: Multi-lag efficiency summary.
        resiliency: Spread recovery speed.
    """
    from scipy.stats import norm

    log_prices = np.log(prices).values
    n = len(log_prices)

    # Returns at two horizons (lagged differences, not n-th order diff)
    ret_short = log_prices[short_period:] - log_prices[:-short_period]
    ret_long = log_prices[long_period:] - log_prices[:-long_period]

    var_short = np.var(ret_short, ddof=1)
    var_long = np.var(ret_long, ddof=1)

    q = long_period / short_period
    vr = var_long / (q * var_short) if var_short > 0 else np.nan

    # Asymptotic z-statistic under IID
    nq = len(ret_short)
    se = np.sqrt(2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * nq))
    z_stat = (vr - 1.0) / se if se > 0 else np.nan
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan

    return {"vr": float(vr), "z_stat": float(z_stat), "p_value": float(p_value)}


# ---------------------------------------------------------------------------
# Enhanced market quality analytics
# ---------------------------------------------------------------------------


def hasbrouck_information_share(
    prices_list: list[pd.Series],
) -> dict[str, NDArray[np.floating]]:
    """Hasbrouck (1995) information share for price discovery analysis.

    Measures each venue's (or instrument's) contribution to the efficient
    price innovation.  The information share for venue *j* is the fraction
    of the total variance of the efficient price innovation attributable to
    that venue.

    The method is based on a Vector Error Correction Model (VECM).  When
    the innovation covariance matrix is non-diagonal, upper and lower
    bounds are computed via Cholesky factorization with different orderings.
    The midpoint of these bounds is reported as the point estimate.

    **When to use**: For analyzing where price discovery occurs across
    multiple venues (e.g., NYSE vs NASDAQ, futures vs spot, ADR vs local
    listing).  Essential for regulatory analysis and optimal execution
    venue selection.

    **Interpretation**:

    - Information shares sum to 1.0 across venues.
    - A venue with information share > 0.5 in a two-venue system is the
      *dominant* price discovery venue.
    - If the upper and lower bounds are far apart, the venues have highly
      correlated innovations and the attribution is ambiguous.

    Parameters:
        prices_list: List of price series from different venues, all
            sharing the same DatetimeIndex.  Must contain at least 2
            venues.

    Returns:
        Dictionary with keys:

        - ``'midpoint'``: Midpoint information shares (best point
          estimate).
        - ``'upper'``: Upper-bound information shares.
        - ``'lower'``: Lower-bound information shares.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> base = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.1))
        >>> venue_a = base + np.random.randn(200) * 0.01
        >>> venue_b = base + np.random.randn(200) * 0.05
        >>> result = hasbrouck_information_share([venue_a, venue_b])
        >>> abs(result['midpoint'].sum() - 1.0) < 0.01
        True

    References:
        Hasbrouck, J. (1995). "One Security, Many Markets: Determining
        the Contributions to Price Discovery." *Journal of Finance*,
        50(4), 1175-1199.

    See Also:
        gonzalo_granger_component: Unique (non-bounded) price discovery measure.
        wraquant.microstructure.toxicity.information_share:
            Simplified variance-based information share.
    """
    n_venues = len(prices_list)
    if n_venues < 2:
        one = np.ones(max(n_venues, 1), dtype=np.float64)
        return {"midpoint": one / max(n_venues, 1), "upper": one / max(n_venues, 1), "lower": one / max(n_venues, 1)}

    # Align and compute returns
    df = pd.concat(prices_list, axis=1).dropna()
    returns = df.diff().dropna()

    # Error correction term: spread between each venue and venue 0
    # Simple VECM: regress returns on lagged spread
    errors = df.subtract(df.iloc[:, 0], axis=0)

    # Residual covariance from a simplified VECM
    # Use raw return covariance as proxy for innovation covariance
    cov_matrix = returns.cov().values

    # Cholesky decomposition for each ordering to get bounds
    upper = np.zeros(n_venues, dtype=np.float64)
    lower = np.ones(n_venues, dtype=np.float64)

    for perm_start in range(n_venues):
        perm = list(range(perm_start, n_venues)) + list(range(perm_start))
        cov_perm = cov_matrix[np.ix_(perm, perm)]

        try:
            L = np.linalg.cholesky(cov_perm)
        except np.linalg.LinAlgError:
            # Add small ridge for numerical stability
            cov_perm += np.eye(n_venues) * 1e-10
            L = np.linalg.cholesky(cov_perm)

        # Information share = (sum of row in L)^2 / total variance
        row_sums = np.sum(L, axis=1)
        total_var = np.sum(row_sums ** 2)

        if total_var > 0:
            shares = row_sums ** 2 / total_var
        else:
            shares = np.ones(n_venues) / n_venues

        # Unpermute
        shares_orig = np.zeros(n_venues, dtype=np.float64)
        for idx_new, idx_orig in enumerate(perm):
            shares_orig[idx_orig] = shares[idx_new]

        upper = np.maximum(upper, shares_orig)
        lower = np.minimum(lower, shares_orig)

    midpoint = (upper + lower) / 2.0

    # Normalize midpoint to sum to 1
    s = midpoint.sum()
    if s > 0:
        midpoint /= s

    # Upper and lower are per-venue bounds; they don't necessarily sum to 1
    # but each individual permutation's shares do.  Keep them as-is.

    return {
        "midpoint": midpoint,
        "upper": upper,
        "lower": lower,
    }


def gonzalo_granger_component(
    prices_list: list[pd.Series],
) -> dict[str, NDArray[np.floating]]:
    """Gonzalo-Granger (1995) permanent-transitory decomposition.

    Decomposes cointegrated price series into a permanent (efficient price)
    component and transitory (pricing error) component.  The permanent
    component weights reveal each venue's contribution to the long-run
    efficient price.

    Unlike Hasbrouck information shares, the GG component shares are unique
    (not dependent on Cholesky ordering).

    **When to use**: As a complement to Hasbrouck information shares for
    price discovery analysis.  Particularly useful when you need a unique
    (not bounded) measure of each venue's price discovery contribution.

    **Interpretation**:

    - GG weights sum to 1.0 across venues.
    - A venue with a large GG weight drives the permanent price -- its
      price innovations are absorbed by the market as a whole.
    - A venue with a small GG weight primarily reflects transitory noise.

    Parameters:
        prices_list: List of cointegrated price series from different
            venues, sharing the same DatetimeIndex.

    Returns:
        Dictionary with keys:

        - ``'gg_weights'``: Gonzalo-Granger permanent component weights
          (one per venue, summing to 1).
        - ``'alpha'``: Error-correction coefficients for each venue.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> base = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.1))
        >>> venue_a = base + np.random.randn(200) * 0.01
        >>> venue_b = base + np.random.randn(200) * 0.05
        >>> result = gonzalo_granger_component([venue_a, venue_b])
        >>> abs(result['gg_weights'].sum() - 1.0) < 1e-10
        True

    References:
        Gonzalo, J. & Granger, C. (1995). "Estimation of Common Long-
        Memory Components in Cointegrated Systems." *Journal of Business
        & Economic Statistics*, 13(1), 27-35.

    See Also:
        hasbrouck_information_share: Cholesky-based information share.
    """
    n_venues = len(prices_list)
    if n_venues < 2:
        return {
            "gg_weights": np.ones(max(n_venues, 1), dtype=np.float64),
            "alpha": np.zeros(max(n_venues, 1), dtype=np.float64),
        }

    df = pd.concat(prices_list, axis=1).dropna()
    returns = df.diff().dropna()

    # Error correction term: spread relative to venue 0
    spread = df.iloc[:, 1:].subtract(df.iloc[:, 0], axis=0)
    spread_lagged = spread.shift(1)

    # Align returns and lagged spread on common index
    common_idx = returns.index.intersection(spread_lagged.dropna().index)
    returns_aligned = returns.loc[common_idx]
    spread_aligned = spread_lagged.loc[common_idx]

    # Estimate alpha: regress each venue's returns on the lagged spread
    # For simplicity with multiple spreads, use the average spread
    if spread_aligned.shape[1] > 1:
        ecm = spread_aligned.mean(axis=1)
    else:
        ecm = spread_aligned.iloc[:, 0]

    alphas = np.zeros(n_venues, dtype=np.float64)
    for j in range(n_venues):
        y = returns_aligned.iloc[:, j].values
        x = ecm.values
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 5:
            continue
        y_c = y[mask]
        x_c = x[mask]
        cov_xy = np.sum((x_c - x_c.mean()) * (y_c - y_c.mean()))
        var_x = np.sum((x_c - x_c.mean()) ** 2)
        if var_x > 1e-15:
            alphas[j] = cov_xy / var_x

    # GG weights: proportional to the orthogonal complement of alpha
    # weight_j = alpha_perp_j = 1 - |alpha_j| / sum(|alpha|)
    # More precisely: GG weight is proportional to how little the venue
    # adjusts to the error correction term
    abs_alpha = np.abs(alphas)
    total_alpha = abs_alpha.sum()

    if total_alpha < 1e-15:
        gg_weights = np.ones(n_venues, dtype=np.float64) / n_venues
    else:
        # Venue that adjusts less has higher weight
        gg_weights = 1.0 - abs_alpha / total_alpha
        gg_sum = gg_weights.sum()
        if gg_sum > 0:
            gg_weights /= gg_sum
        else:
            gg_weights = np.ones(n_venues, dtype=np.float64) / n_venues

    return {
        "gg_weights": gg_weights,
        "alpha": alphas,
    }


def market_efficiency_ratio(
    prices: pd.Series,
    lags: list[int] | None = None,
) -> dict[str, float | dict[int, float]]:
    """Market efficiency ratio based on variance ratio analysis.

    Adapts the Lo-MacKinlay variance ratio test for market quality
    assessment by computing the ratio at multiple lags and summarizing
    the degree of departure from efficient pricing.

    Under an efficient market (random walk), the variance of k-period
    returns equals k times the variance of 1-period returns (VR = 1).
    Departures indicate:

    - **VR > 1**: Positive autocorrelation (momentum, trending).
    - **VR < 1**: Negative autocorrelation (mean reversion,
      microstructure noise).

    **When to use**: For assessing how efficiently a market incorporates
    information.  Useful for comparing market quality across instruments,
    venues, or time periods.

    **Interpretation**: The ``efficiency_score`` is the average absolute
    deviation of variance ratios from 1.0.  Lower is more efficient:

    - **< 0.05**: Highly efficient market.
    - **0.05-0.15**: Moderately efficient.
    - **> 0.15**: Significant inefficiency (either microstructure noise
      or predictable patterns).

    Parameters:
        prices: Price series (levels, not returns).
        lags: List of return horizons to test (default [2, 5, 10, 20]).

    Returns:
        Dictionary with keys:

        - ``'efficiency_score'``: Average |VR - 1| across lags (lower
          is more efficient).
        - ``'variance_ratios'``: Dict mapping each lag to its VR value.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(500) * 0.01)))
        >>> result = market_efficiency_ratio(prices)
        >>> result['efficiency_score'] >= 0
        True

    References:
        Lo, A. W. & MacKinlay, A. C. (1988). "Stock Market Prices Do
        Not Follow Random Walks." *Review of Financial Studies*, 1(1),
        41-66.

    See Also:
        variance_ratio: Single-lag variance ratio test with p-value.
        resiliency: Spread-based market quality measure.
    """
    if lags is None:
        lags = [2, 5, 10, 20]

    log_prices = np.log(prices.values)
    ret_1 = log_prices[1:] - log_prices[:-1]
    var_1 = np.var(ret_1, ddof=1)

    vr_dict: dict[int, float] = {}
    for k in lags:
        if k >= len(log_prices):
            continue
        ret_k = log_prices[k:] - log_prices[:-k]
        var_k = np.var(ret_k, ddof=1)
        vr_dict[k] = float(var_k / (k * var_1)) if var_1 > 0 else np.nan

    deviations = [abs(v - 1.0) for v in vr_dict.values() if not np.isnan(v)]
    efficiency_score = float(np.mean(deviations)) if deviations else np.nan

    return {
        "efficiency_score": efficiency_score,
        "variance_ratios": vr_dict,
    }


def price_impact_regression(
    price_changes: pd.Series,
    signed_volume: pd.Series,
    lags: int = 5,
) -> dict[str, float]:
    """Price impact regression decomposing permanent and temporary effects.

    Regresses price changes on contemporaneous and lagged signed order
    flow to estimate:

    - **Permanent impact**: the long-run effect of a unit of signed
      volume on prices (information content).
    - **Temporary impact**: the transient effect that reverses over time
      (liquidity provision revenue).

    The regression model is::

        dp_t = c + beta_0 * v_t + beta_1 * v_{t-1} + ... + beta_k * v_{t-k} + eps_t

    Permanent impact is the sum of all beta coefficients.  Temporary
    impact is ``beta_0 - permanent_impact``.

    **When to use**: For analyzing the dynamic price impact of trading
    activity.  Essential for optimal execution and transaction cost
    analysis.

    **Interpretation**:

    - **Positive permanent impact**: trades convey information; the
      market adjusts permanently.
    - **Negative temporary impact**: trades cause a transient price
      displacement that reverses (liquidity provider's profit).
    - **R-squared**: fraction of price variation explained by order
      flow; higher values indicate more order-flow-driven pricing.

    Parameters:
        price_changes: Price change (delta-p) series.
        signed_volume: Signed order flow (positive = buys, negative =
            sells).
        lags: Number of lagged order flow terms to include (default 5).

    Returns:
        Dictionary with keys:

        - ``'permanent_impact'``: Sum of all order flow coefficients.
        - ``'temporary_impact'``: ``beta_0 - permanent_impact``.
        - ``'beta_0'``: Contemporaneous impact coefficient.
        - ``'r_squared'``: R-squared of the regression.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> dp = pd.Series(np.random.randn(200) * 0.01)
        >>> sv = pd.Series(np.random.randn(200) * 1000)
        >>> result = price_impact_regression(dp, sv, lags=3)
        >>> 'permanent_impact' in result and 'r_squared' in result
        True

    References:
        Hasbrouck, J. (1991). "Measuring the Information Content of
        Stock Trades." *Journal of Finance*, 46(1), 179-207.

    See Also:
        wraquant.microstructure.liquidity.kyle_lambda:
            Simpler single-coefficient price impact.
        wraquant.microstructure.liquidity.spread_decomposition:
            Spread-based adverse selection decomposition.
    """
    # Build design matrix with contemporaneous + lagged signed volume
    data = pd.DataFrame({"dp": price_changes})
    for lag in range(lags + 1):
        data[f"v_lag{lag}"] = signed_volume.shift(lag)

    data = data.dropna()
    if len(data) < lags + 3:
        return {
            "permanent_impact": np.nan,
            "temporary_impact": np.nan,
            "beta_0": np.nan,
            "r_squared": np.nan,
        }

    y = data["dp"].values
    X = data[[f"v_lag{i}" for i in range(lags + 1)]].values

    # OLS via shared regression module
    from wraquant.stats.regression import ols as _ols

    try:
        ols_result = _ols(y, X, add_constant=True)
    except (np.linalg.LinAlgError, ValueError):
        return {
            "permanent_impact": np.nan,
            "temporary_impact": np.nan,
            "beta_0": np.nan,
            "r_squared": np.nan,
        }

    beta = ols_result["coefficients"]
    r_sq = ols_result["r_squared"]

    # beta[0] is intercept, beta[1] is contemporaneous, beta[2:] are lags
    beta_0 = beta[1]
    permanent = float(np.sum(beta[1:]))
    temporary = float(beta_0 - permanent)

    return {
        "permanent_impact": permanent,
        "temporary_impact": temporary,
        "beta_0": float(beta_0),
        "r_squared": float(r_sq),
    }


def intraday_volatility_pattern(
    prices: pd.Series,
    freq: str = "h",
) -> pd.Series:
    """Estimate the intraday volatility pattern (U-shape or J-shape).

    Computes the average absolute return at each intraday time bucket
    (hourly by default), revealing the well-documented U-shaped pattern
    where volatility is highest at the open and close and lowest at
    midday.

    **When to use**: For understanding the diurnal volatility cycle of a
    market.  Essential for:

    - Optimal execution: schedule trades during low-volatility periods.
    - Risk management: adjust intraday VaR for time-of-day effects.
    - Market-making: widen quotes during high-volatility open/close.

    **Interpretation**: The output is indexed by time-of-day (e.g., hour).
    Peaks at the open and close indicate information-driven volatility
    (overnight information absorption and closing auctions).  A flat
    profile suggests a market dominated by algorithmic flow with little
    information asymmetry.

    Parameters:
        prices: Intraday price series with a DatetimeIndex.
        freq: Resampling frequency for the volatility buckets.  Use
            ``'h'`` for hourly, ``'30min'`` for half-hourly, ``'15min'``
            for 15-minute buckets.

    Returns:
        Average absolute return by time-of-day bucket, indexed by the
        bucket label (e.g., hour of day).

    Example:
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range('2024-01-02 09:30', periods=78, freq='5min')
        >>> prices = pd.Series(100 + np.cumsum(np.random.randn(78) * 0.1),
        ...                     index=idx)
        >>> pattern = intraday_volatility_pattern(prices, freq='h')
        >>> len(pattern) > 0
        True

    References:
        Wood, R. A., McInish, T. H. & Ord, J. K. (1985). "An
        Investigation of Transactions Data for NYSE Stocks." *Journal of
        Finance*, 40(3), 723-739.

        Admati, A. R. & Pfleiderer, P. (1988). "A Theory of Intraday
        Patterns: Volume and Price Variability." *Review of Financial
        Studies*, 1(1), 3-40.

    See Also:
        variance_ratio: Variance ratio test for random walk.
        market_efficiency_ratio: Multi-lag efficiency assessment.
    """
    returns = prices.pct_change().dropna()
    abs_returns = np.abs(returns)

    # Group by time-of-day
    if freq == "h":
        groups = abs_returns.groupby(abs_returns.index.hour)
    elif freq in ("30min", "30T"):
        groups = abs_returns.groupby(
            abs_returns.index.hour * 2 + abs_returns.index.minute // 30
        )
    elif freq in ("15min", "15T"):
        groups = abs_returns.groupby(
            abs_returns.index.hour * 4 + abs_returns.index.minute // 15
        )
    else:
        # Fall back to hour
        groups = abs_returns.groupby(abs_returns.index.hour)

    pattern = groups.mean()
    pattern.name = "intraday_volatility"
    return pattern
