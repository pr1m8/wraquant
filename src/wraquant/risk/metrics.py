"""Core risk and performance metrics.

Provides the standard risk-adjusted return ratios used across portfolio
management, fund evaluation, and strategy research. Each metric
quantifies a different aspect of the risk-return trade-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio.

    The Sharpe ratio measures excess return per unit of total risk
    (standard deviation). It is the most widely cited risk-adjusted
    performance measure in finance.

    When to use:
        Use Sharpe when you want a single number summarising risk-adjusted
        performance. Compare strategies on the same asset class (Sharpe
        is less meaningful across asset classes with different return
        distributions).

    Mathematical formulation:
        SR = (mean(r - r_f) / std(r - r_f)) * sqrt(N)

        where r is the return series, r_f is the per-period risk-free
        rate, and N is periods_per_year.

    How to interpret:
        - SR < 0: strategy loses money on a risk-adjusted basis.
        - 0 < SR < 0.5: poor; barely compensating for risk.
        - 0.5 < SR < 1.0: acceptable for long-only strategies.
        - 1.0 < SR < 2.0: good; typical of well-designed quant strategies.
        - SR > 2.0: excellent; verify this is not overfitting.
        - SR > 3.0: suspicious; likely backtest artifact or very short
          sample.

    Parameters:
        returns: Simple return series (e.g., daily percentage changes
            divided by 100).
        risk_free: Annual risk-free rate (e.g., 0.05 for 5%).
        periods_per_year: Trading periods per year (252 for daily,
            12 for monthly, 52 for weekly).

    Returns:
        Annualized Sharpe ratio as a float.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> daily_returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
        >>> sr = sharpe_ratio(daily_returns, risk_free=0.04)
        >>> isinstance(sr, float)
        True

    Caveats:
        - Assumes returns are IID and normally distributed; both
          assumptions are violated in practice.
        - Penalises upside volatility equally with downside volatility;
          use ``sortino_ratio`` if you only care about downside risk.
        - Annualisation via sqrt(N) is only exact for IID returns.

    See Also:
        sortino_ratio: Uses downside deviation only.
        information_ratio: Measures alpha relative to a benchmark.

    References:
        - Sharpe (1966), "Mutual Fund Performance"
        - Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient
          Frontier"
    """
    excess = returns - risk_free / periods_per_year
    mean_excess = excess.mean()
    std = excess.std()
    if std == 0:
        return 0.0
    return float(mean_excess / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (downside risk only).

    The Sortino ratio replaces total standard deviation with *downside
    deviation* -- the standard deviation of negative excess returns
    only. This avoids penalising strategies for upside volatility,
    making it more appropriate for asymmetric return distributions
    (which most equity strategies exhibit).

    When to use:
        Prefer Sortino over Sharpe when returns are skewed or when you
        only care about downside risk. Particularly useful for options
        strategies, trend-following, and any strategy with convex payoffs.

    Mathematical formulation:
        Sortino = (mean(r - r_f) / DD) * sqrt(N)

        where DD = sqrt(mean(min(r - r_f, 0)^2)) is the downside
        deviation (computed as the second lower partial moment).

    How to interpret:
        Values follow a similar scale to Sharpe, but Sortino is
        typically higher because the denominator (downside deviation)
        is smaller than total standard deviation. A Sortino of 2.0 is
        roughly equivalent to a Sharpe of 1.5 for normally distributed
        returns.

    Parameters:
        returns: Simple return series.
        risk_free: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        Annualized Sortino ratio. Returns ``inf`` when there are no
        negative excess returns and the mean is positive, and ``0.0``
        when the mean is non-positive.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> daily_returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
        >>> sr = sortino_ratio(daily_returns, risk_free=0.04)
        >>> sr > 0
        True

    See Also:
        sharpe_ratio: Uses total standard deviation.
        max_drawdown: Peak-to-trough loss measure.

    References:
        - Sortino & van der Meer (1991), "Downside Risk"
        - Sortino & Satchell (2001), "Managing Downside Risk in
          Financial Markets"
    """
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = np.sqrt((downside**2).mean())
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
) -> float:
    """Information ratio (active return / tracking error).

    The information ratio measures the average active return (alpha)
    relative to the variability of that active return (tracking error).
    It answers: "is the manager consistently adding value, or is the
    alpha noisy and unreliable?"

    When to use:
        Use IR when evaluating a strategy *relative to a benchmark*.
        Sharpe measures absolute risk-adjusted return; IR measures
        *relative* risk-adjusted return. Most relevant for active fund
        managers benchmarked against an index.

    Mathematical formulation:
        IR = mean(r_p - r_b) / std(r_p - r_b)

        where r_p is the portfolio return and r_b is the benchmark
        return. This version is *not* annualized; multiply by
        sqrt(periods_per_year) to annualize.

    How to interpret:
        - IR < 0: underperforming the benchmark.
        - 0 < IR < 0.3: modest skill; hard to distinguish from luck.
        - 0.3 < IR < 0.5: good active management.
        - IR > 0.5: exceptional; sustained alpha generation.
        - IR > 1.0: very rare and likely indicates short sample bias.

    Parameters:
        returns: Portfolio return series.
        benchmark: Benchmark return series (same frequency and index).

    Returns:
        Information ratio (not annualized) as a float.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> portfolio = pd.Series(np.random.normal(0.0006, 0.01, 252))
        >>> benchmark = pd.Series(np.random.normal(0.0004, 0.009, 252))
        >>> ir = information_ratio(portfolio, benchmark)
        >>> isinstance(ir, float)
        True

    See Also:
        sharpe_ratio: Absolute risk-adjusted return.
        hit_ratio: Fraction of positive-return periods.

    References:
        - Grinold & Kahn (2000), "Active Portfolio Management"
    """
    active = returns - benchmark
    te = active.std()
    if te == 0:
        return 0.0
    return float(active.mean() / te)


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown from a price series.

    Maximum drawdown is the largest peak-to-trough decline in the price
    series. It measures the worst historical loss an investor would have
    experienced if they bought at the peak and sold at the trough.

    When to use:
        Use max drawdown as a "worst case" risk measure. It is more
        intuitive than VaR for communicating tail risk to non-technical
        stakeholders. Often used as a hard constraint in portfolio
        optimisation (e.g., "max drawdown must stay below 15%").

    Mathematical formulation:
        MDD = min_t ( (P_t - max_{s<=t} P_s) / max_{s<=t} P_s )

        The result is a negative number (or zero if the price series
        only goes up).

    How to interpret:
        - MDD = -0.10 means the worst peak-to-trough loss was 10%.
        - MDD = -0.50 means the strategy lost half its value at worst.
        - For S&P 500 since 1928, the worst MDD was about -0.86
          (1929-1932). Post-2000, the worst was about -0.57 (GFC).
        - A strategy with a Sharpe of 1.0 and max drawdown of -0.30
          has a Calmar ratio of about 0.33.

    Parameters:
        prices: Price or equity curve series (not returns). Must be
            positive values.

    Returns:
        Maximum drawdown as a negative float (e.g., -0.25 for a 25%
        drawdown). Returns 0.0 if the series is monotonically
        increasing.

    Example:
        >>> import pandas as pd
        >>> prices = pd.Series([100, 110, 105, 95, 108, 102])
        >>> mdd = max_drawdown(prices)
        >>> round(mdd, 4)
        -0.1364

    See Also:
        sharpe_ratio: Risk-adjusted return measure.
        sortino_ratio: Downside-only risk-adjusted return.

    References:
        - Magdon-Ismail & Atiya (2004), "Maximum Drawdown"
    """
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())


def hit_ratio(returns: pd.Series) -> float:
    """Fraction of positive return periods (win rate).

    The hit ratio measures how often the strategy produces a positive
    return. It is the simplest measure of consistency and is often used
    alongside payoff ratio (average win / average loss) to characterise
    a strategy's profile.

    When to use:
        Use hit ratio for quick strategy diagnostics. A trend-following
        strategy typically has a low hit ratio (30-45%) but large
        average wins. A mean-reversion strategy typically has a high
        hit ratio (55-70%) but smaller average wins. Neither is
        inherently better -- what matters is the product of hit ratio
        and payoff ratio.

    How to interpret:
        - 0.50 = coin flip; no directional edge.
        - 0.55 = statistically meaningful edge on daily data.
        - 0.60+ = strong edge; verify you are not overfitting.
        - < 0.40 does not mean a bad strategy if the avg win >> avg loss.

    Parameters:
        returns: Simple return series.

    Returns:
        Hit ratio as a float between 0 and 1.

    Example:
        >>> import pandas as pd
        >>> returns = pd.Series([0.01, -0.005, 0.008, -0.003, 0.012])
        >>> hit_ratio(returns)
        0.6

    See Also:
        sharpe_ratio: Risk-adjusted return measure.
        information_ratio: Alpha per unit of tracking error.
    """
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    return float((clean > 0).sum() / len(clean))
