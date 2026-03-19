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
        wraquant.backtest.tearsheet.comprehensive_tearsheet: Full report including Sharpe.
        wraquant.stats.descriptive.rolling_sharpe: Time-varying Sharpe ratio.

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
        wraquant.backtest.tearsheet.comprehensive_tearsheet: Full report with drawdowns.
        wraquant.stats.descriptive.rolling_drawdown: Time-varying drawdown.

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


def treynor_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Treynor ratio: excess return per unit of systematic (beta) risk.

    The Treynor ratio is similar to the Sharpe ratio but uses beta
    (systematic risk) rather than total standard deviation in the
    denominator. It is the appropriate performance measure for
    well-diversified portfolios where specific risk has been
    diversified away.

    When to use:
        Use Treynor for comparing portfolios that are *part of* a larger
        diversified portfolio (so only systematic risk matters). If the
        portfolio is the investor's entire wealth, use Sharpe instead.

    Mathematical formulation:
        Treynor = (R_p - R_f) / beta_p

        where R_p is annualized portfolio return, R_f is the risk-free
        rate, and beta_p is the portfolio's beta vs the benchmark.

    Parameters:
        returns: Portfolio return series.
        benchmark: Benchmark return series (same frequency and index).
        risk_free: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        Treynor ratio as a float. Higher is better. Negative indicates
        the portfolio underperformed the risk-free rate.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> market = pd.Series(np.random.normal(0.0005, 0.01, 252))
        >>> portfolio = 1.2 * market + np.random.normal(0.0001, 0.005, 252)
        >>> tr = treynor_ratio(portfolio, market, risk_free=0.04)
        >>> isinstance(tr, float)
        True

    See Also:
        sharpe_ratio: Total risk-adjusted return.
        jensens_alpha: Excess return above CAPM prediction.

    References:
        - Treynor (1965), "How to Rate Management of Investment Funds"
    """
    excess_port = returns - risk_free / periods_per_year
    excess_bench = benchmark - risk_free / periods_per_year

    aligned = pd.concat(
        [excess_port.rename("p"), excess_bench.rename("b")], axis=1
    ).dropna()

    cov_pb = np.cov(aligned["p"].values, aligned["b"].values, ddof=1)
    beta = cov_pb[0, 1] / cov_pb[1, 1] if cov_pb[1, 1] != 0 else 0.0

    if beta == 0:
        return 0.0

    ann_return = float(aligned["p"].mean() * periods_per_year)
    return ann_return / beta


def m_squared(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """M-squared (Modigliani-Modigliani) risk-adjusted performance.

    M-squared leverages or deleverages the portfolio to match the
    benchmark's volatility, then measures the excess return at that
    risk level. The result is in return units (basis points / percent),
    making it easier to interpret than the dimensionless Sharpe ratio.

    When to use:
        Use M-squared when you want to compare two portfolios with
        different risk levels on a common scale. M-squared answers:
        "if both portfolios had the same risk as the benchmark, which
        would earn more?"

    Mathematical formulation:
        M^2 = SR_p * sigma_b + R_f

    Parameters:
        returns: Portfolio return series.
        benchmark: Benchmark return series.
        risk_free: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        M-squared as an annualized return (float). Positive means the
        portfolio outperforms the benchmark on a risk-adjusted basis.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> portfolio = pd.Series(np.random.normal(0.0006, 0.01, 252))
        >>> benchmark = pd.Series(np.random.normal(0.0004, 0.009, 252))
        >>> m2 = m_squared(portfolio, benchmark, risk_free=0.04)
        >>> isinstance(m2, float)
        True

    See Also:
        sharpe_ratio: Dimensionless risk-adjusted return.

    References:
        - Modigliani & Modigliani (1997), "Risk-Adjusted Performance"
    """
    excess_bench = benchmark - risk_free / periods_per_year

    sr_port = sharpe_ratio(returns, risk_free, periods_per_year)
    bench_vol = float(excess_bench.std() * np.sqrt(periods_per_year))

    return sr_port * bench_vol + risk_free


def jensens_alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Jensen's alpha: excess return above CAPM-predicted return.

    Jensen's alpha measures the average return of the portfolio in
    excess of what the Capital Asset Pricing Model (CAPM) predicts
    given the portfolio's beta. A positive alpha indicates that the
    manager generated return beyond what the market risk exposure
    would explain.

    Mathematical formulation:
        alpha = R_p - [R_f + beta * (R_m - R_f)]

        where R_p is the portfolio return, R_m is the benchmark return,
        and beta is the portfolio's beta to the benchmark.

    Parameters:
        returns: Portfolio return series.
        benchmark: Benchmark return series.
        risk_free: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        Annualized Jensen's alpha as a float.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> market = pd.Series(np.random.normal(0.0005, 0.01, 252))
        >>> stock = 0.0002 + 1.0 * market + np.random.normal(0, 0.005, 252)
        >>> alpha = jensens_alpha(stock, market, risk_free=0.04)
        >>> isinstance(alpha, float)
        True

    See Also:
        treynor_ratio: Risk-adjusted return using beta.
        appraisal_ratio: Alpha per unit of residual risk.

    References:
        - Jensen (1968), "The Performance of Mutual Funds in the Period
          1945-1964"
    """
    rf_per_period = risk_free / periods_per_year

    excess_port = returns - rf_per_period
    excess_bench = benchmark - rf_per_period

    aligned = pd.concat(
        [excess_port.rename("p"), excess_bench.rename("b")], axis=1
    ).dropna()

    cov_pb = np.cov(aligned["p"].values, aligned["b"].values, ddof=1)
    beta = cov_pb[0, 1] / cov_pb[1, 1] if cov_pb[1, 1] != 0 else 0.0

    # Alpha per period
    alpha_per_period = aligned["p"].mean() - beta * aligned["b"].mean()

    return float(alpha_per_period * periods_per_year)


def appraisal_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Appraisal ratio: Jensen's alpha per unit of residual risk.

    The appraisal ratio (also called the Treynor-Black appraisal ratio)
    measures the manager's alpha relative to the risk taken to achieve
    it (residual/idiosyncratic volatility). A high appraisal ratio
    means the manager generates alpha efficiently.

    Mathematical formulation:
        AR = alpha / sigma_epsilon

        where alpha is Jensen's alpha and sigma_epsilon is the standard
        deviation of the regression residuals (annualized).

    Parameters:
        returns: Portfolio return series.
        benchmark: Benchmark return series.
        risk_free: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        Appraisal ratio as a float. Higher is better.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> market = pd.Series(np.random.normal(0.0005, 0.01, 252))
        >>> stock = 0.0003 + 1.0 * market + np.random.normal(0, 0.005, 252)
        >>> ar = appraisal_ratio(stock, market, risk_free=0.04)
        >>> isinstance(ar, float)
        True

    See Also:
        jensens_alpha: The numerator of the appraisal ratio.
        information_ratio: Alpha per unit of tracking error.

    References:
        - Treynor & Black (1973), "How to Use Security Analysis to
          Improve Portfolio Selection"
    """
    rf_per_period = risk_free / periods_per_year

    excess_port = returns - rf_per_period
    excess_bench = benchmark - rf_per_period

    aligned = pd.concat(
        [excess_port.rename("p"), excess_bench.rename("b")], axis=1
    ).dropna()

    p_vals = aligned["p"].values
    b_vals = aligned["b"].values

    # OLS regression
    cov_pb = np.cov(p_vals, b_vals, ddof=1)
    beta = cov_pb[0, 1] / cov_pb[1, 1] if cov_pb[1, 1] != 0 else 0.0
    alpha_per_period = np.mean(p_vals) - beta * np.mean(b_vals)

    # Residuals
    residuals = p_vals - (alpha_per_period + beta * b_vals)
    residual_vol = float(np.std(residuals, ddof=1) * np.sqrt(periods_per_year))

    alpha_annual = float(alpha_per_period * periods_per_year)

    if residual_vol == 0:
        return 0.0
    return alpha_annual / residual_vol


def capture_ratios(
    returns: pd.Series,
    benchmark: pd.Series,
) -> dict[str, float]:
    """Up-capture and down-capture ratios.

    Capture ratios measure how much of the benchmark's up and down
    movements the portfolio captures. An ideal portfolio has high
    up-capture (>100%) and low down-capture (<100%).

    When to use:
        Use capture ratios for:
        - Evaluating defensive vs aggressive positioning: a portfolio
          with 90% up-capture and 70% down-capture is defensively
          positioned and will outperform in bear markets.
        - Manager selection: compare capture ratios across funds.
        - Style analysis: growth funds typically have high up-capture
          and high down-capture; value funds often have lower both.

    Mathematical formulation:
        Up-capture = (mean(r_p | r_b > 0) / mean(r_b | r_b > 0)) * 100
        Down-capture = (mean(r_p | r_b < 0) / mean(r_b | r_b < 0)) * 100

        Capture ratio = up-capture / down-capture

    Parameters:
        returns: Portfolio return series.
        benchmark: Benchmark return series (same frequency and index).

    Returns:
        Dictionary containing:
        - **up_capture** (*float*) -- Percentage of benchmark's up
          movements captured (>100 = amplified).
        - **down_capture** (*float*) -- Percentage of benchmark's down
          movements captured (<100 = dampened losses).
        - **capture_ratio** (*float*) -- up_capture / down_capture.
          Values > 1 indicate the portfolio adds value through
          asymmetric participation.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> benchmark = pd.Series(np.random.normal(0.0005, 0.01, 252))
        >>> portfolio = 0.8 * benchmark + np.random.normal(0.0001, 0.005, 252)
        >>> caps = capture_ratios(portfolio, benchmark)
        >>> caps["up_capture"] > 0
        True

    See Also:
        treynor_ratio: Systematic risk-adjusted return.
    """
    aligned = pd.concat([returns.rename("p"), benchmark.rename("b")], axis=1).dropna()

    up_mask = aligned["b"] > 0
    down_mask = aligned["b"] < 0

    up_bench_mean = aligned["b"][up_mask].mean()
    up_port_mean = aligned["p"][up_mask].mean()

    down_bench_mean = aligned["b"][down_mask].mean()
    down_port_mean = aligned["p"][down_mask].mean()

    up_capture = (
        float(up_port_mean / up_bench_mean * 100) if up_bench_mean != 0 else 0.0
    )
    down_capture = (
        float(down_port_mean / down_bench_mean * 100) if down_bench_mean != 0 else 0.0
    )
    cap_ratio = up_capture / down_capture if down_capture != 0 else float("inf")

    return {
        "up_capture": up_capture,
        "down_capture": down_capture,
        "capture_ratio": cap_ratio,
    }
