"""Backtesting performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.risk.metrics import (
    max_drawdown as _risk_max_drawdown,
    sharpe_ratio as _risk_sharpe,
    sortino_ratio as _risk_sortino,
)


def performance_summary(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Calculate comprehensive performance metrics.

    Parameters:
        returns: Portfolio return series.
        risk_free: Annual risk-free rate.
        periods_per_year: Number of trading periods per year.

    Returns:
        Dict with performance metrics.
    """
    total_return = float((1 + returns).prod() - 1)
    n_periods = len(returns)
    ann_factor = periods_per_year / n_periods if n_periods > 0 else 1

    ann_return = float((1 + total_return) ** ann_factor - 1)
    ann_vol = float(returns.std() * np.sqrt(periods_per_year))

    sharpe = _risk_sharpe(returns, risk_free=risk_free, periods_per_year=periods_per_year)

    sortino = _risk_sortino(returns, risk_free=risk_free, periods_per_year=periods_per_year)

    # Max drawdown — risk.metrics.max_drawdown expects a price series
    cumulative = (1 + returns).cumprod()
    max_dd = _risk_max_drawdown(cumulative)

    # Calmar (no canonical implementation in risk module yet)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate
    n_positive = int((returns > 0).sum())
    win_rate = n_positive / n_periods if n_periods > 0 else 0.0

    # Profit factor
    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))
    profit_factor = gains / losses if losses > 0 else float("inf")

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_periods": n_periods,
    }


# ------------------------------------------------------------------
# Additional performance metrics
# ------------------------------------------------------------------


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
) -> float:
    """Omega ratio: probability-weighted gain/loss ratio above a threshold.

    The Omega ratio partitions the return distribution at a threshold *L*
    and computes the ratio of the expected gain above *L* to the expected
    loss below *L*.  Unlike the Sharpe ratio it uses the *entire*
    distribution (all moments, not just the first two), making it more
    appropriate for non-normal returns.

    Mathematical formulation:
        Omega(L) = E[max(R - L, 0)] / E[max(L - R, 0)]
                 = sum(max(r_i - L, 0)) / sum(max(L - r_i, 0))

    How to interpret:
        - Omega = 1.0: strategy breaks even relative to the threshold.
        - Omega > 1.0: gains outweigh losses (good).
        - Omega > 2.0: strong risk-adjusted performance.
        - Omega = inf: no returns below the threshold.
        - The higher the better; compare strategies at the same threshold.

    When to use:
        Use Omega when return distributions are skewed or fat-tailed and
        you want a single number that captures the full distribution.
        Prefer Omega over Sharpe for options-based or trend-following
        strategies whose returns are far from Gaussian.

    Parameters:
        returns: Simple return series (e.g., daily returns).
        threshold: Minimum acceptable return per period.  Default 0.

    Returns:
        Omega ratio as a float.  Returns ``inf`` if no returns fall
        below the threshold.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.01, 0.02, -0.005, 0.015, -0.01])
        >>> omega_ratio(r, threshold=0.0)  # doctest: +SKIP
        3.0

    See Also:
        kappa_ratio: Generalized lower-partial-moment ratio (Kappa(1) = Omega - 1).
        sharpe_ratio: Mean/std ratio; only uses first two moments.
    """
    excess = returns - threshold
    gains = float(excess[excess > 0].sum())
    losses = float(abs(excess[excess <= 0].sum()))
    if losses == 0:
        return float("inf")
    return gains / losses


def burke_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Burke ratio: return per unit of drawdown severity.

    The Burke ratio penalises strategies that experience many deep
    drawdowns by dividing the annualised excess return by the square
    root of the sum of squared drawdown depths.  Compared to Calmar
    (which uses only the *worst* drawdown), Burke considers the
    *entire* drawdown history.

    Mathematical formulation:
        Burke = annualized_return / sqrt(sum(d_i^2))

        where d_i are the individual drawdown depths (negative values).

    How to interpret:
        - Higher is better.
        - A strategy with many small drawdowns can have a higher Burke
          than one with a single large drawdown, even if their Calmar
          ratios are identical.
        - There is no universal "good" threshold; use for relative
          comparison between strategies.

    When to use:
        Prefer Burke over Calmar when you want to penalise strategies
        that repeatedly draw down, not just the single worst case.

    Parameters:
        returns: Simple return series.
        periods_per_year: Trading periods per year (252 for daily).

    Returns:
        Burke ratio as a float.  Returns 0.0 if there are no drawdowns.

    Example:
        >>> import pandas as pd, numpy as np
        >>> r = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        >>> burke_ratio(r)  # doctest: +SKIP
        1.23

    See Also:
        ulcer_performance_index: Uses Ulcer Index (RMS of drawdowns) in denominator.
        recovery_factor: Net profit / max drawdown.
    """
    n = len(returns)
    if n == 0:
        return 0.0
    total_return = float((1 + returns).prod() - 1)
    ann_factor = periods_per_year / n if n > 0 else 1
    ann_return = float((1 + total_return) ** ann_factor - 1)

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    dd_sq_sum = float((dd**2).sum())
    if dd_sq_sum == 0:
        return 0.0
    return ann_return / np.sqrt(dd_sq_sum)


def ulcer_performance_index(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Ulcer Performance Index (UPI): return per unit of drawdown pain.

    The UPI (also known as Martin ratio) divides the annualised excess
    return by the Ulcer Index, which is the root-mean-square of
    percentage drawdowns.  The Ulcer Index captures both the depth and
    duration of drawdowns, making UPI a comprehensive pain-adjusted
    return measure.

    Mathematical formulation:
        Ulcer Index = sqrt(mean(d_i^2))
        UPI = annualized_return / Ulcer Index

        where d_i is the drawdown at each point in time.

    How to interpret:
        - Higher is better.
        - UPI > 2.0 is generally considered very good.
        - UPI accounts for drawdown duration (not just depth), so a
          strategy that recovers quickly scores better than one that
          lingers in drawdown.

    When to use:
        Use UPI when you want a risk-adjusted measure that captures
        the investor's real experience of pain (deep, prolonged
        drawdowns hurt more than brief dips).

    Parameters:
        returns: Simple return series.
        periods_per_year: Trading periods per year.

    Returns:
        UPI as a float.  Returns 0.0 if the Ulcer Index is zero.

    Example:
        >>> import pandas as pd, numpy as np
        >>> r = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        >>> ulcer_performance_index(r)  # doctest: +SKIP
        2.5

    See Also:
        burke_ratio: Uses sum of squared drawdowns instead of RMS.
        max_drawdown: Single worst peak-to-trough decline.
    """
    n = len(returns)
    if n == 0:
        return 0.0
    total_return = float((1 + returns).prod() - 1)
    ann_factor = periods_per_year / n if n > 0 else 1
    ann_return = float((1 + total_return) ** ann_factor - 1)

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    ulcer_idx = float(np.sqrt((dd**2).mean()))
    if ulcer_idx == 0:
        return 0.0
    return ann_return / ulcer_idx


def kappa_ratio(
    returns: pd.Series,
    order: int = 2,
    threshold: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Kappa ratio: generalized Sortino using lower partial moments.

    The Kappa ratio family generalises the Sortino ratio by using the
    n-th root of the n-th lower partial moment (LPM) as the risk
    measure.  Special cases:

    - Kappa(1) is equivalent to (Omega - 1) when annualization is
      removed (first lower partial moment = expected shortfall below
      threshold).
    - Kappa(2) is the Sortino ratio (second lower partial moment =
      downside deviation).
    - Kappa(3) penalises large negative returns even more heavily.

    Mathematical formulation:
        LPM_n = mean(max(L - r_i, 0)^n)
        Kappa_n = (annualized_mean_excess) / LPM_n^(1/n)

    How to interpret:
        - Higher is better (more return per unit of downside risk).
        - Higher orders penalise tail risk more severely.
        - Compare strategies using the same order.

    When to use:
        Use Kappa(3) when you especially fear large losses.  Use
        Kappa(2) as a drop-in alternative to Sortino.  Use Kappa(1)
        when you want an Omega-style measure in ratio form.

    Parameters:
        returns: Simple return series.
        order: Moment order n (1, 2, or 3 are typical).
        threshold: Minimum acceptable return per period.
        periods_per_year: Trading periods per year.

    Returns:
        Kappa ratio as a float.  Returns 0.0 if LPM is zero.

    Example:
        >>> import pandas as pd, numpy as np
        >>> r = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        >>> kappa_ratio(r, order=2)  # doctest: +SKIP
        1.8

    See Also:
        omega_ratio: Threshold-based gain/loss ratio.
        sortino_ratio: Equivalent to Kappa(2).
    """
    n = len(returns)
    if n == 0:
        return 0.0
    mean_excess = float(returns.mean() - threshold)
    ann_mean_excess = mean_excess * periods_per_year
    shortfall = np.maximum(threshold - returns, 0.0)
    lpm = float((shortfall**order).mean())
    if lpm == 0:
        return 0.0
    return ann_mean_excess / (lpm ** (1.0 / order))


def tail_ratio(
    returns: pd.Series,
    upper_pct: float = 95.0,
    lower_pct: float = 5.0,
) -> float:
    """Tail ratio: magnitude of right tail vs. left tail.

    The tail ratio measures how large positive outlier returns are
    relative to negative outlier returns.  A value greater than 1 means
    the strategy's best days are proportionally larger than its worst
    days.

    Mathematical formulation:
        Tail ratio = |percentile(R, upper_pct)| / |percentile(R, lower_pct)|

    How to interpret:
        - Tail ratio > 1.0: right tail is fatter (upside surprises
          bigger than downside).  Desirable.
        - Tail ratio = 1.0: symmetric tails.
        - Tail ratio < 1.0: left tail is fatter (downside risk
          dominates).

    When to use:
        Quick diagnostic to check if a strategy has favorable tail
        behaviour.  Combine with ``common_sense_ratio`` for a sanity
        check.

    Parameters:
        returns: Simple return series.
        upper_pct: Upper percentile (default 95).
        lower_pct: Lower percentile (default 5).

    Returns:
        Tail ratio as a float.  Returns ``inf`` if the lower
        percentile is zero.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.03, 0.01, -0.01, 0.02, -0.005])
        >>> tail_ratio(r)  # doctest: +SKIP
        2.0

    See Also:
        common_sense_ratio: Tail ratio multiplied by (1 + Sharpe).
        rachev_ratio: CVaR-based tail comparison.
    """
    upper = float(np.abs(np.percentile(returns, upper_pct)))
    lower = float(np.abs(np.percentile(returns, lower_pct)))
    if lower == 0:
        return float("inf")
    return upper / lower


def common_sense_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Common Sense Ratio: quick sanity check combining Sharpe and tails.

    The Common Sense Ratio multiplies the tail ratio by (1 + Sharpe).
    It combines a measure of tail asymmetry with overall risk-adjusted
    performance into a single number for rapid strategy screening.

    Mathematical formulation:
        CSR = tail_ratio * (1 + Sharpe)

    How to interpret:
        - CSR > 1.0: strategy has both favorable tails and positive
          risk-adjusted return.  Good.
        - CSR < 1.0: either tails are unfavorable or risk-adjusted
          return is poor.
        - Use for fast initial screening; not a substitute for deeper
          analysis.

    When to use:
        Use CSR as a first-pass filter when evaluating many strategies
        simultaneously.

    Parameters:
        returns: Simple return series.
        risk_free: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        Common Sense Ratio as a float.

    Example:
        >>> import pandas as pd, numpy as np
        >>> r = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        >>> common_sense_ratio(r)  # doctest: +SKIP
        1.5

    See Also:
        tail_ratio: Upside vs downside tail magnitude.
        sharpe_ratio: Mean/std risk-adjusted return.
    """
    tr = tail_ratio(returns)
    sr = _risk_sharpe(returns, risk_free=risk_free, periods_per_year=periods_per_year)
    return tr * (1.0 + sr)


def rachev_ratio(
    returns: pd.Series,
    alpha: float = 0.05,
) -> float:
    """Rachev ratio: CVaR of gains over CVaR of losses.

    The Rachev ratio (also called the Conditional Tail Ratio) compares
    the expected size of extreme gains to the expected size of extreme
    losses, using Conditional Value at Risk (CVaR, a.k.a. Expected
    Shortfall).  Unlike the simple tail ratio, Rachev uses the *mean*
    of the tail rather than a single percentile, making it more robust
    to outliers.

    Mathematical formulation:
        CVaR_alpha(gains) = E[R | R > VaR_{1-alpha}(R)]
        CVaR_alpha(losses) = E[-R | R < VaR_alpha(R)]
        Rachev = CVaR_alpha(gains) / CVaR_alpha(losses)

    How to interpret:
        - Rachev > 1.0: expected extreme gains exceed expected extreme
          losses.  The strategy has favorable fat-tail asymmetry.
        - Rachev = 1.0: symmetric tail risk.
        - Rachev < 1.0: tail risk is skewed to the downside.

    When to use:
        Use Rachev when you need a fat-tail-aware gain/loss comparison.
        More robust than ``tail_ratio`` because it averages over the
        tail rather than using a single percentile.

    Parameters:
        returns: Simple return series.
        alpha: Tail probability (default 0.05 = 5 % tails).

    Returns:
        Rachev ratio as a float.  Returns ``inf`` if the loss CVaR is
        zero.

    Example:
        >>> import pandas as pd, numpy as np
        >>> r = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 500))
        >>> rachev_ratio(r, alpha=0.05)  # doctest: +SKIP
        1.1

    See Also:
        tail_ratio: Percentile-based tail comparison.
        omega_ratio: Full-distribution gain/loss ratio.
    """
    sorted_r = np.sort(returns.dropna().values)
    n = len(sorted_r)
    if n == 0:
        return 0.0
    k = max(int(n * alpha), 1)

    # CVaR of losses (left tail): mean of worst k returns, take abs
    cvar_loss = float(np.abs(sorted_r[:k].mean()))

    # CVaR of gains (right tail): mean of best k returns
    cvar_gain = float(sorted_r[-k:].mean())

    if cvar_loss == 0:
        return float("inf")
    return cvar_gain / cvar_loss


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """Gain to Pain ratio: total gains over total absolute losses.

    The Gain to Pain ratio (GPR) divides the sum of all returns by the
    absolute sum of all negative returns.  It provides a simple measure
    of how much total return the strategy generates per unit of pain
    (aggregate losses) experienced.

    Mathematical formulation:
        GPR = sum(r_i) / |sum(r_i where r_i < 0)|

    How to interpret:
        - GPR > 1.0: total gains exceed total losses; strategy is net
          profitable and then some.
        - GPR = 0.5: for every dollar lost, the strategy earned 50
          cents net.
        - GPR < 0: strategy loses money overall.
        - GPR > 1.5 is generally considered good for daily returns.

    When to use:
        Quick profitability diagnostic.  Simpler and more intuitive
        than profit factor (which excludes zero returns and looks at
        gross gains vs. gross losses).

    Parameters:
        returns: Simple return series.

    Returns:
        Gain to Pain ratio as a float.  Returns ``inf`` if there are
        no negative returns.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.02, -0.01, 0.015, -0.005, 0.01])
        >>> gain_to_pain_ratio(r)  # doctest: +SKIP
        2.67

    See Also:
        profit_factor: Gross gains / gross losses (ignores net).
        omega_ratio: Threshold-based gain/loss comparison.
    """
    total = float(returns.sum())
    pain = float(abs(returns[returns < 0].sum()))
    if pain == 0:
        return float("inf")
    return total / pain


def risk_of_ruin(
    win_rate: float,
    payoff_ratio: float,
    ruin_pct: float = 0.5,
    n_trades: int = 1000,
) -> float:
    """Probability of losing a given fraction of capital.

    Estimates the risk of ruin -- the probability that a strategy will
    lose *ruin_pct* of its capital -- given its win rate and average
    payoff ratio, assuming fixed fractional position sizing and
    independent trades.

    Mathematical formulation (simplified):
        RoR = ((1 - edge) / (1 + edge)) ^ capital_units

        where edge = (win_rate * payoff_ratio) - (1 - win_rate) and
        capital_units approximates the number of bet-units to exhaust
        before reaching ruin_pct.

    How to interpret:
        - RoR close to 0: very unlikely to hit ruin level.
        - RoR > 0.05 (5 %): meaningful risk; reduce position size.
        - RoR > 0.20 (20 %): dangerous; the strategy may not survive.
        - A strategy with high win_rate and high payoff_ratio has very
          low risk of ruin.

    When to use:
        Use risk of ruin to decide whether a strategy is survivable
        over *n_trades*.  Combine with Kelly fraction to determine
        appropriate sizing.

    Parameters:
        win_rate: Probability of a winning trade (0 to 1).
        payoff_ratio: Average win / average loss (positive number).
        ruin_pct: Fraction of capital that constitutes ruin (e.g.,
            0.5 = losing 50 % of capital).
        n_trades: Number of trades to consider in the simulation
            window (affects capital_units approximation).

    Returns:
        Estimated probability of ruin (0 to 1).

    Example:
        >>> risk_of_ruin(win_rate=0.55, payoff_ratio=1.5, ruin_pct=0.5)  # doctest: +SKIP
        0.002

    See Also:
        kelly_fraction: Optimal position sizing.
        expectancy: Expected value per trade.
    """
    if not 0 < win_rate < 1:
        raise ValueError("win_rate must be between 0 and 1 (exclusive)")
    if payoff_ratio <= 0:
        raise ValueError("payoff_ratio must be positive")

    edge = win_rate * payoff_ratio - (1.0 - win_rate)
    if edge <= 0:
        return 1.0  # negative edge -> ruin is certain

    # Number of "capital units" approximated from ruin_pct
    capital_units = max(int(-np.log(ruin_pct) / np.log(1.0 + edge) * 10), 1)
    ratio = (1.0 - edge) / (1.0 + edge)
    if ratio >= 1.0:
        return 1.0
    ror = ratio**capital_units
    return float(np.clip(ror, 0.0, 1.0))


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Kelly fraction: optimal bet sizing for geometric growth.

    The Kelly criterion determines the fraction of capital to risk on
    each trade to maximise the long-run geometric growth rate.  The
    full Kelly is aggressive; practitioners typically use fractional
    Kelly (e.g., half-Kelly) to reduce variance.

    Mathematical formulation:
        b = avg_win / avg_loss   (odds ratio)
        f* = (b * p - q) / b

        where p = win_rate, q = 1 - p.

    How to interpret:
        - f* > 0: strategy has positive edge; bet this fraction.
        - f* = 0: no edge; do not bet.
        - f* < 0: negative edge (clamped to 0); avoid this strategy.
        - f* > 0.25: full Kelly is very aggressive; consider using
          half or quarter Kelly.

    When to use:
        Use Kelly to determine the theoretical maximum position size.
        In practice, use fractional Kelly (0.25x to 0.5x) because
        Kelly assumes known and constant edge, which is unrealistic.

    Parameters:
        win_rate: Probability of a winning trade (0 to 1).
        avg_win: Average winning trade magnitude (positive).
        avg_loss: Average losing trade magnitude (positive, i.e.,
            the absolute value of the average loss).

    Returns:
        Optimal fraction of capital to risk, clamped to [0, 1].

    Example:
        >>> kelly_fraction(win_rate=0.55, avg_win=1.5, avg_loss=1.0)
        0.25

    See Also:
        risk_of_ruin: Probability of catastrophic drawdown.
        expectancy: Expected value per trade.
    """
    if not 0 <= win_rate <= 1:
        raise ValueError("win_rate must be between 0 and 1")
    if avg_loss <= 0:
        raise ValueError("avg_loss must be positive")
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    if b == 0:
        return 0.0
    f = (b * win_rate - q) / b
    return float(np.clip(f, 0.0, 1.0))


def expectancy(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Expectancy: expected profit per trade.

    Expectancy combines win rate and payoff to give the average
    expected value of each trade.  A strategy with positive expectancy
    is profitable in the long run (assuming sufficient trades and
    stable edge).

    Mathematical formulation:
        E = (win_rate * avg_win) - ((1 - win_rate) * |avg_loss|)

    How to interpret:
        - E > 0: each trade is expected to be profitable on average.
        - E = 0: break-even strategy.
        - E < 0: losing strategy.
        - E > 0.10 (if avg_loss is normalised to 1.0): reasonable edge.

    When to use:
        Use expectancy alongside system quality number (SQN) and
        profit factor to evaluate a trading system.  Expectancy alone
        does not account for variability of outcomes.

    Parameters:
        win_rate: Probability of a winning trade (0 to 1).
        avg_win: Average winning trade magnitude (positive).
        avg_loss: Average losing trade magnitude (positive, absolute
            value of the average loss).

    Returns:
        Expected value per trade.

    Example:
        >>> expectancy(win_rate=0.6, avg_win=100.0, avg_loss=80.0)
        28.0

    See Also:
        system_quality_number: Expectancy normalised by trade variability.
        kelly_fraction: Optimal sizing given expectancy.
    """
    return win_rate * avg_win - (1.0 - win_rate) * abs(avg_loss)


def profit_factor(returns: pd.Series) -> float:
    """Profit factor: gross profit divided by gross loss.

    The profit factor measures how many dollars the strategy earns for
    every dollar it loses.  It is the simplest measure of a strategy's
    profitability.

    Mathematical formulation:
        PF = sum(r_i where r_i > 0) / |sum(r_i where r_i < 0)|

    How to interpret:
        - PF > 1.0: strategy is profitable.
        - PF = 1.0: break even.
        - PF < 1.0: strategy loses money.
        - PF > 1.5: good.
        - PF > 2.0: very good (verify not overfitting).

    When to use:
        Use as a quick profitability check.  Pair with win rate and
        payoff ratio for a complete picture.

    Parameters:
        returns: Return or P&L series.

    Returns:
        Profit factor as a float.  Returns ``inf`` if there are no
        losses.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.02, -0.01, 0.03, -0.005])
        >>> profit_factor(r)
        ... # doctest: +SKIP
        3.33

    See Also:
        gain_to_pain_ratio: Net return / total losses.
        expectancy: Expected value per trade.
    """
    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))
    if losses == 0:
        return float("inf")
    return gains / losses


def payoff_ratio(
    returns: pd.Series,
) -> float:
    """Payoff ratio: average win divided by average loss.

    The payoff ratio (also called reward-to-risk ratio) measures how
    large the average winning trade is relative to the average losing
    trade.  Combined with win rate, it fully characterises a strategy's
    return profile.

    Mathematical formulation:
        Payoff = mean(r_i where r_i > 0) / |mean(r_i where r_i < 0)|

    How to interpret:
        - Payoff > 1.0: average wins are larger than average losses.
          Common in trend-following strategies.
        - Payoff = 1.0: wins and losses are the same size.
        - Payoff < 1.0: average losses exceed average wins.  The
          strategy must have a high win rate to compensate.
        - Payoff > 2.0 with win rate > 0.40 is a strong system.

    When to use:
        Use alongside win rate.  A low win rate with high payoff
        (trend-following) is as viable as high win rate with low
        payoff (mean-reversion).

    Parameters:
        returns: Return or P&L series.

    Returns:
        Payoff ratio as a float.  Returns ``inf`` if there are no
        losses, and 0.0 if there are no wins.

    Example:
        >>> import pandas as pd
        >>> r = pd.Series([0.02, -0.01, 0.03, -0.005])
        >>> payoff_ratio(r)
        ... # doctest: +SKIP
        3.33

    See Also:
        profit_factor: Gross gains / gross losses.
        expectancy: Combines win rate and payoff into expected value.
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) == 0:
        return 0.0
    if len(losses) == 0:
        return float("inf")
    return float(wins.mean()) / float(abs(losses.mean()))


def recovery_factor(
    returns: pd.Series,
) -> float:
    """Recovery factor: net profit relative to max drawdown.

    Recovery factor measures how many times over the strategy has
    recovered from its worst drawdown.  A high recovery factor
    indicates that the strategy generates returns efficiently relative
    to the drawdown pain it inflicts.

    Mathematical formulation:
        RF = total_return / |max_drawdown|

    How to interpret:
        - RF > 1.0: strategy has earned back more than its worst
          drawdown.
        - RF > 3.0: strong.
        - RF > 5.0: excellent recovery relative to risk.
        - RF < 1.0: strategy has not yet recovered from its worst
          drawdown.

    When to use:
        Use recovery factor when you want to know if the strategy's
        returns justify the drawdown pain.  Useful for comparing
        strategies with different drawdown profiles.

    Parameters:
        returns: Simple return series.

    Returns:
        Recovery factor as a float.  Returns ``inf`` if there is no
        drawdown.

    Example:
        >>> import pandas as pd, numpy as np
        >>> r = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        >>> recovery_factor(r)  # doctest: +SKIP
        3.5

    See Also:
        burke_ratio: Return per sum-of-squared-drawdowns.
        ulcer_performance_index: Return per Ulcer Index.
    """
    total_ret = float((1 + returns).prod() - 1)
    cum = (1 + returns).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    if max_dd == 0:
        return float("inf")
    return total_ret / abs(max_dd)


def system_quality_number(
    pnl: pd.Series,
) -> float:
    """System Quality Number (SQN): Van Tharp's strategy quality metric.

    SQN normalises expectancy by the variability of trade outcomes and
    scales by the square root of the number of trades.  It answers the
    question: "given the consistency and edge of this system, is the
    positive expectancy statistically significant?"

    Mathematical formulation:
        SQN = sqrt(N) * mean(pnl) / std(pnl)

        where N is the number of trades (or periods).

    How to interpret:
        - SQN < 1.6: poor; difficult to trade profitably.
        - 1.6 < SQN < 2.0: below average; marginal edge.
        - 2.0 < SQN < 2.5: average; tradeable with discipline.
        - 2.5 < SQN < 3.0: good; reliable system.
        - 3.0 < SQN < 5.0: excellent.
        - 5.0 < SQN < 7.0: superb.
        - SQN > 7.0: holy grail (verify not overfitting).

    When to use:
        Use SQN to evaluate whether a system's edge is statistically
        meaningful given the number of observations.  Particularly
        useful when comparing systems with different numbers of trades.

    Parameters:
        pnl: Per-trade or per-period P&L series.

    Returns:
        SQN as a float.  Returns 0.0 if the standard deviation is
        zero.

    Example:
        >>> import pandas as pd
        >>> pnl = pd.Series([100, -50, 80, -30, 120, -40, 90, -20])
        >>> system_quality_number(pnl)  # doctest: +SKIP
        1.7

    See Also:
        expectancy: Mean expected value per trade.
        kelly_fraction: Optimal position sizing from edge.
    """
    pnl = pnl.dropna()
    n = len(pnl)
    if n == 0:
        return 0.0
    std = float(pnl.std())
    if std == 0:
        return 0.0
    return float(np.sqrt(n) * pnl.mean() / std)
