"""Comprehensive stress testing for portfolio risk analysis.

Stress testing answers "what if?" questions that historical VaR and CVaR
cannot address. While VaR extrapolates from the empirical distribution,
stress tests evaluate specific adverse scenarios -- including scenarios
that have never occurred in the sample.

This module provides seven complementary stress testing approaches:

1. **Scenario-based** (``stress_test_returns``) -- apply user-defined
   additive shocks (e.g., "what if every day's return is 10% worse?").
2. **Historical replay** (``historical_stress_test``) -- measure
   portfolio performance during known crises (GFC, COVID, dot-com).
3. **Volatility scaling** (``vol_stress_test``) -- scale return
   dispersion by multipliers (1.5x, 2x, 3x) while preserving the mean.
4. **Spot stress** (``spot_stress_test``) -- shift price levels by
   percentage amounts (-30% to +10%).
5. **Sensitivity ladder** (``sensitivity_ladder``) -- P&L sensitivity
   to a single factor across a range of shock levels.
6. **Reverse stress test** (``reverse_stress_test``) -- find the
   scenarios that produce a target loss (regulatory requirement).
7. **Joint stress test** (``joint_stress_test``) -- simultaneous
   volatility, spot, and correlation shocks.
8. **Marginal contribution** (``marginal_stress_contribution``) --
   identify the asset contributing most to stress loss.

How to interpret stress test results:
    Stress tests produce point estimates, not probability distributions.
    The output tells you "if X happens, the P&L impact is Y."  They are
    valuable for:
    - Setting position limits and stop-losses.
    - Capital adequacy assessment (CCAR, DFAST).
    - Identifying concentrated risk exposures.
    - Communicating tail risk to stakeholders.

References:
    - Berkowitz (2000), "A Coherent Framework for Stress-Testing"
    - McNeil, Frey & Embrechts (2005), "Quantitative Risk Management"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def stress_test_returns(
    returns: pd.Series | pd.DataFrame,
    scenarios: dict[str, float],
) -> dict[str, Any]:
    """Apply user-defined additive shock scenarios to a return series.

    Each scenario name maps to an additive shift applied uniformly to
    all returns. The function computes the stressed mean, stressed VaR
    (5th percentile), and stressed CVaR for every scenario. This is
    the simplest stress test and is useful for quick what-if analysis.

    When to use:
        Use when you want to evaluate the impact of a uniform adverse
        shift in returns. For example, "what if every daily return is
        10bp worse due to a funding cost shock?" For more realistic
        scenario analysis, use ``historical_stress_test`` (replays real
        crises) or ``joint_stress_test`` (simultaneous multi-factor
        shocks).

    How to interpret:
        Compare ``stressed_var_95`` and ``stressed_cvar_95`` across
        scenarios to identify which shock level pushes your portfolio
        into unacceptable loss territory. If a moderate shock (-5%)
        already produces a severe stressed CVaR, the portfolio has
        insufficient risk budget.

    Parameters:
        returns: Historical return series (Series) or multi-asset returns
            (DataFrame). For DataFrames, the cross-asset mean is used.
        scenarios: Mapping of scenario name to additive return shock
            (e.g. ``{"crash": -0.10, "boom": 0.05}``). A shock of
            -0.10 subtracts 10% from every observation.

    Returns:
        Dict with keys:

        * ``"scenario_results"`` -- dict mapping scenario name to a dict
          with ``"stressed_mean"``, ``"stressed_var_95"`` (5th percentile
          of stressed returns), ``"stressed_cvar_95"`` (mean of returns
          below the 5th percentile).
        * ``"base_mean"`` -- mean of the original (unstressed) returns.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
        >>> result = stress_test_returns(returns, {"mild": -0.005, "severe": -0.02})
        >>> result["scenario_results"]["severe"]["stressed_mean"] < result["base_mean"]
        True

    See Also:
        historical_stress_test: Replay known crisis periods.
        vol_stress_test: Scale volatility by multipliers.
        joint_stress_test: Simultaneous multi-factor shocks.
    """
    if isinstance(returns, pd.DataFrame):
        base_vals = returns.mean(axis=1).dropna().values
    else:
        base_vals = returns.dropna().values

    base_mean = float(np.mean(base_vals))
    results: dict[str, dict[str, float]] = {}

    for name, shock in scenarios.items():
        stressed = base_vals + shock
        var_95 = float(np.percentile(stressed, 5))
        tail = stressed[stressed <= var_95]
        cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95
        results[name] = {
            "stressed_mean": float(np.mean(stressed)),
            "stressed_var_95": var_95,
            "stressed_cvar_95": cvar_95,
        }

    return {
        "scenario_results": results,
        "base_mean": base_mean,
    }


# Pre-defined crisis date ranges (start, end) as ISO-format strings.
_CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "gfc_2008": ("2008-09-01", "2009-03-31"),
    "covid_2020": ("2020-02-19", "2020-03-23"),
    "dot_com_2000": ("2000-03-10", "2002-10-09"),
    "euro_debt_2011": ("2011-07-01", "2011-11-30"),
    "taper_tantrum_2013": ("2013-05-22", "2013-09-05"),
    "vol_mageddon_2018": ("2018-02-02", "2018-02-08"),
    "flash_crash_2010": ("2010-05-06", "2010-05-06"),
}


def historical_stress_test(
    returns: pd.Series | pd.DataFrame,
    crisis_periods: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Test portfolio returns against known historical crisis periods.

    Replays the portfolio through actual historical crises and reports
    cumulative return, max drawdown, and mean daily return for each
    period. This is the most intuitive form of stress testing because
    the scenarios are real events that stakeholders can relate to.

    When to use:
        Use historical stress testing for:
        - Board and regulator presentations ("how would we have
          performed in the GFC?").
        - Identifying whether the portfolio's risk profile has
          improved or deteriorated relative to past crises.
        - Calibrating position limits against known worst cases.

    How to interpret:
        Compare ``cumulative_return`` and ``max_drawdown`` across
        crises. A portfolio that survived the GFC with only -15%
        cumulative return has very different tail risk from one that
        lost -45%. The ``periods_found`` list tells you which crises
        overlap with your data -- crises not found are silently skipped.

    Built-in crisis periods (used when *crisis_periods* is None):
        - GFC 2008: 2008-09-01 to 2009-03-31
        - COVID 2020: 2020-02-19 to 2020-03-23
        - Dot-Com 2000: 2000-03-10 to 2002-10-09
        - Euro Debt 2011: 2011-07-01 to 2011-11-30
        - Taper Tantrum 2013: 2013-05-22 to 2013-09-05
        - Volmageddon 2018: 2018-02-02 to 2018-02-08
        - Flash Crash 2010: 2010-05-06

    Parameters:
        returns: Return series with a ``DatetimeIndex``.
        crisis_periods: Mapping of crisis name to ``(start, end)`` date
            strings. Periods not covered by the data are skipped.

    Returns:
        Dict with keys:

        * ``"crisis_results"`` -- dict mapping crisis name to a dict
          with ``"cumulative_return"`` (compounded return over the
          crisis), ``"max_drawdown"`` (worst peak-to-trough within
          the crisis), ``"mean_daily_return"``, ``"n_days"``.
        * ``"periods_found"`` -- list of crisis names that overlap
          with the data.

    Example:
        >>> import pandas as pd, numpy as np
        >>> idx = pd.bdate_range("2008-01-01", "2009-12-31")
        >>> returns = pd.Series(np.random.normal(-0.001, 0.02, len(idx)), index=idx)
        >>> result = historical_stress_test(returns)
        >>> "gfc_2008" in result["periods_found"]
        True

    See Also:
        stress_test_returns: User-defined additive shocks.
        reverse_stress_test: Find scenarios that produce a target loss.
    """
    if crisis_periods is None:
        crisis_periods = _CRISIS_PERIODS

    if isinstance(returns, pd.DataFrame):
        ret = returns.mean(axis=1)
    else:
        ret = returns

    crisis_results: dict[str, dict[str, float]] = {}
    periods_found: list[str] = []

    for name, (start, end) in crisis_periods.items():
        mask = (ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(end))
        subset = ret.loc[mask]
        if len(subset) == 0:
            continue

        periods_found.append(name)
        from wraquant.risk.metrics import max_drawdown as _max_drawdown

        cum_return = float(np.prod(1 + subset.values) - 1)
        cum_prices = pd.Series(np.cumprod(1 + subset.values))
        max_dd = _max_drawdown(cum_prices)

        crisis_results[name] = {
            "cumulative_return": cum_return,
            "max_drawdown": max_dd,
            "mean_daily_return": float(np.mean(subset.values)),
            "n_days": int(len(subset)),
        }

    return {
        "crisis_results": crisis_results,
        "periods_found": periods_found,
    }


def vol_stress_test(
    returns: pd.Series | pd.DataFrame,
    vol_shocks: list[float] | None = None,
) -> dict[str, Any]:
    """Stress test by scaling return volatility with multipliers.

    Demeaned returns are scaled by each multiplier, then the mean is
    re-added. This preserves the expected return while increasing (or
    decreasing) dispersion. The technique is useful for asking "what
    happens to VaR and CVaR if volatility doubles?"

    When to use:
        Use volatility stress tests to:
        - Assess margin adequacy under elevated vol regimes.
        - Calibrate dynamic position sizing rules.
        - Compare the portfolio's sensitivity to vol scaling
          (a diversified portfolio should be less sensitive than a
          concentrated one).

    How to interpret:
        The ``stressed_vol`` should scale linearly with the multiplier
        (by construction). The key outputs are ``stressed_var_95`` and
        ``stressed_cvar_95``: if doubling vol (multiplier 2.0) causes
        CVaR to more than double, the portfolio has convex (nonlinear)
        tail exposure.

    Parameters:
        returns: Historical return series.
        vol_shocks: List of volatility multipliers (e.g. ``[1.5, 2.0, 3.0]``).
            Defaults to ``[1.5, 2.0, 2.5, 3.0]``.

    Returns:
        Dict with keys:

        * ``"vol_results"`` -- dict mapping multiplier (as string) to
          ``"stressed_vol"``, ``"stressed_var_95"``,
          ``"stressed_cvar_95"``, ``"stressed_mean"``.
        * ``"base_vol"`` -- volatility of the original returns.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0.0005, 0.01, 500))
        >>> result = vol_stress_test(returns, vol_shocks=[1.5, 2.0])
        >>> result["vol_results"]["2.0"]["stressed_vol"] > result["base_vol"]
        True

    See Also:
        stress_test_returns: Additive shock scenarios.
        joint_stress_test: Combined vol, spot, and correlation shocks.
    """
    if vol_shocks is None:
        vol_shocks = [1.5, 2.0, 2.5, 3.0]

    if isinstance(returns, pd.DataFrame):
        vals = returns.mean(axis=1).dropna().values
    else:
        vals = returns.dropna().values

    mu = np.mean(vals)
    base_vol = float(np.std(vals, ddof=1))
    demeaned = vals - mu

    vol_results: dict[str, dict[str, float]] = {}
    for mult in vol_shocks:
        stressed = demeaned * mult + mu
        var_95 = float(np.percentile(stressed, 5))
        tail = stressed[stressed <= var_95]
        cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

        vol_results[str(mult)] = {
            "stressed_vol": float(np.std(stressed, ddof=1)),
            "stressed_var_95": var_95,
            "stressed_cvar_95": cvar_95,
            "stressed_mean": float(np.mean(stressed)),
        }

    return {
        "vol_results": vol_results,
        "base_vol": base_vol,
    }


def spot_stress_test(
    prices: pd.Series | pd.DataFrame,
    spot_shocks: list[float] | None = None,
) -> dict[str, Any]:
    """Shift spot (price) levels by specified percentage amounts.

    Each shock is applied as a multiplicative factor to the final price
    (e.g., -0.10 means a 10% drop from the last price). This is useful
    for mark-to-market stress testing of current positions.

    When to use:
        Use spot stress tests for:
        - Options portfolio Greeks analysis (delta P&L under spot move).
        - Margin calculation under adverse spot scenarios.
        - Reporting to counterparties ("what is my exposure if the
          underlying drops 20%?").

    How to interpret:
        The ``shocked_price`` shows the resulting price level after the
        shock. For a DataFrame (multi-asset), the same percentage shock
        is applied to each asset's last price. ``price_change`` is the
        absolute dollar change.

    Parameters:
        prices: Price series or DataFrame of asset prices.
        spot_shocks: List of percentage shocks (e.g. ``[-0.30, -0.20,
            -0.10, 0.10, 0.20]``). Defaults to
            ``[-0.30, -0.20, -0.10, -0.05, 0.05, 0.10]``.

    Returns:
        Dict with keys:

        * ``"spot_results"`` -- dict mapping shock (as string) to
          ``"shocked_price"``, ``"price_change"``, ``"pct_change"``.
        * ``"base_price"`` -- the last observed price.

    Example:
        >>> import pandas as pd
        >>> prices = pd.Series([100.0, 102.0, 101.0, 103.0])
        >>> result = spot_stress_test(prices, spot_shocks=[-0.10, 0.10])
        >>> result["spot_results"]["-0.1"]["shocked_price"]
        92.7

    See Also:
        vol_stress_test: Scale volatility by multipliers.
        sensitivity_ladder: P&L sensitivity to a single factor.
    """
    if spot_shocks is None:
        spot_shocks = [-0.30, -0.20, -0.10, -0.05, 0.05, 0.10]

    if isinstance(prices, pd.DataFrame):
        last_prices = prices.iloc[-1]
        spot_results: dict[str, Any] = {}
        for shock in spot_shocks:
            shocked = last_prices * (1 + shock)
            spot_results[str(shock)] = {
                "shocked_price": shocked.to_dict(),
                "price_change": (shocked - last_prices).to_dict(),
                "pct_change": shock,
            }
        return {
            "spot_results": spot_results,
            "base_price": last_prices.to_dict(),
        }

    last_price = float(prices.iloc[-1])
    spot_results = {}
    for shock in spot_shocks:
        shocked_price = last_price * (1 + shock)
        spot_results[str(shock)] = {
            "shocked_price": shocked_price,
            "price_change": shocked_price - last_price,
            "pct_change": shock,
        }

    return {
        "spot_results": spot_results,
        "base_price": last_price,
    }


def sensitivity_ladder(
    portfolio_returns: pd.Series,
    factor_returns: pd.Series,
    shock_range: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """Compute portfolio P&L across a range of factor shocks.

    Fits a linear regression of portfolio returns on a single factor,
    then uses the estimated beta to project the portfolio P&L impact
    at each shock level. The result is a "ladder" -- a table of factor
    values and corresponding portfolio returns.

    When to use:
        Use sensitivity ladders to:
        - Understand how exposed the portfolio is to a single risk
          factor (e.g., S&P 500, 10Y yield, oil price).
        - Construct hedging ratios (the beta tells you how much
          factor exposure to neutralise).
        - Present risk to traders and PMs in an intuitive format.

    Mathematical formulation:
        Step 1: Fit r_p = alpha + beta * r_f + epsilon via OLS.
        Step 2: For each shock s, estimate P&L = alpha + beta * s.

    How to interpret:
        The ``ladder`` maps each factor shock to the estimated portfolio
        return. A high ``beta`` means the portfolio is very sensitive
        to the factor. ``r_squared`` tells you how much of the
        portfolio's variance is explained by this factor; if R^2 is
        low (<0.3), the ladder is unreliable because other factors
        dominate.

    Parameters:
        portfolio_returns: Portfolio return series.
        factor_returns: Single-factor return series (same index).
        shock_range: Array of factor shock values (e.g.
            ``np.linspace(-0.10, 0.10, 21)``). Defaults to
            ``np.linspace(-0.10, 0.10, 21)``.

    Returns:
        Dict with keys:

        * ``"ladder"`` -- dict mapping shock level (float) to estimated
          portfolio P&L.
        * ``"beta"`` -- regression beta (sensitivity).
        * ``"alpha"`` -- regression intercept (return when factor = 0).
        * ``"r_squared"`` -- R-squared of the regression.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> factor = pd.Series(np.random.normal(0, 0.01, 252))
        >>> portfolio = 0.8 * factor + np.random.normal(0, 0.005, 252)
        >>> result = sensitivity_ladder(portfolio, factor)
        >>> abs(result["beta"] - 0.8) < 0.2  # beta close to true value
        True

    See Also:
        spot_stress_test: Direct price-level shocks.
        joint_stress_test: Multi-factor simultaneous shocks.
    """
    if shock_range is None:
        shock_range = np.linspace(-0.10, 0.10, 21)
    shock_range = np.asarray(shock_range)

    from wraquant.stats.regression import ols as _ols

    # Align and clean
    aligned = pd.concat(
        [portfolio_returns.rename("port"), factor_returns.rename("factor")],
        axis=1,
    ).dropna()
    y = aligned["port"].values
    x = aligned["factor"].values

    # OLS regression via shared module
    ols_result = _ols(y, x, add_constant=True)
    intercept = float(ols_result["coefficients"][0])
    slope = float(ols_result["coefficients"][1])
    r_value_sq = ols_result["r_squared"]

    ladder: dict[float, float] = {}
    for shock in shock_range:
        ladder[float(shock)] = float(intercept + slope * shock)

    return {
        "ladder": ladder,
        "beta": float(slope),
        "alpha": float(intercept),
        "r_squared": float(r_value_sq),
    }


def reverse_stress_test(
    returns: pd.Series | pd.DataFrame,
    target_loss: float,
    n_sims: int = 10000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Find scenarios that produce at least the specified target loss.

    Reverse stress testing inverts the usual question: instead of
    "what is the loss under scenario X?", it asks "what scenarios
    produce a loss of at least Y?" This is a regulatory requirement
    under ICAAP/SREP and is valuable for identifying the portfolio's
    breaking point.

    When to use:
        Use reverse stress tests when you need to:
        - Identify the conditions under which the portfolio breaches
          a risk limit (e.g., -20% annual loss).
        - Satisfy regulatory requirements for reverse stress testing.
        - Understand how "unlikely" a catastrophic loss really is.

    How to interpret:
        ``probability`` is the fraction of simulated paths that hit
        the target loss. A probability of 0.01 means a 1% chance of
        the target loss under the fitted normal model. ``avg_loss``
        and ``worst_loss`` characterise the severity of qualifying
        scenarios. ``threshold_percentile`` places the target loss
        in the simulated distribution (e.g., 2nd percentile means
        the target is a 1-in-50 event).

    Caveats:
        The simulation assumes normally distributed returns (fitted
        from the historical sample). For fat-tailed assets, the true
        probability of extreme losses is higher than estimated here.
        Consider using ``filtered_historical_simulation`` from the
        ``monte_carlo`` sub-module for more realistic tails.

    Parameters:
        returns: Historical return series.
        target_loss: Target cumulative loss as a negative number
            (e.g. ``-0.20`` for a 20% loss).
        n_sims: Number of Monte Carlo paths to simulate.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:

        * ``"scenarios_found"`` -- number of simulated paths that hit
          the target.
        * ``"probability"`` -- estimated probability of hitting the
          target.
        * ``"avg_loss"`` -- mean loss across qualifying scenarios.
        * ``"worst_loss"`` -- worst loss observed in qualifying
          scenarios.
        * ``"threshold_percentile"`` -- percentile at which the target
          loss sits in the simulated distribution.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0.0003, 0.01, 252))
        >>> result = reverse_stress_test(returns, target_loss=-0.30, n_sims=5000, seed=42)
        >>> result["probability"] >= 0
        True

    See Also:
        historical_stress_test: Replay known crisis periods.
        stress_test_returns: User-defined additive shocks.
    """
    if isinstance(returns, pd.DataFrame):
        vals = returns.mean(axis=1).dropna().values
    else:
        vals = returns.dropna().values

    mu = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=1))
    n_periods = len(vals)

    rng = np.random.default_rng(seed)
    sims = rng.normal(mu, sigma, size=(n_sims, n_periods))
    cum_returns = np.prod(1 + sims, axis=1) - 1

    hit_mask = cum_returns <= target_loss
    n_found = int(np.sum(hit_mask))

    if n_found > 0:
        losses = cum_returns[hit_mask]
        avg_loss = float(np.mean(losses))
        worst_loss = float(np.min(losses))
    else:
        avg_loss = 0.0
        worst_loss = 0.0

    # Where does target_loss sit?
    threshold_pct = float(sp_stats.percentileofscore(cum_returns, target_loss))

    return {
        "scenarios_found": n_found,
        "probability": n_found / n_sims,
        "avg_loss": avg_loss,
        "worst_loss": worst_loss,
        "threshold_percentile": threshold_pct,
    }


def joint_stress_test(
    returns: pd.DataFrame,
    vol_shock: float = 2.0,
    spot_shock: float = -0.10,
    correlation_shock: float = 0.0,
) -> dict[str, Any]:
    """Apply combined volatility, spot, and correlation shocks.

    Real crises involve simultaneous increases in volatility, drops in
    asset prices, and spikes in correlation (diversification breaks
    down when you need it most). This function applies all three shocks
    simultaneously to produce a stressed covariance matrix and stressed
    expected returns.

    When to use:
        Use joint stress tests for:
        - Portfolio optimisation stress testing: feed the stressed
          covariance matrix into a mean-variance optimiser.
        - Capital adequacy under combined adverse conditions.
        - Comparing diversification benefits under normal vs. stressed
          conditions (correlation shock toward 1.0 eliminates
          diversification).

    Procedure:
        1. Scale demeaned returns by *vol_shock* (volatility multiplier).
        2. Shift the mean by *spot_shock* (additive level shift).
        3. Blend the correlation matrix toward uniform correlation:
           stressed_corr = (1 - c) * corr + c * ones_matrix,
           where c = correlation_shock.

    How to interpret:
        The stressed covariance matrix (``stressed_cov``) reflects
        the combined effect of all three shocks. Pass it to
        ``wraquant.opt`` for stress-aware portfolio construction.
        Compare ``stressed_vol`` / ``base_vol`` to verify the vol
        scaling. Compare ``stressed_corr`` to the base correlation
        to see how diversification degrades.

    Parameters:
        returns: Multi-asset return DataFrame (columns = assets).
        vol_shock: Volatility multiplier (e.g. 2.0 = double vol).
        spot_shock: Additive shift to mean returns (e.g. -0.10 =
            subtract 10% from each asset's mean return).
        correlation_shock: Blend factor toward perfect correlation.
            0 = unchanged, 0.5 = halfway to perfect correlation,
            1 = all pairwise correlations set to 1.0.

    Returns:
        Dict with keys:

        * ``"stressed_mean"`` -- stressed mean return per asset (dict).
        * ``"stressed_vol"`` -- stressed volatility per asset (dict).
        * ``"stressed_corr"`` -- stressed correlation matrix (ndarray).
        * ``"stressed_cov"`` -- stressed covariance matrix (ndarray).
        * ``"base_mean"`` -- original mean returns (dict).
        * ``"base_vol"`` -- original volatilities (dict).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame({
        ...     "SPY": np.random.normal(0.0005, 0.01, 252),
        ...     "TLT": np.random.normal(0.0002, 0.005, 252),
        ... })
        >>> result = joint_stress_test(returns, vol_shock=2.0, correlation_shock=0.5)
        >>> result["stressed_vol"]["SPY"] > result["base_vol"]["SPY"]
        True

    See Also:
        vol_stress_test: Volatility scaling only.
        marginal_stress_contribution: Identify worst-contributing asset.
    """
    clean = returns.dropna()
    assets = clean.columns.tolist()
    vals = clean.values

    mu = np.mean(vals, axis=0)
    sigma = np.std(vals, axis=0, ddof=1)
    corr = np.corrcoef(vals, rowvar=False)

    # Vol shock: scale standard deviations
    stressed_sigma = sigma * vol_shock

    # Spot shock: shift mean
    stressed_mu = mu + spot_shock

    # Correlation shock: blend toward unit correlation
    ones_mat = np.ones_like(corr)
    stressed_corr = (1 - correlation_shock) * corr + correlation_shock * ones_mat
    # Ensure diagonal stays at 1
    np.fill_diagonal(stressed_corr, 1.0)

    # Reconstruct covariance
    D = np.diag(stressed_sigma)
    stressed_cov = D @ stressed_corr @ D

    return {
        "stressed_mean": dict(zip(assets, stressed_mu.tolist(), strict=False)),
        "stressed_vol": dict(zip(assets, stressed_sigma.tolist(), strict=False)),
        "stressed_corr": stressed_corr,
        "stressed_cov": stressed_cov,
        "base_mean": dict(zip(assets, mu.tolist(), strict=False)),
        "base_vol": dict(zip(assets, sigma.tolist(), strict=False)),
    }


def marginal_stress_contribution(
    portfolio_weights: np.ndarray,
    returns: pd.DataFrame,
    scenario: dict[str, float],
) -> dict[str, Any]:
    """Identify which asset contributes most to portfolio stress loss.

    Decomposes the total portfolio loss under a stress scenario into
    per-asset contributions. This is essential for understanding
    *where* the risk is concentrated and deciding which positions to
    hedge or reduce.

    When to use:
        Use marginal stress contribution after running a stress test
        to answer: "which position is killing the portfolio under
        this scenario?" This guides targeted hedging decisions (e.g.,
        buy puts on the worst-contributing asset).

    How to interpret:
        ``asset_contributions`` shows each asset's dollar P&L under the
        scenario (weight_i * scenario_return_i). ``pct_contributions``
        normalises these to sum to 1.0, showing the *fraction* of
        total loss attributable to each asset. ``worst_asset`` is the
        asset with the most negative contribution.

    Parameters:
        portfolio_weights: Weight vector aligned with ``returns.columns``.
            Must have the same length as the number of columns.
        returns: Multi-asset return DataFrame (columns = assets).
        scenario: Mapping of asset name to shocked return value. Assets
            not in the scenario use their historical mean return.

    Returns:
        Dict with keys:

        * ``"total_stress_loss"`` -- portfolio return under the scenario.
        * ``"asset_contributions"`` -- dict mapping asset name to its
          P&L contribution (weight * return).
        * ``"pct_contributions"`` -- dict mapping asset name to its
          percentage contribution to total loss.
        * ``"worst_asset"`` -- name of the asset contributing the most
          loss.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame({
        ...     "AAPL": np.random.normal(0.001, 0.02, 100),
        ...     "MSFT": np.random.normal(0.001, 0.015, 100),
        ... })
        >>> weights = np.array([0.6, 0.4])
        >>> scenario = {"AAPL": -0.15, "MSFT": -0.05}
        >>> result = marginal_stress_contribution(weights, returns, scenario)
        >>> result["worst_asset"]
        'AAPL'

    See Also:
        joint_stress_test: Generate stressed parameters for scenarios.
        wraquant.risk.portfolio.risk_contribution: Euler risk
            decomposition (non-scenario-based).
    """
    assets = returns.columns.tolist()
    means = returns.mean().values

    # Build scenario vector
    scenario_vec = means.copy()
    for asset, shock_val in scenario.items():
        if asset in assets:
            idx = assets.index(asset)
            scenario_vec[idx] = shock_val

    # Per-asset contribution = weight_i * scenario_return_i
    contributions = portfolio_weights * scenario_vec
    total_loss = float(np.sum(contributions))

    asset_contribs = dict(zip(assets, contributions.tolist(), strict=False))

    # Percentage contributions
    if abs(total_loss) > 1e-15:
        pct = contributions / total_loss
    else:
        pct = np.zeros_like(contributions)
    pct_contribs = dict(zip(assets, pct.tolist(), strict=False))

    # Worst asset is the one with the most negative contribution
    worst_idx = int(np.argmin(contributions))
    worst_asset = assets[worst_idx]

    return {
        "total_stress_loss": total_loss,
        "asset_contributions": asset_contribs,
        "pct_contributions": pct_contribs,
        "worst_asset": worst_asset,
    }


def correlation_stress(
    returns: pd.DataFrame,
    shock_levels: list[float] | None = None,
) -> dict[str, Any]:
    """Stress test portfolio by increasing pairwise correlations.

    Blends the empirical correlation matrix toward perfect correlation
    at various shock levels, recomputes the covariance matrix, and
    measures the resulting portfolio volatility. This reveals how much
    diversification benefit the portfolio loses as correlations rise.

    When to use:
        Use correlation stress for:
        - Evaluating diversification fragility: how much does portfolio
          risk increase if diversification breaks down?
        - Regulatory stress testing: correlation breakdown is a standard
          CCAR scenario.
        - Risk committee presentations: "if all correlations jump to 0.8,
          our portfolio vol goes from X% to Y%."

    Parameters:
        returns: Multi-asset return DataFrame (columns = assets).
        shock_levels: Blend factors toward perfect correlation. 0 =
            unchanged, 1 = all correlations set to 1.0. Defaults to
            ``[0.0, 0.25, 0.5, 0.75, 1.0]``.

    Returns:
        Dict with keys:

        * ``"results"`` -- dict mapping shock level to a dict with
          ``"portfolio_vol"`` (equal-weighted portfolio volatility),
          ``"avg_correlation"`` (mean off-diagonal correlation),
          ``"stressed_corr"`` (the stressed correlation matrix).
        * ``"base_vol"`` -- equal-weighted portfolio vol with no shock.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame({
        ...     "A": np.random.normal(0, 0.01, 252),
        ...     "B": np.random.normal(0, 0.012, 252),
        ...     "C": np.random.normal(0, 0.008, 252),
        ... })
        >>> result = correlation_stress(returns, shock_levels=[0.0, 0.5, 1.0])
        >>> result["results"][1.0]["portfolio_vol"] >= result["base_vol"]
        True

    See Also:
        joint_stress_test: Combined vol, spot, and correlation shocks.
        vol_stress_test: Volatility-only scaling.
    """
    if shock_levels is None:
        shock_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    from wraquant.risk.portfolio import portfolio_volatility as _portfolio_vol

    clean = returns.dropna()
    n_assets = clean.shape[1]
    vols = clean.std().values
    corr = clean.corr().values
    eq_weights = np.ones(n_assets) / n_assets

    results: dict[float, dict[str, Any]] = {}
    base_vol = None

    for level in shock_levels:
        ones_mat = np.ones_like(corr)
        stressed_corr = (1 - level) * corr + level * ones_mat
        np.fill_diagonal(stressed_corr, 1.0)

        # Reconstruct cov
        D = np.diag(vols)
        stressed_cov = D @ stressed_corr @ D

        port_vol = _portfolio_vol(eq_weights, stressed_cov)
        if level == 0.0:
            base_vol = port_vol

        # Average off-diagonal correlation
        mask = ~np.eye(n_assets, dtype=bool)
        avg_corr = float(np.mean(stressed_corr[mask]))

        results[level] = {
            "portfolio_vol": port_vol,
            "avg_correlation": avg_corr,
            "stressed_corr": stressed_corr,
        }

    if base_vol is None:
        base_vol = 0.0

    return {
        "results": results,
        "base_vol": base_vol,
    }


def liquidity_stress(
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
    liquidity_haircuts: dict[str, float] | None = None,
    portfolio_value: float = 1_000_000.0,
) -> dict[str, Any]:
    """Estimate liquidation cost under adverse market conditions.

    Models the cost of unwinding a portfolio under stressed liquidity
    conditions. If volume data is provided, uses a market-impact model;
    otherwise, applies user-defined haircuts to each asset.

    When to use:
        Use liquidity stress for:
        - Estimating portfolio liquidation costs during crises.
        - Measuring liquidity-adjusted VaR (LVaR).
        - Satisfying regulatory requirements for liquidity stress testing
          (e.g., SEC Rule 22e-4 for mutual funds).

    Parameters:
        returns: Multi-asset return DataFrame (columns = assets).
        volumes: Optional DataFrame of trading volumes (same shape and
            index as *returns*). If provided, liquidity cost is estimated
            as spread * sqrt(position / ADV).
        liquidity_haircuts: Optional dict mapping asset name to
            liquidation cost (e.g., ``{"AAPL": 0.001, "ILLIQ": 0.05}``).
            If neither *volumes* nor *haircuts* are provided, uses
            volatility as a proxy.
        portfolio_value: Total portfolio value for position sizing.

    Returns:
        Dict with keys:

        * ``"total_cost"`` -- Estimated total liquidation cost ($).
        * ``"total_cost_pct"`` -- Cost as a fraction of portfolio value.
        * ``"asset_costs"`` -- dict mapping asset to its liquidation cost.
        * ``"days_to_liquidate"`` -- estimated days to liquidate if
          limited to 10% of ADV per day (only if *volumes* provided).

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame({
        ...     "A": np.random.normal(0, 0.01, 252),
        ...     "B": np.random.normal(0, 0.02, 252),
        ... })
        >>> result = liquidity_stress(returns, portfolio_value=1_000_000)
        >>> result["total_cost"] > 0
        True

    See Also:
        vol_stress_test: Volatility scaling stress test.
        wraquant.execution.cost: Transaction cost modeling.
    """
    clean = returns.dropna()
    assets = clean.columns.tolist()
    n_assets = len(assets)
    eq_weights = np.ones(n_assets) / n_assets
    position_values = eq_weights * portfolio_value

    asset_costs: dict[str, float] = {}
    days_to_liquidate: dict[str, float] = {}

    if liquidity_haircuts is not None:
        for asset in assets:
            haircut = liquidity_haircuts.get(asset, 0.01)
            cost = position_values[assets.index(asset)] * haircut
            asset_costs[asset] = cost

    elif volumes is not None:
        vol_clean = volumes.reindex(clean.index).dropna()
        for i, asset in enumerate(assets):
            if asset in vol_clean.columns:
                adv = float(vol_clean[asset].mean())
                asset_vol = float(clean[asset].std())
                position = position_values[i]
                # Square-root market impact: cost ~ sigma * sqrt(Q/V)
                if adv > 0:
                    impact = asset_vol * np.sqrt(position / adv)
                    cost = position * impact
                    # Days to liquidate at 10% of ADV
                    days = position / (0.1 * adv) if adv > 0 else float("inf")
                    days_to_liquidate[asset] = float(days)
                else:
                    cost = position * asset_vol
                asset_costs[asset] = float(cost)
            else:
                asset_costs[asset] = float(position_values[i] * clean[asset].std())
    else:
        # Use volatility as proxy for bid-ask spread
        for i, asset in enumerate(assets):
            asset_vol = float(clean[asset].std())
            # Stressed spread ~ 3x daily vol
            cost = position_values[i] * asset_vol * 3
            asset_costs[asset] = float(cost)

    total_cost = sum(asset_costs.values())
    total_cost_pct = total_cost / portfolio_value if portfolio_value > 0 else 0.0

    result: dict[str, Any] = {
        "total_cost": total_cost,
        "total_cost_pct": total_cost_pct,
        "asset_costs": asset_costs,
    }
    if days_to_liquidate:
        result["days_to_liquidate"] = days_to_liquidate

    return result


# Pre-defined scenario library
_SCENARIO_LIBRARY: dict[str, dict[str, float]] = {
    "gfc_2008": {
        "equity_shock": -0.40,
        "credit_spread_widening": 0.03,
        "vol_multiplier": 3.0,
        "correlation_shock": 0.7,
        "description_rate_change": -0.02,
    },
    "covid_2020": {
        "equity_shock": -0.34,
        "credit_spread_widening": 0.025,
        "vol_multiplier": 4.0,
        "correlation_shock": 0.6,
        "description_rate_change": -0.015,
    },
    "dot_com_2000": {
        "equity_shock": -0.49,
        "credit_spread_widening": 0.02,
        "vol_multiplier": 2.0,
        "correlation_shock": 0.3,
        "description_rate_change": -0.03,
    },
    "rate_hike_2022": {
        "equity_shock": -0.25,
        "credit_spread_widening": 0.015,
        "vol_multiplier": 1.5,
        "correlation_shock": 0.5,
        "description_rate_change": 0.03,
    },
    "stagflation": {
        "equity_shock": -0.30,
        "credit_spread_widening": 0.02,
        "vol_multiplier": 2.0,
        "correlation_shock": 0.4,
        "description_rate_change": 0.02,
    },
    "flash_crash": {
        "equity_shock": -0.10,
        "credit_spread_widening": 0.005,
        "vol_multiplier": 5.0,
        "correlation_shock": 0.8,
        "description_rate_change": -0.005,
    },
    "em_crisis": {
        "equity_shock": -0.35,
        "credit_spread_widening": 0.04,
        "vol_multiplier": 2.5,
        "correlation_shock": 0.5,
        "description_rate_change": 0.01,
    },
}


def scenario_library(
    returns: pd.DataFrame,
    scenarios: list[str] | None = None,
) -> dict[str, Any]:
    """Apply pre-defined crisis scenarios from the built-in library.

    Provides a curated set of stress scenarios calibrated to historical
    crises. Each scenario specifies equity shocks, volatility multipliers,
    and correlation shocks. The function applies each scenario to the
    provided returns and reports the stressed portfolio metrics.

    When to use:
        Use the scenario library for:
        - Quick stress testing without designing custom scenarios.
        - Regulatory reporting: standard scenarios that regulators
          expect to see.
        - Benchmarking: compare your portfolio's sensitivity to
          well-known crises.

    Available scenarios:
        - ``"gfc_2008"`` -- Global Financial Crisis
        - ``"covid_2020"`` -- COVID-19 crash
        - ``"dot_com_2000"`` -- Dot-com bubble burst
        - ``"rate_hike_2022"`` -- 2022 rate hiking cycle
        - ``"stagflation"`` -- Stagflation scenario
        - ``"flash_crash"`` -- Flash crash (intraday)
        - ``"em_crisis"`` -- Emerging markets crisis

    Parameters:
        returns: Multi-asset return DataFrame (columns = assets).
        scenarios: List of scenario names from the library. If None,
            all scenarios are applied.

    Returns:
        Dict with keys:

        * ``"scenario_results"`` -- dict mapping scenario name to a dict
          with ``"stressed_portfolio_return"`` (equity shock applied),
          ``"stressed_vol"`` (vol-scaled portfolio volatility),
          ``"scenario_params"`` (the raw scenario parameters).
        * ``"available_scenarios"`` -- list of all available scenario names.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.DataFrame({
        ...     "SPY": np.random.normal(0.0005, 0.01, 252),
        ...     "TLT": np.random.normal(0.0002, 0.005, 252),
        ... })
        >>> result = scenario_library(returns, scenarios=["gfc_2008", "covid_2020"])
        >>> "gfc_2008" in result["scenario_results"]
        True

    See Also:
        historical_stress_test: Replay actual crisis returns.
        joint_stress_test: Custom multi-factor stress test.
    """
    if scenarios is None:
        scenarios = list(_SCENARIO_LIBRARY.keys())

    from wraquant.risk.portfolio import portfolio_volatility as _portfolio_volatility

    clean = returns.dropna()
    n_assets = clean.shape[1]
    eq_weights = np.ones(n_assets) / n_assets

    base_vol = _portfolio_volatility(eq_weights, clean.cov().values)

    results: dict[str, dict[str, Any]] = {}

    for name in scenarios:
        if name not in _SCENARIO_LIBRARY:
            continue

        params = _SCENARIO_LIBRARY[name]
        equity_shock = params.get("equity_shock", 0.0)
        vol_mult = params.get("vol_multiplier", 1.0)

        stressed_return = float(clean.mean().mean() + equity_shock)
        stressed_vol = base_vol * vol_mult

        results[name] = {
            "stressed_portfolio_return": stressed_return,
            "stressed_vol": float(stressed_vol),
            "scenario_params": params,
        }

    return {
        "scenario_results": results,
        "available_scenarios": list(_SCENARIO_LIBRARY.keys()),
    }
