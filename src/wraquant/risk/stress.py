"""Comprehensive stress testing for portfolio risk analysis.

Provides scenario-based, historical, volatility, and spot stress tests,
reverse stress testing, sensitivity ladders, joint stress tests, and
marginal stress contribution analysis.
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

    Each scenario name maps to an additive shift applied uniformly to all
    returns.  The function computes the stressed mean, stressed VaR (5th
    percentile), and stressed CVaR for every scenario.

    Parameters:
        returns: Historical return series (Series) or multi-asset returns
            (DataFrame).
        scenarios: Mapping of scenario name to additive return shock
            (e.g. ``{"crash": -0.10, "boom": 0.05}``).

    Returns:
        Dict with keys:

        * ``"scenario_results"`` -- dict mapping scenario name to a dict
          with ``"stressed_mean"``, ``"stressed_var_95"``,
          ``"stressed_cvar_95"``.
        * ``"base_mean"`` -- mean of the original returns.
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

    If *crisis_periods* is ``None``, a built-in set of crises is used
    (GFC 2008, COVID 2020, Dot-Com 2000, Euro Debt 2011, etc.).

    Parameters:
        returns: Return series with a ``DatetimeIndex``.
        crisis_periods: Mapping of crisis name to ``(start, end)`` date
            strings.  Periods not covered by the data are skipped.

    Returns:
        Dict with keys:

        * ``"crisis_results"`` -- dict mapping crisis name to a dict
          with ``"cumulative_return"``, ``"max_drawdown"``,
          ``"mean_daily_return"``, ``"n_days"``.
        * ``"periods_found"`` -- list of crisis names that overlap with
          the data.
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
        mask = (ret.index >= pd.Timestamp(start)) & (
            ret.index <= pd.Timestamp(end)
        )
        subset = ret.loc[mask]
        if len(subset) == 0:
            continue

        periods_found.append(name)
        cum_return = float(np.prod(1 + subset.values) - 1)
        cum_prices = np.cumprod(1 + subset.values)
        running_max = np.maximum.accumulate(cum_prices)
        drawdowns = (cum_prices - running_max) / running_max
        max_dd = float(np.min(drawdowns))

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
    re-added.  This preserves the mean while increasing (or decreasing)
    dispersion.

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
    (e.g. -0.10 means a 10 % drop from the last price).

    Parameters:
        prices: Price series or DataFrame of asset prices.
        spot_shocks: List of percentage shocks (e.g. ``[-0.30, -0.20,
            -0.10, 0.10, 0.20]``).  Defaults to
            ``[-0.30, -0.20, -0.10, -0.05, 0.05, 0.10]``.

    Returns:
        Dict with keys:

        * ``"spot_results"`` -- dict mapping shock (as string) to
          ``"shocked_price"``, ``"price_change"``, ``"pct_change"``.
        * ``"base_price"`` -- the last observed price.
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

    Fits a linear regression of portfolio returns on the factor, then
    uses the beta to estimate the P&L impact of each shock level.

    Parameters:
        portfolio_returns: Portfolio return series.
        factor_returns: Single-factor return series (same index).
        shock_range: Array of factor shock values (e.g.
            ``np.linspace(-0.10, 0.10, 21)``).  Defaults to
            ``np.linspace(-0.10, 0.10, 21)``.

    Returns:
        Dict with keys:

        * ``"ladder"`` -- dict mapping shock level (float) to estimated
          portfolio P&L.
        * ``"beta"`` -- regression beta.
        * ``"alpha"`` -- regression intercept.
        * ``"r_squared"`` -- R-squared of the regression.
    """
    if shock_range is None:
        shock_range = np.linspace(-0.10, 0.10, 21)
    shock_range = np.asarray(shock_range)

    # Align and clean
    aligned = pd.concat(
        [portfolio_returns.rename("port"), factor_returns.rename("factor")],
        axis=1,
    ).dropna()
    y = aligned["port"].values
    x = aligned["factor"].values

    # OLS regression
    slope, intercept, r_value, _p_value, _std_err = sp_stats.linregress(x, y)

    ladder: dict[float, float] = {}
    for shock in shock_range:
        ladder[float(shock)] = float(intercept + slope * shock)

    return {
        "ladder": ladder,
        "beta": float(slope),
        "alpha": float(intercept),
        "r_squared": float(r_value**2),
    }


def reverse_stress_test(
    returns: pd.Series | pd.DataFrame,
    target_loss: float,
    n_sims: int = 10000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Find scenarios that produce at least the specified target loss.

    Simulates returns from a fitted normal distribution and identifies
    paths where the cumulative loss meets or exceeds *target_loss*.

    Parameters:
        returns: Historical return series.
        target_loss: Target cumulative loss as a negative number
            (e.g. ``-0.20`` for a 20 % loss).
        n_sims: Number of Monte Carlo paths to simulate.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:

        * ``"scenarios_found"`` -- number of simulated paths that hit
          the target.
        * ``"probability"`` -- estimated probability of hitting the
          target.
        * ``"avg_loss"`` -- mean loss across qualifying scenarios.
        * ``"worst_loss"`` -- worst loss observed.
        * ``"threshold_percentile"`` -- percentile at which the target
          loss sits in the simulated distribution.
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

    Procedure:

    1. Scale demeaned returns by *vol_shock* (volatility multiplier).
    2. Shift the mean by *spot_shock* (additive level shift).
    3. Adjust the correlation matrix toward uniform correlation by
       blending toward a matrix of ones (``correlation_shock`` in
       ``[0, 1]`` where 0 = no change, 1 = perfect correlation).

    Parameters:
        returns: Multi-asset return DataFrame.
        vol_shock: Volatility multiplier (e.g. 2.0 = double vol).
        spot_shock: Additive shift to mean returns.
        correlation_shock: Blend factor toward perfect correlation
            (0 = unchanged, 1 = perfect positive correlation).

    Returns:
        Dict with keys:

        * ``"stressed_mean"`` -- stressed mean return per asset.
        * ``"stressed_vol"`` -- stressed volatility per asset.
        * ``"stressed_corr"`` -- stressed correlation matrix.
        * ``"stressed_cov"`` -- stressed covariance matrix.
        * ``"base_mean"`` -- original mean returns.
        * ``"base_vol"`` -- original volatilities.
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
        "stressed_mean": dict(zip(assets, stressed_mu.tolist())),
        "stressed_vol": dict(zip(assets, stressed_sigma.tolist())),
        "stressed_corr": stressed_corr,
        "stressed_cov": stressed_cov,
        "base_mean": dict(zip(assets, mu.tolist())),
        "base_vol": dict(zip(assets, sigma.tolist())),
    }


def marginal_stress_contribution(
    portfolio_weights: np.ndarray,
    returns: pd.DataFrame,
    scenario: dict[str, float],
) -> dict[str, Any]:
    """Identify which asset contributes most to stress loss.

    Computes each asset's contribution to the total portfolio stress
    loss under the given scenario.

    Parameters:
        portfolio_weights: Weight vector aligned with ``returns.columns``.
        returns: Multi-asset return DataFrame.
        scenario: Mapping of asset name to shocked return value.
            Assets not in the scenario use their historical mean.

    Returns:
        Dict with keys:

        * ``"total_stress_loss"`` -- portfolio return under the scenario.
        * ``"asset_contributions"`` -- dict mapping asset name to its
          P&L contribution.
        * ``"pct_contributions"`` -- dict mapping asset name to its
          percentage contribution to total loss (sums to 1).
        * ``"worst_asset"`` -- name of the asset contributing the most
          loss.
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

    asset_contribs = dict(zip(assets, contributions.tolist()))

    # Percentage contributions
    if abs(total_loss) > 1e-15:
        pct = contributions / total_loss
    else:
        pct = np.zeros_like(contributions)
    pct_contribs = dict(zip(assets, pct.tolist()))

    # Worst asset is the one with the most negative contribution
    worst_idx = int(np.argmin(contributions))
    worst_asset = assets[worst_idx]

    return {
        "total_stress_loss": total_loss,
        "asset_contributions": asset_contribs,
        "pct_contributions": pct_contribs,
        "worst_asset": worst_asset,
    }
