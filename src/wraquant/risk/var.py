"""Value-at-Risk and Conditional VaR (Expected Shortfall) estimation.

Provides the two most important tail-risk measures in quantitative risk
management. VaR is a regulatory standard (Basel II/III); CVaR (Expected
Shortfall) is preferred by Basel IV and is mathematically superior
because it is a *coherent* risk measure (satisfies sub-additivity).

Both measures can be estimated via historical simulation (non-parametric)
or parametric (Gaussian) methods. For heavy-tailed distributions
(equities, credit), historical simulation is generally more accurate;
for smooth risk surfaces (rates), parametric is often sufficient.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Estimate Value-at-Risk (VaR).

    VaR answers the question: "With X% confidence, what is the maximum
    loss I should expect over one period?" More precisely, VaR is the
    (1 - confidence) quantile of the return distribution, flipped to
    a positive loss number.

    When to use:
        Use VaR for regulatory reporting, margin calculations, and
        setting position limits. Choose historical VaR when you have
        enough data (>500 observations) and want to capture fat tails
        without distributional assumptions. Choose parametric VaR when
        data is scarce or when you need analytical sensitivities (e.g.,
        delta-normal VaR for a derivatives book).

    Mathematical formulation:
        Historical: VaR_alpha = -quantile(returns, 1 - alpha)
        Parametric: VaR_alpha = -(mu + sigma * Phi^{-1}(1 - alpha))

        where alpha is the confidence level, mu and sigma are the
        sample mean and standard deviation, and Phi^{-1} is the
        standard normal inverse CDF.

    How to interpret:
        A 95% daily VaR of 0.02 means: "on 95% of days, the portfolio
        loses less than 2%. On the remaining 5% of days, the loss
        *exceeds* 2%." VaR says nothing about *how much* worse the loss
        can be beyond the threshold -- that is what CVaR captures.

    Parameters:
        returns: Simple return series (e.g., daily percentage changes).
        confidence: Confidence level (e.g., 0.95 for 95%, 0.99 for 99%).
            Basel III uses 0.99; internal risk management often uses 0.95.
        method: Estimation method:
            - ``"historical"`` -- empirical quantile (non-parametric,
              default). No distributional assumption; captures fat tails.
            - ``"parametric"`` -- Gaussian assumption. Smooth but
              underestimates tail risk for leptokurtic returns.

    Returns:
        VaR as a positive float representing the loss threshold. For
        example, 0.025 means a 2.5% loss at the given confidence level.

    Raises:
        ValueError: If *method* is not recognized.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0, 0.01, 1000))
        >>> var_95 = value_at_risk(returns, confidence=0.95)
        >>> var_95 > 0
        True

    Caveats:
        - VaR is *not* sub-additive: the VaR of a portfolio can exceed
          the sum of individual VaRs.  Use ``conditional_var`` for a
          coherent measure.
        - Historical VaR is sensitive to the sample window; recent
          crises dominate short windows.
        - Parametric VaR severely underestimates tail risk for fat-
          tailed distributions (equities, credit).

    See Also:
        conditional_var: Expected loss beyond the VaR threshold (CVaR).
        garch_var: GARCH-based time-varying VaR using conditional volatility.
        wraquant.vol.models.garch_fit: Fit GARCH model for conditional vol.
        wraquant.risk.stress.stress_test_returns: Scenario-based analysis.

    References:
        - Jorion (2006), "Value at Risk: The New Benchmark"
        - Basel Committee on Banking Supervision (2019), "Minimum capital
          requirements for market risk"
    """
    clean = returns.dropna().values

    if method == "historical":
        var = float(np.percentile(clean, (1 - confidence) * 100))
    elif method == "parametric":
        mu = clean.mean()
        sigma = clean.std()
        var = float(sp_stats.norm.ppf(1 - confidence, loc=mu, scale=sigma))
    else:
        msg = f"Unknown VaR method: {method!r}"
        raise ValueError(msg)

    return -var


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Estimate Conditional VaR (Expected Shortfall / CVaR).

    CVaR answers: "given that the loss exceeds VaR, what is the
    *expected* loss?" It captures the severity of tail losses, not just
    their threshold. Unlike VaR, CVaR is a *coherent* risk measure
    (Artzner et al. 1999) -- it satisfies sub-additivity, meaning the
    CVaR of a portfolio is at most the sum of individual CVaRs.

    When to use:
        CVaR is preferred over VaR for:
        - Portfolio optimisation (mean-CVaR optimisation is convex).
        - Regulatory capital under Basel IV / FRTB.
        - Any situation where you care about tail *severity*, not just
          tail *frequency*.
        Use historical CVaR with long samples (>1000 obs) and parametric
        CVaR when you need smooth gradients or have short data.

    Mathematical formulation:
        Historical: CVaR_alpha = -mean(returns | returns <= VaR_quantile)
        Parametric: CVaR_alpha = -(mu - sigma * phi(z_alpha) / (1 - alpha))

        where z_alpha = Phi^{-1}(1 - alpha), phi is the standard normal
        PDF, and Phi is the CDF.

    How to interpret:
        A 95% daily CVaR of 0.035 means: "on the worst 5% of days, the
        *average* loss is 3.5%." CVaR is always >= VaR at the same
        confidence level. For normal distributions, 95% CVaR is about
        1.25x the 95% VaR. For fat-tailed distributions, the ratio is
        much larger -- this ratio itself is a useful diagnostic of tail
        heaviness.

    Parameters:
        returns: Simple return series.
        confidence: Confidence level (e.g., 0.95 for 95%).
        method: Estimation method:
            - ``"historical"`` -- mean of returns in the tail
              (default). Non-parametric; captures fat tails.
            - ``"parametric"`` -- Gaussian formula. Smooth but
              underestimates tail risk for heavy-tailed distributions.

    Returns:
        CVaR as a positive float representing the expected tail loss.
        For example, 0.035 means an expected loss of 3.5% in the tail.

    Raises:
        ValueError: If *method* is not recognized.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0, 0.01, 1000))
        >>> cvar = conditional_var(returns, confidence=0.95)
        >>> var = value_at_risk(returns, confidence=0.95)
        >>> cvar >= var  # CVaR is always >= VaR
        True

    See Also:
        value_at_risk: The VaR threshold itself.
        wraquant.risk.monte_carlo.importance_sampling_var: Variance-
            reduced tail estimation.

    References:
        - Artzner et al. (1999), "Coherent Measures of Risk"
        - Rockafellar & Uryasev (2000), "Optimization of Conditional
          Value-at-Risk"
    """
    clean = returns.dropna().values

    if method == "historical":
        cutoff = np.percentile(clean, (1 - confidence) * 100)
        tail = clean[clean <= cutoff]
        cvar = float(-tail.mean()) if len(tail) > 0 else 0.0
    elif method == "parametric":
        mu = clean.mean()
        sigma = clean.std()
        alpha = 1 - confidence
        z = sp_stats.norm.ppf(alpha)
        cvar = float(-(mu - sigma * sp_stats.norm.pdf(z) / alpha))
    else:
        msg = f"Unknown CVaR method: {method!r}"
        raise ValueError(msg)

    return cvar


def garch_var(
    returns: pd.Series | np.ndarray,
    alpha: float = 0.05,
    vol_model: str = "GARCH",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    horizon: int = 1,
) -> dict[str, Any]:
    """Value at Risk using GARCH conditional volatility.

    Combines GARCH volatility forecasting with parametric VaR,
    producing time-varying risk estimates that adapt to current
    market conditions. Superior to static VaR in volatile markets.

    The conditional VaR at time t is:
        VaR_t = -mu + sigma_t * z_alpha

    where sigma_t is the GARCH-forecasted volatility and z_alpha
    is the quantile of the fitted error distribution.

    Parameters:
        returns: Return series.
        alpha: Significance level (0.05 = 95% VaR).
        vol_model: GARCH variant ("GARCH", "EGARCH", "GJR").
        p: GARCH lag order.
        q: ARCH lag order.
        dist: Error distribution ("normal", "t", "skewt").
        horizon: Forecast horizon in periods.

    Returns:
        Dictionary containing:
        - **var** (*pd.Series*) -- Time-varying VaR series.
        - **cvar** (*pd.Series*) -- Time-varying CVaR/ES series.
        - **conditional_vol** (*pd.Series*) -- GARCH conditional volatility.
        - **breaches** (*pd.Series*) -- Boolean where actual loss exceeded VaR.
        - **breach_rate** (*float*) -- Fraction of breaches (should be ~alpha).
        - **garch_params** (*dict*) -- Fitted GARCH parameters.

    Example:
        >>> from wraquant.risk.var import garch_var
        >>> result = garch_var(returns, alpha=0.05, vol_model="GJR", dist="t")
        >>> print(f"Breach rate: {result['breach_rate']:.3f} (target: 0.050)")
    """
    from wraquant.vol.models import egarch_fit, garch_fit, gjr_garch_fit

    # Fit the appropriate GARCH variant
    _fit_funcs = {
        "GARCH": garch_fit,
        "EGARCH": egarch_fit,
        "GJR": gjr_garch_fit,
    }
    fit_fn = _fit_funcs.get(vol_model.upper())
    if fit_fn is None:
        msg = f"Unknown vol_model: {vol_model!r}. Choose from {list(_fit_funcs)}"
        raise ValueError(msg)

    fit_result = fit_fn(returns, p=p, q=q, dist=dist)

    cond_vol = fit_result["conditional_volatility"]
    std_resid = fit_result["standardized_residuals"]
    params = fit_result["params"]

    # Compute mean return
    ret_arr = np.asarray(returns, dtype=np.float64).ravel()
    mu = float(np.nanmean(ret_arr))

    # Scale conditional vol for multi-period horizon
    cond_vol_h = cond_vol * np.sqrt(horizon)

    # Determine z_alpha based on distribution
    if dist == "normal":
        z_alpha = sp_stats.norm.ppf(alpha)
        # CVaR multiplier: E[Z | Z < z_alpha] for standard normal
        cvar_mult = -sp_stats.norm.pdf(z_alpha) / alpha
    elif dist == "t":
        nu = params.get("nu", 5.0)
        z_alpha = sp_stats.t.ppf(alpha, df=nu)
        cvar_mult = (
            -sp_stats.t.pdf(z_alpha, df=nu) / alpha * ((nu + z_alpha**2) / (nu - 1))
        )
    elif dist == "skewt":
        # For skewed-t, fall back to empirical quantile of standardized residuals
        valid_resid = std_resid[np.isfinite(std_resid)]
        z_alpha = float(np.percentile(valid_resid, alpha * 100))
        tail = valid_resid[valid_resid <= z_alpha]
        cvar_mult = float(np.mean(tail)) if len(tail) > 0 else z_alpha
    else:
        z_alpha = sp_stats.norm.ppf(alpha)
        cvar_mult = -sp_stats.norm.pdf(z_alpha) / alpha

    # Compute time-varying VaR and CVaR
    var_series = -(mu * horizon) + cond_vol_h * (-z_alpha)
    if dist == "skewt":
        cvar_series = -(mu * horizon) + cond_vol_h * (-cvar_mult)
    else:
        cvar_series = -(mu * horizon) + cond_vol_h * (-cvar_mult)

    # Make sure we have pandas Series output
    if isinstance(cond_vol, pd.Series):
        var_series = pd.Series(var_series, index=cond_vol.index, name="garch_var")
        cvar_series = pd.Series(cvar_series, index=cond_vol.index, name="garch_cvar")

    # Compute breaches: actual loss exceeded VaR
    # Align lengths (returns may be longer/shorter than cond_vol)
    n = min(len(ret_arr), len(var_series))
    actual_losses = -ret_arr[-n:]
    var_values = np.asarray(var_series)[-n:]

    breaches_arr = actual_losses > var_values
    if isinstance(var_series, pd.Series):
        breaches = pd.Series(breaches_arr, index=var_series.index[-n:], name="breach")
    else:
        breaches = pd.Series(breaches_arr, name="breach")

    breach_rate = float(np.mean(breaches_arr))

    return {
        "var": var_series,
        "cvar": cvar_series,
        "conditional_vol": cond_vol,
        "breaches": breaches,
        "breach_rate": breach_rate,
        "garch_params": params,
    }


def greeks_var(
    portfolio_greeks: dict[str, float],
    spot: float,
    vol: float,
    rf: float = 0.0,
    dt: float = 1 / 252,
    spot_shock: float = 0.01,
    vol_shock: float = 0.01,
    n_scenarios: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """VaR approximation using portfolio Greeks (delta-gamma-vega).

    Approximates portfolio P&L using a second-order Taylor expansion
    with Greeks from the ``price/`` module, then estimates VaR from the
    resulting P&L distribution.  This bridges ``price/`` (Greeks
    computation) and ``risk/`` (VaR estimation).

    The P&L approximation is:

        PnL approx delta * dS + 0.5 * gamma * dS^2 + vega * d_sigma + theta * dt

    where dS and d_sigma are simulated from normal distributions with
    standard deviations *spot_shock * spot* and *vol_shock* respectively.

    When to use:
        Use delta-gamma-vega VaR for options portfolios where the P&L
        is nonlinear in the underlying.  Standard (delta-only) VaR
        underestimates risk for portfolios with significant gamma or
        vega exposure.  Full revaluation VaR is more accurate but much
        slower; this method is a fast approximation.

    Parameters:
        portfolio_greeks: Dictionary with portfolio-level Greeks.
            Required keys: ``'delta'``, ``'gamma'``.
            Optional keys: ``'vega'``, ``'theta'``.
        spot: Current spot price of the underlying.
        vol: Current implied volatility (annualised, e.g. 0.20 for 20%).
        rf: Risk-free rate (annualised). Used for drift in spot dynamics.
        dt: Time step as a fraction of a year (default 1/252 for one
            trading day).
        spot_shock: Standard deviation of the spot return used for
            simulation (default 0.01 = 1%).  Typically set to
            ``vol * sqrt(dt)`` for a realistic one-day shock.
        vol_shock: Standard deviation of the volatility change
            (default 0.01 = 1 vol point).
        n_scenarios: Number of Monte Carlo scenarios (default 10,000).
        confidence: VaR confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:

        - ``'var'`` (*float*) -- Estimated VaR (positive = loss).
        - ``'cvar'`` (*float*) -- Estimated CVaR / Expected Shortfall.
        - ``'mean_pnl'`` (*float*) -- Mean P&L across scenarios.
        - ``'std_pnl'`` (*float*) -- Standard deviation of P&L.
        - ``'delta_component'`` (*float*) -- VaR contribution from delta.
        - ``'gamma_component'`` (*float*) -- VaR contribution from gamma.
        - ``'vega_component'`` (*float*) -- VaR contribution from vega.
        - ``'theta_component'`` (*float*) -- Deterministic theta P&L.

    Example:
        >>> greeks = {'delta': 100, 'gamma': -50, 'vega': 200, 'theta': -10}
        >>> result = greeks_var(greeks, spot=100, vol=0.20, seed=42)
        >>> result['var'] > 0
        True
        >>> result['cvar'] >= result['var']
        True

    Notes:
        For a single option, compute Greeks with ``wraquant.price.greeks``
        and pass them here.  For a portfolio, sum the Greeks across
        positions first.

    See Also:
        value_at_risk: Standard return-based VaR.
        wraquant.price.greeks: Compute option Greeks.
    """
    rng = np.random.default_rng(seed)

    delta = portfolio_greeks.get("delta", 0.0)
    gamma = portfolio_greeks.get("gamma", 0.0)
    vega = portfolio_greeks.get("vega", 0.0)
    theta = portfolio_greeks.get("theta", 0.0)

    # Simulate spot and vol changes
    dS = rng.normal(0, spot_shock * spot, size=n_scenarios)
    d_sigma = rng.normal(0, vol_shock, size=n_scenarios)

    # Taylor expansion P&L
    delta_pnl = delta * dS
    gamma_pnl = 0.5 * gamma * dS ** 2
    vega_pnl = vega * d_sigma
    theta_pnl = theta * dt

    total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl

    # VaR and CVaR from the P&L distribution (losses are negative PnL)
    loss_quantile = np.percentile(total_pnl, (1 - confidence) * 100)
    var_estimate = -loss_quantile
    tail = total_pnl[total_pnl <= loss_quantile]
    cvar_estimate = float(-np.mean(tail)) if len(tail) > 0 else var_estimate

    return {
        "var": float(var_estimate),
        "cvar": float(cvar_estimate),
        "mean_pnl": float(np.mean(total_pnl)),
        "std_pnl": float(np.std(total_pnl)),
        "delta_component": float(np.std(delta_pnl)),
        "gamma_component": float(np.std(gamma_pnl)),
        "vega_component": float(np.std(vega_pnl)),
        "theta_component": float(theta_pnl),
    }
