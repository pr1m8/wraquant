"""Volatility models — GARCH family, stochastic vol, and Hawkes processes.

This module provides deep integration with the ``arch`` library for GARCH-family
models, plus pure implementations of stochastic volatility and Hawkes processes
for modeling volatility clustering and self-exciting dynamics.

All GARCH-family functions return rich result dictionaries including fitted
parameters, conditional volatility series, diagnostic statistics, and the
underlying model object for further analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize as sp_optimize
from scipy import stats as sp_stats

from wraquant.core.decorators import requires_extra

__all__ = [
    "ewma_volatility",
    "garch_fit",
    "egarch_fit",
    "gjr_garch_fit",
    "figarch_fit",
    "garch_forecast",
    "dcc_fit",
    "realized_garch",
    "harch_fit",
    "news_impact_curve",
    "volatility_persistence",
    "hawkes_process",
    "stochastic_vol_sv",
    "gaussian_mixture_vol",
    "vol_surface_svi",
    "variance_risk_premium",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_returns_array(returns: pd.Series | np.ndarray) -> np.ndarray:
    """Convert returns input to a 1-D float64 numpy array."""
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if len(arr) < 10:
        msg = "Need at least 10 observations for volatility modeling."
        raise ValueError(msg)
    return arr


def _ljung_box_squared(
    std_resid: np.ndarray,
    lags: int = 10,
) -> dict[str, Any]:
    """Ljung-Box test on squared standardized residuals.

    Tests whether squared residuals exhibit serial correlation, which
    would indicate remaining ARCH effects not captured by the model.
    """
    sq = std_resid**2
    n = len(sq)
    sq_demeaned = sq - sq.mean()

    acf_vals = np.correlate(sq_demeaned, sq_demeaned, mode="full")
    acf_vals = acf_vals[n - 1 :]
    acf_vals = acf_vals / acf_vals[0]

    nlags = min(lags, n - 1)
    rho = acf_vals[1 : nlags + 1]
    ks = np.arange(1, nlags + 1)
    q_stat = float(n * (n + 2) * np.sum(rho**2 / (n - ks)))
    p_value = float(1 - sp_stats.chi2.cdf(q_stat, df=nlags))

    return {
        "statistic": q_stat,
        "p_value": p_value,
        "lags": nlags,
        "adequate": p_value > 0.05,
    }


def _compute_half_life(persistence: float) -> float:
    """Compute half-life of volatility shocks from persistence parameter.

    Half-life = ln(0.5) / ln(persistence). Returns inf if persistence >= 1.
    """
    if persistence >= 1.0 or persistence <= 0.0:
        return float("inf")
    return float(-np.log(2) / np.log(persistence))


def _build_garch_result(
    fit_result: Any,
    *,
    model_name: str,
    scale: float = 100.0,
) -> dict[str, Any]:
    """Build a standardized result dictionary from an arch fit result.

    Parameters:
        fit_result: The fitted arch model result object.
        model_name: Name of the model (e.g., "GARCH(1,1)").
        scale: The scaling factor applied to returns before fitting.

    Returns:
        Standardized result dictionary.
    """
    params = dict(fit_result.params)
    cond_vol = fit_result.conditional_volatility / scale
    std_resid = fit_result.std_resid

    # Compute persistence from parameters
    alpha_sum = sum(v for k, v in params.items() if k.startswith("alpha"))
    beta_sum = sum(v for k, v in params.items() if k.startswith("beta"))
    gamma_sum = sum(
        v * 0.5 for k, v in params.items() if k.startswith("gamma")
    )
    persistence = alpha_sum + beta_sum + gamma_sum

    omega = params.get("omega", 0.0)
    if persistence < 1.0 and omega > 0:
        uncond_var = omega / (1 - persistence) / (scale**2)
    else:
        uncond_var = float("inf")

    half_life = _compute_half_life(persistence)

    # Ljung-Box on squared standardized residuals
    valid_resid = std_resid[np.isfinite(std_resid)]
    lb = _ljung_box_squared(valid_resid) if len(valid_resid) > 20 else None

    return {
        "model_name": model_name,
        "params": params,
        "conditional_volatility": pd.Series(
            cond_vol,
            index=(
                fit_result.conditional_volatility.index
                if hasattr(fit_result.conditional_volatility, "index")
                else None
            ),
            name="conditional_volatility",
        ),
        "standardized_residuals": std_resid,
        "aic": float(fit_result.aic),
        "bic": float(fit_result.bic),
        "log_likelihood": float(fit_result.loglikelihood),
        "persistence": float(persistence),
        "half_life": half_life,
        "unconditional_variance": float(uncond_var),
        "ljung_box": lb,
        "model": fit_result,
    }


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------


def ewma_volatility(
    returns: pd.Series,
    span: int = 30,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Exponentially weighted moving average volatility.

    The EWMA model of J.P. Morgan's RiskMetrics (1996) assigns
    exponentially decaying weights to past squared returns. The decay
    factor lambda = 1 - 2/(span+1).

    .. math::

        \\sigma_t^2 = \\lambda \\sigma_{t-1}^2
                     + (1 - \\lambda) r_{t-1}^2

    Parameters:
        returns: Return series (not prices).
        span: EWMA span parameter. The decay factor is computed as
            ``lambda = 1 - 2/(span+1)``. RiskMetrics uses span ~= 73
            (lambda = 0.94 for daily data). Default 30.
        annualize: Whether to annualize the volatility by multiplying
            by ``sqrt(periods_per_year)``.
        periods_per_year: Trading periods per year. Use 252 for daily,
            52 for weekly, 12 for monthly.

    Returns:
        EWMA volatility series with same index as input.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> returns = pd.Series(rng.normal(0, 0.01, 500))
        >>> vol = ewma_volatility(returns, span=30)
        >>> vol.iloc[-1] > 0
        True

    Notes:
        RiskMetrics (1996) recommended lambda = 0.94 for daily data
        and lambda = 0.97 for monthly data. The EWMA model is a
        restricted IGARCH(1,1) with zero intercept.

    See Also:
        garch_fit: More flexible parametric volatility model.
        realized_volatility: Non-parametric rolling window estimator.
    """
    var = returns.ewm(span=span).var()
    vol = np.sqrt(var)
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol


# ---------------------------------------------------------------------------
# GARCH(p, q)
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def garch_fit(
    returns: pd.Series | np.ndarray,
    p: int = 1,
    q: int = 1,
    mean: str = "Constant",
    dist: str = "normal",
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit a GARCH(p, q) model to a return series.

    The Generalized Autoregressive Conditional Heteroskedasticity model
    of Bollerslev (1986) captures volatility clustering -- the tendency
    for large moves to follow large moves. The conditional variance is:

    .. math::

        \\sigma_t^2 = \\omega + \\sum_{i=1}^{p} \\alpha_i \\epsilon_{t-i}^2
                     + \\sum_{j=1}^{q} \\beta_j \\sigma_{t-j}^2

    Parameters:
        returns: Return series (not prices). Should be percentage returns
            or log returns, typically in the range [-10, 10].
        p: Order of the GARCH (lagged variance) terms. Higher values
            capture longer volatility memory. Default 1 is standard.
        q: Order of the ARCH (lagged squared residual) terms. Default 1.
        mean: Mean model specification. Options:

            - ``"Constant"``: constant mean (most common)
            - ``"Zero"``: zero mean (for demeaned returns)
            - ``"ARX"``: autoregressive with exogenous variables
        dist: Error distribution. Options:

            - ``"normal"``: Gaussian (fastest, least flexible)
            - ``"t"``: Student's t (captures fat tails)
            - ``"skewt"``: Hansen's skewed t (tails + asymmetry)
            - ``"ged"``: Generalized Error Distribution
        **kwargs: Additional keyword arguments passed to
            ``arch.arch_model()`` (e.g., ``lags`` for ARX mean).

    Returns:
        Dictionary containing:

        - **model_name** (*str*) -- Model identifier, e.g. ``"GARCH(1,1)"``.
        - **params** (*dict*) -- Fitted parameters (omega, alpha, beta,
          plus distribution params if non-normal).
        - **conditional_volatility** (*pd.Series*) -- Time series of
          estimated conditional standard deviation. Multiply by sqrt(252)
          to annualize if returns are daily.
        - **standardized_residuals** (*np.ndarray*) -- Residuals divided
          by conditional std dev. Should be ~i.i.d. if model is correct.
        - **aic** (*float*) -- Akaike Information Criterion (lower = better).
        - **bic** (*float*) -- Bayesian Information Criterion.
        - **log_likelihood** (*float*) -- Maximized log-likelihood.
        - **persistence** (*float*) -- Sum of alpha + beta. Values near 1
          indicate high volatility persistence (IGARCH if exactly 1).
        - **half_life** (*float*) -- Number of periods for a volatility
          shock to decay to half its initial impact.
        - **unconditional_variance** (*float*) -- Long-run variance
          omega / (1 - alpha - beta).
        - **ljung_box** (*dict*) -- Ljung-Box test on squared standardized
          residuals. If p-value > 0.05, model captures vol clustering.
        - **model** -- The fitted ``arch`` model result object for
          further analysis (forecasting, simulation, etc.).

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import garch_fit
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 1, 1000)
        >>> result = garch_fit(returns, p=1, q=1, dist="normal")
        >>> 0 < result['persistence'] < 2
        True

    Notes:
        The GARCH(1,1) model is the workhorse of volatility modeling.
        Persistence (alpha + beta) near 1 is common for financial returns.
        If persistence > 1, consider IGARCH or FIGARCH.

        Returns are internally scaled by 100 before fitting to improve
        numerical stability (the ``arch`` library convention). All outputs
        are rescaled back to the original units.

        Reference: Bollerslev, T. (1986). "Generalized Autoregressive
        Conditional Heteroskedasticity." *Journal of Econometrics*,
        31, 307--327.

    See Also:
        egarch_fit: Asymmetric GARCH capturing leverage effect.
        gjr_garch_fit: Alternative asymmetric specification.
        garch_forecast: Multi-step ahead volatility forecasting.
        news_impact_curve: Visualize asymmetric shock response.
    """
    from arch import arch_model

    ret = _to_returns_array(returns)
    scale = 100.0

    am = arch_model(
        ret * scale,
        vol="GARCH",
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        **kwargs,
    )
    fit = am.fit(disp="off")

    return _build_garch_result(
        fit, model_name=f"GARCH({p},{q})", scale=scale
    )


# ---------------------------------------------------------------------------
# EGARCH
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def egarch_fit(
    returns: pd.Series | np.ndarray,
    p: int = 1,
    q: int = 1,
    mean: str = "Constant",
    dist: str = "normal",
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit an EGARCH(p, q) model (exponential GARCH).

    The EGARCH model of Nelson (1991) captures the *leverage effect* --
    negative shocks increase volatility more than positive shocks of
    equal magnitude. The model specifies the log of conditional variance,
    so positivity of variance is guaranteed without parameter constraints:

    .. math::

        \\ln \\sigma_t^2 = \\omega
            + \\sum_{i=1}^{q} \\alpha_i \\left(
                \\left|z_{t-i}\\right| - E\\left|z_{t-i}\\right|
              \\right)
            + \\sum_{i=1}^{q} \\gamma_i z_{t-i}
            + \\sum_{j=1}^{p} \\beta_j \\ln \\sigma_{t-j}^2

    where :math:`z_t = \\epsilon_t / \\sigma_t` are standardized residuals
    and :math:`\\gamma_i` captures the asymmetric (leverage) response.

    Parameters:
        returns: Return series (not prices).
        p: Order of GARCH (lagged log-variance) terms.
        q: Order of ARCH (asymmetric innovation) terms.
        mean: Mean model specification (``"Constant"``, ``"Zero"``,
            ``"ARX"``).
        dist: Error distribution (``"normal"``, ``"t"``, ``"skewt"``,
            ``"ged"``).
        **kwargs: Additional keyword arguments passed to
            ``arch.arch_model()``.

    Returns:
        Dictionary with the same structure as :func:`garch_fit`.
        The ``params`` dict includes ``gamma`` terms capturing the
        leverage effect. A negative gamma means negative shocks
        increase volatility more than positive shocks.

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import egarch_fit
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 1, 1000)
        >>> result = egarch_fit(returns, p=1, q=1)
        >>> result['conditional_volatility'].iloc[-1] > 0
        True

    Notes:
        The EGARCH model is preferred when:

        1. Leverage effects are important (equity markets).
        2. Parameter positivity constraints are problematic.

        The log specification means persistence is measured by the
        sum of beta coefficients. Unconditional variance may not
        have a closed form depending on the distribution.

        Reference: Nelson, D.B. (1991). "Conditional Heteroskedasticity
        in Asset Returns: A New Approach." *Econometrica*, 59(2),
        347--370.

    See Also:
        garch_fit: Symmetric GARCH model.
        gjr_garch_fit: Alternative asymmetric GARCH.
        news_impact_curve: Compare asymmetric responses across models.
    """
    from arch import arch_model

    ret = _to_returns_array(returns)
    scale = 100.0

    am = arch_model(
        ret * scale,
        vol="EGARCH",
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        **kwargs,
    )
    fit = am.fit(disp="off")

    return _build_garch_result(
        fit, model_name=f"EGARCH({p},{q})", scale=scale
    )


# ---------------------------------------------------------------------------
# GJR-GARCH
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def gjr_garch_fit(
    returns: pd.Series | np.ndarray,
    p: int = 1,
    q: int = 1,
    o: int = 1,
    mean: str = "Constant",
    dist: str = "normal",
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit a GJR-GARCH(p, o, q) model (Glosten-Jagannathan-Runkle).

    The GJR-GARCH model of Glosten, Jagannathan, and Runkle (1993)
    augments the standard GARCH with an indicator function that
    activates on negative shocks, capturing the *leverage effect*:

    .. math::

        \\sigma_t^2 = \\omega
            + \\sum_{i=1}^{q} \\alpha_i \\epsilon_{t-i}^2
            + \\sum_{i=1}^{o} \\gamma_i \\epsilon_{t-i}^2
              I(\\epsilon_{t-i} < 0)
            + \\sum_{j=1}^{p} \\beta_j \\sigma_{t-j}^2

    where :math:`I(\\cdot)` is the indicator function. The gamma
    coefficient captures the extra volatility increase from negative
    shocks (bad news).

    Parameters:
        returns: Return series (not prices).
        p: Order of GARCH (lagged variance) terms.
        q: Order of ARCH (lagged squared residual) terms.
        o: Order of asymmetric (threshold) terms. Default 1.
        mean: Mean model specification.
        dist: Error distribution.
        **kwargs: Additional keyword arguments passed to
            ``arch.arch_model()``.

    Returns:
        Dictionary with the same structure as :func:`garch_fit`.
        The ``params`` dict includes ``gamma[1]`` capturing the
        asymmetric response. Positive gamma means negative shocks
        increase volatility more.

        The persistence for GJR-GARCH is
        alpha + beta + 0.5 * gamma (assuming symmetric distribution).

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import gjr_garch_fit
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 1, 1000)
        >>> result = gjr_garch_fit(returns, p=1, q=1)
        >>> 'gamma[1]' in result['params'] or 'gamma' in str(result['params'])
        True

    Notes:
        The GJR-GARCH is equivalent to TARCH (Threshold ARCH) when
        applied to the variance (not standard deviation). It is the
        default asymmetric model in many risk management applications.

        For stationarity, the persistence
        alpha + beta + 0.5 * gamma < 1 is required.

        Reference: Glosten, L.R., Jagannathan, R., and Runkle, D.E.
        (1993). "On the Relation between the Expected Value and the
        Volatility of the Nominal Excess Return on Stocks."
        *Journal of Finance*, 48(5), 1779--1801.

    See Also:
        egarch_fit: Log-variance based asymmetric model.
        garch_fit: Symmetric baseline GARCH.
        news_impact_curve: Compare asymmetric responses.
    """
    from arch import arch_model

    ret = _to_returns_array(returns)
    scale = 100.0

    am = arch_model(
        ret * scale,
        vol="GARCH",
        p=p,
        o=o,
        q=q,
        dist=dist,
        mean=mean,
        **kwargs,
    )
    fit = am.fit(disp="off")

    return _build_garch_result(
        fit, model_name=f"GJR-GARCH({p},{o},{q})", scale=scale
    )


# ---------------------------------------------------------------------------
# FIGARCH
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def figarch_fit(
    returns: pd.Series | np.ndarray,
    p: int = 1,
    q: int = 1,
    mean: str = "Constant",
    dist: str = "normal",
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit a FIGARCH(p, d, q) model (Fractionally Integrated GARCH).

    The FIGARCH model of Baillie, Bollerslev, and Mikkelsen (1996)
    allows for long memory in the conditional variance process. The
    fractional differencing parameter *d* is estimated from the data
    and captures the slow hyperbolic decay of volatility autocorrelations
    that is frequently observed in financial time series.

    .. math::

        \\sigma_t^2 = \\omega + \\beta \\sigma_{t-1}^2
                     + [1 - \\beta L - (1-L)^d (1 - \\alpha L)]
                       \\epsilon_t^2

    where *d* in (0, 1) is the fractional integration parameter, and
    *L* is the lag operator.

    Parameters:
        returns: Return series (not prices).
        p: Order of GARCH terms (typically 1).
        q: Order of ARCH terms (typically 1).
        mean: Mean model specification.
        dist: Error distribution.
        **kwargs: Additional keyword arguments passed to
            ``arch.arch_model()``.

    Returns:
        Dictionary with the same structure as :func:`garch_fit`.
        The ``params`` dict includes the fractional integration
        parameter ``d``. Values of *d* near 0 imply short memory
        (like GARCH), values near 0.5 imply long memory.

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import figarch_fit
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 1, 1000)
        >>> result = figarch_fit(returns, p=1, q=1)
        >>> result['aic'] < 0 or result['aic'] > 0  # AIC is finite
        True

    Notes:
        FIGARCH nests GARCH (d=0) and IGARCH (d=1). Empirically, most
        financial return series have d in (0.3, 0.5), suggesting long
        memory in volatility.

        The ``arch`` library implements FIGARCH via the ``"FIGARCH"``
        volatility specification. The ``power`` parameter defaults to
        2.0 (standard FIGARCH).

        Reference: Baillie, R.T., Bollerslev, T., and Mikkelsen, H.O.
        (1996). "Fractionally Integrated Generalized Autoregressive
        Conditional Heteroskedasticity." *Journal of Econometrics*,
        74(1), 3--30.

    See Also:
        garch_fit: Standard GARCH without long memory.
        harch_fit: Heterogeneous ARCH as alternative long-memory model.
        volatility_persistence: Measure persistence and half-life.
    """
    from arch import arch_model

    ret = _to_returns_array(returns)
    scale = 100.0

    am = arch_model(
        ret * scale,
        vol="FIGARCH",
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        **kwargs,
    )
    fit = am.fit(disp="off")

    return _build_garch_result(
        fit, model_name=f"FIGARCH({p},d,{q})", scale=scale
    )


# ---------------------------------------------------------------------------
# HARCH
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def harch_fit(
    returns: pd.Series | np.ndarray,
    lags: list[int] | None = None,
    mean: str = "Constant",
    dist: str = "normal",
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit a HARCH model (Heterogeneous ARCH).

    The HARCH model of Mueller et al. (1997) aggregates squared returns
    over multiple time horizons to capture the heterogeneous nature of
    market participants operating at different frequencies (intraday
    traders, daily, weekly, monthly):

    .. math::

        \\sigma_t^2 = \\omega + \\sum_{j=1}^{J} \\alpha_j
            \\left( \\sum_{i=1}^{l_j} \\epsilon_{t-i} \\right)^2 / l_j

    where :math:`l_1 < l_2 < \\ldots < l_J` are the lag lengths
    corresponding to different time horizons.

    Parameters:
        returns: Return series (not prices).
        lags: List of lag lengths representing different time horizons.
            Default is ``[1, 5, 22]`` corresponding to daily, weekly,
            and monthly horizons for daily data.
        mean: Mean model specification.
        dist: Error distribution.
        **kwargs: Additional keyword arguments passed to
            ``arch.arch_model()``.

    Returns:
        Dictionary with the same structure as :func:`garch_fit`.
        The ``params`` dict includes coefficients for each
        aggregation horizon.

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import harch_fit
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 1, 1000)
        >>> result = harch_fit(returns, lags=[1, 5, 22])
        >>> result['conditional_volatility'].iloc[-1] > 0
        True

    Notes:
        HARCH is particularly useful for modeling high-frequency data
        where different market participants operate on vastly different
        time scales. It provides a parsimonious alternative to high-order
        ARCH or GARCH models.

        Reference: Mueller, U.A., Dacorogna, M.M., Dave, R.D.,
        Olsen, R.B., Pictet, O.V., and von Weizsacker, J.E. (1997).
        "Volatilities of Different Time Resolutions." *Journal of
        Empirical Finance*, 4(2-3), 213--239.

    See Also:
        garch_fit: Standard GARCH model.
        figarch_fit: Long-memory GARCH via fractional integration.
    """
    from arch import arch_model

    if lags is None:
        lags = [1, 5, 22]

    ret = _to_returns_array(returns)
    scale = 100.0

    am = arch_model(
        ret * scale,
        vol="HARCH",
        lags=lags,
        dist=dist,
        mean=mean,
        **kwargs,
    )
    fit = am.fit(disp="off")

    return _build_garch_result(
        fit,
        model_name=f"HARCH({lags})",
        scale=scale,
    )


# ---------------------------------------------------------------------------
# GARCH Forecast
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def garch_forecast(
    returns: pd.Series | np.ndarray,
    p: int = 1,
    q: int = 1,
    mean: str = "Constant",
    dist: str = "normal",
    horizon: int = 10,
    method: str = "analytic",
    simulations: int = 1000,
    **kwargs: Any,
) -> dict[str, Any]:
    """Multi-step ahead volatility forecast from a GARCH model.

    Fits a GARCH(p,q) model and produces multi-step forecasts of
    conditional variance, with optional confidence intervals via
    simulation.

    Parameters:
        returns: Return series (not prices).
        p: GARCH lag order.
        q: ARCH lag order.
        mean: Mean model specification.
        dist: Error distribution.
        horizon: Number of steps ahead to forecast. Default 10.
        method: Forecasting method. Options:

            - ``"analytic"``: Closed-form multi-step variance forecast.
              Fastest and exact for GARCH. Does not provide intervals.
            - ``"simulation"``: Monte Carlo simulation of future paths.
              Provides confidence intervals but slower.
        simulations: Number of simulation paths (only used when
            ``method="simulation"``). Default 1000.
        **kwargs: Additional keyword arguments passed to
            ``arch.arch_model()``.

    Returns:
        Dictionary containing:

        - **model_name** (*str*) -- ``"GARCH(p,q) forecast"``.
        - **forecast_variance** (*np.ndarray*) -- Forecasted conditional
          variance for each horizon step, shape ``(horizon,)``.
        - **forecast_volatility** (*np.ndarray*) -- Square root of
          forecast variance, shape ``(horizon,)``.
        - **confidence_intervals** (*dict | None*) -- If method is
          ``"simulation"``, contains ``"lower_5"`` and ``"upper_95"``
          arrays. None for analytic method.
        - **fit_result** (*dict*) -- Full GARCH fit result from
          :func:`garch_fit`.

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import garch_forecast
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 1, 1000)
        >>> fc = garch_forecast(returns, horizon=5)
        >>> len(fc['forecast_volatility']) == 5
        True

    Notes:
        Analytic forecasts assume the model is correctly specified and
        use the recursive formula for multi-step variance:

        .. math::

            E[\\sigma_{T+h}^2 | \\mathcal{F}_T] = \\omega \\frac{1 -
            (\\alpha + \\beta)^h}{1 - \\alpha - \\beta}
            + (\\alpha + \\beta)^h \\sigma_{T+1}^2

        Simulation forecasts draw from the fitted error distribution
        and roll the GARCH recursion forward, providing a distribution
        of future paths.

    See Also:
        garch_fit: Fit the underlying GARCH model.
        volatility_persistence: Analyze shock decay properties.
    """
    from arch import arch_model

    ret = _to_returns_array(returns)
    scale = 100.0

    am = arch_model(
        ret * scale,
        vol="GARCH",
        p=p,
        q=q,
        dist=dist,
        mean=mean,
        **kwargs,
    )
    fit = am.fit(disp="off")

    fit_result = _build_garch_result(
        fit, model_name=f"GARCH({p},{q})", scale=scale
    )

    if method == "simulation":
        forecasts = fit.forecast(
            horizon=horizon,
            method="simulation",
            simulations=simulations,
        )
        fv = forecasts.variance.iloc[-1].values / (scale**2)
        # Gather simulation paths for confidence intervals
        sim_var = forecasts.simulations.variances[-1] / (scale**2)
        lower = np.percentile(sim_var, 5, axis=0)
        upper = np.percentile(sim_var, 95, axis=0)
        ci = {
            "lower_5": np.sqrt(lower),
            "upper_95": np.sqrt(upper),
        }
    else:
        forecasts = fit.forecast(horizon=horizon, method="analytic")
        fv = forecasts.variance.iloc[-1].values / (scale**2)
        ci = None

    return {
        "model_name": f"GARCH({p},{q}) forecast",
        "forecast_variance": fv,
        "forecast_volatility": np.sqrt(fv),
        "confidence_intervals": ci,
        "fit_result": fit_result,
    }


# ---------------------------------------------------------------------------
# DCC-GARCH
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def dcc_fit(
    returns: pd.DataFrame | np.ndarray,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> dict[str, Any]:
    """Fit a DCC-GARCH model for multivariate dynamic correlations.

    The Dynamic Conditional Correlation model of Engle (2002) estimates
    time-varying correlations between multiple asset returns using a
    two-step procedure:

    1. Fit univariate GARCH(p,q) to each series.
    2. Estimate DCC parameters (a, b) via MLE on standardized residuals.

    The DCC correlation dynamics are:

    .. math::

        Q_t = (1 - a - b) \\bar{Q} + a \\, z_{t-1} z_{t-1}^\\top
              + b \\, Q_{t-1}

        R_t = \\text{diag}(Q_t)^{-1/2} \\, Q_t \\,
              \\text{diag}(Q_t)^{-1/2}

    where :math:`\\bar{Q}` is the unconditional correlation matrix
    of standardized residuals.

    Parameters:
        returns: Return data, either a DataFrame with k columns (one per
            asset) or an array of shape ``(T, k)``.
        p: GARCH lag order for each univariate model.
        q: ARCH lag order for each univariate model.
        dist: Error distribution for univariate GARCH fits.

    Returns:
        Dictionary containing:

        - **dcc_params** (*dict*) -- DCC parameters ``{"a": ..., "b": ...}``.
        - **univariate_results** (*list[dict]*) -- Per-asset GARCH fit
          results from :func:`garch_fit`.
        - **conditional_correlations** (*np.ndarray*) -- Array of shape
          ``(T, k, k)`` with time-varying correlation matrices.
        - **conditional_covariances** (*np.ndarray*) -- Array of shape
          ``(T, k, k)`` with time-varying covariance matrices.
        - **conditional_volatilities** (*np.ndarray*) -- Array of shape
          ``(T, k)`` with per-asset conditional volatilities.
        - **standardized_residuals** (*np.ndarray*) -- Array of shape
          ``(T, k)``.
        - **qbar** (*np.ndarray*) -- Unconditional correlation matrix.

    Example:
        >>> import numpy as np, pandas as pd
        >>> from wraquant.vol.models import dcc_fit
        >>> rng = np.random.default_rng(42)
        >>> ret = pd.DataFrame({
        ...     'A': rng.normal(0, 0.01, 500),
        ...     'B': rng.normal(0, 0.015, 500),
        ... })
        >>> result = dcc_fit(ret)
        >>> result['conditional_correlations'].shape == (500, 2, 2)
        True

    Notes:
        This implementation uses the ``arch`` library for the univariate
        GARCH step, then applies the pure scipy/numpy DCC estimation
        from ``wraquant.risk.dcc``.

        The DCC model is widely used in portfolio management,
        value-at-risk computation, and hedging applications where
        time-varying correlations are important.

        Reference: Engle, R. (2002). "Dynamic Conditional Correlation:
        A Simple Class of Multivariate Generalized Autoregressive
        Conditional Heteroskedasticity Models." *Journal of Business
        & Economic Statistics*, 20(3), 339--350.

    See Also:
        garch_fit: Univariate GARCH estimation.
        garch_forecast: Volatility forecasting.
    """
    from arch import arch_model

    if isinstance(returns, pd.DataFrame):
        ret_arr = returns.values.astype(np.float64)
    else:
        ret_arr = np.asarray(returns, dtype=np.float64)

    if ret_arr.ndim != 2 or ret_arr.shape[1] < 2:
        msg = "DCC requires multivariate returns with at least 2 series."
        raise ValueError(msg)

    T, k = ret_arr.shape
    scale = 100.0

    # Step 1: Fit univariate GARCH to each series
    uni_results = []
    cond_vols = np.empty((T, k))
    std_resids = np.empty((T, k))

    for j in range(k):
        am = arch_model(
            ret_arr[:, j] * scale,
            vol="GARCH",
            p=p,
            q=q,
            dist=dist,
            mean="Constant",
        )
        fit = am.fit(disp="off")
        uni_result = _build_garch_result(
            fit, model_name=f"GARCH({p},{q})", scale=scale
        )
        uni_results.append(uni_result)
        cond_vols[:, j] = fit.conditional_volatility / scale
        std_resids[:, j] = fit.std_resid

    # Step 2: Estimate DCC parameters via MLE
    qbar = np.corrcoef(std_resids, rowvar=False)

    def _dcc_neg_loglik(params: np.ndarray) -> float:
        a, b = params
        if a < 0 or b < 0 or a + b >= 1:
            return 1e12
        c = 1 - a - b
        Qt = qbar.copy()
        ll = 0.0
        for t in range(1, T):
            et = std_resids[t - 1].reshape(-1, 1)
            Qt = c * qbar + a * (et @ et.T) + b * Qt
            d = np.sqrt(np.diag(Qt))
            if np.any(d <= 0):
                return 1e12
            D_inv = np.diag(1.0 / d)
            Rt = D_inv @ Qt @ D_inv
            det_Rt = np.linalg.det(Rt)
            if det_Rt <= 0:
                return 1e12
            zt = std_resids[t]
            quad = zt @ np.linalg.solve(Rt, zt) - zt @ zt
            ll += -0.5 * (np.log(det_Rt) + quad)
        return -ll

    res = sp_optimize.minimize(
        _dcc_neg_loglik,
        x0=np.array([0.01, 0.95]),
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8},
    )
    a_hat, b_hat = float(res.x[0]), float(res.x[1])
    a_hat = max(a_hat, 1e-6)
    b_hat = max(b_hat, 1e-6)
    if a_hat + b_hat >= 1.0:
        s = a_hat + b_hat
        a_hat = a_hat / s * 0.999
        b_hat = b_hat / s * 0.999

    # Step 3: Reconstruct time-varying correlations and covariances
    c = 1 - a_hat - b_hat
    Qt = qbar.copy()
    cond_corr = np.empty((T, k, k))
    cond_cov = np.empty((T, k, k))
    cond_corr[0] = qbar.copy()
    D0 = np.diag(cond_vols[0])
    cond_cov[0] = D0 @ qbar @ D0

    for t in range(1, T):
        et = std_resids[t - 1].reshape(-1, 1)
        Qt = c * qbar + a_hat * (et @ et.T) + b_hat * Qt
        d = np.sqrt(np.diag(Qt))
        d = np.where(d <= 0, 1e-10, d)
        D_inv = np.diag(1.0 / d)
        Rt = D_inv @ Qt @ D_inv
        np.clip(Rt, -1, 1, out=Rt)
        np.fill_diagonal(Rt, 1.0)
        cond_corr[t] = Rt
        Dt = np.diag(cond_vols[t])
        cond_cov[t] = Dt @ Rt @ Dt

    return {
        "dcc_params": {"a": a_hat, "b": b_hat},
        "univariate_results": uni_results,
        "conditional_correlations": cond_corr,
        "conditional_covariances": cond_cov,
        "conditional_volatilities": cond_vols,
        "standardized_residuals": std_resids,
        "qbar": qbar,
    }


# ---------------------------------------------------------------------------
# Realized GARCH
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def realized_garch(
    returns: pd.Series | np.ndarray,
    realized_vol: pd.Series | np.ndarray,
    p: int = 1,
    q: int = 1,
    mean: str = "Constant",
    dist: str = "normal",
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit a Realized GARCH model augmented with realized volatility.

    The Realized GARCH model of Hansen, Huang, and Shek (2012) augments
    the standard GARCH with a measurement equation linking the conditional
    variance to a realized volatility measure, improving forecasting
    performance compared to standard GARCH.

    This implementation uses the ``arch`` library's ARX mean model with
    lagged realized volatility as an exogenous regressor, capturing the
    predictive information in the realized measure.

    Parameters:
        returns: Return series (not prices).
        realized_vol: Corresponding realized volatility measure (must
            have the same length as *returns*). Can be any realized
            measure: realized variance, Parkinson, Garman-Klass, etc.
        p: GARCH lag order.
        q: ARCH lag order.
        mean: Mean model specification. Default ``"Constant"`` uses
            a constant plus the realized vol regressor.
        dist: Error distribution.
        **kwargs: Additional keyword arguments passed to
            ``arch.arch_model()``.

    Returns:
        Dictionary with the same structure as :func:`garch_fit`,
        plus:

        - **realized_vol_used** (*np.ndarray*) -- The realized volatility
          series used in the model.

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import realized_garch
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 0.01, 500)
        >>> rv = np.abs(returns).rolling(20).mean() if hasattr(returns, 'rolling') else np.convolve(np.abs(returns), np.ones(20)/20, mode='same')
        >>> # result = realized_garch(returns, rv)  # doctest: +SKIP

    Notes:
        The realized measure provides information beyond what the
        GARCH filter can extract from squared returns alone, leading
        to improved volatility forecasts -- especially at shorter
        horizons.

        Reference: Hansen, P.R., Huang, Z., and Shek, H.H. (2012).
        "Realized GARCH: A Joint Model for Returns and Realized Measures
        of Volatility." *Journal of Applied Econometrics*, 27(6),
        877--906.

    See Also:
        garch_fit: Standard GARCH without realized measures.
        garch_forecast: Multi-step forecasting.
    """
    from arch import arch_model

    ret = _to_returns_array(returns)
    rv = np.asarray(realized_vol, dtype=np.float64).ravel()

    if len(rv) != len(ret):
        msg = (
            f"returns and realized_vol must have the same length, "
            f"got {len(ret)} and {len(rv)}."
        )
        raise ValueError(msg)

    scale = 100.0

    # Construct lagged realized vol as exogenous variable
    rv_lagged = np.empty_like(rv)
    rv_lagged[0] = rv[0]
    rv_lagged[1:] = rv[:-1]

    try:
        am = arch_model(
            ret * scale,
            vol="GARCH",
            p=p,
            q=q,
            mean="ARX",
            lags=0,
            dist=dist,
            x=pd.DataFrame({"rv_lag": rv_lagged * scale}),
            **kwargs,
        )
        fit = am.fit(disp="off")
    except Exception:
        # Fallback to constant mean if ARX fails
        am = arch_model(
            ret * scale,
            vol="GARCH",
            p=p,
            q=q,
            mean="Constant",
            dist=dist,
            **kwargs,
        )
        fit = am.fit(disp="off")

    result = _build_garch_result(
        fit, model_name=f"RealizedGARCH({p},{q})", scale=scale
    )
    result["realized_vol_used"] = rv
    return result


# ---------------------------------------------------------------------------
# News Impact Curve
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def news_impact_curve(
    returns: pd.Series | np.ndarray,
    model_type: str = "GARCH",
    p: int = 1,
    q: int = 1,
    n_points: int = 100,
    shock_range: float = 3.0,
    dist: str = "normal",
) -> dict[str, Any]:
    """Compute the news impact curve for a fitted GARCH-family model.

    The news impact curve shows how the next-period conditional variance
    responds to shocks of different sizes and signs. For symmetric models
    (GARCH), the curve is a parabola centered at zero. For asymmetric
    models (EGARCH, GJR-GARCH), the curve is steeper for negative shocks,
    visualizing the *leverage effect*.

    Parameters:
        returns: Return series for model estimation.
        model_type: Type of GARCH model. Options:

            - ``"GARCH"``: Standard symmetric model.
            - ``"EGARCH"``: Nelson's exponential GARCH.
            - ``"GJR"``: GJR-GARCH (Glosten-Jagannathan-Runkle).
        p: GARCH lag order.
        q: ARCH lag order.
        n_points: Number of shock values to evaluate. Default 100.
        shock_range: Range of shocks in units of unconditional standard
            deviation. Default 3.0 (evaluates from -3*sigma to +3*sigma).
        dist: Error distribution for the model fit.

    Returns:
        Dictionary containing:

        - **shocks** (*np.ndarray*) -- Array of shock values (epsilon)
          evaluated, shape ``(n_points,)``.
        - **conditional_variance** (*np.ndarray*) -- Corresponding
          next-period conditional variance for each shock.
        - **model_type** (*str*) -- Type of model fitted.
        - **params** (*dict*) -- Fitted model parameters.

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import news_impact_curve
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 1, 1000)
        >>> nic = news_impact_curve(returns, model_type="GJR")
        >>> nic['shocks'].shape == nic['conditional_variance'].shape
        True

    Notes:
        The news impact curve was introduced by Engle and Ng (1993) as a
        diagnostic tool for comparing the asymmetric response of different
        volatility models to news (shocks).

        For GARCH(1,1): :math:`\\sigma_{t+1}^2 = \\omega + \\alpha
        \\epsilon_t^2 + \\beta \\sigma_t^2` (symmetric parabola)

        For GJR-GARCH: the curve has a steeper slope for
        :math:`\\epsilon_t < 0` when :math:`\\gamma > 0`.

        Reference: Engle, R.F. and Ng, V.K. (1993). "Measuring and
        Testing the Impact of News on Volatility." *Journal of Finance*,
        48(5), 1749--1778.

    See Also:
        garch_fit: Symmetric GARCH estimation.
        egarch_fit: Asymmetric EGARCH estimation.
        gjr_garch_fit: GJR-GARCH estimation.
    """
    from arch import arch_model

    ret = _to_returns_array(returns)
    scale = 100.0
    scaled_ret = ret * scale

    # Fit the requested model
    model_type_upper = model_type.upper()
    if model_type_upper == "GJR":
        am = arch_model(
            scaled_ret,
            vol="GARCH",
            p=p,
            o=1,
            q=q,
            dist=dist,
            mean="Constant",
        )
    elif model_type_upper == "EGARCH":
        am = arch_model(
            scaled_ret,
            vol="EGARCH",
            p=p,
            q=q,
            dist=dist,
            mean="Constant",
        )
    else:
        am = arch_model(
            scaled_ret,
            vol="GARCH",
            p=p,
            q=q,
            dist=dist,
            mean="Constant",
        )

    fit = am.fit(disp="off")
    params = dict(fit.params)

    # Generate shock range
    uncond_std = np.std(scaled_ret)
    shocks = np.linspace(
        -shock_range * uncond_std,
        shock_range * uncond_std,
        n_points,
    )

    # Compute NIC based on model type
    omega = params.get("omega", 0.0)

    if model_type_upper == "EGARCH":
        # EGARCH: log(sigma^2) = omega + alpha(|z| - E|z|) + gamma*z + beta*log(sigma^2)
        # At unconditional variance, sigma^2 = exp(omega / (1-beta))
        beta_val = params.get("beta[1]", 0.0)
        alpha_val = params.get("alpha[1]", 0.0)
        gamma_val = params.get("gamma[1]", 0.0)

        # Unconditional log-variance
        if abs(1 - beta_val) > 1e-10:
            uncond_log_var = omega / (1 - beta_val)
        else:
            uncond_log_var = np.log(np.var(scaled_ret))

        # News impact: at unconditional variance level
        z = shocks / np.sqrt(np.exp(uncond_log_var))
        expected_abs_z = np.sqrt(2 / np.pi)  # E[|z|] for standard normal
        log_var = (
            omega
            + alpha_val * (np.abs(z) - expected_abs_z)
            + gamma_val * z
            + beta_val * uncond_log_var
        )
        cond_var = np.exp(log_var)
    elif model_type_upper == "GJR":
        # GJR: sigma^2 = omega + alpha*eps^2 + gamma*eps^2*I(eps<0) + beta*sigma^2
        alpha_val = params.get("alpha[1]", 0.0)
        beta_val = params.get("beta[1]", 0.0)
        gamma_val = params.get("gamma[1]", 0.0)

        # At unconditional variance level
        persistence = alpha_val + beta_val + 0.5 * gamma_val
        if persistence < 1.0 and omega > 0:
            uncond_var = omega / (1 - persistence)
        else:
            uncond_var = np.var(scaled_ret)

        indicator = (shocks < 0).astype(float)
        cond_var = (
            omega
            + alpha_val * shocks**2
            + gamma_val * shocks**2 * indicator
            + beta_val * uncond_var
        )
    else:
        # Standard GARCH: sigma^2 = omega + alpha*eps^2 + beta*sigma^2
        alpha_val = params.get("alpha[1]", 0.0)
        beta_val = params.get("beta[1]", 0.0)

        persistence = alpha_val + beta_val
        if persistence < 1.0 and omega > 0:
            uncond_var = omega / (1 - persistence)
        else:
            uncond_var = np.var(scaled_ret)

        cond_var = omega + alpha_val * shocks**2 + beta_val * uncond_var

    # Rescale back
    return {
        "shocks": shocks / scale,
        "conditional_variance": cond_var / (scale**2),
        "model_type": model_type_upper,
        "params": params,
    }


# ---------------------------------------------------------------------------
# Volatility Persistence
# ---------------------------------------------------------------------------


def volatility_persistence(
    params: dict[str, Any] | None = None,
    *,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    omega: float | None = None,
) -> dict[str, float]:
    """Compute volatility persistence metrics from GARCH parameters.

    Computes the half-life of volatility shocks, the persistence
    parameter, and the unconditional (long-run) variance implied by
    the GARCH model.

    Parameters:
        params: Dictionary of fitted GARCH parameters (e.g., from
            :func:`garch_fit`). If provided, alpha, beta, gamma, and
            omega are extracted automatically.
        alpha: Sum of ARCH coefficients. Overrides ``params`` if both
            are provided.
        beta: Sum of GARCH coefficients. Overrides ``params`` if both
            are provided.
        gamma: Sum of asymmetric (GJR) coefficients. Default 0.
        omega: GARCH intercept. Required for unconditional variance.

    Returns:
        Dictionary containing:

        - **persistence** (*float*) -- alpha + beta + 0.5*gamma. Values
          near 1 imply very persistent volatility.
        - **half_life** (*float*) -- Periods for a shock to decay by 50%.
          ``ln(0.5) / ln(persistence)``.
        - **unconditional_variance** (*float*) -- Long-run variance
          ``omega / (1 - persistence)``. Infinity if persistence >= 1.
        - **unconditional_volatility** (*float*) -- Square root of
          unconditional variance.
        - **mean_reversion_speed** (*float*) -- 1 - persistence. Higher
          values mean faster reversion.

    Example:
        >>> from wraquant.vol.models import volatility_persistence
        >>> vp = volatility_persistence(alpha=0.05, beta=0.93, omega=0.01)
        >>> vp['persistence']
        0.98
        >>> vp['half_life'] > 30
        True

    Notes:
        For standard GARCH, persistence = alpha + beta.
        For GJR-GARCH, persistence = alpha + beta + 0.5 * gamma.
        For EGARCH, persistence is the sum of beta coefficients.

        A persistence > 1 implies non-stationarity (IGARCH behavior).
        Most equity series have persistence in [0.95, 0.999].

    See Also:
        garch_fit: Estimate GARCH parameters.
        garch_forecast: Use persistence for multi-step forecasting.
    """
    if params is not None:
        if alpha is None:
            alpha = sum(
                v for k, v in params.items() if k.startswith("alpha")
            )
        if beta is None:
            beta = sum(
                v for k, v in params.items() if k.startswith("beta")
            )
        if gamma is None:
            gamma = sum(
                v for k, v in params.items() if k.startswith("gamma")
            )
        if omega is None:
            omega = params.get("omega", 0.0)

    alpha = alpha or 0.0
    beta = beta or 0.0
    gamma = gamma or 0.0
    omega = omega or 0.0

    persistence = alpha + beta + 0.5 * gamma
    half_life = _compute_half_life(persistence)

    if persistence < 1.0 and omega > 0:
        uncond_var = omega / (1 - persistence)
    else:
        uncond_var = float("inf")

    uncond_vol = np.sqrt(uncond_var) if np.isfinite(uncond_var) else float("inf")

    return {
        "persistence": float(persistence),
        "half_life": half_life,
        "unconditional_variance": float(uncond_var),
        "unconditional_volatility": float(uncond_vol),
        "mean_reversion_speed": float(1 - persistence),
    }


# ---------------------------------------------------------------------------
# Hawkes Process
# ---------------------------------------------------------------------------


def hawkes_process(
    events: np.ndarray,
    *,
    max_time: float | None = None,
    n_iter: int = 500,
) -> dict[str, Any]:
    """Fit a univariate Hawkes (self-exciting) point process via MLE.

    The Hawkes process models *volatility clustering* as a self-exciting
    point process where each event temporarily increases the intensity
    (rate) of future events. This is directly analogous to how large
    price moves tend to cluster in financial markets.

    The intensity (conditional rate) is:

    .. math::

        \\lambda(t) = \\mu + \\sum_{t_i < t} \\alpha \\exp(-\\beta (t - t_i))

    where :math:`\\mu` is the background intensity, :math:`\\alpha` is the
    jump size per event, and :math:`\\beta` is the exponential decay rate.

    Parameters:
        events: Array of event times (e.g., times of large returns or
            trades). Must be sorted in ascending order.
        max_time: End of observation window. If None, uses
            ``max(events) * 1.01``.
        n_iter: Maximum iterations for the optimizer. Default 500.

    Returns:
        Dictionary containing:

        - **mu** (*float*) -- Background (baseline) intensity.
        - **alpha** (*float*) -- Excitation magnitude. Larger alpha means
          each event causes a bigger spike in future event intensity.
        - **beta** (*float*) -- Decay rate. Larger beta means faster
          return to baseline intensity after an event.
        - **branching_ratio** (*float*) -- alpha / beta. Must be < 1
          for the process to be stationary. Values near 1 indicate
          strong self-excitation (volatility clustering).
        - **half_life** (*float*) -- Time for excitation to decay by 50%:
          ``ln(2) / beta``.
        - **log_likelihood** (*float*) -- Maximized log-likelihood.
        - **intensity** (*np.ndarray*) -- Estimated intensity function
          evaluated at each event time.

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import hawkes_process
        >>> rng = np.random.default_rng(42)
        >>> # Simulate clustered events
        >>> events = np.sort(rng.exponential(1, 100).cumsum())
        >>> result = hawkes_process(events)
        >>> result['mu'] > 0
        True

    Notes:
        The branching ratio alpha/beta controls the degree of
        self-excitation. A ratio near 1 means the process is barely
        stationary and events strongly cluster. In financial markets,
        branching ratios of 0.5-0.8 are common for trade arrivals.

        The Hawkes process is equivalent to an inhomogeneous Poisson
        process whose intensity depends on the history of past events.
        It was originally developed for earthquake modeling (Hawkes, 1971)
        and has been widely applied to high-frequency finance.

        Reference: Hawkes, A.G. (1971). "Spectra of some self-exciting
        and mutually exciting point processes." *Biometrika*, 58(1),
        83--90.

    See Also:
        stochastic_vol_sv: Continuous-time stochastic volatility model.
        gaussian_mixture_vol: Regime-based volatility clustering.
    """
    events = np.asarray(events, dtype=np.float64)
    events = np.sort(events)
    n = len(events)
    if n < 5:
        msg = "Need at least 5 events for Hawkes process estimation."
        raise ValueError(msg)

    if max_time is None:
        max_time = events[-1] * 1.01

    def _neg_loglik(params: np.ndarray) -> float:
        mu, alpha, beta = params
        if mu <= 0 or alpha <= 0 or beta <= 0:
            return 1e12
        if alpha / beta >= 1.0:
            return 1e12

        # Log-likelihood for exponential Hawkes process
        ll = 0.0
        A = 0.0  # recursive intensity contribution

        for i in range(n):
            if i > 0:
                dt = events[i] - events[i - 1]
                A = np.exp(-beta * dt) * (A + 1)

            lam_i = mu + alpha * A
            if lam_i <= 0:
                return 1e12
            ll += np.log(lam_i)

        # Compensator: integral of lambda over [0, max_time]
        compensator = mu * max_time
        for i in range(n):
            compensator += (alpha / beta) * (
                1 - np.exp(-beta * (max_time - events[i]))
            )

        ll -= compensator
        return -ll

    # Initial parameter estimate
    avg_rate = n / max_time
    x0 = np.array([avg_rate * 0.5, avg_rate * 0.3, 1.0])

    res = sp_optimize.minimize(
        _neg_loglik,
        x0,
        method="Nelder-Mead",
        options={"maxiter": n_iter, "xatol": 1e-8, "fatol": 1e-8},
    )

    mu, alpha, beta = res.x
    mu = max(float(mu), 1e-10)
    alpha = max(float(alpha), 1e-10)
    beta = max(float(beta), 1e-10)

    # Compute intensity at each event time
    intensity = np.empty(n)
    A = 0.0
    for i in range(n):
        if i > 0:
            dt = events[i] - events[i - 1]
            A = np.exp(-beta * dt) * (A + 1)
        intensity[i] = mu + alpha * A

    branching = alpha / beta

    return {
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": float(branching),
        "half_life": float(np.log(2) / beta),
        "log_likelihood": float(-res.fun),
        "intensity": intensity,
    }


# ---------------------------------------------------------------------------
# Stochastic Volatility (particle filter / quasi-MLE)
# ---------------------------------------------------------------------------


def stochastic_vol_sv(
    returns: pd.Series | np.ndarray,
    n_particles: int = 500,
    n_iter: int = 20,
) -> dict[str, Any]:
    """Fit a basic stochastic volatility model via particle filter.

    The discrete-time stochastic volatility (SV) model treats
    log-volatility as a latent AR(1) process:

    .. math::

        y_t &= \\exp(h_t / 2) \\, \\epsilon_t, \\quad
            \\epsilon_t \\sim N(0, 1) \\\\
        h_t &= \\mu + \\phi (h_{t-1} - \\mu) + \\sigma_\\eta \\eta_t,
            \\quad \\eta_t \\sim N(0, 1)

    where :math:`h_t` is the log-variance, :math:`\\phi` controls
    persistence, :math:`\\mu` is the long-run mean of log-variance,
    and :math:`\\sigma_\\eta` is the volatility of volatility.

    Parameters:
        returns: Return series (not prices).
        n_particles: Number of particles for the bootstrap particle
            filter. More particles = more accurate but slower.
            Default 500.
        n_iter: Number of iterations for parameter learning via
            particle marginal MH or iterative filtering. Default 20.

    Returns:
        Dictionary containing:

        - **mu** (*float*) -- Long-run mean of log-variance.
        - **phi** (*float*) -- Persistence of log-variance AR(1).
          Values near 1 indicate very persistent volatility.
        - **sigma_eta** (*float*) -- Volatility of volatility.
        - **filtered_volatility** (*np.ndarray*) -- Filtered estimate
          of exp(h_t/2), the conditional standard deviation.
        - **log_variance** (*np.ndarray*) -- Filtered log-variance h_t.
        - **log_likelihood** (*float*) -- Approximate log-likelihood
          from the particle filter.

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import stochastic_vol_sv
        >>> rng = np.random.default_rng(42)
        >>> returns = rng.normal(0, 0.01, 500)
        >>> result = stochastic_vol_sv(returns, n_particles=200, n_iter=5)
        >>> result['phi'] > 0
        True

    Notes:
        The SV model is an alternative to GARCH that treats volatility
        as an unobserved latent process rather than a deterministic
        function of past returns. This is arguably more realistic but
        harder to estimate.

        This implementation uses a bootstrap particle filter (sequential
        Monte Carlo) for likelihood evaluation and a simple grid-based
        parameter search. For production Bayesian SV estimation, consider
        PyMC or Stan.

        Reference: Taylor, S.J. (1982). "Financial Returns Modelled by
        the Product of Two Stochastic Processes." In *Time Series
        Analysis: Theory and Practice 1*, O.D. Anderson (ed.), 203--226.

    See Also:
        garch_fit: Deterministic volatility filtering (GARCH).
        gaussian_mixture_vol: Regime-switching volatility.
        hawkes_process: Point process for event clustering.
    """
    ret = _to_returns_array(returns)
    T = len(ret)

    # Avoid log(0) issues
    ret_safe = np.where(ret == 0, 1e-10, ret)

    def _particle_filter(
        mu: float,
        phi: float,
        sigma_eta: float,
        rng: np.random.Generator,
    ) -> tuple[float, np.ndarray]:
        """Run bootstrap particle filter, return (log-lik, filtered h)."""
        N = n_particles
        # Initialize particles from stationary distribution
        if abs(phi) < 1.0:
            h_std = sigma_eta / np.sqrt(1 - phi**2)
        else:
            h_std = sigma_eta * 5
        particles = rng.normal(mu, h_std, N)
        weights = np.ones(N) / N

        ll = 0.0
        filtered_h = np.empty(T)

        for t in range(T):
            # Weight update: p(y_t | h_t) = N(0, exp(h_t))
            log_vol = particles / 2.0
            log_w = (
                -log_vol
                - 0.5 * ret_safe[t] ** 2 / np.exp(particles)
            )
            log_w -= np.max(log_w)  # numerical stability
            weights = np.exp(log_w)
            w_sum = np.sum(weights)
            if w_sum <= 0:
                weights = np.ones(N) / N
                w_sum = 1.0
            else:
                weights /= w_sum
                ll += np.log(w_sum / N)

            # Filtered estimate
            filtered_h[t] = np.sum(weights * particles)

            # Resample
            indices = rng.choice(N, size=N, p=weights)
            particles = particles[indices]

            # Propagate
            particles = (
                mu + phi * (particles - mu)
                + sigma_eta * rng.normal(0, 1, N)
            )

        return ll, filtered_h

    # Parameter search via grid (simple quasi-MLE)
    rng = np.random.default_rng(42)

    # Use log-squared returns as initial estimate of h
    log_y2 = np.log(ret_safe**2 + 1e-20)
    mu_init = float(np.mean(log_y2))

    best_ll = -np.inf
    best_params = (mu_init, 0.95, 0.1)
    best_h = np.full(T, mu_init)

    # Coarse grid search
    for phi_c in [0.8, 0.9, 0.95, 0.98]:
        for sig_c in [0.05, 0.1, 0.2, 0.5]:
            for _ in range(n_iter):
                ll, h = _particle_filter(mu_init, phi_c, sig_c, rng)
                if ll > best_ll:
                    best_ll = ll
                    best_params = (mu_init, phi_c, sig_c)
                    best_h = h.copy()

    mu_hat, phi_hat, sigma_eta_hat = best_params

    return {
        "mu": float(mu_hat),
        "phi": float(phi_hat),
        "sigma_eta": float(sigma_eta_hat),
        "filtered_volatility": np.exp(best_h / 2),
        "log_variance": best_h,
        "log_likelihood": float(best_ll),
    }


# ---------------------------------------------------------------------------
# Gaussian Mixture Vol
# ---------------------------------------------------------------------------


@requires_extra("ml")
def gaussian_mixture_vol(
    returns: pd.Series | np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit a Gaussian Mixture Model for regime-dependent volatility.

    Models the return distribution as a mixture of Gaussian components,
    each representing a different volatility regime. This captures
    the empirical observation that returns alternate between calm
    (low-vol) and turbulent (high-vol) periods.

    .. math::

        f(r_t) = \\sum_{k=1}^{K} \\pi_k \\,
            \\mathcal{N}(r_t; \\mu_k, \\sigma_k^2)

    Parameters:
        returns: Return series (not prices).
        n_components: Number of Gaussian components (regimes).
            Default 2 (low-vol and high-vol). Use 3 for an additional
            crisis regime.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:

        - **weights** (*np.ndarray*) -- Mixing weights (pi_k) for each
          component, shape ``(n_components,)``.
        - **means** (*np.ndarray*) -- Mean return in each regime.
        - **volatilities** (*np.ndarray*) -- Volatility (std dev) in
          each regime, sorted from lowest to highest.
        - **regime_probabilities** (*np.ndarray*) -- Posterior probability
          of each regime for each observation, shape ``(T, n_components)``.
        - **regime_labels** (*np.ndarray*) -- Most likely regime for
          each observation, shape ``(T,)``.
        - **aic** (*float*) -- Akaike Information Criterion.
        - **bic** (*float*) -- Bayesian Information Criterion.
        - **model** -- Fitted sklearn GaussianMixture object.

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import gaussian_mixture_vol
        >>> rng = np.random.default_rng(42)
        >>> low_vol = rng.normal(0, 0.005, 300)
        >>> high_vol = rng.normal(0, 0.02, 200)
        >>> returns = np.concatenate([low_vol, high_vol])
        >>> result = gaussian_mixture_vol(returns, n_components=2)
        >>> len(result['volatilities']) == 2
        True

    Notes:
        The GMM approach is simpler than hidden Markov models but does
        not model transition dynamics between regimes. For Markov
        regime-switching, see ``wraquant.regimes``.

        The number of components can be selected via AIC/BIC by fitting
        models with different *n_components* values.

    See Also:
        stochastic_vol_sv: Continuous latent volatility process.
        garch_fit: GARCH-based volatility modeling.
    """
    from sklearn.mixture import GaussianMixture

    ret = _to_returns_array(returns).reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        max_iter=200,
        n_init=5,
    )
    gmm.fit(ret)

    means = gmm.means_.ravel()
    vols = np.sqrt(gmm.covariances_.ravel())
    weights = gmm.weights_

    # Sort components by volatility (ascending)
    order = np.argsort(vols)
    means = means[order]
    vols = vols[order]
    weights = weights[order]

    probs = gmm.predict_proba(ret)
    probs = probs[:, order]
    labels = np.argmax(probs, axis=1)

    return {
        "weights": weights,
        "means": means,
        "volatilities": vols,
        "regime_probabilities": probs,
        "regime_labels": labels,
        "aic": float(gmm.aic(ret)),
        "bic": float(gmm.bic(ret)),
        "model": gmm,
    }


# ---------------------------------------------------------------------------
# SVI Implied Vol Surface
# ---------------------------------------------------------------------------


def vol_surface_svi(
    strikes: np.ndarray,
    forward: float,
    total_implied_var: np.ndarray,
    time_to_expiry: float,
) -> dict[str, Any]:
    """Fit the SVI parameterization to an implied volatility smile.

    The Stochastic Volatility Inspired (SVI) model of Gatheral (2004)
    parameterizes the total implied variance as a function of
    log-moneyness:

    .. math::

        w(k) = a + b \\left( \\rho (k - m)
               + \\sqrt{(k - m)^2 + \\sigma^2} \\right)

    where :math:`k = \\ln(K/F)` is log-moneyness, and the five
    parameters (a, b, rho, m, sigma) control the level, slope,
    curvature, translation, and minimum variance of the smile.

    Parameters:
        strikes: Array of strike prices.
        forward: Forward price of the underlying.
        total_implied_var: Array of total implied variances (IV^2 * T)
            corresponding to each strike. Same length as *strikes*.
        time_to_expiry: Time to expiry in years.

    Returns:
        Dictionary containing:

        - **params** (*dict*) -- Fitted SVI parameters:
          ``a`` (level), ``b`` (slope), ``rho`` (rotation, -1 to 1),
          ``m`` (translation), ``sigma`` (smoothing, > 0).
        - **fitted_total_var** (*np.ndarray*) -- Fitted total implied
          variance at each strike.
        - **fitted_iv** (*np.ndarray*) -- Fitted implied volatility
          at each strike (sqrt(w / T)).
        - **residuals** (*np.ndarray*) -- Fitting residuals.
        - **rmse** (*float*) -- Root mean squared error of the fit.

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import vol_surface_svi
        >>> strikes = np.array([90, 95, 100, 105, 110], dtype=float)
        >>> forward = 100.0
        >>> iv = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
        >>> T = 0.25
        >>> total_var = iv**2 * T
        >>> result = vol_surface_svi(strikes, forward, total_var, T)
        >>> result['rmse'] < 0.01
        True

    Notes:
        The SVI parameterization is widely used by practitioners for
        interpolating and extrapolating the volatility smile. It
        naturally produces realistic smile shapes and can be calibrated
        to satisfy no-arbitrage constraints (Gatheral and Jacquier, 2014).

        Key properties:

        - ``rho < 0``: negative skew (typical for equities).
        - ``b`` controls the overall slope of the wings.
        - ``sigma`` controls the curvature near the money.

        Reference: Gatheral, J. (2004). "A Parsimonious Arbitrage-Free
        Implied Volatility Parameterization." Presentation at Global
        Derivatives & Risk Management, Madrid.

    See Also:
        variance_risk_premium: Compute VRP from implied and realized vol.
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    total_implied_var = np.asarray(total_implied_var, dtype=np.float64)

    if len(strikes) != len(total_implied_var):
        msg = "strikes and total_implied_var must have the same length."
        raise ValueError(msg)

    k = np.log(strikes / forward)  # log-moneyness

    def _svi(params: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = params
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))

    def _objective(params: np.ndarray) -> float:
        a, b, rho, m, sigma = params
        if b < 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e12
        fitted = _svi(params)
        if np.any(fitted < 0):
            return 1e12
        return float(np.sum((fitted - total_implied_var) ** 2))

    # Initialize with reasonable guesses
    atm_var = float(np.interp(0, k, total_implied_var))
    x0 = np.array([atm_var, 0.1, -0.3, 0.0, 0.1])

    res = sp_optimize.minimize(
        _objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-10},
    )

    a, b, rho, m, sigma = res.x
    fitted_var = _svi(res.x)
    residuals = total_implied_var - fitted_var
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # Convert to implied vol
    fitted_iv = np.sqrt(np.maximum(fitted_var, 0) / time_to_expiry)

    return {
        "params": {
            "a": float(a),
            "b": float(b),
            "rho": float(rho),
            "m": float(m),
            "sigma": float(sigma),
        },
        "fitted_total_var": fitted_var,
        "fitted_iv": fitted_iv,
        "residuals": residuals,
        "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# Variance Risk Premium
# ---------------------------------------------------------------------------


def variance_risk_premium(
    implied_vol: pd.Series | np.ndarray,
    realized_vol: pd.Series | np.ndarray,
    annualized: bool = True,
) -> dict[str, Any]:
    """Compute the variance risk premium (VRP).

    The variance risk premium is the difference between risk-neutral
    (implied) and physical (realized) variance. A positive VRP indicates
    that investors pay a premium for volatility protection, which is
    the empirical norm for equity indices.

    .. math::

        VRP_t = IV_t^2 - RV_t^2

    A consistently positive VRP is the economic basis for volatility
    selling strategies (e.g., selling straddles, variance swaps).

    Parameters:
        implied_vol: Implied volatility series (e.g., VIX / 100 for
            S&P 500). Should be in the same units as *realized_vol*
            (both annualized or both not).
        realized_vol: Realized volatility series. Same length and
            alignment as *implied_vol*.
        annualized: Whether the input volatilities are annualized.
            If True, the VRP is in annualized variance units.

    Returns:
        Dictionary containing:

        - **vrp** (*np.ndarray*) -- Variance risk premium time series.
          Positive values = implied > realized (normal).
        - **mean_vrp** (*float*) -- Average VRP over the sample.
        - **vrp_ratio** (*np.ndarray*) -- IV / RV ratio. Values > 1
          indicate implied exceeds realized.
        - **pct_positive** (*float*) -- Fraction of observations where
          VRP > 0. Typically 60--80% for equity indices.
        - **vol_spread** (*np.ndarray*) -- Simple difference IV - RV
          (in volatility, not variance units).

    Example:
        >>> import numpy as np
        >>> from wraquant.vol.models import variance_risk_premium
        >>> iv = np.array([0.20, 0.22, 0.18, 0.25, 0.19])
        >>> rv = np.array([0.15, 0.17, 0.16, 0.20, 0.14])
        >>> result = variance_risk_premium(iv, rv)
        >>> result['mean_vrp'] > 0
        True

    Notes:
        The VRP is one of the most robust risk premiums in financial
        markets. It reflects the cost of insurance against volatility
        spikes and is related to jump risk, model uncertainty, and
        aggregate risk aversion.

        Reference: Bollerslev, T., Tauchen, G., and Zhou, H. (2009).
        "Expected Stock Returns and Variance Risk Premia."
        *Review of Financial Studies*, 22(11), 4463--4492.

    See Also:
        garch_fit: Estimate realized conditional volatility.
        vol_surface_svi: Implied volatility surface fitting.
    """
    iv = np.asarray(implied_vol, dtype=np.float64).ravel()
    rv = np.asarray(realized_vol, dtype=np.float64).ravel()

    if len(iv) != len(rv):
        msg = (
            f"implied_vol and realized_vol must have the same length, "
            f"got {len(iv)} and {len(rv)}."
        )
        raise ValueError(msg)

    iv_var = iv**2
    rv_var = rv**2
    vrp = iv_var - rv_var

    # Avoid division by zero in ratio
    rv_safe = np.where(rv > 0, rv, 1e-10)

    valid = np.isfinite(vrp)
    pct_pos = float(np.mean(vrp[valid] > 0)) if valid.any() else 0.0

    return {
        "vrp": vrp,
        "mean_vrp": float(np.mean(vrp[valid])) if valid.any() else 0.0,
        "vrp_ratio": iv / rv_safe,
        "pct_positive": pct_pos,
        "vol_spread": iv - rv,
    }
