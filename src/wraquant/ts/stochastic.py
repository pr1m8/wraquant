"""Stochastic process forecasting for financial time series.

Provides parameter estimation and forecasting for continuous-time
stochastic processes commonly used in quantitative finance:

- **Ornstein-Uhlenbeck (OU)**: mean-reverting diffusion for spreads,
  rates, and volatility.
- **Merton Jump-Diffusion**: GBM with Poisson jumps for fat-tailed
  asset returns.
- **Regime-Switching**: separate models per hidden regime, blended by
  regime probabilities.
- **Vector Autoregression (VAR)**: multi-asset linear forecasting with
  impulse response and variance decomposition.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats as sp_stats

from wraquant.core.decorators import requires_extra


# ---------------------------------------------------------------------------
# ornstein_uhlenbeck_forecast
# ---------------------------------------------------------------------------


def ornstein_uhlenbeck_forecast(
    data: pd.Series,
    h: int = 10,
    dt: float | None = None,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
) -> dict:
    """Forecast using an Ornstein-Uhlenbeck (OU) mean-reverting process.

    The OU process is the continuous-time analogue of an AR(1) process
    and is the workhorse model for mean-reverting financial quantities.

    Use for mean-reverting assets like spreads, interest rates,
    volatility (VIX), pairs-trading residuals, or any series that
    fluctuates around a long-run equilibrium.

    SDE:
        ``dX_t = theta * (mu - X_t) * dt + sigma * dW_t``

    where:
        - ``theta`` > 0: speed of mean reversion (higher = faster).
        - ``mu``: long-run mean level.
        - ``sigma``: volatility of the diffusion.
        - ``W_t``: standard Brownian motion.

    The **half-life** of mean reversion is ``ln(2) / theta``, giving
    the expected time for a deviation to halve.

    Parameter estimation:
        Parameters are estimated via Maximum Likelihood on the
        discrete-time transition density, which is Gaussian:
        ``X_{t+dt} | X_t ~ N(mu + (X_t - mu) * exp(-theta*dt),
        sigma^2 / (2*theta) * (1 - exp(-2*theta*dt)))``

    Parameters:
        data: Time series of observed values.
        h: Number of steps to forecast (default 10).
        dt: Time step between observations.  If ``None``, inferred
            from the index (1/252 for business days, 1.0 otherwise).
        n_simulations: Number of Monte Carlo paths for confidence
            bands (default 1000).
        confidence_level: Confidence level for intervals
            (default 0.95).

    Returns:
        Dictionary with:
        - ``params``: dict with ``theta``, ``mu``, ``sigma``.
        - ``half_life``: half-life of mean reversion (in observation
          units).
        - ``forecast``: pd.Series of expected path (conditional mean).
        - ``confidence_intervals``: dict with ``lower`` and ``upper``
          pd.Series.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> # Simulate OU process
        >>> n, theta, mu, sigma = 500, 5.0, 100.0, 2.0
        >>> x = np.zeros(n); x[0] = 100.0
        >>> dt = 1 / 252
        >>> for i in range(1, n):
        ...     x[i] = x[i-1] + theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * rng.normal()
        >>> data = pd.Series(x)
        >>> result = ornstein_uhlenbeck_forecast(data, h=20)
        >>> result['half_life'] > 0
        True

    References:
        Uhlenbeck, G.E. & Ornstein, L.S. (1930). On the theory of
        Brownian motion. *Physical Review*, 36(5), 823.
    """
    y = data.values.astype(float)
    n = len(y)

    # Infer dt
    if dt is None:
        if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
            freq = pd.infer_freq(data.index)
            if freq is not None and freq in ("B", "D"):
                dt = 1.0 / 252.0
            else:
                median_diff = np.median(
                    np.diff(data.index.astype(np.int64)) / 1e9 / 86400
                )
                dt = float(median_diff) / 252.0 if median_diff > 0 else 1.0
        else:
            dt = 1.0

    # MLE estimation via discrete-time transition density
    # X_{t+1} | X_t ~ N(X_t * exp(-theta*dt) + mu*(1-exp(-theta*dt)),
    #                     sigma^2/(2*theta)*(1-exp(-2*theta*dt)))
    x_prev = y[:-1]
    x_next = y[1:]

    def neg_log_lik(params: np.ndarray) -> float:
        th, m, s = params
        if th <= 0 or s <= 0:
            return 1e12
        exp_neg = np.exp(-th * dt)
        mean = x_prev * exp_neg + m * (1 - exp_neg)
        var = (s**2) / (2 * th) * (1 - np.exp(-2 * th * dt))
        if var <= 0:
            return 1e12
        residuals = x_next - mean
        nll = 0.5 * np.sum(
            np.log(2 * np.pi * var) + residuals**2 / var
        )
        return float(nll)

    # Initial guesses from AR(1) regression
    slope, intercept = np.polyfit(x_prev, x_next, 1)
    phi = max(min(slope, 0.9999), 0.0001)
    theta0 = -np.log(phi) / dt
    mu0 = intercept / (1 - phi)
    resid_var = np.var(x_next - (slope * x_prev + intercept))
    sigma0 = np.sqrt(
        max(resid_var * 2 * theta0 / (1 - np.exp(-2 * theta0 * dt)), 1e-8)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimize.minimize(
            neg_log_lik,
            x0=[theta0, mu0, sigma0],
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
        )
    theta_hat, mu_hat, sigma_hat = result.x
    theta_hat = max(theta_hat, 1e-6)
    sigma_hat = max(sigma_hat, 1e-8)

    half_life = np.log(2) / theta_hat

    # Conditional mean forecast
    x_last = y[-1]
    steps = np.arange(1, h + 1)
    exp_decay = np.exp(-theta_hat * dt * steps)
    mean_fcast = mu_hat + (x_last - mu_hat) * exp_decay

    # Conditional variance
    var_fcast = (
        sigma_hat**2 / (2 * theta_hat) * (1 - np.exp(-2 * theta_hat * dt * steps))
    )
    std_fcast = np.sqrt(np.maximum(var_fcast, 0))

    z = sp_stats.norm.ppf(0.5 + confidence_level / 2)

    # Build forecast index
    if isinstance(data.index, pd.DatetimeIndex) and data.index.freq is not None:
        idx = pd.date_range(
            start=data.index[-1] + data.index.freq,
            periods=h,
            freq=data.index.freq,
        )
    elif isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
        freq = pd.infer_freq(data.index)
        if freq is not None:
            idx = pd.date_range(
                start=data.index[-1] + pd.tseries.frequencies.to_offset(freq),
                periods=h,
                freq=freq,
            )
        else:
            idx = pd.RangeIndex(h)
    else:
        last = data.index[-1]
        if isinstance(last, (int, np.integer)):
            idx = pd.RangeIndex(start=last + 1, stop=last + 1 + h)
        else:
            idx = pd.RangeIndex(h)

    return {
        "params": {
            "theta": float(theta_hat),
            "mu": float(mu_hat),
            "sigma": float(sigma_hat),
        },
        "half_life": float(half_life),
        "forecast": pd.Series(mean_fcast, index=idx, name="ou_forecast"),
        "confidence_intervals": {
            "lower": pd.Series(
                mean_fcast - z * std_fcast, index=idx, name="lower"
            ),
            "upper": pd.Series(
                mean_fcast + z * std_fcast, index=idx, name="upper"
            ),
        },
    }


# ---------------------------------------------------------------------------
# jump_diffusion_forecast
# ---------------------------------------------------------------------------


def jump_diffusion_forecast(
    data: pd.Series,
    h: int = 10,
    dt: float | None = None,
    n_paths: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict:
    """Merton jump-diffusion model forecast via Monte Carlo simulation.

    Extends geometric Brownian motion (GBM) by adding Poisson-driven
    jumps, capturing the fat tails and sudden moves observed in real
    asset returns.

    SDE:
        ``dS/S = (mu - lambda*k) * dt + sigma * dW + J * dN``

    where:
        - ``mu``: drift of the diffusion component.
        - ``sigma``: diffusion volatility.
        - ``lambda``: jump intensity (expected jumps per unit time).
        - ``J``: jump size, ``ln(1+J) ~ N(mu_j, sigma_j^2)``.
        - ``N_t``: Poisson process with intensity ``lambda``.
        - ``k = exp(mu_j + sigma_j^2/2) - 1``.

    Parameter estimation:
        A moment-matching approach is used on log-returns:
        - Jump intensity estimated from excess kurtosis.
        - Jump mean and variance from the non-Gaussian tail structure.

    Parameters:
        data: Price series (positive values).
        h: Forecast horizon in steps (default 10).
        dt: Time step between observations (default 1/252 for daily).
        n_paths: Number of Monte Carlo simulation paths (default 1000).
        confidence_level: Confidence level for intervals
            (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with:
        - ``params``: dict with ``mu``, ``sigma``, ``lambda_``,
          ``mu_j``, ``sigma_j``.
        - ``forecast_paths``: np.ndarray of shape ``(n_paths, h)``
          with simulated price paths.
        - ``mean_forecast``: pd.Series of mean across paths.
        - ``confidence_intervals``: dict with ``lower`` and ``upper``
          pd.Series.

    Example:
        >>> import pandas as pd, numpy as np
        >>> prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(200) * 0.01)))
        >>> result = jump_diffusion_forecast(prices, h=10, n_paths=500, seed=42)
        >>> result['forecast_paths'].shape
        (500, 10)

    References:
        Merton, R.C. (1976). Option pricing when underlying stock
        returns are discontinuous. *Journal of Financial Economics*,
        3(1-2), 125-144.
    """
    rng = np.random.default_rng(seed)
    prices = data.values.astype(float)
    log_returns = np.diff(np.log(prices))
    n = len(log_returns)

    if dt is None:
        dt = 1.0 / 252.0

    # Moment-matching estimation
    mu_total = float(np.mean(log_returns) / dt)
    sigma_total = float(np.std(log_returns) / np.sqrt(dt))
    kurt = float(sp_stats.kurtosis(log_returns, fisher=True))
    skew_val = float(sp_stats.skew(log_returns))

    # Estimate jump parameters from excess kurtosis
    # If kurtosis > 0, attribute excess to jumps
    if kurt > 0.5:
        lambda_ = max(kurt / (dt * 10), 0.5)  # jump intensity
        sigma_j = max(abs(skew_val) * 0.1, 0.01)
        mu_j = -sigma_j**2 / 2  # risk-neutral adjustment
        sigma_diff = max(
            np.sqrt(max(sigma_total**2 - lambda_ * sigma_j**2, 1e-8)),
            sigma_total * 0.5,
        )
    else:
        lambda_ = 0.5
        mu_j = 0.0
        sigma_j = 0.01
        sigma_diff = sigma_total

    k = np.exp(mu_j + sigma_j**2 / 2) - 1
    mu_diff = mu_total + lambda_ * k

    # Monte Carlo simulation
    s0 = prices[-1]
    paths = np.zeros((n_paths, h))

    for i in range(n_paths):
        s = s0
        for t_step in range(h):
            # Diffusion
            dw = rng.normal(0, np.sqrt(dt))
            # Jumps
            n_jumps = rng.poisson(lambda_ * dt)
            jump = 0.0
            if n_jumps > 0:
                jump = np.sum(rng.normal(mu_j, sigma_j, n_jumps))
            log_return = (
                (mu_diff - 0.5 * sigma_diff**2 - lambda_ * k) * dt
                + sigma_diff * dw
                + jump
            )
            s = s * np.exp(log_return)
            paths[i, t_step] = s

    # Summary statistics
    mean_path = np.mean(paths, axis=0)
    alpha_q = (1 - confidence_level) / 2
    lower = np.quantile(paths, alpha_q, axis=0)
    upper = np.quantile(paths, 1 - alpha_q, axis=0)

    # Build forecast index
    if isinstance(data.index, pd.DatetimeIndex) and data.index.freq is not None:
        idx = pd.date_range(
            start=data.index[-1] + data.index.freq,
            periods=h,
            freq=data.index.freq,
        )
    elif isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
        freq = pd.infer_freq(data.index)
        if freq is not None:
            idx = pd.date_range(
                start=data.index[-1] + pd.tseries.frequencies.to_offset(freq),
                periods=h,
                freq=freq,
            )
        else:
            idx = pd.RangeIndex(h)
    else:
        last = data.index[-1]
        if isinstance(last, (int, np.integer)):
            idx = pd.RangeIndex(start=last + 1, stop=last + 1 + h)
        else:
            idx = pd.RangeIndex(h)

    return {
        "params": {
            "mu": float(mu_diff),
            "sigma": float(sigma_diff),
            "lambda_": float(lambda_),
            "mu_j": float(mu_j),
            "sigma_j": float(sigma_j),
        },
        "forecast_paths": paths,
        "mean_forecast": pd.Series(
            mean_path, index=idx, name="jd_mean_forecast"
        ),
        "confidence_intervals": {
            "lower": pd.Series(lower, index=idx, name="lower"),
            "upper": pd.Series(upper, index=idx, name="upper"),
        },
    }


# ---------------------------------------------------------------------------
# regime_switching_forecast
# ---------------------------------------------------------------------------


def regime_switching_forecast(
    data: pd.Series,
    n_regimes: int = 2,
    h: int = 10,
) -> dict:
    """Forecast using a Markov regime-switching model.

    Fits a Markov-switching autoregression where the series parameters
    (mean, variance) switch between discrete hidden regimes.  Forecasts
    are blended across regimes weighted by the predicted regime
    probabilities.

    When to use:
        - Series that alternate between distinct states (e.g., bull/bear
          markets, high/low volatility regimes).
        - When a single-model forecast is inadequate because the data
          generation process changes over time.

    Parameters:
        data: Time series to forecast.
        n_regimes: Number of hidden regimes (default 2).
        h: Forecast horizon (default 10).

    Returns:
        Dictionary with:
        - ``forecast``: pd.Series of blended point forecasts.
        - ``regime_forecasts``: dict mapping regime index to its
          pd.Series forecast.
        - ``regime_probs``: np.ndarray of shape ``(h, n_regimes)``
          with predicted regime probabilities.
        - ``blended``: same as ``forecast`` (for clarity).

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> # Bull/bear regime data
        >>> regimes = np.concatenate([rng.normal(0.1, 0.5, 150),
        ...                           rng.normal(-0.05, 1.0, 150)])
        >>> data = pd.Series(np.cumsum(regimes))
        >>> result = regime_switching_forecast(data, n_regimes=2, h=10)
        >>> result['regime_probs'].shape[1]
        2

    References:
        Hamilton, J.D. (1989). A New Approach to the Economic Analysis
        of Nonstationary Time Series and the Business Cycle.
        *Econometrica*, 57(2), 357-384.
    """
    from statsmodels.tsa.regime_switching.markov_autoregression import (
        MarkovAutoregression,
    )

    y = data.values.astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MarkovAutoregression(
            data, k_regimes=n_regimes, order=1, switching_ar=False
        )
        fit = model.fit(maxiter=200, disp=False)

    # Smoothed regime probabilities at the last observation
    smoothed = fit.smoothed_marginal_probabilities
    last_probs = smoothed.iloc[-1].values  # shape (n_regimes,)

    # Transition matrix
    trans_mat = fit.regime_transition  # shape (n_regimes, n_regimes)
    # In statsmodels, regime_transition[i, j] = P(regime j at t+1 | regime i at t)
    # We need P(regime at t+1) = trans_mat.T @ P(regime at t)

    # Regime-specific parameters
    regime_means = np.array(
        [fit.params[f"const[{k}]"] for k in range(n_regimes)]
    )
    # AR coefficient (non-switching)
    ar_coef = fit.params.get("x1", 0.0)
    if isinstance(ar_coef, (pd.Series, np.ndarray)):
        ar_coef = float(ar_coef)

    # Forecast
    regime_probs = np.zeros((h, n_regimes))
    current_probs = last_probs.copy()
    last_val = y[-1]

    regime_fcasts = np.zeros((n_regimes, h))
    for step in range(h):
        # Update regime probabilities
        current_probs = trans_mat @ current_probs
        current_probs = current_probs / current_probs.sum()
        regime_probs[step] = current_probs

        for k in range(n_regimes):
            if step == 0:
                regime_fcasts[k, step] = regime_means[k] + ar_coef * last_val
            else:
                regime_fcasts[k, step] = (
                    regime_means[k] + ar_coef * regime_fcasts[k, step - 1]
                )

    # Blended forecast
    blended = np.sum(regime_fcasts * regime_probs.T, axis=0)

    # Build index
    if isinstance(data.index, pd.DatetimeIndex) and data.index.freq is not None:
        idx = pd.date_range(
            start=data.index[-1] + data.index.freq,
            periods=h,
            freq=data.index.freq,
        )
    elif isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
        freq = pd.infer_freq(data.index)
        if freq is not None:
            idx = pd.date_range(
                start=data.index[-1] + pd.tseries.frequencies.to_offset(freq),
                periods=h,
                freq=freq,
            )
        else:
            idx = pd.RangeIndex(h)
    else:
        last_idx = data.index[-1]
        if isinstance(last_idx, (int, np.integer)):
            idx = pd.RangeIndex(start=last_idx + 1, stop=last_idx + 1 + h)
        else:
            idx = pd.RangeIndex(h)

    regime_forecast_dict = {
        k: pd.Series(
            regime_fcasts[k], index=idx, name=f"regime_{k}_forecast"
        )
        for k in range(n_regimes)
    }

    fcast_series = pd.Series(blended, index=idx, name="regime_forecast")

    return {
        "forecast": fcast_series,
        "regime_forecasts": regime_forecast_dict,
        "regime_probs": regime_probs,
        "blended": fcast_series,
    }


# ---------------------------------------------------------------------------
# var_forecast
# ---------------------------------------------------------------------------


def var_forecast(
    data: pd.DataFrame,
    h: int = 10,
    maxlags: int | None = None,
    ic: str = "aic",
) -> dict:
    """Vector Autoregression (VAR) for multi-asset forecasting.

    VAR models each variable as a linear function of its own lags and
    the lags of all other variables.  This captures lead-lag
    relationships (e.g., one asset predicting another) and provides
    impulse response functions (IRF) and forecast error variance
    decomposition (FEVD).

    When to use:
        - Forecasting multiple related time series simultaneously
          (e.g., a portfolio of assets, yield curve factors).
        - Analysing dynamic interactions: which variables Granger-cause
          which.
        - Computing impulse responses: how a shock to one variable
          propagates to others.

    Math:
        ``Y_t = c + A_1 * Y_{t-1} + ... + A_p * Y_{t-p} + e_t``

        where ``Y_t`` is a (k x 1) vector, ``A_i`` are (k x k)
        coefficient matrices, and ``e_t ~ N(0, Sigma)``.

    Parameters:
        data: DataFrame with multiple columns (one per variable).
            Should be stationary (difference if needed).
        h: Forecast horizon (default 10).
        maxlags: Maximum lag order to consider. If ``None``, uses
            ``12 * (nobs/100)^(1/4)`` rule of thumb.
        ic: Information criterion for lag selection:
            ``"aic"``, ``"bic"``, ``"hqic"``, ``"fpe"``
            (default ``"aic"``).

    Returns:
        Dictionary with:
        - ``forecast``: pd.DataFrame of point forecasts.
        - ``irf``: impulse response function results (dict of
          DataFrames keyed by shocked variable).
        - ``fevd``: forecast error variance decomposition (dict of
          DataFrames keyed by target variable).
        - ``granger_causality``: dict of p-values for Granger
          causality tests.
        - ``lag_order``: selected lag order.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> n = 200
        >>> x1 = np.cumsum(rng.normal(0, 1, n))
        >>> x2 = 0.3 * np.roll(x1, 1) + rng.normal(0, 1, n)
        >>> data = pd.DataFrame({'x1': np.diff(x1), 'x2': np.diff(x2)})
        >>> result = var_forecast(data, h=5)
        >>> result['forecast'].shape[0]
        5

    References:
        Lutkepohl, H. (2005). *New Introduction to Multiple Time
        Series Analysis*. Springer.
    """
    from statsmodels.tsa.api import VAR as StatsVAR

    var_model = StatsVAR(data)

    if maxlags is None:
        maxlags = max(int(12 * (len(data) / 100) ** 0.25), 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = var_model.fit(maxlags=maxlags, ic=ic)

    lag_order = fit.k_ar
    columns = list(data.columns)

    # Forecast
    fcast = fit.forecast(data.values[-lag_order:], steps=h)
    if isinstance(data.index, pd.DatetimeIndex) and data.index.freq is not None:
        idx = pd.date_range(
            start=data.index[-1] + data.index.freq,
            periods=h,
            freq=data.index.freq,
        )
    else:
        idx = pd.RangeIndex(h)
    forecast_df = pd.DataFrame(fcast, columns=columns, index=idx)

    # Impulse Response Functions
    irf_result = fit.irf(periods=h)
    irf_dict: dict[str, pd.DataFrame] = {}
    for i, col in enumerate(columns):
        irf_dict[col] = pd.DataFrame(
            irf_result.irfs[:, :, i],
            columns=columns,
            index=np.arange(h + 1),
        )

    # Forecast Error Variance Decomposition
    fevd_result = fit.fevd(periods=h)
    fevd_dict: dict[str, pd.DataFrame] = {}
    decomp = fevd_result.decomp
    n_periods_fevd = decomp.shape[0]
    for i, col in enumerate(columns):
        fevd_dict[col] = pd.DataFrame(
            decomp[:, i, :],
            columns=columns,
            index=np.arange(1, n_periods_fevd + 1),
        )

    # Granger causality tests
    gc_dict: dict[str, float] = {}
    for caused in columns:
        for causing in columns:
            if caused == causing:
                continue
            key = f"{causing} -> {caused}"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gc_test = fit.test_causality(
                        caused, [causing], kind="f"
                    )
                gc_dict[key] = float(gc_test.pvalue)
            except Exception:  # noqa: BLE001
                gc_dict[key] = float("nan")

    return {
        "forecast": forecast_df,
        "irf": irf_dict,
        "fevd": fevd_dict,
        "granger_causality": gc_dict,
        "lag_order": lag_order,
    }
