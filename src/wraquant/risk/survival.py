"""Survival analysis estimators for financial applications.

Survival analysis models the time until an event occurs -- in finance,
this could be time-to-default, time until a drawdown ends, time between
large losses, or fund lifetime. The key challenge is *censoring*: not
all subjects experience the event during the observation period.

This module provides pure numpy/scipy implementations of the standard
survival analysis toolkit:

Non-parametric estimators:
    - ``kaplan_meier``: the Kaplan-Meier product-limit estimator of
      the survival function S(t) = P(T > t). The most common survival
      curve estimator. Handles right-censored data.
    - ``nelson_aalen``: cumulative hazard estimator H(t). Related to
      Kaplan-Meier via S(t) = exp(-H(t)) but more natural for hazard
      rate estimation.
    - ``hazard_rate``: kernel-smoothed instantaneous hazard rate from
      Nelson-Aalen increments. Useful for visualising how default risk
      changes over time.

Semi-parametric model:
    - ``cox_partial_likelihood``: Cox proportional hazards model.
      Estimates the effect of covariates on the hazard rate without
      specifying the baseline hazard. The workhorse of survival
      regression: "does leverage, size, or profitability affect
      time-to-default?"

Parametric models:
    - ``exponential_survival``: S(t) = exp(-lambda * t). Assumes
      constant hazard (memoryless). Simple but often too restrictive.
    - ``weibull_survival``: S(t) = exp(-(t/lambda)^k). Generalises
      exponential (k=1). k < 1 = decreasing hazard (burn-in),
      k > 1 = increasing hazard (aging/wear-out), which corresponds
      to increasing default risk with time for distressed firms.

Hypothesis testing:
    - ``log_rank_test``: compares two survival curves. Use to test
      whether two groups (e.g., investment-grade vs. high-yield) have
      significantly different survival distributions.

Utility:
    - ``median_survival_time``: smallest t where S(t) <= 0.5.

Financial applications:
    - **Credit risk**: model time-to-default with Cox PH, using
      leverage, profitability, and market indicators as covariates.
    - **Drawdown analysis**: model time to recover from a drawdown;
      Weibull shape > 1 suggests recovery becomes less likely over time.
    - **Fund closure**: Kaplan-Meier curves for hedge fund lifetimes,
      stratified by strategy type.
    - **Trade duration**: model how long a position is held before
      hitting a stop-loss or take-profit.

References:
    - Cox (1972), "Regression Models and Life-Tables"
    - Kaplan & Meier (1958), "Nonparametric Estimation from Incomplete
      Observations"
    - Lando (2004), "Credit Risk Modeling: Theory and Applications"
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

__all__ = [
    "cox_partial_likelihood",
    "exponential_survival",
    "hazard_rate",
    "kaplan_meier",
    "log_rank_test",
    "median_survival_time",
    "nelson_aalen",
    "weibull_survival",
]


def kaplan_meier(
    durations: np.ndarray,
    event_observed: np.ndarray,
) -> dict[str, np.ndarray]:
    """Kaplan-Meier survival curve estimator.

    Non-parametric estimator of the survival function from censored
    duration data.

    Parameters:
        durations: 1-D array of observed durations (time-to-event or
            time-to-censoring).
        event_observed: 1-D boolean/int array where 1 (True) indicates
            the event occurred and 0 (False) indicates right-censoring.

    Returns:
        Dictionary with keys:

        - ``timeline``: Sorted unique event times.
        - ``survival``: Survival probability at each event time.
        - ``variance``: Greenwood's variance estimate at each event time.
    """
    durations = np.asarray(durations, dtype=float)
    event_observed = np.asarray(event_observed, dtype=bool)
    if durations.shape != event_observed.shape:
        raise ValueError("durations and event_observed must have the same shape")

    # Sort by duration
    order = np.argsort(durations)
    t_sorted = durations[order]
    e_sorted = event_observed[order]

    unique_times = np.unique(t_sorted)
    len(durations)

    survival = np.ones(len(unique_times))
    variance = np.zeros(len(unique_times))
    s = 1.0
    var_sum = 0.0

    for idx, t in enumerate(unique_times):
        # Number at risk at time t
        at_risk = np.sum(t_sorted >= t)
        # Number of events at time t
        events = np.sum((t_sorted == t) & e_sorted)

        if at_risk > 0 and events > 0:
            s *= 1.0 - events / at_risk
            var_sum += (
                events / (at_risk * (at_risk - events)) if at_risk > events else 0.0
            )

        survival[idx] = s
        variance[idx] = s**2 * var_sum

    return {
        "timeline": unique_times,
        "survival": survival,
        "variance": variance,
    }


def nelson_aalen(
    durations: np.ndarray,
    event_observed: np.ndarray,
) -> dict[str, np.ndarray]:
    """Nelson-Aalen cumulative hazard estimator.

    Non-parametric estimator of the cumulative hazard function.

    Parameters:
        durations: 1-D array of observed durations.
        event_observed: 1-D boolean/int array indicating event occurrence.

    Returns:
        Dictionary with keys:

        - ``timeline``: Sorted unique event times.
        - ``cumulative_hazard``: Cumulative hazard estimate at each time.
        - ``variance``: Variance estimate at each time.
    """
    durations = np.asarray(durations, dtype=float)
    event_observed = np.asarray(event_observed, dtype=bool)
    if durations.shape != event_observed.shape:
        raise ValueError("durations and event_observed must have the same shape")

    order = np.argsort(durations)
    t_sorted = durations[order]
    e_sorted = event_observed[order]

    unique_times = np.unique(t_sorted)
    cum_hazard = np.zeros(len(unique_times))
    variance = np.zeros(len(unique_times))

    h = 0.0
    v = 0.0

    for idx, t in enumerate(unique_times):
        at_risk = np.sum(t_sorted >= t)
        events = np.sum((t_sorted == t) & e_sorted)

        if at_risk > 0:
            h += events / at_risk
            v += events / at_risk**2

        cum_hazard[idx] = h
        variance[idx] = v

    return {
        "timeline": unique_times,
        "cumulative_hazard": cum_hazard,
        "variance": variance,
    }


def hazard_rate(
    durations: np.ndarray,
    event_observed: np.ndarray,
    bandwidth: float | None = None,
) -> dict[str, np.ndarray]:
    """Kernel-smoothed hazard rate estimate.

    Applies Epanechnikov kernel smoothing to the Nelson-Aalen increments
    to produce a smooth hazard rate function.

    Parameters:
        durations: 1-D array of observed durations.
        event_observed: 1-D boolean/int array indicating event occurrence.
        bandwidth: Kernel bandwidth. If ``None``, uses Silverman's rule
            of thumb.

    Returns:
        Dictionary with keys:

        - ``timeline``: Evaluation grid (same as Nelson-Aalen times).
        - ``hazard``: Smoothed hazard rate at each time point.
    """
    na = nelson_aalen(durations, event_observed)
    times = na["timeline"]
    cum_h = na["cumulative_hazard"]

    if len(times) < 2:
        return {"timeline": times, "hazard": np.zeros_like(times)}

    # Nelson-Aalen increments
    increments = np.diff(cum_h, prepend=0.0)

    if bandwidth is None:
        # Silverman's rule of thumb
        std = np.std(durations[np.asarray(event_observed, dtype=bool)])
        n = np.sum(np.asarray(event_observed, dtype=bool))
        if std > 0 and n > 0:
            bandwidth = 1.06 * std * n ** (-1.0 / 5.0)
        else:
            bandwidth = 1.0

    # Epanechnikov kernel smoothing
    smoothed = np.zeros(len(times))
    for i, t in enumerate(times):
        u = (times - t) / bandwidth
        kernel = np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)
        w = kernel / bandwidth
        smoothed[i] = np.sum(w * increments)

    # Clip to non-negative
    smoothed = np.maximum(smoothed, 0.0)

    return {"timeline": times, "hazard": smoothed}


def cox_partial_likelihood(
    durations: np.ndarray,
    event_observed: np.ndarray,
    covariates: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> dict[str, np.ndarray | float]:
    """Cox proportional hazards model via Newton-Raphson.

    Fits the Cox PH model by maximising the partial likelihood.  This is
    a simplified implementation (Breslow method for ties).

    Parameters:
        durations: 1-D array of observed durations.
        event_observed: 1-D boolean/int array indicating event occurrence.
        covariates: 2-D array of shape ``(n_subjects, n_covariates)``.
        max_iter: Maximum Newton-Raphson iterations.
        tol: Convergence tolerance on the gradient norm.

    Returns:
        Dictionary with keys:

        - ``beta``: Estimated regression coefficients.
        - ``se``: Standard errors of the coefficients.
        - ``log_partial_likelihood``: Maximised log partial likelihood.
        - ``n_iter``: Number of iterations to convergence.
    """
    durations = np.asarray(durations, dtype=float)
    event_observed = np.asarray(event_observed, dtype=bool)
    X = np.asarray(covariates, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape
    if durations.shape[0] != n or event_observed.shape[0] != n:
        raise ValueError("Inconsistent array lengths")

    # Sort by duration descending (for risk set computation)
    order = np.argsort(-durations)
    durations[order]
    e = event_observed[order]
    X = X[order]

    beta = np.zeros(p)
    iteration = 0

    for iteration in range(max_iter):
        exp_xb = np.exp(X @ beta)

        # Cumulative sums from the end (since we sorted descending,
        # cumsum gives the risk set sums)
        risk_sum = np.cumsum(exp_xb)
        risk_sum_x = np.cumsum((exp_xb[:, None] * X), axis=0)  # (n, p)
        risk_sum_xx = np.zeros((n, p, p))
        for i in range(n):
            xi = X[i : i + 1]  # (1, p)
            risk_sum_xx[i] = exp_xb[i] * (xi.T @ xi)
        risk_sum_xx = np.cumsum(risk_sum_xx, axis=0)  # (n, p, p)

        # Gradient and Hessian of log partial likelihood
        gradient = np.zeros(p)
        hessian = np.zeros((p, p))

        for i in range(n):
            if not e[i]:
                continue
            rs = risk_sum[i]
            if rs < 1e-15:
                continue
            rsx = risk_sum_x[i]
            rsxx = risk_sum_xx[i]

            gradient += X[i] - rsx / rs
            hessian -= rsxx / rs - np.outer(rsx, rsx) / rs**2

        if np.linalg.norm(gradient) < tol:
            break

        # Newton-Raphson step
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        beta -= step

    # Standard errors from the inverse of the observed information
    try:
        inv_hessian = np.linalg.inv(-hessian)
        se = np.sqrt(np.maximum(np.diag(inv_hessian), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    # Compute final log partial likelihood
    exp_xb = np.exp(X @ beta)
    risk_sum = np.cumsum(exp_xb)
    lpl = 0.0
    for i in range(n):
        if e[i]:
            rs = risk_sum[i]
            if rs > 0:
                lpl += X[i] @ beta - np.log(rs)

    return {
        "beta": beta,
        "se": se,
        "log_partial_likelihood": float(lpl),
        "n_iter": iteration + 1 if iteration < max_iter else max_iter,
    }


def exponential_survival(
    lambda_param: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Exponential survival function S(t) = exp(-lambda * t).

    Parameters:
        lambda_param: Hazard rate (constant).
        t: Time point(s) at which to evaluate.

    Returns:
        Survival probability at each *t*.
    """
    if lambda_param < 0:
        raise ValueError("lambda_param must be non-negative")
    t = np.asarray(t, dtype=float)
    result = np.exp(-lambda_param * t)
    return float(result) if result.ndim == 0 else result


def weibull_survival(
    lambda_param: float,
    k: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Weibull survival function S(t) = exp(-(t / lambda)^k).

    Parameters:
        lambda_param: Scale parameter (> 0).
        k: Shape parameter (> 0).  k=1 reduces to exponential.
        t: Time point(s) at which to evaluate.

    Returns:
        Survival probability at each *t*.
    """
    if lambda_param <= 0:
        raise ValueError("lambda_param must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    t = np.asarray(t, dtype=float)
    result = np.exp(-((t / lambda_param) ** k))
    return float(result) if result.ndim == 0 else result


def log_rank_test(
    durations1: np.ndarray,
    event1: np.ndarray,
    durations2: np.ndarray,
    event2: np.ndarray,
) -> dict[str, float]:
    """Log-rank test comparing two survival curves.

    Tests the null hypothesis that the survival functions of two groups
    are identical.

    Parameters:
        durations1: Durations for group 1.
        event1: Event indicators for group 1.
        durations2: Durations for group 2.
        event2: Event indicators for group 2.

    Returns:
        Dictionary with keys:

        - ``test_statistic``: Chi-squared test statistic.
        - ``p_value``: P-value from chi-squared distribution with 1 df.
        - ``observed1``: Total observed events in group 1.
        - ``expected1``: Expected events in group 1 under H0.
    """
    d1 = np.asarray(durations1, dtype=float)
    e1 = np.asarray(event1, dtype=bool)
    d2 = np.asarray(durations2, dtype=float)
    e2 = np.asarray(event2, dtype=bool)

    # Combine all unique event times
    all_times = np.unique(np.concatenate([d1[e1], d2[e2]]))

    obs1_total = 0.0
    exp1_total = 0.0
    var_total = 0.0

    for t in all_times:
        # At risk in each group at time t
        n1 = np.sum(d1 >= t)
        n2 = np.sum(d2 >= t)
        n = n1 + n2

        if n == 0:
            continue

        # Events at time t in each group
        o1 = np.sum((d1 == t) & e1)
        o2 = np.sum((d2 == t) & e2)
        o = o1 + o2

        # Expected events under H0
        e1_expected = o * n1 / n

        obs1_total += o1
        exp1_total += e1_expected

        # Variance
        if n > 1:
            var_total += o * (n1 / n) * (n2 / n) * (n - o) / (n - 1)

    if var_total < 1e-15:
        return {
            "test_statistic": 0.0,
            "p_value": 1.0,
            "observed1": float(obs1_total),
            "expected1": float(exp1_total),
        }

    chi2 = (obs1_total - exp1_total) ** 2 / var_total
    p_value = 1.0 - sp_stats.chi2.cdf(chi2, df=1)

    return {
        "test_statistic": float(chi2),
        "p_value": float(p_value),
        "observed1": float(obs1_total),
        "expected1": float(exp1_total),
    }


def median_survival_time(
    durations: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """Median survival time from the Kaplan-Meier estimator.

    The median survival time is the smallest time *t* at which the
    estimated survival function drops to or below 0.5.

    Parameters:
        durations: 1-D array of observed durations.
        event_observed: 1-D boolean/int array indicating event occurrence.

    Returns:
        Median survival time, or ``np.inf`` if the survival curve never
        drops to 0.5.
    """
    km = kaplan_meier(durations, event_observed)
    timeline = km["timeline"]
    survival = km["survival"]

    idx = np.where(survival <= 0.5)[0]
    if len(idx) == 0:
        return float(np.inf)

    return float(timeline[idx[0]])
