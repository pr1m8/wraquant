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

    The Kaplan-Meier (KM) estimator is the standard non-parametric
    method for estimating the survival function S(t) = P(T > t) from
    potentially censored data.  "Censored" means some subjects have
    not yet experienced the event at the time of observation.

    In finance, this answers questions like:
        - "What fraction of bonds survive to year 5 without defaulting?"
        - "How long do hedge funds typically survive before closing?"
        - "What is the probability a drawdown lasts longer than 60 days?"

    Interpretation:
        - **survival**: S(t) is the probability of surviving beyond
          time t.  A steep drop indicates a period of high hazard.
        - **variance** (Greenwood's formula): Use to construct 95%
          confidence bands as S(t) +/- 1.96 * sqrt(variance(t)).
        - The median survival time is where S(t) first drops below 0.5.
        - A flat survival curve = low hazard rate (few events).
        - A curve that drops quickly early = high initial hazard
          (e.g., new funds failing in the first year).

    Parameters:
        durations: 1-D array of observed durations (time-to-event or
            time-to-censoring).
        event_observed: 1-D boolean/int array where 1 (True) indicates
            the event occurred and 0 (False) indicates right-censoring.

    Returns:
        Dictionary with keys:

        - **timeline** (*ndarray*) -- Sorted unique event times.
        - **survival** (*ndarray*) -- Survival probability S(t) at each
          event time.
        - **variance** (*ndarray*) -- Greenwood's variance estimate for
          constructing confidence bands.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> durations = rng.exponential(5.0, 200)  # avg survival 5 years
        >>> events = rng.binomial(1, 0.7, 200)     # 70% observed, 30% censored
        >>> km = kaplan_meier(durations, events)
        >>> print(f"5-year survival: {km['survival'][km['timeline'] <= 5][-1]:.2f}")

    See Also:
        nelson_aalen: Cumulative hazard estimator.
        log_rank_test: Compare two survival curves.
    """
    from wraquant.core._coerce import coerce_array

    durations = coerce_array(durations, name="durations")
    event_observed = np.asarray(event_observed, dtype=bool)
    if durations.shape != event_observed.shape:
        raise ValueError("durations and event_observed must have the same shape")

    # Sort by duration
    order = np.argsort(durations)
    t_sorted = durations[order]
    e_sorted = event_observed[order]

    unique_times = np.unique(t_sorted)

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

    Estimates the cumulative hazard function H(t), which is related to
    the survival function by S(t) = exp(-H(t)).  While the Kaplan-Meier
    directly estimates S(t), the Nelson-Aalen estimator is more natural
    for estimating the hazard rate and for models where the hazard is
    the primary quantity of interest.

    Interpretation:
        - H(t) represents the accumulated risk up to time t.
        - The slope of H(t) is the instantaneous hazard rate: steep
          segments indicate periods of high risk.
        - A linear H(t) suggests a constant hazard rate (exponential
          survival).
        - A concave H(t) suggests a decreasing hazard (survival gets
          easier over time).
        - A convex H(t) suggests an increasing hazard (risk
          accelerates -- typical for aging/wear-out or credit
          deterioration).

    Parameters:
        durations: 1-D array of observed durations.
        event_observed: 1-D boolean/int array indicating event occurrence.

    Returns:
        Dictionary with keys:

        - **timeline** (*ndarray*) -- Sorted unique event times.
        - **cumulative_hazard** (*ndarray*) -- H(t) at each time.
        - **variance** (*ndarray*) -- Variance estimate at each time.

    See Also:
        kaplan_meier: Direct survival function estimator.
        hazard_rate: Smoothed instantaneous hazard from Nelson-Aalen.
    """
    from wraquant.core._coerce import coerce_array

    durations = coerce_array(durations, name="durations")
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
    to produce a smooth instantaneous hazard rate function h(t).

    The hazard rate answers: "Given that a subject has survived to time
    t, what is the instantaneous probability of the event?" This is
    more informative than the cumulative survival function for
    understanding *when* risk is highest.

    Interpretation:
        - A flat hazard rate means constant risk (exponential model).
        - An increasing hazard means risk accelerates with time
          (typical for credit deterioration, infrastructure aging).
        - A decreasing hazard means early failures dominate and
          survivors become stronger ("infant mortality").
        - A bathtub-shaped hazard (decreasing then increasing) is
          common in reliability engineering.

    Parameters:
        durations: 1-D array of observed durations.
        event_observed: 1-D boolean/int array indicating event occurrence.
        bandwidth: Kernel bandwidth. If ``None``, uses Silverman's rule.
            Larger bandwidth = smoother curve. Smaller = more detail
            but noisier.

    Returns:
        Dictionary with keys:

        - **timeline** (*ndarray*) -- Evaluation grid.
        - **hazard** (*ndarray*) -- Smoothed hazard rate at each point.

    See Also:
        nelson_aalen: The cumulative hazard from which this is derived.
        kaplan_meier: Survival function estimator.
    """
    from wraquant.core._coerce import coerce_array

    durations = coerce_array(durations, name="durations")
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

    The Cox PH model is the workhorse of survival regression.  It
    estimates the effect of covariates on the hazard rate without
    specifying the baseline hazard function (semi-parametric).  The
    model assumes the hazard ratio is constant over time (proportional
    hazards assumption).

    In finance: "Does leverage, profitability, or market beta affect
    the hazard of default, controlling for other factors?"

    Interpretation:
        - **beta[j]** is the log hazard ratio for covariate j.
          exp(beta[j]) > 1 means the covariate increases the hazard
          (bad for survival).  exp(beta[j]) < 1 means it decreases
          the hazard (protective).
        - **se[j]**: Standard error. beta[j] / se[j] gives a z-statistic.
          |z| > 1.96 is significant at the 5% level.
        - **log_partial_likelihood**: Higher (less negative) = better
          fit.  Use for comparing nested models via likelihood ratio
          tests.

    Red flags:
        - Very large beta (|beta| > 5): possible separation/convergence
          issues.
        - n_iter = max_iter: did not converge, results unreliable.
        - se contains NaN: Hessian is singular, model is degenerate.

    Parameters:
        durations: 1-D array of observed durations.
        event_observed: 1-D boolean/int array indicating event occurrence.
        covariates: 2-D array of shape ``(n_subjects, n_covariates)``.
        max_iter: Maximum Newton-Raphson iterations.
        tol: Convergence tolerance on the gradient norm.

    Returns:
        Dictionary with keys:

        - **beta** (*ndarray*) -- Regression coefficients. exp(beta)
          gives hazard ratios.
        - **se** (*ndarray*) -- Standard errors of coefficients.
        - **log_partial_likelihood** (*float*) -- Maximised log partial
          likelihood.
        - **n_iter** (*int*) -- Number of iterations to convergence.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 200
        >>> leverage = rng.uniform(0.2, 0.8, n)
        >>> durations = rng.exponential(5 / (1 + leverage), n)
        >>> events = np.ones(n, dtype=bool)
        >>> result = cox_partial_likelihood(durations, events, leverage.reshape(-1, 1))
        >>> print(f"Leverage HR: {np.exp(result['beta'][0]):.2f}")

    See Also:
        kaplan_meier: Non-parametric survival curve (no covariates).
        weibull_survival: Parametric survival model.
    """
    from wraquant.core._coerce import coerce_array

    durations = coerce_array(durations, name="durations")
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

    The exponential model assumes a constant hazard rate -- the
    probability of the event in the next instant is the same
    regardless of how long the subject has already survived.  This is
    the "memoryless" property.

    Interpretation:
        - lambda = 0.1 means roughly a 10% chance of the event per
          unit of time.
        - Mean survival time = 1 / lambda.
        - If the hazard is actually increasing or decreasing over time,
          this model is too simplistic.  Use ``weibull_survival`` instead.

    Parameters:
        lambda_param: Hazard rate (constant, > 0). Higher = faster
            time to event.
        t: Time point(s) at which to evaluate.

    Returns:
        Survival probability at each *t*.

    See Also:
        weibull_survival: Generalises exponential with time-varying hazard.
    """
    from wraquant.core._coerce import coerce_array

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

    The Weibull distribution generalises the exponential by allowing
    the hazard rate to increase or decrease over time.  It is the
    most commonly used parametric survival model in practice.

    Interpretation of the shape parameter k:
        - **k = 1**: Constant hazard (reduces to exponential). The
          event is equally likely at any time.
        - **k < 1**: Decreasing hazard ("burn-in"). Early failures
          are most common; survivors become stronger.  Typical for
          infant mortality in manufactured goods.
        - **k > 1**: Increasing hazard ("aging"). Risk increases over
          time.  Typical for credit deterioration in distressed firms
          or aging infrastructure.

    In finance:
        - k > 1 for time-to-default: firms that have survived a long
          time in distress become more likely to default (debt maturity
          approaches, liquidity dries up).
        - k < 1 for drawdown recovery: if a drawdown has already
          lasted a long time, recovery becomes more likely (mean
          reversion kicks in).

    Parameters:
        lambda_param: Scale parameter (> 0). Larger = longer survival.
        k: Shape parameter (> 0). k=1 is exponential. k>1 is
            increasing hazard. k<1 is decreasing hazard.
        t: Time point(s) at which to evaluate.

    Returns:
        Survival probability at each *t*.

    See Also:
        exponential_survival: Simplest case (k=1).
        kaplan_meier: Non-parametric alternative.
    """
    from wraquant.core._coerce import coerce_array

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
    are identical.  This is the standard test for comparing survival
    experiences between groups.

    In finance: "Do investment-grade bonds have significantly different
    time-to-default than high-yield bonds?" or "Do value stocks have
    different drawdown durations than growth stocks?"

    Interpretation:
        - **p_value < 0.05**: reject H0 -- the two groups have
          significantly different survival experiences.
        - **observed1 >> expected1**: Group 1 has more events than
          expected (worse survival).
        - **observed1 << expected1**: Group 1 has fewer events than
          expected (better survival).
        - The test is most powerful when the hazard ratio is constant
          (proportional hazards).  For crossing survival curves, the
          Wilcoxon (Breslow) test may be more appropriate.

    Parameters:
        durations1: Durations for group 1.
        event1: Event indicators for group 1.
        durations2: Durations for group 2.
        event2: Event indicators for group 2.

    Returns:
        Dictionary with keys:

        - **test_statistic** (*float*) -- Chi-squared statistic (1 df).
        - **p_value** (*float*) -- P-value. < 0.05 rejects equality.
        - **observed1** (*float*) -- Total observed events in group 1.
        - **expected1** (*float*) -- Expected events under H0.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> d1 = rng.exponential(5.0, 100)  # group 1: avg survival 5y
        >>> d2 = rng.exponential(3.0, 100)  # group 2: avg survival 3y
        >>> e1 = np.ones(100, dtype=bool)
        >>> e2 = np.ones(100, dtype=bool)
        >>> result = log_rank_test(d1, e1, d2, e2)
        >>> print(f"p-value: {result['p_value']:.4f}")  # should be small
    """
    from wraquant.core._coerce import coerce_array

    d1 = coerce_array(durations1, name="durations1")
    e1 = np.asarray(event1, dtype=bool)
    d2 = coerce_array(durations2, name="durations2")
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
    estimated survival function drops to or below 0.5 -- i.e., the
    time by which half the subjects have experienced the event.

    Interpretation:
        - This is the "half-life" of the population.
        - More robust than mean survival (which is heavily influenced
          by censoring and long survivors).
        - Returns np.inf if the survival curve never reaches 0.5,
          which happens with heavy censoring or if more than half
          the subjects never experience the event.

    In finance:
        - "The median time-to-default for CCC-rated firms is 2.3 years."
        - "The median drawdown recovery time for equity portfolios
          is 45 trading days."

    Parameters:
        durations: 1-D array of observed durations.
        event_observed: 1-D boolean/int array indicating event occurrence.

    Returns:
        Median survival time, or ``np.inf`` if the survival curve
        never drops to 0.5 (too much censoring).
    """
    from wraquant.core._coerce import coerce_array

    durations = coerce_array(durations, name="durations")
    km = kaplan_meier(durations, event_observed)
    timeline = km["timeline"]
    survival = km["survival"]

    idx = np.where(survival <= 0.5)[0]
    if len(idx) == 0:
        return float(np.inf)

    return float(timeline[idx[0]])
