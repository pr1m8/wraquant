"""Change-point detection methods for structural break analysis.

Provides Bayesian online change-point detection (Adams & MacKay 2007),
PELT (Pruned Exact Linear Time), binary segmentation, sliding-window
change-point detection, and CUSUM control charts.  These tools identify
abrupt shifts in the statistical properties of a time series -- a
critical prerequisite for regime-aware investing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra


def online_changepoint(
    data: pd.Series,
    hazard: float = 0.005,
) -> pd.Series:
    """Bayesian online change-point detection.

    Implements the algorithm from Adams & MacKay (2007). Uses a normal
    model with unknown mean and known variance (estimated from data).

    Parameters:
        data: Time series to monitor.
        hazard: Prior probability that a change-point occurs at each
            time step (constant hazard rate ``1/mean_run_length``).

    Returns:
        Series of estimated run lengths (the most probable run length
        at each time step). A drop to zero indicates a detected
        change-point.
    """
    clean = data.dropna()
    values = clean.values
    T = len(values)

    # Prior parameters (Normal-Gamma conjugate)
    mu0 = values.mean()
    kappa0 = 1.0
    alpha0 = 1.0
    beta0 = float(values.var()) if values.var() > 0 else 1.0

    # Run-length probabilities: R[t, r] = P(r_t = r | data_{1:t})
    # We only keep current and use log probabilities for stability.
    run_length_probs = np.zeros(T + 1)
    run_length_probs[0] = 1.0

    # Sufficient statistics for each run length
    mu_params = np.array([mu0])
    kappa_params = np.array([kappa0])
    alpha_params = np.array([alpha0])
    beta_params = np.array([beta0])

    max_run_lengths = np.zeros(T, dtype=int)

    for t in range(T):
        x = values[t]

        # Predictive probability under each run-length hypothesis
        # Student-t distribution
        df = 2 * alpha_params
        scale = beta_params * (kappa_params + 1) / (alpha_params * kappa_params)
        scale = np.maximum(scale, 1e-10)

        # Evaluate predictive log probability
        pred_probs = _student_t_pdf(x, mu_params, scale, df)

        # Growth probabilities
        growth_probs = run_length_probs[: len(pred_probs)] * pred_probs * (1 - hazard)

        # Change-point probability
        cp_prob = np.sum(run_length_probs[: len(pred_probs)] * pred_probs * hazard)

        # Update run length distribution
        new_run_length_probs = np.zeros(t + 2)
        new_run_length_probs[0] = cp_prob
        new_run_length_probs[1 : len(growth_probs) + 1] = growth_probs

        # Normalise
        total = new_run_length_probs.sum()
        if total > 0:
            new_run_length_probs /= total
        run_length_probs = new_run_length_probs

        # Update sufficient statistics
        new_mu = (kappa_params * mu_params + x) / (kappa_params + 1)
        new_kappa = kappa_params + 1
        new_alpha = alpha_params + 0.5
        new_beta = beta_params + kappa_params * (x - mu_params) ** 2 / (
            2 * (kappa_params + 1)
        )

        # Prepend prior for new run (r=0)
        mu_params = np.concatenate([[mu0], new_mu])
        kappa_params = np.concatenate([[kappa0], new_kappa])
        alpha_params = np.concatenate([[alpha0], new_alpha])
        beta_params = np.concatenate([[beta0], new_beta])

        max_run_lengths[t] = int(np.argmax(run_length_probs))

    return pd.Series(max_run_lengths, index=clean.index, name="run_length")


def _student_t_pdf(
    x: float,
    mu: np.ndarray,
    scale: np.ndarray,
    df: np.ndarray,
) -> np.ndarray:
    """Evaluate the Student-t PDF for vectorised parameters."""
    from scipy.special import gammaln

    z = (x - mu) ** 2 / scale
    log_prob = (
        gammaln((df + 1) / 2)
        - gammaln(df / 2)
        - 0.5 * np.log(np.pi * df * scale)
        - (df + 1) / 2 * np.log1p(z / df)
    )
    return np.exp(log_prob)


# ---------------------------------------------------------------------------
# PELT (Pruned Exact Linear Time) changepoint detection
# ---------------------------------------------------------------------------


@requires_extra("regimes")
def pelt_changepoint(
    data: pd.Series | np.ndarray,
    *,
    penalty: str | float = "bic",
    model: str = "l2",
    min_size: int = 5,
    jump: int = 1,
    n_bkps: int | None = None,
) -> dict[str, Any]:
    """Pruned Exact Linear Time (PELT) optimal changepoint detection.

    PELT finds the exact global optimum of the penalised cost function
    in O(n) expected time (under mild assumptions).  It is the
    recommended method when you want **optimal** changepoints rather
    than an approximate or sequential solution.

    The algorithm minimises:

    .. math::

        \\sum_{i=0}^{K} C(y_{\\tau_i : \\tau_{i+1}}) + \\beta \\, K

    where *C* is the segment cost, *K* is the number of changepoints,
    and *beta* is the penalty.

    **Interpretation guidance:**

    - A changepoint at index *t* means the statistical properties of
      the series changed between observation *t-1* and *t*.
    - ``segment_stats`` gives the mean, variance, and length of each
      segment so you can characterise the regimes between breaks.
    - ``confidence`` is a heuristic based on the cost reduction at
      each changepoint relative to the global cost.  Higher values
      indicate more confident breaks.

    Parameters:
        data: Time series to analyse.  If a ``pd.Series``, the index
            is preserved in the output.
        penalty: Penalty method or explicit float value.
            Supported strings:

            - ``"bic"`` -- Bayesian Information Criterion (default).
              Works well when the number of changepoints is unknown.
            - ``"aic"`` -- Akaike Information Criterion.  Tends to
              over-segment; prefer BIC unless you want sensitivity.
            - ``"mbic"`` -- Modified BIC (Liu et al., 2016).

            If a float is given, it is used directly as the penalty
            value *beta*.
        model: Cost function for each segment.  Options:

            - ``"l2"`` -- least-squares cost (Gaussian change in mean).
            - ``"l1"`` -- absolute deviation cost (robust to outliers).
            - ``"rbf"`` -- kernel cost (nonparametric, captures complex
              distribution changes).
            - ``"linear"`` -- linear regression cost (for trend breaks).
            - ``"normal"`` -- full Gaussian cost (mean and variance).
            - ``"ar"`` -- autoregressive model cost (order-1 AR).
        min_size: Minimum segment length.  Shorter segments are
            prohibited.  Increase to suppress spurious breaks.
        jump: Subsample factor for candidate changepoints.  ``jump=1``
            considers every index; larger values trade accuracy for
            speed on very long series.
        n_bkps: If given, ignore ``penalty`` and find exactly this
            many changepoints (uses the Pelt ``predict`` with ``n_bkps``).

    Returns:
        Dictionary with:

        - **changepoints** (list[int]): Detected changepoint indices
          (0-based). The last element is always the series length.
        - **n_changepoints** (int): Number of detected changepoints
          (excludes the trailing length marker).
        - **segment_stats** (list[dict]): Per-segment statistics with
          keys ``start``, ``end``, ``mean``, ``var``, ``length``.
        - **confidence** (np.ndarray): Heuristic confidence score for
          each changepoint (between 0 and 1).
        - **penalty_value** (float): Effective penalty value used.
        - **model** (str): Cost model used.
        - **cost** (float): Total cost of the segmentation.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(0)
        >>> data = np.concatenate([
        ...     rng.normal(0, 1, 200),
        ...     rng.normal(5, 1, 200),
        ...     rng.normal(0, 2, 200),
        ... ])
        >>> result = pelt_changepoint(pd.Series(data))
        >>> print(result["changepoints"])  # ~ [200, 400, 600]
        >>> print(result["segment_stats"])

    References:
        Killick, R., Fearnhead, P. & Eckley, I. A. (2012). "Optimal
        Detection of Changepoints with a Linear Computational Cost."
        *J. Amer. Statist. Assoc.*, 107(500).

    See Also:
        binary_segmentation: Hierarchical top-down changepoint detection.
        window_changepoint: Sliding-window change score time series.
        online_changepoint: Bayesian online detection (sequential).
    """
    import ruptures as rpt

    values = _extract_values(data)
    T = len(values)

    # Build the cost model and algorithm
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(values)

    # Determine penalty
    if n_bkps is not None:
        # Use Dynp for exact n_bkps
        algo_exact = rpt.Dynp(model=model, min_size=min_size, jump=jump).fit(values)
        bkps = algo_exact.predict(n_bkps=n_bkps)
        pen_val = 0.0
    elif isinstance(penalty, (int, float)):
        pen_val = float(penalty)
        bkps = algo.predict(pen=pen_val)
    else:
        # String penalty
        pen_val = _compute_penalty(values, penalty, model)
        bkps = algo.predict(pen=pen_val)

    # Segment statistics
    boundaries = [0] + bkps
    segment_stats = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        seg = values[start:end]
        segment_stats.append({
            "start": start,
            "end": end,
            "mean": float(np.mean(seg)),
            "var": float(np.var(seg, ddof=1)) if len(seg) > 1 else 0.0,
            "length": end - start,
        })

    # Confidence heuristic: compare cost reduction at each changepoint
    # vs the total single-segment cost
    cost_model = rpt.costs.cost_factory(model)
    cost_model.fit(values)
    total_cost_one_seg = cost_model.error(0, T)
    seg_costs = [cost_model.error(boundaries[i], boundaries[i + 1])
                 for i in range(len(boundaries) - 1)]
    total_seg_cost = sum(seg_costs)

    # Confidence per changepoint
    n_cp = len(bkps) - 1  # exclude trailing T
    confidence = np.zeros(max(n_cp, 0))
    if n_cp > 0 and total_cost_one_seg > 0:
        cost_reduction = total_cost_one_seg - total_seg_cost
        for i in range(n_cp):
            # Merge segments i and i+1 and measure cost increase
            merged_cost = cost_model.error(boundaries[i], boundaries[i + 2])
            split_cost = seg_costs[i] + seg_costs[i + 1]
            reduction_i = merged_cost - split_cost
            confidence[i] = float(np.clip(
                reduction_i / total_cost_one_seg, 0.0, 1.0
            ))

    return {
        "changepoints": bkps,
        "n_changepoints": n_cp,
        "segment_stats": segment_stats,
        "confidence": confidence,
        "penalty_value": pen_val,
        "model": model,
        "cost": total_seg_cost,
    }


# ---------------------------------------------------------------------------
# Binary Segmentation
# ---------------------------------------------------------------------------


@requires_extra("regimes")
def binary_segmentation(
    data: pd.Series | np.ndarray,
    *,
    n_bkps: int | None = None,
    penalty: str | float = "bic",
    model: str = "l2",
    min_size: int = 5,
    jump: int = 1,
) -> dict[str, Any]:
    """Binary segmentation with hierarchical changepoint ordering.

    Binary segmentation is a fast, greedy, top-down algorithm that
    recursively splits the series at the point of maximum cost
    improvement.  Although sub-optimal compared to PELT, it is
    computationally faster and produces a natural **hierarchy** of
    changepoints ordered by significance.

    **Interpretation guidance:**

    - ``hierarchical_order[0]`` is the single most important
      changepoint in the series.  If you can only act on one break,
      act on this one.
    - The ordering degrades gracefully: if you want *K* changepoints,
      take the first *K* entries of ``hierarchical_order``.

    Parameters:
        data: Time series to analyse.
        n_bkps: Number of changepoints to detect.  If ``None``, the
            penalty is used to determine the number automatically.
        penalty: Penalty method or explicit float.  Same options as
            :func:`pelt_changepoint`.  Ignored when ``n_bkps`` is set.
        model: Cost function (``"l1"``, ``"l2"``, ``"rbf"``,
            ``"linear"``, ``"normal"``).
        min_size: Minimum segment length.
        jump: Candidate subsample factor.

    Returns:
        Dictionary with:

        - **changepoints** (list[int]): Detected changepoints (last
          element is always series length).
        - **n_changepoints** (int): Number of changepoints.
        - **hierarchical_order** (list[int]): Changepoints sorted from
          most significant (largest cost improvement) to least.
        - **segment_stats** (list[dict]): Per-segment ``start``,
          ``end``, ``mean``, ``var``, ``length``.
        - **model** (str): Cost model used.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> data = np.concatenate([
        ...     rng.normal(0, 1, 200),
        ...     rng.normal(5, 2, 100),
        ...     rng.normal(0, 1, 200),
        ... ])
        >>> result = binary_segmentation(pd.Series(data), n_bkps=2)
        >>> # First entry is the dominant changepoint
        >>> print(result["hierarchical_order"])

    References:
        Scott, A. J. & Knott, M. (1974). "A Cluster Analysis Method
        for Grouping Means in the Analysis of Variance." *Biometrics*.

    See Also:
        pelt_changepoint: Globally optimal changepoint detection.
        window_changepoint: Sliding-window approach.
    """
    import ruptures as rpt

    values = _extract_values(data)
    T = len(values)

    algo = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(values)

    if n_bkps is not None:
        bkps = algo.predict(n_bkps=n_bkps)
    else:
        if isinstance(penalty, (int, float)):
            pen_val = float(penalty)
        else:
            pen_val = _compute_penalty(values, penalty, model)
        bkps = algo.predict(pen=pen_val)

    # Build hierarchical order via the greedy split sequence.
    # Binary segmentation internally records splits in order of
    # cost improvement.  We approximate this by computing the cost
    # reduction each changepoint contributes.
    cost_model = rpt.costs.cost_factory(model)
    cost_model.fit(values)

    boundaries = sorted(set([0] + bkps))
    cp_list = sorted(set(bkps) - {T})

    cost_reductions: list[tuple[float, int]] = []
    for cp in cp_list:
        # Find the segment that this cp splits
        idx_in_bounds = boundaries.index(cp)
        if idx_in_bounds == 0:
            left = 0
        else:
            left = boundaries[idx_in_bounds - 1]
        if idx_in_bounds + 1 < len(boundaries):
            right = boundaries[idx_in_bounds + 1]
        else:
            right = T

        merged = cost_model.error(left, right)
        split = cost_model.error(left, cp) + cost_model.error(cp, right)
        cost_reductions.append((merged - split, cp))

    cost_reductions.sort(key=lambda x: -x[0])
    hierarchical_order = [cp for _, cp in cost_reductions]

    # Segment statistics
    seg_boundaries = [0] + sorted(cp_list) + [T]
    seg_boundaries = sorted(set(seg_boundaries))
    segment_stats = []
    for i in range(len(seg_boundaries) - 1):
        start = seg_boundaries[i]
        end = seg_boundaries[i + 1]
        seg = values[start:end]
        segment_stats.append({
            "start": start,
            "end": end,
            "mean": float(np.mean(seg)),
            "var": float(np.var(seg, ddof=1)) if len(seg) > 1 else 0.0,
            "length": end - start,
        })

    return {
        "changepoints": bkps,
        "n_changepoints": len(cp_list),
        "hierarchical_order": hierarchical_order,
        "segment_stats": segment_stats,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Sliding-window changepoint detection
# ---------------------------------------------------------------------------


def window_changepoint(
    data: pd.Series | np.ndarray,
    *,
    width: int = 50,
    model: str = "l2",
    min_size: int = 5,
    jump: int = 1,
    n_bkps: int | None = None,
    penalty: str | float = "bic",
) -> dict[str, Any]:
    """Sliding-window changepoint detection with change-score time series.

    Compares the distributions on either side of a sliding window to
    produce a continuous **change score** at every time step.  Peaks
    in this score indicate likely changepoints.  This approach is well
    suited for online/streaming applications where you want a
    continuously updated change intensity signal.

    **Interpretation guidance:**

    - The ``change_score`` series is analogous to a "structural break
      intensity" indicator.  Threshold it or combine it with other
      signals to trigger regime-switch alerts.
    - The score is non-negative and unitless; higher values indicate
      stronger evidence for a distributional shift.

    Parameters:
        data: Time series to analyse.
        width: Half-width of the sliding window.  Total window is
            ``2 * width``.  Larger windows are more robust but less
            responsive.
        model: Cost function (``"l2"``, ``"l1"``, ``"rbf"``, etc.).
        min_size: Minimum segment length.
        jump: Candidate subsample factor.
        n_bkps: If given, return exactly this many changepoints.
        penalty: Penalty method or float.  Used when ``n_bkps``
            is ``None``.

    Returns:
        Dictionary with:

        - **change_score** (pd.Series or np.ndarray): Change score at
          each time step.  Same type as input.
        - **changepoints** (list[int]): Detected changepoint indices.
        - **n_changepoints** (int): Number of changepoints.
        - **width** (int): Window half-width used.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> data = np.concatenate([
        ...     rng.normal(0, 1, 200),
        ...     rng.normal(3, 1, 200),
        ... ])
        >>> result = window_changepoint(pd.Series(data), width=30)
        >>> score = result["change_score"]
        >>> # Plot score to see the peak near index 200

    References:
        Truong, C., Oudre, L. & Vayer, N. (2020). "Selective review
        of offline change point detection methods." *Signal Processing*.

    See Also:
        pelt_changepoint: Offline optimal detection.
        online_changepoint: Bayesian online detection.
    """
    import ruptures as rpt

    values = _extract_values(data)
    T = len(values)

    algo = rpt.Window(
        width=width, model=model, min_size=min_size, jump=jump,
    ).fit(values)

    if n_bkps is not None:
        bkps = algo.predict(n_bkps=n_bkps)
    else:
        if isinstance(penalty, (int, float)):
            pen_val = float(penalty)
        else:
            pen_val = _compute_penalty(values, penalty, model)
        bkps = algo.predict(pen=pen_val)

    # Extract the change score from the cost matrix
    # The score at each point is the cost reduction of splitting there
    cost_model = rpt.costs.cost_factory(model)
    cost_model.fit(values)

    change_score = np.zeros(T)
    for t in range(width, T - width):
        left_cost = cost_model.error(max(0, t - width), t)
        right_cost = cost_model.error(t, min(T, t + width))
        merged_cost = cost_model.error(max(0, t - width), min(T, t + width))
        change_score[t] = max(0.0, merged_cost - left_cost - right_cost)

    cp_list = sorted(set(bkps) - {T})

    if isinstance(data, pd.Series):
        change_score_out = pd.Series(
            change_score, index=data.index, name="change_score",
        )
    else:
        change_score_out = change_score

    return {
        "change_score": change_score_out,
        "changepoints": bkps,
        "n_changepoints": len(cp_list),
        "width": width,
    }


# ---------------------------------------------------------------------------
# CUSUM control chart
# ---------------------------------------------------------------------------


def cusum_control_chart(
    data: pd.Series | np.ndarray,
    *,
    target: float | None = None,
    std_est: float | None = None,
    k: float = 0.5,
    h: float = 5.0,
) -> dict[str, Any]:
    """CUSUM (Cumulative Sum) control chart for process monitoring.

    The CUSUM chart monitors the cumulative sum of deviations from a
    target value.  It is one of the most powerful sequential tests for
    detecting small, persistent shifts in the mean of a process.

    The upper and lower CUSUM statistics are:

    .. math::

        S^+_t = \\max(0,\\; S^+_{t-1} + z_t - k)

        S^-_t = \\max(0,\\; S^-_{t-1} - z_t - k)

    where :math:`z_t = (x_t - \\mu_0) / \\sigma` is the standardised
    observation, *k* is the allowance (slack) parameter, and *h* is
    the decision threshold.  An alarm is triggered when either
    :math:`S^+_t > h` or :math:`S^-_t > h`.

    **Interpretation guidance:**

    - ``upper_cusum`` rising indicates the process mean is drifting
      **above** target.  ``lower_cusum`` rising indicates drift
      **below** target.
    - ``alarm_points`` are the times at which the CUSUM crossed the
      control limit.  These are your structural break candidates.
    - ``arl`` (average run length) is the expected number of
      observations between false alarms under the null hypothesis.
      Higher ARL means fewer false positives.  With *k* = 0.5 and
      *h* = 5, the in-control ARL is approximately 465 (for a
      standard normal process).

    Parameters:
        data: Time series to monitor.
        target: Target (in-control) mean.  If ``None``, the overall
            sample mean is used.
        std_est: Standard deviation estimate.  If ``None``, the sample
            standard deviation is used.
        k: Allowance (slack) parameter, in units of standard deviation.
            Controls the size of the shift you want to detect.  Default
            0.5 detects a 1-sigma shift optimally.
        h: Decision interval (control limit), in units of standard
            deviation.  Larger *h* reduces false alarms but increases
            detection delay.

    Returns:
        Dictionary with:

        - **upper_cusum** (np.ndarray): Upper CUSUM statistic at each
          time step.
        - **lower_cusum** (np.ndarray): Lower CUSUM statistic.
        - **control_limit** (float): The *h* threshold used.
        - **alarm_points** (list[int]): Indices where an alarm was
          triggered (CUSUM exceeded *h*).
        - **alarm_direction** (list[str]): Direction of each alarm
          (``"upper"`` or ``"lower"``).
        - **arl** (float): Estimated average run length (between
          alarms) based on the observed alarm sequence.  ``inf`` if
          no alarms were triggered.
        - **target** (float): Target mean used.
        - **std** (float): Standard deviation estimate used.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> # In-control for 200 obs, then shift mean by +1 sigma
        >>> data = np.concatenate([
        ...     rng.normal(0, 1, 200),
        ...     rng.normal(1, 1, 200),
        ... ])
        >>> result = cusum_control_chart(pd.Series(data), target=0, std_est=1)
        >>> print(result["alarm_points"])  # first alarm near index 200
        >>> print(f"ARL: {result['arl']:.0f}")

    References:
        Page, E. S. (1954). "Continuous Inspection Schemes."
        *Biometrika*, 41(1/2).

        Montgomery, D. C. (2009). *Introduction to Statistical Quality
        Control*. 6th ed. Wiley.

    See Also:
        online_changepoint: Bayesian approach to sequential detection.
        window_changepoint: Sliding-window approach.
    """
    values = _extract_values(data)
    T = len(values)

    if target is None:
        target = float(np.mean(values))
    if std_est is None:
        std_est = float(np.std(values, ddof=1))
        if std_est < 1e-12:
            std_est = 1.0

    # Standardise
    z = (values - target) / std_est

    upper_cusum = np.zeros(T)
    lower_cusum = np.zeros(T)
    alarm_points: list[int] = []
    alarm_direction: list[str] = []

    for t in range(T):
        if t == 0:
            upper_cusum[t] = max(0.0, z[t] - k)
            lower_cusum[t] = max(0.0, -z[t] - k)
        else:
            upper_cusum[t] = max(0.0, upper_cusum[t - 1] + z[t] - k)
            lower_cusum[t] = max(0.0, lower_cusum[t - 1] - z[t] - k)

        if upper_cusum[t] > h:
            alarm_points.append(t)
            alarm_direction.append("upper")
            upper_cusum[t] = 0.0  # reset after alarm
        if lower_cusum[t] > h:
            alarm_points.append(t)
            alarm_direction.append("lower")
            lower_cusum[t] = 0.0  # reset after alarm

    # Average run length
    if len(alarm_points) >= 2:
        diffs = np.diff(alarm_points)
        arl = float(np.mean(diffs))
    elif len(alarm_points) == 1:
        arl = float(alarm_points[0])
    else:
        arl = float("inf")

    return {
        "upper_cusum": upper_cusum,
        "lower_cusum": lower_cusum,
        "control_limit": h,
        "alarm_points": alarm_points,
        "alarm_direction": alarm_direction,
        "arl": arl,
        "target": target,
        "std": std_est,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_values(data: pd.Series | np.ndarray) -> np.ndarray:
    """Convert input to a clean 1-D float64 numpy array."""
    if isinstance(data, pd.Series):
        arr = data.dropna().values.astype(np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64).flatten()
        arr = arr[~np.isnan(arr)]
    return arr


def _compute_penalty(values: np.ndarray, method: str, model: str) -> float:
    """Compute a penalty value from a named method.

    Supports ``"bic"``, ``"aic"``, and ``"mbic"``.
    """
    T = len(values)
    n_params = 2  # mean + variance per segment for most models
    if model in ("normal",):
        n_params = 2
    elif model in ("l1", "l2"):
        n_params = 1

    if method == "bic":
        return n_params * np.log(T)
    elif method == "aic":
        return 2.0 * n_params
    elif method == "mbic":
        return 3.0 * n_params * np.log(T)
    else:
        raise ValueError(
            f"Unknown penalty method: {method!r}. "
            f"Choose from 'bic', 'aic', 'mbic' or pass a float."
        )
