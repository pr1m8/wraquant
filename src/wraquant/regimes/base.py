"""Regime detection abstractions and unified interfaces.

Provides a standardized framework for regime detection so that all
methods (HMM, GMM, Markov-switching, changepoint) return consistent
results and can be swapped without changing downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from wraquant.regimes.hmm import (
    _compute_steady_state,
    regime_statistics as _regime_statistics,
)


# ---------------------------------------------------------------------------
# RegimeResult — standardized container
# ---------------------------------------------------------------------------


@dataclass
class RegimeResult:
    """Standardized container for regime detection results.

    All regime detection functions return this object, ensuring
    consistent access to states, probabilities, and statistics
    regardless of the detection method used.

    Attributes:
        states: Integer regime labels (T,). State 0 is always the
            lowest-volatility regime.
        probabilities: Posterior regime probabilities (T, K).
            Row t gives P(regime=k | data) at time t.
        transition_matrix: (K, K) matrix where entry (i,j) is
            P(next_regime=j | current_regime=i). Rows sum to 1.
        n_regimes: Number of detected regimes.
        means: Per-regime mean returns. Shape (K,) for univariate,
            (K, n_assets) for multivariate.
        covariances: Per-regime variance/covariance. Shape (K,) for
            univariate, (K, n_assets, n_assets) for multivariate.
        statistics: pd.DataFrame with per-regime summary stats
            (mean, std, sharpe, max_drawdown, duration, pct_time).
        method: String identifying the detection method used.
        model: The underlying fitted model object (for advanced use).
        metadata: Additional method-specific results.
    """

    states: np.ndarray
    probabilities: np.ndarray
    transition_matrix: np.ndarray
    n_regimes: int
    means: np.ndarray
    covariances: np.ndarray
    statistics: pd.DataFrame
    method: str
    model: Any = None
    metadata: dict = field(default_factory=dict)

    @property
    def current_regime(self) -> int:
        """Most likely current regime (last observation)."""
        return int(self.states[-1])

    @property
    def current_probabilities(self) -> np.ndarray:
        """Regime probabilities for the most recent observation."""
        return self.probabilities[-1]

    @property
    def steady_state(self) -> np.ndarray:
        """Long-run (ergodic) regime distribution."""
        return _compute_steady_state(self.transition_matrix)

    @property
    def expected_durations(self) -> np.ndarray:
        """Expected duration in each regime: 1 / (1 - p_ii)."""
        return 1.0 / np.maximum(1.0 - np.diag(self.transition_matrix), 1e-10)

    def filter_signals(
        self,
        signals: pd.Series | np.ndarray,
    ) -> pd.Series | np.ndarray:
        """Filter trading signals based on regime probabilities.

        Delegates to ``wraquant.backtest.position.regime_signal_filter``
        to zero out signals in unfavourable regimes.

        Parameters:
            signals: Raw trading signal series (same length as states).

        Returns:
            Regime-filtered signals.
        """
        from wraquant.backtest.position import regime_signal_filter

        return regime_signal_filter(signals, self.probabilities)

    def plot(self) -> Any:
        """Plot regime detection results using viz dashboard.

        Returns:
            Plotly figure object from ``viz.dashboard.regime_dashboard``.
        """
        from wraquant.viz.dashboard import regime_dashboard

        return regime_dashboard(self)

    def summary(self) -> str:
        """Human-readable summary of regime detection results.

        Returns:
            Multi-line string with regime statistics.
        """
        lines = [
            f"RegimeResult (method={self.method}, n_regimes={self.n_regimes})",
            f"  Current regime: {self.current_regime}",
            f"  Expected durations: {self.expected_durations}",
        ]
        if self.statistics is not None and len(self.statistics) > 0:
            lines.append(f"  Statistics:\n{self.statistics.to_string()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RegimeDetector — base class / protocol
# ---------------------------------------------------------------------------


class RegimeDetector(ABC):
    """Base class for regime detection methods.

    Subclass this to create custom regime detectors that integrate
    seamlessly with wraquant's regime framework.

    A detector has a scikit-learn-like API: ``fit`` learns from data,
    ``predict`` assigns regimes to (possibly new) data, and
    ``predict_proba`` returns soft assignments. The ``to_result``
    method packages everything into a :class:`RegimeResult`.
    """

    @abstractmethod
    def fit(self, returns: pd.Series | pd.DataFrame | np.ndarray) -> "RegimeDetector":
        """Fit the detector to return data.

        Parameters:
            returns: Return series or multi-asset return DataFrame.

        Returns:
            self, for method chaining.
        """

    @abstractmethod
    def predict(self, returns: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
        """Assign regime labels to each observation.

        Parameters:
            returns: Return series.

        Returns:
            Integer state labels, shape (T,).
        """

    @abstractmethod
    def predict_proba(self, returns: pd.Series | pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return posterior regime probabilities.

        Parameters:
            returns: Return series.

        Returns:
            Probabilities array, shape (T, K).
        """

    @abstractmethod
    def to_result(self) -> RegimeResult:
        """Package fitted results into a RegimeResult.

        Returns:
            RegimeResult with all standard fields populated.
        """


# ---------------------------------------------------------------------------
# detect_regimes — unified high-level function
# ---------------------------------------------------------------------------


def detect_regimes(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    method: str = "hmm",
    n_regimes: int = 2,
    **kwargs: Any,
) -> RegimeResult:
    """Detect market regimes using the specified method.

    This is the primary entry point for regime detection. It dispatches
    to the appropriate specialized function and returns a standardized
    RegimeResult regardless of method.

    Parameters:
        returns: Return series or multi-asset return DataFrame.
        method: Detection method. Options:
            - ``"hmm"`` -- Gaussian Hidden Markov Model (default).
              Best for: capturing temporal regime persistence.
            - ``"gmm"`` -- Gaussian Mixture Model.
              Best for: quick classification without temporal structure.
            - ``"ms_regression"`` -- Markov-switching regression.
              Best for: regime-dependent mean and variance modeling.
            - ``"changepoint"`` -- Bayesian online changepoint detection.
              Best for: detecting abrupt structural breaks.
            - ``"kmeans"`` -- K-means on rolling statistics.
              Best for: simple, fast, interpretable regime labels.
        n_regimes: Number of regimes to detect.
        **kwargs: Additional arguments passed to the underlying method.

    Returns:
        RegimeResult with standardized fields.

    Example:
        >>> result = detect_regimes(returns, method="hmm", n_regimes=2)
        >>> print(f"Current regime: {result.current_regime}")
        >>> print(f"Regime probabilities: {result.current_probabilities}")
        >>> print(result.statistics)
    """
    dispatch = {
        "hmm": _detect_hmm,
        "gmm": _detect_gmm,
        "ms_regression": _detect_ms_regression,
        "changepoint": _detect_changepoint,
        "kmeans": _detect_kmeans,
    }

    detector = dispatch.get(method)
    if detector is None:
        raise ValueError(
            f"Unknown method: {method!r}. Choose from {sorted(dispatch)}."
        )
    return detector(returns, n_regimes=n_regimes, **kwargs)


# ---------------------------------------------------------------------------
# regime_report — comprehensive analysis
# ---------------------------------------------------------------------------


def regime_report(
    returns: pd.Series | pd.DataFrame,
    result: RegimeResult,
    risk_free_rate: float = 0.0,
    annualization: int = 252,
) -> dict[str, Any]:
    """Generate a comprehensive regime analysis report.

    Parameters:
        returns: Return series used for detection.
        result: RegimeResult from detect_regimes or any detection function.
        risk_free_rate: Annual risk-free rate for Sharpe calculations.
        annualization: Trading days per year.

    Returns:
        Dictionary containing:
        - **summary** -- pd.DataFrame with per-regime statistics
        - **current_regime** -- Current regime info (label, probability,
          duration so far)
        - **transition_analysis** -- Expected durations, steady state,
          visit counts
        - **regime_history** -- pd.DataFrame with date, regime,
          probability columns
        - **allocation_suggestion** -- Suggested portfolio adjustment
          based on regime
        - **risk_assessment** -- Per-regime VaR, CVaR, max drawdown
    """
    from wraquant.regimes.hmm import regime_transition_analysis

    r_arr = _to_1d_array(returns)

    # --- summary ---
    summary = result.statistics.copy()

    # --- current_regime info ---
    cur = result.current_regime
    cur_probs = result.current_probabilities

    # How long have we been in the current regime?
    states = result.states
    duration_so_far = 0
    for s in reversed(states):
        if int(s) == cur:
            duration_so_far += 1
        else:
            break

    current_regime_info = {
        "label": cur,
        "probability": float(cur_probs[cur]) if cur < len(cur_probs) else 0.0,
        "duration_so_far": duration_so_far,
    }

    # --- transition_analysis ---
    trans = regime_transition_analysis(states, result.transition_matrix)

    transition_analysis = {
        "expected_durations": result.expected_durations,
        "steady_state": result.steady_state,
        "visit_counts": trans["regime_counts"],
    }

    # --- regime_history ---
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        idx = returns.index if isinstance(returns, pd.Series) else returns.index
        # Align length (returns may have been cleaned of NaNs)
        if len(idx) == len(states):
            hist_index = idx
        else:
            hist_index = range(len(states))
    else:
        hist_index = range(len(states))

    regime_history = pd.DataFrame(
        {
            "regime": states,
            **{
                f"prob_{k}": result.probabilities[:, k]
                for k in range(result.n_regimes)
            },
        },
        index=hist_index,
    )

    # --- allocation_suggestion ---
    allocation_suggestion = _suggest_allocation(result)

    # --- risk_assessment ---
    risk_assessment = _compute_risk_assessment(
        r_arr, states, result.n_regimes, annualization
    )

    return {
        "summary": summary,
        "current_regime": current_regime_info,
        "transition_analysis": transition_analysis,
        "regime_history": regime_history,
        "allocation_suggestion": allocation_suggestion,
        "risk_assessment": risk_assessment,
    }


# ---------------------------------------------------------------------------
# Internal dispatch functions
# ---------------------------------------------------------------------------


def _detect_hmm(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    n_regimes: int = 2,
    **kwargs: Any,
) -> RegimeResult:
    """Dispatch to fit_gaussian_hmm and wrap as RegimeResult."""
    from wraquant.regimes.hmm import fit_gaussian_hmm

    # fit_gaussian_hmm expects univariate input
    series_input = _to_series_or_array(returns)
    r_arr = _to_1d_array(returns)

    hmm_kwargs = {
        "n_states": n_regimes,
        "n_init": kwargs.pop("n_init", 10),
        "n_iter": kwargs.pop("n_iter", 200),
        "covariance_type": kwargs.pop("covariance_type", "full"),
        "tol": kwargs.pop("tol", 1e-4),
        "random_state": kwargs.pop("random_state", 42),
    }
    raw = fit_gaussian_hmm(series_input, **hmm_kwargs)

    stats = _regime_statistics(r_arr, raw["states"])

    return RegimeResult(
        states=raw["states"],
        probabilities=raw["state_probs"],
        transition_matrix=raw["transition_matrix"],
        n_regimes=n_regimes,
        means=raw["means"],
        covariances=raw["covariances"],
        statistics=stats,
        method="hmm",
        model=raw["model"],
        metadata={
            "log_likelihood": raw["log_likelihood"],
            "aic": raw["aic"],
            "bic": raw["bic"],
            "startprob": raw["startprob"],
            "index": raw["index"],
        },
    )


def _detect_gmm(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    n_regimes: int = 2,
    **kwargs: Any,
) -> RegimeResult:
    """Dispatch to gaussian_mixture_regimes and wrap as RegimeResult."""
    from wraquant.regimes.hmm import gaussian_mixture_regimes

    series_input = _to_series_or_array(returns)
    r_arr = _to_1d_array(returns)

    gmm_kwargs = {
        "n_components": n_regimes,
        "covariance_type": kwargs.pop("covariance_type", "full"),
        "n_init": kwargs.pop("n_init", 10),
        "random_state": kwargs.pop("random_state", 42),
    }
    raw = gaussian_mixture_regimes(series_input, **gmm_kwargs)

    stats = _regime_statistics(r_arr, raw["states"])

    # GMMs have no true transition matrix; build one empirically
    transmat = _empirical_transition_matrix(raw["states"], n_regimes)

    return RegimeResult(
        states=raw["states"],
        probabilities=raw["state_probs"],
        transition_matrix=transmat,
        n_regimes=n_regimes,
        means=raw["means"],
        covariances=raw["covariances"],
        statistics=stats,
        method="gmm",
        model=raw["model"],
        metadata={
            "weights": raw["weights"],
            "aic": raw["aic"],
            "bic": raw["bic"],
            "index": raw["index"],
        },
    )


def _detect_ms_regression(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    n_regimes: int = 2,
    **kwargs: Any,
) -> RegimeResult:
    """Dispatch to fit_ms_regression and wrap as RegimeResult."""
    from wraquant.regimes.hmm import fit_ms_regression

    series_input = _to_series_or_array(returns)
    r_arr = _to_1d_array(returns)

    ms_kwargs = {
        "k_regimes": n_regimes,
        "trend": kwargs.pop("trend", "c"),
        "switching_variance": kwargs.pop("switching_variance", True),
        "exog": kwargs.pop("exog", None),
        "n_iter": kwargs.pop("n_iter", 200),
    }
    raw = fit_ms_regression(series_input, **ms_kwargs)

    stats = _regime_statistics(r_arr, raw["states"])

    # Extract means and covariances from regime_params
    means = np.array([raw["regime_params"].get(f"mean_{k}", 0.0) for k in range(n_regimes)])
    covariances = np.array(
        [raw["regime_params"].get(f"sigma2_{k}", 0.0) for k in range(n_regimes)]
    )

    return RegimeResult(
        states=raw["states"],
        probabilities=raw["smoothed_probs"],
        transition_matrix=raw["transition_matrix"],
        n_regimes=n_regimes,
        means=means,
        covariances=covariances,
        statistics=stats,
        method="ms_regression",
        model=raw["model_result"],
        metadata={
            "filtered_probs": raw["filtered_probs"],
            "regime_params": raw["regime_params"],
            "log_likelihood": raw["log_likelihood"],
            "aic": raw["aic"],
            "bic": raw["bic"],
            "summary": raw["summary"],
        },
    )


def _detect_changepoint(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    n_regimes: int = 2,
    **kwargs: Any,
) -> RegimeResult:
    """Dispatch to online_changepoint and wrap as RegimeResult."""
    from wraquant.regimes.changepoint import online_changepoint

    # Ensure we have a pd.Series for online_changepoint
    if isinstance(returns, pd.DataFrame):
        series = returns.iloc[:, 0]
    elif isinstance(returns, np.ndarray):
        series = pd.Series(returns.flatten())
    else:
        series = returns

    hazard = kwargs.pop("hazard", 0.005)
    run_lengths = online_changepoint(series, hazard=hazard)

    r_arr = _to_1d_array(returns)
    T = len(r_arr)

    # Convert run lengths into regime labels using changepoints
    # A drop in run length indicates a changepoint
    changepoints = [0]
    rl_vals = run_lengths.values
    for i in range(1, len(rl_vals)):
        if rl_vals[i] < rl_vals[i - 1]:
            changepoints.append(i)

    # Assign segment-based labels using K-means-like clustering on
    # segment means/variances, capped at n_regimes
    segments: list[tuple[int, int]] = []
    for i in range(len(changepoints)):
        start = changepoints[i]
        end = changepoints[i + 1] if i + 1 < len(changepoints) else T
        segments.append((start, end))

    # Compute segment statistics
    seg_means = []
    seg_vars = []
    for start, end in segments:
        seg_r = r_arr[start:end]
        seg_means.append(float(np.mean(seg_r)))
        seg_vars.append(float(np.var(seg_r)) if len(seg_r) > 1 else 0.0)

    # Cluster segments by variance into n_regimes groups
    seg_vars_arr = np.array(seg_vars)
    if len(seg_vars_arr) >= n_regimes:
        # Use quantile-based assignment
        thresholds = np.linspace(0, 100, n_regimes + 1)[1:-1]
        percentiles = np.percentile(seg_vars_arr, thresholds)
        seg_labels = np.digitize(seg_vars_arr, percentiles)
    else:
        seg_labels = np.arange(len(seg_vars_arr)) % n_regimes

    # Map segment labels to time series
    states = np.zeros(T, dtype=int)
    for idx, (start, end) in enumerate(segments):
        states[start:end] = seg_labels[idx]

    # Build probabilities (hard assignments for changepoint)
    actual_n_regimes = max(n_regimes, int(states.max()) + 1)
    probabilities = np.zeros((T, actual_n_regimes))
    for t in range(T):
        probabilities[t, states[t]] = 1.0

    # Trim to requested n_regimes
    probabilities = probabilities[:, :n_regimes]

    # Compute stats and transition matrix
    stats = _regime_statistics(r_arr, states)
    transmat = _empirical_transition_matrix(states, n_regimes)

    unique_states = np.unique(states)
    means_arr = np.array([float(np.mean(r_arr[states == k])) for k in range(n_regimes) if k in unique_states])
    covs_arr = np.array(
        [
            float(np.var(r_arr[states == k], ddof=1)) if np.sum(states == k) > 1 else 0.0
            for k in range(n_regimes)
            if k in unique_states
        ]
    )

    # Pad if fewer unique states than n_regimes
    if len(means_arr) < n_regimes:
        means_arr = np.pad(means_arr, (0, n_regimes - len(means_arr)))
        covs_arr = np.pad(covs_arr, (0, n_regimes - len(covs_arr)))

    return RegimeResult(
        states=states,
        probabilities=probabilities,
        transition_matrix=transmat,
        n_regimes=n_regimes,
        means=means_arr,
        covariances=covs_arr,
        statistics=stats,
        method="changepoint",
        model=None,
        metadata={
            "run_lengths": run_lengths,
            "changepoints": changepoints,
            "hazard": hazard,
        },
    )


def _detect_kmeans(
    returns: pd.Series | pd.DataFrame | np.ndarray,
    n_regimes: int = 2,
    **kwargs: Any,
) -> RegimeResult:
    """K-means on rolling statistics for simple regime detection."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    r_arr = _to_1d_array(returns)
    T = len(r_arr)

    window = kwargs.pop("window", min(21, T // 4))
    random_state = kwargs.pop("random_state", 42)

    # Compute rolling features
    roll_mean = pd.Series(r_arr).rolling(window, min_periods=1).mean().values
    roll_std = pd.Series(r_arr).rolling(window, min_periods=1).std().fillna(0).values

    features = np.column_stack([roll_mean, roll_std])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
    raw_states = km.fit_predict(features_scaled)

    # Re-order states by ascending volatility
    cluster_vars = np.array(
        [float(np.var(r_arr[raw_states == k], ddof=1)) if np.sum(raw_states == k) > 1 else 0.0 for k in range(n_regimes)]
    )
    order = np.argsort(cluster_vars)
    state_map = {int(old): new for new, old in enumerate(order)}
    states = np.array([state_map[int(s)] for s in raw_states])

    # Soft assignments based on distance to cluster centers
    distances = km.transform(features_scaled)  # (T, K)
    # Convert distances to pseudo-probabilities (inverse distance weighting)
    inv_dist = 1.0 / np.maximum(distances, 1e-10)
    probabilities = inv_dist / inv_dist.sum(axis=1, keepdims=True)
    # Re-order columns
    probabilities = probabilities[:, order]

    transmat = _empirical_transition_matrix(states, n_regimes)
    stats = _regime_statistics(r_arr, states)

    means = np.array([float(np.mean(r_arr[states == k])) for k in range(n_regimes)])
    covariances = np.array(
        [float(np.var(r_arr[states == k], ddof=1)) if np.sum(states == k) > 1 else 0.0 for k in range(n_regimes)]
    )

    return RegimeResult(
        states=states,
        probabilities=probabilities,
        transition_matrix=transmat,
        n_regimes=n_regimes,
        means=means,
        covariances=covariances,
        statistics=stats,
        method="kmeans",
        model=km,
        metadata={
            "scaler": scaler,
            "features": features,
            "window": window,
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_1d_array(
    returns: pd.Series | pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Convert returns input to a clean 1-D numpy array."""
    if isinstance(returns, pd.DataFrame):
        arr = returns.iloc[:, 0].dropna().values
    elif isinstance(returns, pd.Series):
        arr = returns.dropna().values
    else:
        arr = np.asarray(returns, dtype=np.float64).flatten()
        arr = arr[~np.isnan(arr)]
    return np.asarray(arr, dtype=np.float64)


def _to_series_or_array(
    returns: pd.Series | pd.DataFrame | np.ndarray,
) -> pd.Series | np.ndarray:
    """Extract univariate series from potentially multivariate input.

    If *returns* is a DataFrame, returns the first column as a Series.
    Otherwise returns the input unchanged.
    """
    if isinstance(returns, pd.DataFrame):
        return returns.iloc[:, 0]
    return returns


def _empirical_transition_matrix(states: np.ndarray, n_regimes: int) -> np.ndarray:
    """Compute empirical transition matrix from state sequence."""
    transmat = np.zeros((n_regimes, n_regimes))
    s = np.asarray(states, dtype=int).flatten()
    for t in range(len(s) - 1):
        i, j = int(s[t]), int(s[t + 1])
        if i < n_regimes and j < n_regimes:
            transmat[i, j] += 1

    # Normalise rows (handle rows with zero transitions)
    row_sums = transmat.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    transmat = transmat / row_sums

    return transmat


def _suggest_allocation(result: RegimeResult) -> dict[str, Any]:
    """Suggest portfolio allocation adjustments based on current regime."""
    cur = result.current_regime
    n = result.n_regimes
    cur_prob = float(result.current_probabilities[cur]) if cur < len(result.current_probabilities) else 0.0

    # Determine regime character by volatility ordering
    # State 0 = lowest vol, state n-1 = highest vol
    if n == 2:
        if cur == 0:
            suggestion = "risk-on"
            description = (
                "Low-volatility regime detected. Favour equity exposure "
                "and growth assets."
            )
        else:
            suggestion = "risk-off"
            description = (
                "High-volatility regime detected. Favour defensive assets, "
                "increase cash/bond allocation."
            )
    elif n >= 3:
        if cur == 0:
            suggestion = "risk-on"
            description = "Lowest-volatility regime. Full risk allocation."
        elif cur == n - 1:
            suggestion = "risk-off"
            description = "Highest-volatility regime. Maximum defensiveness."
        else:
            suggestion = "neutral"
            description = f"Intermediate regime {cur}. Moderate allocation."
    else:
        suggestion = "unknown"
        description = "Single regime detected."

    return {
        "suggestion": suggestion,
        "description": description,
        "confidence": cur_prob,
        "current_regime": cur,
    }


def _compute_risk_assessment(
    returns: np.ndarray,
    states: np.ndarray,
    n_regimes: int,
    annualization: int = 252,
) -> pd.DataFrame:
    """Compute per-regime risk metrics."""
    records = []
    for k in range(n_regimes):
        mask = states == k
        regime_r = returns[mask]
        n_obs = int(mask.sum())

        if n_obs < 2:
            records.append(
                {
                    "regime": k,
                    "VaR_95": 0.0,
                    "CVaR_95": 0.0,
                    "max_drawdown": 0.0,
                    "annualized_vol": 0.0,
                }
            )
            continue

        var_95 = float(np.percentile(regime_r, 5))
        cvar_mask = regime_r <= var_95
        cvar_95 = float(np.mean(regime_r[cvar_mask])) if cvar_mask.any() else var_95

        # Max drawdown within regime
        cumulative = np.cumprod(1.0 + regime_r)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / np.maximum(running_max, 1e-12)
        max_dd = float(np.min(drawdowns))

        ann_vol = float(np.std(regime_r, ddof=1) * np.sqrt(annualization))

        records.append(
            {
                "regime": k,
                "VaR_95": var_95,
                "CVaR_95": cvar_95,
                "max_drawdown": max_dd,
                "annualized_vol": ann_vol,
            }
        )

    return pd.DataFrame(records).set_index("regime")
