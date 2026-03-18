"""Regime quality assessment and method comparison.

Evaluates how *good* a set of detected regimes actually is by measuring
stability, separability, and predictability.  These scores help you
decide whether your regime model is capturing meaningful market structure
or just fitting noise.

Three orthogonal quality dimensions
------------------------------------

1. **Stability** (:func:`regime_stability_score`): Are the regimes
   persistent?  Rapidly flickering regimes are hard to trade.
2. **Separation** (:func:`regime_separation_score`): Are the regimes
   statistically distinct?  Overlapping distributions offer no
   actionable information.
3. **Predictability** (:func:`regime_predictability`): Can regime
   transitions be forecast?  A model is useful only if it gives you
   a head-start on the next shift.

Use :func:`compare_regime_methods` to run all three metrics across
multiple detection algorithms on the same data.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regime Stability Score
# ---------------------------------------------------------------------------


def regime_stability_score(
    states: np.ndarray | pd.Series,
    transition_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure how stable the detected regimes are.

    Stability quantifies the temporal persistence of regimes.  Stable
    regimes last many periods and transitions are infrequent.  A
    stability score close to 1 indicates regimes that are highly
    persistent; a score near 0 indicates rapid, noisy switching that
    is likely untradable.

    **How it works:**

    Three sub-metrics are combined (equal-weighted average):

    1. **Average duration** (normalised): longer average stays = more
       stable.  Normalised as ``1 - 1/avg_duration`` so it maps to
       [0, 1].
    2. **Transition frequency** (inverse): fewer transitions per unit
       time = more stable.  Computed as ``1 - n_transitions / (T-1)``.
    3. **Probability entropy** (inverse): when the transition matrix
       rows have low entropy (i.e., dominated by the diagonal), the
       chain is persistent.  Normalised by ``log(K)`` where *K* is
       the number of regimes.

    **Interpretation guidance:**

    - **> 0.8**: Highly stable regimes, suitable for discrete portfolio
      switches.
    - **0.5 -- 0.8**: Moderate stability, consider probability-weighted
      blending rather than hard switches.
    - **< 0.5**: Unstable, likely overfitting or too many regimes.

    Parameters:
        states: Integer regime labels, shape ``(T,)``.
        transition_matrix: Optional ``(K, K)`` transition matrix.  If
            ``None``, an empirical matrix is computed from *states*.

    Returns:
        Dictionary with:

        - **stability_score** (float): Composite stability score in [0, 1].
        - **avg_duration** (float): Average regime duration in periods.
        - **transition_frequency** (float): Fraction of time steps with
          a regime change.
        - **probability_entropy** (float): Mean row entropy of the
          transition matrix (nats).
        - **n_transitions** (int): Total number of regime transitions.
        - **per_regime_duration** (dict[int, float]): Average duration
          per regime.

    Example:
        >>> states = np.array([0]*100 + [1]*100 + [0]*100)
        >>> result = regime_stability_score(states)
        >>> print(f"Stability: {result['stability_score']:.2f}")
        0.99

    See Also:
        regime_separation_score: Distribution distance between regimes.
        compare_regime_methods: Multi-method comparison.
    """
    s = np.asarray(states, dtype=int).flatten()
    T = len(s)
    unique_states = np.unique(s)
    K = len(unique_states)

    # Transition count
    transitions = np.sum(s[1:] != s[:-1])
    transition_freq = transitions / max(T - 1, 1)

    # Per-regime durations
    durations: dict[int, list[int]] = {int(k): [] for k in unique_states}
    current_state = int(s[0])
    current_len = 1
    for t in range(1, T):
        if int(s[t]) == current_state:
            current_len += 1
        else:
            durations[current_state].append(current_len)
            current_state = int(s[t])
            current_len = 1
    durations[current_state].append(current_len)

    per_regime_duration = {
        k: float(np.mean(v)) if v else 0.0 for k, v in durations.items()
    }
    all_durations = [d for dlist in durations.values() for d in dlist]
    avg_duration = float(np.mean(all_durations)) if all_durations else 1.0

    # Transition matrix
    if transition_matrix is None:
        transition_matrix = _empirical_transmat(s, K)

    # Probability entropy: mean row entropy
    eps = 1e-12
    row_entropies = []
    for row in transition_matrix:
        p = row[row > eps]
        entropy = -float(np.sum(p * np.log(p)))
        row_entropies.append(entropy)
    mean_entropy = float(np.mean(row_entropies))
    max_entropy = np.log(K) if K > 1 else 1.0

    # Sub-scores
    duration_score = 1.0 - 1.0 / max(avg_duration, 1.0)
    freq_score = 1.0 - transition_freq
    entropy_score = 1.0 - mean_entropy / max(max_entropy, eps)

    stability = float(np.mean([duration_score, freq_score, entropy_score]))

    return {
        "stability_score": np.clip(stability, 0.0, 1.0),
        "avg_duration": avg_duration,
        "transition_frequency": transition_freq,
        "probability_entropy": mean_entropy,
        "n_transitions": int(transitions),
        "per_regime_duration": per_regime_duration,
    }


# ---------------------------------------------------------------------------
# Regime Separation Score
# ---------------------------------------------------------------------------


def regime_separation_score(
    returns: np.ndarray | pd.Series,
    states: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Measure how well-separated the regime distributions are.

    Separation quantifies whether the regimes have meaningfully
    distinct statistical properties.  High separation means the
    regimes carry actionable information; low separation means the
    classification is arbitrary.

    **How it works:**

    For each pair of regimes *(i, j)*:

    1. **Bhattacharyya distance**: measures divergence between two
       Gaussian distributions.  Larger = more separated.
    2. **Overlap coefficient**: fraction of probability mass that
       overlaps when the two regime distributions are superimposed.
       Smaller = more separated.

    The composite score is the average pairwise Bhattacharyya distance,
    normalised to [0, 1] via a sigmoid transform.

    **Interpretation guidance:**

    - **> 0.7**: Well-separated regimes with distinct return
      distributions.
    - **0.3 -- 0.7**: Moderate separation.  Regimes differ but there
      is meaningful overlap.
    - **< 0.3**: Poor separation.  Consider reducing the number of
      regimes.

    Parameters:
        returns: Return series, shape ``(T,)``.
        states: Integer regime labels, shape ``(T,)``.

    Returns:
        Dictionary with:

        - **separation_score** (float): Composite separation in [0, 1].
        - **pairwise_bhattacharyya** (dict[tuple, float]): Bhattacharyya
          distance for each regime pair ``(i, j)``.
        - **pairwise_overlap** (dict[tuple, float]): Overlap coefficient
          for each pair.
        - **per_regime_stats** (dict[int, dict]): Mean and std per
          regime.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> returns = np.concatenate([
        ...     rng.normal(0.01, 0.01, 200),
        ...     rng.normal(-0.02, 0.03, 200),
        ... ])
        >>> states = np.array([0]*200 + [1]*200)
        >>> result = regime_separation_score(returns, states)
        >>> print(f"Separation: {result['separation_score']:.2f}")

    See Also:
        regime_stability_score: Temporal persistence of regimes.
        compare_regime_methods: Multi-method comparison.
    """
    r = np.asarray(returns, dtype=np.float64).flatten()
    s = np.asarray(states, dtype=int).flatten()

    if len(r) != len(s):
        raise ValueError(
            f"returns and states must have same length, got {len(r)} vs {len(s)}"
        )

    unique_states = sorted(np.unique(s))
    K = len(unique_states)

    # Per-regime statistics
    per_regime_stats: dict[int, dict[str, float]] = {}
    for k in unique_states:
        mask = s == k
        rk = r[mask]
        per_regime_stats[int(k)] = {
            "mean": float(np.mean(rk)),
            "std": float(np.std(rk, ddof=1)) if len(rk) > 1 else 0.0,
            "n": int(len(rk)),
        }

    # Pairwise metrics
    pairwise_bhattacharyya: dict[tuple[int, int], float] = {}
    pairwise_overlap: dict[tuple[int, int], float] = {}

    for i_idx in range(K):
        for j_idx in range(i_idx + 1, K):
            ki, kj = unique_states[i_idx], unique_states[j_idx]
            mu_i = per_regime_stats[int(ki)]["mean"]
            mu_j = per_regime_stats[int(kj)]["mean"]
            s_i = max(per_regime_stats[int(ki)]["std"], 1e-12)
            s_j = max(per_regime_stats[int(kj)]["std"], 1e-12)

            # Bhattacharyya distance for Gaussians
            var_i, var_j = s_i ** 2, s_j ** 2
            avg_var = (var_i + var_j) / 2.0
            db = 0.25 * (mu_i - mu_j) ** 2 / avg_var + 0.5 * np.log(
                avg_var / np.sqrt(var_i * var_j)
            )
            pairwise_bhattacharyya[(int(ki), int(kj))] = float(db)

            # Overlap coefficient (Bhattacharyya coefficient = exp(-db))
            bc = float(np.exp(-db))
            pairwise_overlap[(int(ki), int(kj))] = bc

    # Composite score: sigmoid of average Bhattacharyya distance
    if pairwise_bhattacharyya:
        avg_db = float(np.mean(list(pairwise_bhattacharyya.values())))
        # Sigmoid: maps (0, inf) -> (0, 1), with db=2 -> ~0.76
        separation = float(1.0 - np.exp(-avg_db))
    else:
        separation = 0.0

    return {
        "separation_score": np.clip(separation, 0.0, 1.0),
        "pairwise_bhattacharyya": pairwise_bhattacharyya,
        "pairwise_overlap": pairwise_overlap,
        "per_regime_stats": per_regime_stats,
    }


# ---------------------------------------------------------------------------
# Regime Predictability
# ---------------------------------------------------------------------------


def regime_predictability(
    states: np.ndarray | pd.Series,
    transition_matrix: np.ndarray | None = None,
    test_fraction: float = 0.3,
) -> dict[str, Any]:
    """Assess whether regime transitions can be predicted.

    A regime model is useful for trading only if you can anticipate
    transitions before they happen (or at least classify the current
    regime in real time).  This function evaluates predictability
    using transition-probability-based forecasting.

    **How it works:**

    1. Split the state sequence into train / test at
       ``(1 - test_fraction) * T``.
    2. Estimate the transition matrix from the training set.
    3. For each test observation, predict the next state as the
       most probable successor under the estimated transition matrix.
    4. Compute out-of-sample accuracy and compare to a naive
       baseline (always predict the most common state).

    **Interpretation guidance:**

    - **accuracy > baseline + 0.1**: The transition structure adds
      predictive value.  Use regime probabilities to adjust
      portfolio weights.
    - **accuracy ~ baseline**: No predictive value from transitions.
      Consider whether the regime labeling is too noisy or the
      number of states is wrong.
    - **economic_value**: A rough estimate of the annualised Sharpe
      improvement from perfect regime prediction vs buy-and-hold.
      This uses per-regime Sharpe ratios and is indicative only.

    Parameters:
        states: Integer regime labels, shape ``(T,)``.
        transition_matrix: If provided, used instead of the
            training-set estimate.
        test_fraction: Fraction of data reserved for testing.

    Returns:
        Dictionary with:

        - **predictability_score** (float): Composite predictability
          in [0, 1].  Combines accuracy lift and transition regularity.
        - **accuracy** (float): Out-of-sample one-step prediction
          accuracy.
        - **baseline_accuracy** (float): Accuracy of always predicting
          the most common state.
        - **accuracy_lift** (float): ``accuracy - baseline_accuracy``.
        - **transition_matrix_train** (np.ndarray): Transition matrix
          estimated from training data.

    Example:
        >>> # Highly predictable two-state chain
        >>> rng = np.random.default_rng(0)
        >>> states = np.zeros(500, dtype=int)
        >>> for t in range(1, 500):
        ...     if states[t-1] == 0:
        ...         states[t] = 0 if rng.random() < 0.95 else 1
        ...     else:
        ...         states[t] = 1 if rng.random() < 0.90 else 0
        >>> result = regime_predictability(states)
        >>> print(f"Accuracy: {result['accuracy']:.2f}")

    See Also:
        regime_stability_score: Regime persistence.
        regime_separation_score: Distribution distinctness.
        compare_regime_methods: Multi-method comparison.
    """
    s = np.asarray(states, dtype=int).flatten()
    T = len(s)
    split = int(T * (1.0 - test_fraction))
    split = max(split, 2)
    split = min(split, T - 2)

    s_train = s[:split]
    s_test = s[split:]
    T_test = len(s_test)

    unique_states = np.unique(s)
    K = len(unique_states)

    # Estimate transition matrix from training data
    if transition_matrix is None:
        tm = _empirical_transmat(s_train, K)
    else:
        tm = np.array(transition_matrix, dtype=np.float64)

    # Out-of-sample one-step prediction
    correct = 0
    for t in range(T_test - 1):
        current = int(s_test[t])
        actual_next = int(s_test[t + 1])
        if current < K:
            predicted = int(np.argmax(tm[current]))
        else:
            predicted = 0
        if predicted == actual_next:
            correct += 1

    accuracy = correct / max(T_test - 1, 1)

    # Baseline: always predict most common state in test
    from collections import Counter

    counts = Counter(s_test.tolist())
    most_common = counts.most_common(1)[0][0]
    baseline_correct = sum(
        1 for t in range(T_test - 1) if int(s_test[t + 1]) == most_common
    )
    baseline_accuracy = baseline_correct / max(T_test - 1, 1)

    accuracy_lift = accuracy - baseline_accuracy

    # Composite predictability score
    # Combines accuracy lift (clipped to [0, 0.5], scaled to [0, 1])
    # and diagonal dominance of the transition matrix
    diag_strength = float(np.mean(np.diag(tm))) if K > 1 else 0.5
    lift_score = np.clip(accuracy_lift * 2.0, 0.0, 1.0)
    predictability = float(np.mean([lift_score, diag_strength]))

    return {
        "predictability_score": np.clip(predictability, 0.0, 1.0),
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_lift": accuracy_lift,
        "transition_matrix_train": tm,
    }


# ---------------------------------------------------------------------------
# Compare Regime Methods
# ---------------------------------------------------------------------------


def compare_regime_methods(
    returns: pd.Series | np.ndarray,
    methods: dict[str, Callable[..., Any]] | None = None,
    method_kwargs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Compare multiple regime detection methods on the same data.

    Runs each method on the provided return series and evaluates
    stability, separation, and predictability.  Returns a DataFrame
    for easy comparison.

    **Interpretation guidance:**

    - Use this to pick the best regime model for your asset or
      strategy.  The "best" model depends on your use-case:
      high **stability** for discrete allocation switches,
      high **separation** for regime-conditional signals,
      high **predictability** for transition timing.

    Parameters:
        returns: Return series, shape ``(T,)``.
        methods: Dictionary mapping method names to callables.  Each
            callable must accept ``returns`` and return a tuple
            ``(states, transition_matrix)`` where *states* is an
            integer array and *transition_matrix* is ``(K, K)`` or
            ``None``.

            If ``None``, a default set is used (requires
            ``detect_regimes`` from ``wraquant.regimes.base``).
        method_kwargs: Per-method keyword arguments.  Keys should match
            *methods* keys.

    Returns:
        pd.DataFrame indexed by method name with columns:

        - ``stability``, ``separation``, ``predictability``
        - ``avg_duration``, ``n_transitions``, ``n_regimes``
        - ``composite`` (equal-weighted average of the three scores)

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> returns = np.concatenate([
        ...     rng.normal(0.01, 0.01, 200),
        ...     rng.normal(-0.02, 0.03, 200),
        ... ])
        >>> df = compare_regime_methods(pd.Series(returns))
        >>> print(df.sort_values("composite", ascending=False))

    See Also:
        regime_stability_score: Stability metric details.
        regime_separation_score: Separation metric details.
        regime_predictability: Predictability metric details.
    """
    r = np.asarray(returns, dtype=np.float64).flatten()
    method_kwargs = method_kwargs or {}

    if methods is None:
        methods = _default_methods()

    records = []
    for name, method_fn in methods.items():
        kwargs = method_kwargs.get(name, {})
        try:
            result = method_fn(returns, **kwargs)
            if isinstance(result, tuple):
                states_arr, tm = result
            elif isinstance(result, dict):
                states_arr = result.get("states", np.zeros(len(r), dtype=int))
                tm = result.get("transition_matrix", None)
            else:
                # Assume it's a RegimeResult
                states_arr = result.states
                tm = result.transition_matrix
        except Exception as exc:
            # Record failure without crashing the comparison
            records.append({
                "method": name,
                "stability": float("nan"),
                "separation": float("nan"),
                "predictability": float("nan"),
                "avg_duration": float("nan"),
                "n_transitions": 0,
                "n_regimes": 0,
                "composite": float("nan"),
                "error": str(exc),
            })
            continue

        states_arr = np.asarray(states_arr, dtype=int).flatten()
        K = len(np.unique(states_arr))

        stab = regime_stability_score(states_arr, tm)
        sep = regime_separation_score(r, states_arr)
        pred = regime_predictability(states_arr, tm)

        composite = float(np.mean([
            stab["stability_score"],
            sep["separation_score"],
            pred["predictability_score"],
        ]))

        records.append({
            "method": name,
            "stability": stab["stability_score"],
            "separation": sep["separation_score"],
            "predictability": pred["predictability_score"],
            "avg_duration": stab["avg_duration"],
            "n_transitions": stab["n_transitions"],
            "n_regimes": K,
            "composite": composite,
        })

    df = pd.DataFrame(records)
    if "method" in df.columns:
        df = df.set_index("method")
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empirical_transmat(states: np.ndarray, n_regimes: int) -> np.ndarray:
    """Compute empirical transition matrix from a state sequence."""
    tm = np.zeros((n_regimes, n_regimes))
    s = np.asarray(states, dtype=int).flatten()
    for t in range(len(s) - 1):
        i, j = int(s[t]), int(s[t + 1])
        if i < n_regimes and j < n_regimes:
            tm[i, j] += 1
    row_sums = tm.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    return tm / row_sums


def _default_methods() -> dict[str, Callable[..., Any]]:
    """Build default method dictionary using detect_regimes."""
    from wraquant.regimes.base import detect_regimes

    def _make_detector(method: str) -> Callable[..., Any]:
        def _detect(returns: Any, **kwargs: Any) -> Any:
            return detect_regimes(returns, method=method, **kwargs)
        return _detect

    return {
        "hmm": _make_detector("hmm"),
        "gmm": _make_detector("gmm"),
        "kmeans": _make_detector("kmeans"),
    }
