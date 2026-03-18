"""Optimal stopping theory for finance.

Longstaff-Schwartz American option pricing, binomial American options,
optimal exit thresholds, sequential testing, and change-point detection.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "longstaff_schwartz",
    "binomial_american",
    "optimal_exit_threshold",
    "sequential_probability_ratio",
    "cusum_stopping",
    "secretary_problem_threshold",
]


# ---------------------------------------------------------------------------
# Longstaff-Schwartz
# ---------------------------------------------------------------------------

def longstaff_schwartz(
    paths: ArrayLike,
    strike: float,
    rf_rate: float,
    dt: float,
    basis_functions: str = "laguerre",
    option_type: str = "put",
) -> float:
    """Longstaff-Schwartz algorithm for American option pricing.

    Parameters
    ----------
    paths : array_like
        Simulated price paths of shape ``(n_paths, n_steps + 1)``.
        Column 0 is the initial price.
    strike : float
        Strike price.
    rf_rate : float
        Risk-free rate (annualised, continuously compounded).
    dt : float
        Time step (in years).
    basis_functions : {'laguerre', 'polynomial'}, optional
        Basis for the continuation-value regression (default ``'laguerre'``).
    option_type : {'put', 'call'}, optional
        Option type (default ``'put'``).

    Returns
    -------
    float
        Estimated American option price.
    """
    paths = np.asarray(paths, dtype=float)
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1
    discount = np.exp(-rf_rate * dt)

    # Payoff at each node
    if option_type == "put":
        payoff_fn: Callable[[np.ndarray], np.ndarray] = lambda s: np.maximum(strike - s, 0.0)
    else:
        payoff_fn = lambda s: np.maximum(s - strike, 0.0)

    # Cash flow matrix: stores the time at which the option is exercised
    # and the corresponding payoff
    cashflows = payoff_fn(paths[:, -1])
    exercise_time = np.full(n_paths, n_steps, dtype=int)

    # Backward induction
    for t in range(n_steps - 1, 0, -1):
        S_t = paths[:, t]
        intrinsic = payoff_fn(S_t)
        itm = intrinsic > 0

        if itm.sum() == 0:
            continue

        # Discounted future cash flows for ITM paths
        steps_to_cf = exercise_time[itm] - t
        Y = cashflows[itm] * discount ** steps_to_cf

        X = S_t[itm]

        # Build basis
        basis = _build_basis(X, basis_functions)

        # Regression
        try:
            coeffs = np.linalg.lstsq(basis, Y, rcond=None)[0]
            continuation = basis @ coeffs
        except np.linalg.LinAlgError:
            continuation = Y

        # Exercise decision
        exercise_now = intrinsic[itm] >= continuation
        itm_indices = np.where(itm)[0]
        ex_idx = itm_indices[exercise_now]

        cashflows[ex_idx] = intrinsic[itm][exercise_now]
        exercise_time[ex_idx] = t

    # Discount all cash flows to time 0
    discounted = cashflows * discount ** exercise_time
    return float(np.mean(discounted))


def _build_basis(
    x: np.ndarray,
    basis_type: str,
    degree: int = 3,
) -> np.ndarray:
    """Build basis functions for the LSM regression."""
    if basis_type == "laguerre":
        # First few Laguerre polynomials
        L0 = np.ones_like(x)
        L1 = 1.0 - x
        L2 = 0.5 * (x ** 2 - 4 * x + 2)
        return np.column_stack([L0, L1, L2])
    elif basis_type == "polynomial":
        return np.column_stack([x ** i for i in range(degree + 1)])
    else:
        raise ValueError(
            f"Unknown basis {basis_type!r}; use 'laguerre' or 'polynomial'."
        )


# ---------------------------------------------------------------------------
# Binomial American option
# ---------------------------------------------------------------------------

def binomial_american(
    spot: float,
    strike: float,
    rf_rate: float,
    vol: float,
    T: float,
    n_steps: int,
    option_type: str = "put",
) -> float:
    """Price an American option using the CRR binomial tree.

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Strike price.
    rf_rate : float
        Risk-free rate (annualised).
    vol : float
        Volatility (annualised).
    T : float
        Time to expiration (years).
    n_steps : int
        Number of tree steps.
    option_type : {'put', 'call'}, optional
        Option type (default ``'put'``).

    Returns
    -------
    float
        American option price.
    """
    dt = T / n_steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(rf_rate * dt) - d) / (u - d)
    disc = np.exp(-rf_rate * dt)

    # Terminal asset prices
    j = np.arange(n_steps + 1, dtype=float)
    prices = spot * u ** (n_steps - j) * d ** j

    # Terminal payoff
    if option_type == "put":
        values = np.maximum(strike - prices, 0.0)
    else:
        values = np.maximum(prices - strike, 0.0)

    # Backward induction with early exercise
    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        prices_step = spot * u ** (step - np.arange(step + 1, dtype=float)) * d ** np.arange(step + 1, dtype=float)
        if option_type == "put":
            intrinsic = np.maximum(strike - prices_step, 0.0)
        else:
            intrinsic = np.maximum(prices_step - strike, 0.0)
        values = np.maximum(values, intrinsic)

    return float(values[0])


# ---------------------------------------------------------------------------
# Optimal exit threshold (Ornstein-Uhlenbeck)
# ---------------------------------------------------------------------------

def optimal_exit_threshold(
    mu: float,
    sigma: float,
    transaction_cost: float,
) -> dict[str, float]:
    r"""Compute the optimal exit threshold for an OU mean-reverting process.

    For a process :math:`dX = -\mu\,X\,dt + \sigma\,dW`, the optimal
    exit threshold balances expected profit against the cost of trading.

    Uses the analytical approximation:

    .. math::
        x^* \approx \sigma \sqrt{\frac{2}{\mu}}
        \ln\!\left(\frac{\sigma}{\sqrt{2\mu}\,c}\right)

    when the threshold is large relative to *transaction_cost*.

    Parameters
    ----------
    mu : float
        Mean-reversion speed (> 0).
    sigma : float
        Volatility of the OU process (> 0).
    transaction_cost : float
        Round-trip transaction cost per unit.

    Returns
    -------
    dict
        ``entry_threshold``  – optimal entry (in units of process value).
        ``exit_threshold``   – optimal exit threshold.
        ``expected_profit``  – approximate expected profit per trade.
    """
    if mu <= 0 or sigma <= 0:
        raise ValueError("mu and sigma must be positive.")

    # Approximate optimal threshold
    ratio = sigma / (np.sqrt(2.0 * mu) * max(transaction_cost, 1e-12))
    if ratio > 1.0:
        x_star = sigma * np.sqrt(2.0 / mu) * np.sqrt(np.log(ratio))
    else:
        x_star = transaction_cost * 2.0  # fallback

    # Expected profit: threshold minus costs
    expected_profit = x_star - transaction_cost

    return {
        "entry_threshold": 0.0,  # enter when X crosses zero (mean)
        "exit_threshold": float(x_star),
        "expected_profit": float(max(expected_profit, 0.0)),
    }


# ---------------------------------------------------------------------------
# Sequential Probability Ratio Test
# ---------------------------------------------------------------------------

def sequential_probability_ratio(
    observations: ArrayLike,
    h0_dist: tuple[str, dict[str, float]],
    h1_dist: tuple[str, dict[str, float]],
    alpha: float = 0.05,
    beta: float = 0.05,
) -> dict[str, float | int | str]:
    """Sequential Probability Ratio Test (SPRT).

    Sequentially tests H0 vs H1 with controlled error probabilities.

    Parameters
    ----------
    observations : array_like
        Sequentially observed data.
    h0_dist : tuple
        ``(distribution_name, params)`` for H0.
        Supported: ``('normal', {'mu': ..., 'sigma': ...})``.
    h1_dist : tuple
        ``(distribution_name, params)`` for H1.
    alpha : float, optional
        Type I error probability (default 0.05).
    beta : float, optional
        Type II error probability (default 0.05).

    Returns
    -------
    dict
        ``decision``      – ``'reject_h0'``, ``'accept_h0'``, or ``'inconclusive'``.
        ``stopping_time`` – index at which the decision was reached.
        ``log_ratio``     – final log-likelihood ratio.
    """
    observations = np.asarray(observations, dtype=float)

    # Wald boundaries
    A = np.log((1.0 - beta) / alpha)   # upper boundary (reject H0)
    B = np.log(beta / (1.0 - alpha))   # lower boundary (accept H0)

    cumulative_lr = 0.0
    stopping_time = len(observations)
    decision = "inconclusive"

    for i, x in enumerate(observations):
        lr = _log_likelihood_ratio(x, h0_dist, h1_dist)
        cumulative_lr += lr

        if cumulative_lr >= A:
            decision = "reject_h0"
            stopping_time = i + 1
            break
        elif cumulative_lr <= B:
            decision = "accept_h0"
            stopping_time = i + 1
            break

    return {
        "decision": decision,
        "stopping_time": stopping_time,
        "log_ratio": float(cumulative_lr),
    }


def _log_likelihood_ratio(
    x: float,
    h0_dist: tuple[str, dict[str, float]],
    h1_dist: tuple[str, dict[str, float]],
) -> float:
    """Log-likelihood ratio for a single observation."""
    from scipy.stats import norm as sp_norm

    name0, params0 = h0_dist
    name1, params1 = h1_dist

    if name0 == "normal" and name1 == "normal":
        ll0 = sp_norm.logpdf(x, loc=params0["mu"], scale=params0["sigma"])
        ll1 = sp_norm.logpdf(x, loc=params1["mu"], scale=params1["sigma"])
        return float(ll1 - ll0)
    else:
        raise ValueError(f"Unsupported distributions: {name0}, {name1}")


# ---------------------------------------------------------------------------
# CUSUM stopping rule
# ---------------------------------------------------------------------------

def cusum_stopping(
    observations: ArrayLike,
    target_mean: float,
    threshold: float,
) -> dict[str, float | int | bool]:
    """CUSUM stopping rule for detecting a mean shift.

    Monitors cumulative sum of deviations from *target_mean* and triggers
    a stop when the CUSUM statistic exceeds *threshold*.

    Parameters
    ----------
    observations : array_like
        Sequential observations.
    target_mean : float
        Expected mean under the null (no-change) hypothesis.
    threshold : float
        CUSUM threshold for triggering a detection.

    Returns
    -------
    dict
        ``detected``      – whether a change was detected.
        ``stopping_time`` – index of detection (or length of data if none).
        ``cusum_pos``     – final positive CUSUM statistic.
        ``cusum_neg``     – final negative CUSUM statistic.
        ``cusum_values``  – array of max(cusum_pos, |cusum_neg|) at each step.
    """
    observations = np.asarray(observations, dtype=float)
    n = len(observations)

    cusum_pos = 0.0
    cusum_neg = 0.0
    detected = False
    stopping_time = n
    cusum_values = np.zeros(n)

    for i in range(n):
        cusum_pos = max(0.0, cusum_pos + (observations[i] - target_mean))
        cusum_neg = min(0.0, cusum_neg + (observations[i] - target_mean))
        cusum_values[i] = max(cusum_pos, abs(cusum_neg))

        if cusum_pos > threshold or abs(cusum_neg) > threshold:
            detected = True
            stopping_time = i + 1
            break

    return {
        "detected": detected,
        "stopping_time": stopping_time,
        "cusum_pos": float(cusum_pos),
        "cusum_neg": float(cusum_neg),
        "cusum_values": cusum_values[: stopping_time],
    }


# ---------------------------------------------------------------------------
# Secretary problem
# ---------------------------------------------------------------------------

def secretary_problem_threshold(
    n_candidates: int,
) -> dict[str, float | int]:
    """Compute the optimal stopping threshold for the secretary problem.

    The classical 1/e rule: reject the first *r - 1* candidates, then
    hire the first one who is better than all previously seen.

    Parameters
    ----------
    n_candidates : int
        Total number of candidates.

    Returns
    -------
    dict
        ``threshold``         – number of candidates to unconditionally reject.
        ``optimal_fraction``  – threshold / n_candidates.
        ``success_probability`` – probability of selecting the best candidate.
    """
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive.")

    if n_candidates == 1:
        return {
            "threshold": 0,
            "optimal_fraction": 0.0,
            "success_probability": 1.0,
        }

    # Find r that maximises the probability of selecting the best
    best_prob = 0.0
    best_r = 1

    for r in range(1, n_candidates + 1):
        # P(win) = (r-1)/n * sum_{i=r}^{n} 1/(i-1)
        prob = 0.0
        for i in range(r, n_candidates + 1):
            prob += 1.0 / (i - 1) if i > 1 else 1.0
        prob *= (r - 1) / n_candidates
        if r == 1:
            # If r=1, we pick the first candidate: prob = 1/n
            prob = 1.0 / n_candidates
        if prob > best_prob:
            best_prob = prob
            best_r = r

    return {
        "threshold": best_r - 1,  # reject first (r-1) candidates
        "optimal_fraction": float((best_r - 1) / n_candidates),
        "success_probability": float(best_prob),
    }
