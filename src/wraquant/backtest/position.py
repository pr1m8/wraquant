"""Enhanced position sizing and weight management utilities.

Provides position sizing algorithms (fixed-fraction, Kelly, vol-targeting,
risk parity, equal risk contribution), signal inversion, weight clipping,
and rebalance threshold logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize as sp_opt

__all__ = [
    "PositionSizer",
    "invert_signal",
    "clip_weights",
    "rebalance_threshold",
]


class PositionSizer:
    """Collection of position-sizing algorithms.

    All methods are stateless class methods so the sizer can be used as a
    lightweight namespace without instantiation.

    Example
    -------
    >>> PositionSizer.fixed_fraction(100_000, 0.02)
    2000.0
    """

    @staticmethod
    def fixed_fraction(equity: float, risk_pct: float) -> float:
        """Fixed-fraction position sizing.

        Parameters
        ----------
        equity : float
            Current portfolio equity.
        risk_pct : float
            Fraction of equity to risk (e.g., 0.02 for 2 %).

        Returns
        -------
        float
            Dollar amount to allocate.
        """
        if equity < 0:
            raise ValueError("equity must be non-negative")
        if not 0 <= risk_pct <= 1:
            raise ValueError("risk_pct must be between 0 and 1")
        return equity * risk_pct

    @staticmethod
    def kelly_criterion(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Kelly criterion optimal fraction.

        Parameters
        ----------
        win_rate : float
            Probability of a winning trade (0-1).
        avg_win : float
            Average winning trade return (positive).
        avg_loss : float
            Average losing trade return (positive magnitude).

        Returns
        -------
        float
            Optimal fraction of capital to risk.  Clamped to ``[0, 1]``.
        """
        if not 0 <= win_rate <= 1:
            raise ValueError("win_rate must be between 0 and 1")
        if avg_win < 0:
            raise ValueError("avg_win must be non-negative")
        if avg_loss <= 0:
            raise ValueError("avg_loss must be positive")
        b = avg_win / avg_loss  # odds ratio
        kelly = (win_rate * b - (1 - win_rate)) / b
        return float(np.clip(kelly, 0.0, 1.0))

    @staticmethod
    def volatility_targeting(
        returns: pd.Series,
        target_vol: float,
        lookback: int = 20,
    ) -> float:
        """Volatility-targeting position scalar.

        Computes the leverage / de-leverage factor so that the
        portfolio's annualised volatility approximates *target_vol*.

        Parameters
        ----------
        returns : pd.Series
            Recent asset returns.
        target_vol : float
            Desired annualised volatility (e.g., 0.10 for 10 %).
        lookback : int
            Number of recent periods for vol estimation.

        Returns
        -------
        float
            Scalar multiplier for the position size.
        """
        if len(returns) < lookback:
            return 1.0
        recent = returns.iloc[-lookback:]
        realized_vol = float(recent.std() * np.sqrt(252))
        if realized_vol <= 0:
            return 1.0
        return target_vol / realized_vol

    @staticmethod
    def risk_parity_weights(cov_matrix: pd.DataFrame | NDArray[np.floating]) -> NDArray[np.floating]:
        """Risk-parity portfolio weights.

        Each asset contributes equally to total portfolio risk.  Uses an
        inverse-volatility heuristic that provides a good approximation
        for portfolios with moderate correlations.

        Parameters
        ----------
        cov_matrix : pd.DataFrame or np.ndarray
            Covariance matrix of asset returns (n x n).

        Returns
        -------
        np.ndarray
            Weight vector summing to 1.
        """
        cov = np.asarray(cov_matrix, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov_matrix must be a square matrix")
        vols = np.sqrt(np.diag(cov))
        if np.any(vols <= 0):
            raise ValueError("All diagonal entries of cov_matrix must be positive")
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return weights

    @staticmethod
    def equal_risk_contribution(
        cov_matrix: pd.DataFrame | NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Equal Risk Contribution (ERC) portfolio weights.

        Solves for weights such that each asset's marginal contribution
        to total portfolio variance is identical.

        Parameters
        ----------
        cov_matrix : pd.DataFrame or np.ndarray
            Covariance matrix of asset returns (n x n).

        Returns
        -------
        np.ndarray
            Weight vector summing to 1.
        """
        cov = np.asarray(cov_matrix, dtype=float)
        n = cov.shape[0]
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov_matrix must be a square matrix")

        # Objective: minimise sum of (w_i*(Cov@w)_i - w_j*(Cov@w)_j)^2
        def _objective(w: NDArray[np.floating]) -> float:
            w = w / w.sum()  # normalise
            sigma_w = cov @ w
            rc = w * sigma_w  # risk contributions
            target = rc.sum() / n
            return float(np.sum((rc - target) ** 2))

        w0 = np.ones(n) / n
        bounds = [(1e-6, 1.0)] * n
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        result = sp_opt.minimize(
            _objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        weights = result.x / result.x.sum()
        return weights


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------


def invert_signal(
    signal: pd.Series | pd.DataFrame | NDArray[np.floating],
) -> pd.Series | pd.DataFrame | NDArray[np.floating]:
    """Flip long/short signals (multiply by -1).

    Parameters
    ----------
    signal : pd.Series, pd.DataFrame, or np.ndarray
        Signal values where positive = long, negative = short.

    Returns
    -------
    Same type as input
        Inverted signal.
    """
    return -1 * signal


def clip_weights(
    weights: pd.Series | NDArray[np.floating],
    min_w: float = 0.0,
    max_w: float = 1.0,
) -> pd.Series | NDArray[np.floating]:
    """Clip portfolio weights and re-normalise to sum to 1.

    Parameters
    ----------
    weights : pd.Series or np.ndarray
        Raw portfolio weights.
    min_w : float
        Minimum allowed weight per asset.
    max_w : float
        Maximum allowed weight per asset.

    Returns
    -------
    Same type as input
        Clipped and re-normalised weights.
    """
    arr = np.asarray(weights, dtype=float).copy()
    for _ in range(20):
        arr = np.clip(arr, min_w, max_w)
        total = arr.sum()
        if total > 0:
            arr = arr / total
        if np.all(arr >= min_w - 1e-12) and np.all(arr <= max_w + 1e-12):
            break
    if isinstance(weights, pd.Series):
        return pd.Series(arr, index=weights.index, name=weights.name)
    return arr


def rebalance_threshold(
    current_weights: pd.Series | NDArray[np.floating],
    target_weights: pd.Series | NDArray[np.floating],
    threshold: float = 0.05,
) -> bool:
    """Check whether portfolio drift exceeds a rebalance threshold.

    Parameters
    ----------
    current_weights : pd.Series or np.ndarray
        Current portfolio weights.
    target_weights : pd.Series or np.ndarray
        Target portfolio weights.
    threshold : float
        Maximum absolute drift allowed before rebalancing.

    Returns
    -------
    bool
        ``True`` if any weight has drifted beyond *threshold* and a
        rebalance is recommended.
    """
    diff = np.abs(np.asarray(current_weights, dtype=float) - np.asarray(target_weights, dtype=float))
    return bool(np.max(diff) > threshold + 1e-12)


def risk_parity_position(
    cov_matrix: pd.DataFrame | NDArray[np.floating],
    target_vol: float | None = None,
) -> NDArray[np.floating]:
    """Position sizing using risk parity (equal risk contribution).

    Computes portfolio weights such that each asset contributes equally
    to total portfolio risk, then optionally scales the weights so that
    the portfolio's annualised volatility matches ``target_vol``.

    This is a convenience wrapper around
    ``PositionSizer.equal_risk_contribution`` that adds volatility
    targeting and returns a clean weight vector.

    Mathematical formulation:
        For each asset *i*, the risk contribution is:
            RC_i = w_i * (Cov @ w)_i

        We solve for *w* such that RC_i = RC_j for all i, j, subject to
        sum(w) = 1.

        If ``target_vol`` is provided, the weights are scaled by
        ``target_vol / portfolio_vol``.

    How to interpret:
        - Weights will be higher for lower-volatility assets and lower
          for higher-volatility assets.
        - In a diagonal covariance matrix, risk parity reduces to
          inverse volatility weighting.
        - With non-zero correlations, the optimiser also accounts for
          diversification benefit.

    When to use:
        Use risk parity when you want a balanced portfolio where no
        single asset dominates the risk budget.  Particularly popular
        for multi-asset and all-weather portfolios.

    Parameters:
        cov_matrix: Covariance matrix of asset returns (n x n).
            Can be a pandas DataFrame or numpy array.
        target_vol: Target annualised portfolio volatility (e.g.,
            0.10 for 10 %).  If ``None``, weights sum to 1 without
            vol scaling.

    Returns:
        Weight array.  Sums to 1 if ``target_vol`` is ``None``;
        otherwise scaled to achieve the target volatility.

    Example:
        >>> import numpy as np
        >>> cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        >>> w = risk_parity_position(cov)
        >>> abs(w.sum() - 1.0) < 1e-6
        True
        >>> w[0] > w[1]  # lower-vol asset gets more weight
        True

    See Also:
        PositionSizer.equal_risk_contribution: Core ERC optimiser.
        PositionSizer.risk_parity_weights: Simpler inverse-vol heuristic.
        regime_conditional_sizing: Adjust weights based on market regime.
    """
    weights = PositionSizer.equal_risk_contribution(cov_matrix)

    if target_vol is not None:
        cov = np.asarray(cov_matrix, dtype=float)
        port_var = float(weights @ cov @ weights)
        port_vol_ann = np.sqrt(port_var * 252)
        if port_vol_ann > 0:
            scale = target_vol / port_vol_ann
            weights = weights * scale

    return weights


def regime_conditional_sizing(
    base_weights: NDArray[np.floating] | pd.Series,
    regime_probabilities: dict[str, float],
    risk_multipliers: dict[str, float],
) -> NDArray[np.floating]:
    """Adjust position sizes based on current regime probabilities.

    Scales base portfolio weights by a regime-dependent risk multiplier.
    The effective multiplier is a probability-weighted average of the
    per-regime multipliers, ensuring smooth transitions between regimes.

    Mathematical formulation:
        effective_multiplier = sum(P(regime_i) * multiplier_i)
        adjusted_weights = base_weights * effective_multiplier

    How to interpret:
        - If the current regime is "high_vol" with probability 0.8 and
          the risk multiplier for "high_vol" is 0.5, the weights will
          be scaled down significantly.
        - If the regime is "normal" (multiplier = 1.0), weights remain
          unchanged.
        - Multipliers > 1.0 increase exposure (e.g., during low-vol
          regimes).

    When to use:
        Use regime-conditional sizing when your strategy should adjust
        leverage or exposure based on market conditions.  Pair with
        regime detection (HMM, rolling volatility, etc.) to
        automatically reduce risk during turbulent markets.

    Parameters:
        base_weights: Base portfolio weights (array or Series).
        regime_probabilities: Mapping of regime name to probability
            (e.g., ``{"normal": 0.3, "high_vol": 0.7}``).
            Probabilities should sum to 1 but are not strictly
            enforced.
        risk_multipliers: Mapping of regime name to risk multiplier
            (e.g., ``{"normal": 1.0, "high_vol": 0.5, "low_vol": 1.5}``).
            Regimes in ``regime_probabilities`` that are missing from
            ``risk_multipliers`` default to a multiplier of 1.0.

    Returns:
        Adjusted weight array.  May not sum to 1 (the multiplier acts
        as a leverage/de-leverage factor).

    Example:
        >>> import numpy as np
        >>> base = np.array([0.5, 0.3, 0.2])
        >>> probs = {"normal": 0.3, "high_vol": 0.7}
        >>> mults = {"normal": 1.0, "high_vol": 0.5}
        >>> adj = regime_conditional_sizing(base, probs, mults)
        >>> adj.sum() < base.sum()  # scaled down due to high_vol
        True

    See Also:
        risk_parity_position: Risk-parity-based weight computation.
        PositionSizer.volatility_targeting: Vol-targeting scalar.
    """
    effective_mult = sum(
        prob * risk_multipliers.get(regime, 1.0)
        for regime, prob in regime_probabilities.items()
    )
    arr = np.asarray(base_weights, dtype=float)
    return arr * effective_mult
