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
