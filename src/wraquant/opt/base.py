"""Base classes for optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class Constraint:
    """Optimization constraint specification.

    Parameters:
        type: Constraint type ('eq' for equality, 'ineq' for inequality).
        fun: Constraint function.
        name: Human-readable name.
    """

    type: str  # 'eq' or 'ineq'
    fun: callable  # type: ignore[type-arg]
    name: str = ""


@dataclass
class Objective:
    """Optimization objective specification.

    Parameters:
        fun: Objective function to minimize.
        name: Human-readable name.
    """

    fun: callable  # type: ignore[type-arg]
    name: str = ""


@dataclass
class OptimizationResult:
    """Result of a portfolio optimization.

    Parameters:
        weights: Optimal portfolio weights.
        expected_return: Expected portfolio return.
        volatility: Portfolio volatility (std dev).
        sharpe_ratio: Portfolio Sharpe ratio.
        asset_names: Names of assets.
        metadata: Additional solver-specific information.
    """

    weights: npt.NDArray[np.floating]
    expected_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    asset_names: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Return weights as {asset_name: weight} dict."""
        if self.asset_names:
            return dict(zip(self.asset_names, self.weights.tolist(), strict=False))
        return {f"asset_{i}": w for i, w in enumerate(self.weights.tolist())}
