"""Pricing and fixed income prompt templates."""
from __future__ import annotations
from typing import Any


def register_pricing_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def option_pricing(underlying: str = "AAPL", strike: float = 150, expiry: str = "2024-06-21") -> list[dict]:
        """Option pricing: BS, Greeks, vol smile, Heston calibration."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Price a {underlying} option (K={strike}, expiry={expiry}):

1. Get current price from workspace data.
2. price_option with Black-Scholes — baseline price.
3. compute_greeks — delta, gamma, theta, vega, rho.
4. Estimate implied volatility if market price available.
5. Vol smile across strikes — fit SABR model.
6. Heston calibration if multiple strikes/maturities available.
7. Compare BS vs Heston prices — how much does stochastic vol matter?
8. Summary: fair value, key Greeks, vol smile shape.
"""}}]

    @mcp.prompt()
    def yield_curve_analysis(dataset: str = "bond_data") -> list[dict]:
        """Yield curve construction and analysis."""
        return [{"role": "user", "content": {"type": "text", "text": f"""
Yield curve analysis from {dataset}:

1. Bootstrap zero curve from bond prices/yields.
2. Interpolate at standard maturities (1y, 2y, 5y, 10y, 30y).
3. Compute forward rates — what does the market expect?
4. Duration and convexity for each maturity.
5. Key rate durations — sensitivity to each point on the curve.
6. Curve shape: normal, flat, or inverted? What does it signal?
7. Rate scenario analysis: parallel shift, steepener, flattener.
8. Summary: curve shape, forward rate expectations, rate risk.
"""}}]

    @mcp.prompt()
    def exotic_pricing() -> list[dict]:
        """Price exotic options via characteristic functions and FBSDEs."""
        return [{"role": "user", "content": {"type": "text", "text": """
Exotic option pricing:

1. Choose model: Heston, Variance Gamma, NIG, or CGMY.
2. Calibrate model to market data (vol surface).
3. Price European via characteristic function (FFT or COS method).
4. Compare across models — which fits the market best?
5. For American exercise: use FBSDE solver (reflected BSDE).
6. For high-dimensional (basket): use deep BSDE solver.
7. Greeks via bump-and-revalue.
8. Summary: model-dependent prices, which model to trust?
"""}}]
