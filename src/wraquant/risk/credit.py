"""Credit risk models and default probability estimation.

Credit risk is the risk that a borrower fails to meet its obligations.
This module provides tools spanning three major approaches:

1. **Structural models** -- model the firm's equity as a contingent
   claim on its assets. Default occurs when asset value falls below
   the debt barrier.

   - ``merton_model``: the foundational structural model (Merton 1974).
     Treats equity as a European call option on total firm assets.
     Iteratively solves for implied asset value and volatility, then
     computes distance-to-default and default probability.

2. **Credit scoring** -- statistical models that predict default from
   accounting ratios or market data.

   - ``altman_z_score``: the original 1968 Altman Z-Score for publicly
     traded manufacturing firms. Combines five accounting ratios into
     a single score that classifies firms as "safe" (Z > 2.99),
     "grey zone" (1.81-2.99), or "distress" (Z < 1.81).

3. **Reduced-form / intensity models** -- model default as a random
   event driven by a hazard rate (default intensity).

   - ``default_probability``: cumulative default probability from a
     rating transition matrix raised to the power of the horizon.
   - ``credit_spread``: implied spread from PD and recovery rate.
   - ``cds_spread``: fair CDS premium from a constant hazard rate,
     integrating protection and premium legs.
   - ``loss_given_default``: LGD = exposure * (1 - recovery rate).
   - ``expected_loss``: EL = PD * LGD * EAD -- the central formula
     of regulatory capital calculation (Basel II IRB).

How to choose:
    - For public equities with observable stock prices: ``merton_model``
      gives market-implied default probabilities that update daily.
    - For quick screening of financial health: ``altman_z_score`` using
      balance sheet data.
    - For pricing CDS or credit-linked instruments: ``cds_spread``
      with calibrated hazard rates.
    - For portfolio credit risk (e.g., a loan book): ``default_probability``
      from rating agency transition matrices + ``expected_loss``.

References:
    - Merton (1974), "On the Pricing of Corporate Debt"
    - Altman (1968), "Financial Ratios, Discriminant Analysis and the
      Prediction of Corporate Bankruptcy"
    - Lando (2004), "Credit Risk Modeling: Theory and Applications"
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


__all__ = [
    "altman_z_score",
    "cds_spread",
    "credit_spread",
    "default_probability",
    "expected_loss",
    "loss_given_default",
    "merton_model",
]


def merton_model(
    equity: float,
    debt: float,
    vol: float,
    rf_rate: float,
    maturity: float,
) -> dict[str, float]:
    """Merton structural credit risk model.

    Models firm equity as a European call option on total assets with
    strike equal to the face value of debt.  Uses an iterative procedure
    to back out implied asset value and asset volatility, then computes
    distance-to-default and default probability.

    Parameters:
        equity: Current market value of equity.
        debt: Face value of outstanding debt (strike).
        vol: Equity volatility (annualized).
        rf_rate: Continuous risk-free rate (annualized).
        maturity: Time to maturity of debt in years.

    Returns:
        Dictionary with keys:

        - ``asset_value``: Implied total asset value.
        - ``asset_vol``: Implied asset volatility.
        - ``d1``, ``d2``: Black-Scholes d1 and d2.
        - ``distance_to_default``: d2 under the physical measure.
        - ``default_probability``: N(-d2), probability of default.
        - ``credit_spread``: Implied credit spread over the risk-free rate.
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if equity <= 0:
        raise ValueError("equity must be positive")
    if debt <= 0:
        raise ValueError("debt must be positive")

    sqrt_t = np.sqrt(maturity)

    # Initial guesses for iterative solver
    asset_value = equity + debt
    asset_vol = vol * equity / asset_value

    # Iterative solution (fixed-point iteration)
    for _ in range(200):
        d1 = (np.log(asset_value / debt) + (rf_rate + 0.5 * asset_vol**2) * maturity) / (
            asset_vol * sqrt_t
        )
        d2 = d1 - asset_vol * sqrt_t

        # Equity = V * N(d1) - D * exp(-r*T) * N(d2)
        equity_implied = asset_value * sp_stats.norm.cdf(d1) - debt * np.exp(
            -rf_rate * maturity
        ) * sp_stats.norm.cdf(d2)

        # Update asset volatility using the relationship:
        # sigma_E * E = N(d1) * sigma_A * V
        nd1 = sp_stats.norm.cdf(d1)
        if nd1 > 1e-15:
            asset_vol_new = vol * equity / (nd1 * asset_value)
        else:
            asset_vol_new = asset_vol

        asset_value_new = equity + debt * np.exp(-rf_rate * maturity) * sp_stats.norm.cdf(d2)

        if abs(asset_value_new - asset_value) < 1e-8 and abs(asset_vol_new - asset_vol) < 1e-8:
            asset_value = asset_value_new
            asset_vol = asset_vol_new
            break

        asset_value = asset_value_new
        asset_vol = asset_vol_new

    # Final d1 and d2
    d1 = (np.log(asset_value / debt) + (rf_rate + 0.5 * asset_vol**2) * maturity) / (
        asset_vol * sqrt_t
    )
    d2 = d1 - asset_vol * sqrt_t

    default_prob = float(sp_stats.norm.cdf(-d2))

    # Credit spread: risky yield minus risk-free rate
    # D_risky = V - E  (market value of debt)
    debt_market = asset_value - equity
    if debt_market > 0 and debt > 0:
        risky_yield = -np.log(debt_market / debt) / maturity
        spread = risky_yield - rf_rate
    else:
        spread = 0.0

    return {
        "asset_value": float(asset_value),
        "asset_vol": float(asset_vol),
        "d1": float(d1),
        "d2": float(d2),
        "distance_to_default": float(d2),
        "default_probability": default_prob,
        "credit_spread": float(max(spread, 0.0)),
    }


def altman_z_score(
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    sales: float,
) -> dict[str, float | str]:
    """Altman Z-Score for bankruptcy prediction.

    Uses the original 1968 Altman Z-Score model for publicly traded
    manufacturing firms.

    Parameters:
        working_capital: Current assets minus current liabilities.
        total_assets: Total assets.
        retained_earnings: Cumulative retained earnings.
        ebit: Earnings before interest and taxes.
        market_cap: Market capitalisation of equity.
        total_liabilities: Total liabilities.
        sales: Net sales / revenue.

    Returns:
        Dictionary with keys:

        - ``z_score``: The computed Z-Score.
        - ``zone``: One of ``"safe"`` (Z > 2.99), ``"grey"``
          (1.81 <= Z <= 2.99), or ``"distress"`` (Z < 1.81).
        - ``x1`` .. ``x5``: Individual component ratios.
    """
    if total_assets <= 0:
        raise ValueError("total_assets must be positive")
    if total_liabilities <= 0:
        raise ValueError("total_liabilities must be positive")

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities
    x5 = sales / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if z > 2.99:
        zone = "safe"
    elif z < 1.81:
        zone = "distress"
    else:
        zone = "grey"

    return {
        "z_score": float(z),
        "zone": zone,
        "x1": float(x1),
        "x2": float(x2),
        "x3": float(x3),
        "x4": float(x4),
        "x5": float(x5),
    }


def default_probability(
    rating_transitions: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Cumulative default probability from a rating transition matrix.

    The last row/column of the transition matrix is assumed to represent
    the *default* (absorbing) state.

    Parameters:
        rating_transitions: Square transition matrix of shape ``(n, n)``
            where element ``[i, j]`` is the one-period probability of
            migrating from rating *i* to rating *j*.  Rows must sum to 1.
        horizon: Number of periods to compound the matrix over.

    Returns:
        1-D array of cumulative default probabilities for each non-default
        rating at the given *horizon*.  Length is ``n - 1`` where *n* is the
        number of ratings (including default).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    mat = np.asarray(rating_transitions, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("rating_transitions must be a square matrix")

    # Compound the matrix over the horizon
    compounded = np.linalg.matrix_power(mat, horizon)

    # Default probabilities: last column, excluding the default state row
    return compounded[:-1, -1].copy()


def credit_spread(
    default_prob: float,
    recovery_rate: float,
    rf_rate: float = 0.0,
) -> float:
    """Implied credit spread from a default probability.

    Uses the approximation: ``spread ~ -ln(1 - default_prob * (1 - R))``
    with an adjustment for the risk-free rate not being needed in the
    simple reduced-form relationship.

    Parameters:
        default_prob: Annualized probability of default.
        recovery_rate: Recovery rate in [0, 1].
        rf_rate: Risk-free rate (unused in simple model but accepted for
            API consistency).

    Returns:
        Annualized credit spread (as a fraction, not basis points).
    """
    if not 0 <= default_prob <= 1:
        raise ValueError("default_prob must be in [0, 1]")
    if not 0 <= recovery_rate <= 1:
        raise ValueError("recovery_rate must be in [0, 1]")

    lgd = 1.0 - recovery_rate
    spread = -np.log(1.0 - default_prob * lgd)
    return float(spread)


def loss_given_default(
    exposure: float,
    recovery_rate: float,
) -> float:
    """Expected loss given default.

    Parameters:
        exposure: Exposure at default (EAD).
        recovery_rate: Expected recovery rate in [0, 1].

    Returns:
        Loss given default = exposure * (1 - recovery_rate).
    """
    if not 0 <= recovery_rate <= 1:
        raise ValueError("recovery_rate must be in [0, 1]")
    return float(exposure * (1.0 - recovery_rate))


def expected_loss(
    pd_val: float,
    lgd: float,
    ead: float,
) -> float:
    """Expected loss (EL = PD x LGD x EAD).

    Parameters:
        pd_val: Probability of default.
        lgd: Loss given default (as a fraction of EAD).
        ead: Exposure at default.

    Returns:
        Expected loss.
    """
    if not 0 <= pd_val <= 1:
        raise ValueError("pd_val must be in [0, 1]")
    if lgd < 0:
        raise ValueError("lgd must be non-negative")
    return float(pd_val * lgd * ead)


def cds_spread(
    default_intensity: float,
    recovery_rate: float,
    maturity: float,
) -> float:
    """Fair CDS spread from a constant hazard rate (default intensity).

    Under a simple reduced-form model with constant hazard rate *lambda*
    and flat term structure, the fair CDS premium (annualised) is:

        ``spread = lambda * (1 - R)``

    For more accuracy, this function uses the continuous-time formula
    integrating protection and premium legs assuming quarterly payments.

    Parameters:
        default_intensity: Constant hazard rate (lambda), annualised.
        recovery_rate: Recovery rate in [0, 1].
        maturity: CDS maturity in years.

    Returns:
        Annualised CDS spread (as a fraction).
    """
    if default_intensity < 0:
        raise ValueError("default_intensity must be non-negative")
    if not 0 <= recovery_rate <= 1:
        raise ValueError("recovery_rate must be in [0, 1]")
    if maturity <= 0:
        raise ValueError("maturity must be positive")

    # Simple approximation: spread = hazard_rate * (1 - R)
    # More precise: integrate survival-weighted cashflows
    n_steps = max(int(maturity * 4), 1)  # quarterly steps
    dt = maturity / n_steps
    lam = default_intensity

    protection_leg = 0.0
    premium_leg = 0.0

    for i in range(1, n_steps + 1):
        t = i * dt
        surv = np.exp(-lam * t)
        surv_prev = np.exp(-lam * (t - dt))

        # Protection leg: (1 - R) * prob of default in [t-dt, t]
        protection_leg += (1.0 - recovery_rate) * (surv_prev - surv)

        # Premium leg: spread * dt * survival probability at t
        premium_leg += dt * surv

    if premium_leg < 1e-15:
        return float(lam * (1.0 - recovery_rate))

    return float(protection_leg / premium_leg)
