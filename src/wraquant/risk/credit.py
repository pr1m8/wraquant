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
    strike equal to the face value of debt.  The key insight: equity
    holders have a call option on the firm's assets -- they get the
    upside above the debt level but can walk away (default) if assets
    fall below debt.

    The model iteratively solves for the unobservable asset value and
    asset volatility from the observable equity value and equity
    volatility, using the Black-Scholes option pricing relationship.

    Interpretation:
        - **distance_to_default** (DD): How many standard deviations
          the firm's asset value is above the default barrier.
          DD > 4: very safe. DD 2-4: investment grade. DD 1-2: high
          yield. DD < 1: distress.  Moody's KMV uses this as the
          primary input to their EDF (Expected Default Frequency) model.
        - **default_probability**: N(-DD), the probability that asset
          value drifts below debt by maturity.  This is a risk-neutral
          probability -- real-world default rates are typically lower.
        - **asset_vol**: Implied asset volatility.  Higher = more
          default risk.  Asset vol is always lower than equity vol
          because equity is a levered claim.
        - **credit_spread**: The yield premium investors should demand
          for holding risky debt.  Compare to market CDS spreads to
          detect mispricing.

    When to use:
        - Market-implied default probabilities from daily equity data.
        - Screening for distressed firms (DD < 2).
        - As a factor in credit scoring models.
        - For relative value: compare Merton-implied spreads to market
          CDS spreads.

    Red flags:
        - DD < 1: firm is in acute distress.
        - Asset vol > 50%: inputs may be unreliable.
        - Equity vol is stale or missing: model won't converge.

    Parameters:
        equity: Current market value of equity (market cap).
        debt: Face value of outstanding debt (the "strike").
        vol: Equity volatility (annualized, e.g., 0.30 for 30%).
        rf_rate: Continuous risk-free rate (annualized).
        maturity: Time to maturity of debt in years (typically 1).

    Returns:
        Dictionary with keys:

        - **asset_value** (*float*) -- Implied total asset value.
        - **asset_vol** (*float*) -- Implied asset volatility.
        - **d1**, **d2** (*float*) -- Black-Scholes d1 and d2.
        - **distance_to_default** (*float*) -- d2, number of std devs
          above the default barrier.
        - **default_probability** (*float*) -- N(-d2), risk-neutral
          probability of default.
        - **credit_spread** (*float*) -- Implied credit spread over
          the risk-free rate (annualized).

    Example:
        >>> result = merton_model(equity=50e6, debt=40e6, vol=0.35,
        ...                       rf_rate=0.04, maturity=1.0)
        >>> print(f"DD: {result['distance_to_default']:.2f}")
        >>> print(f"PD: {result['default_probability']:.4f}")
        >>> print(f"Spread: {result['credit_spread']*10000:.0f} bps")

    See Also:
        altman_z_score: Accounting-based bankruptcy predictor.
        cds_spread: Reduced-form CDS pricing.

    Notes:
        Reference: Merton, R.C. (1974). "On the Pricing of Corporate
        Debt: The Risk Structure of Interest Rates." *Journal of
        Finance*, 29(2), 449-470.
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
        d1 = (
            np.log(asset_value / debt) + (rf_rate + 0.5 * asset_vol**2) * maturity
        ) / (asset_vol * sqrt_t)
        d2 = d1 - asset_vol * sqrt_t

        # Equity = V * N(d1) - D * exp(-r*T) * N(d2)
        asset_value * sp_stats.norm.cdf(d1) - debt * np.exp(
            -rf_rate * maturity
        ) * sp_stats.norm.cdf(d2)

        # Update asset volatility using the relationship:
        # sigma_E * E = N(d1) * sigma_A * V
        nd1 = sp_stats.norm.cdf(d1)
        if nd1 > 1e-15:
            asset_vol_new = vol * equity / (nd1 * asset_value)
        else:
            asset_vol_new = asset_vol

        asset_value_new = equity + debt * np.exp(
            -rf_rate * maturity
        ) * sp_stats.norm.cdf(d2)

        if (
            abs(asset_value_new - asset_value) < 1e-8
            and abs(asset_vol_new - asset_vol) < 1e-8
        ):
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

    The Z-Score combines five accounting ratios into a single
    discriminant score that classifies firms by financial health.
    Despite being from 1968, it remains one of the most widely used
    credit screening tools.

    Interpretation:
        - **Z > 2.99** ("safe"): Firm is financially healthy.
          Default probability is very low (< 1% over 2 years).
        - **1.81 <= Z <= 2.99** ("grey zone"): Ambiguous.
          Firm could go either way. Warrants deeper analysis.
        - **Z < 1.81** ("distress"): High bankruptcy risk.
          Historically, ~95% of firms that defaulted had Z < 1.81
          one year prior.

    Component interpretation:
        - **x1** (WC/TA): Liquidity. Negative = current liabilities
          exceed current assets.
        - **x2** (RE/TA): Cumulative profitability and firm age.
          Young firms have low retained earnings.
        - **x3** (EBIT/TA): Operating profitability.
        - **x4** (Market Cap/TL): Market leverage.
        - **x5** (Sales/TA): Asset turnover efficiency.

    When to use:
        - Quick screening of a large universe of firms.
        - As a factor in multi-factor credit models.
        - For early warning systems.

    Limitations:
        - Designed for publicly traded manufacturing firms.
        - Does not capture market-implied information (use
          ``merton_model`` for that).
        - Accounting data can be manipulated.

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

    Computes the probability of eventually defaulting within `horizon`
    periods, starting from each non-default rating.  This is done by
    raising the one-period transition matrix to the `horizon`-th power
    and reading off the default column.

    Interpretation:
        - The output is a vector where each element is the cumulative
          default probability for a given starting rating.
        - AAA will have the smallest PD; CCC the largest.
        - Compare to historical default rates published by Moody's
          or S&P to calibrate.
        - Use with ``expected_loss`` for portfolio credit risk.

    When to use:
        - Converting a rating agency transition matrix into PDs for
          capital calculations.
        - Stress testing: modify the transition matrix (increase
          downgrade probabilities) and re-compute PDs.

    The last row/column of the transition matrix is assumed to represent
    the *default* (absorbing) state.

    Parameters:
        rating_transitions: Square transition matrix of shape ``(n, n)``
            where element ``[i, j]`` is the one-period probability of
            migrating from rating *i* to rating *j*.  Rows must sum to 1.
        horizon: Number of periods to compound over (e.g., 5 for 5-year
            cumulative PD).

    Returns:
        1-D array of cumulative default probabilities for each non-default
        rating, length ``n - 1``.

    Example:
        >>> import numpy as np
        >>> # Simple 3-state matrix: AAA, BBB, Default
        >>> T = np.array([[0.95, 0.04, 0.01],
        ...               [0.02, 0.90, 0.08],
        ...               [0.00, 0.00, 1.00]])
        >>> pd_5yr = default_probability(T, horizon=5)
        >>> print(f"5yr PD from AAA: {pd_5yr[0]:.4f}")
        >>> print(f"5yr PD from BBB: {pd_5yr[1]:.4f}")
    """
    from wraquant.core._coerce import coerce_array

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

    Converts a default probability and recovery rate into the yield
    spread that compensates investors for bearing credit risk.  This
    is the theoretical "fair value" spread -- compare to market spreads
    to identify cheap or expensive credit.

    The formula: spread = -ln(1 - PD * LGD), where LGD = 1 - R.
    For small PD, this simplifies to spread ~ PD * LGD.

    Interpretation:
        - Output is annualized as a decimal (0.01 = 100 bps).
        - Multiply by 10,000 for basis points.
        - If market spread > model spread: bond is cheap (excess
          compensation for credit risk).
        - If market spread < model spread: bond is expensive or the
          model PD is too high.

    Parameters:
        default_prob: Annualized probability of default (e.g., 0.02
            for 2% annual PD).
        recovery_rate: Recovery rate in [0, 1]. Investment grade
            typically 0.40-0.50; high yield 0.25-0.40.
        rf_rate: Risk-free rate (unused in simple model but accepted
            for API consistency).

    Returns:
        Annualized credit spread as a decimal fraction. Multiply by
        10,000 for basis points.

    Example:
        >>> spread = credit_spread(0.02, 0.40)
        >>> print(f"Spread: {spread*10000:.0f} bps")  # ~120 bps
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

    LGD = EAD * (1 - Recovery Rate).  This is the dollar amount you
    expect to lose if the borrower defaults.

    Interpretation:
        - A recovery rate of 0.40 means you recover 40 cents on the
          dollar; LGD is 60% of exposure.
        - Recovery rates vary by seniority: secured senior ~65%,
          unsecured senior ~45%, subordinated ~25%.
        - Use with ``expected_loss`` for Basel II/III capital calculations.

    Parameters:
        exposure: Exposure at default (EAD) -- the amount at risk.
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

    The expected loss is the central formula of credit risk management
    and Basel II/III regulatory capital calculation.  It represents the
    average loss you expect from a credit exposure.

    Interpretation:
        - EL is the mean of the loss distribution.  It should be
          covered by pricing (loan margins, bond spreads) rather than
          capital reserves.
        - Capital reserves cover the unexpected loss (UL), which is
          the tail beyond EL.
        - For a portfolio, EL is additive: sum over all exposures.

    Parameters:
        pd_val: Probability of default (annualized, e.g., 0.02 for 2%).
        lgd: Loss given default as a fraction of EAD (e.g., 0.45 for
            45% loss rate).
        ead: Exposure at default (dollar amount at risk).

    Returns:
        Expected loss in the same units as *ead*.

    Example:
        >>> el = expected_loss(pd_val=0.02, lgd=0.45, ead=1_000_000)
        >>> print(f"Expected loss: ${el:,.0f}")  # $9,000
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

    Computes the breakeven CDS premium by equating the expected
    protection leg (what the protection seller pays at default) with
    the expected premium leg (what the protection buyer pays over time).

    Under a constant hazard rate model, the fair spread is approximately
    lambda * (1 - R), but this function uses the exact continuous-time
    formula with quarterly premium payments for greater accuracy.

    Interpretation:
        - The output is the annualized spread (decimal). Multiply by
          10,000 for basis points.
        - Compare to market CDS spreads to detect relative value.
        - If model spread > market spread: protection is cheap (market
          underestimates default risk).
        - If model spread < market spread: protection is expensive or
          there is a risk premium.
        - CDS spreads are approximately equal to bond spreads over
          swaps (CDS-bond basis ~ 0 in normal markets).

    When to use:
        - Pricing CDS contracts given a calibrated hazard rate.
        - Calibrating hazard rates from market CDS spreads (invert
          numerically).
        - Converting between PD and spread for credit analysis.

    Parameters:
        default_intensity: Constant hazard rate (lambda), annualized.
            E.g., 0.02 means a 2% probability of default per year.
        recovery_rate: Recovery rate in [0, 1]. Standard assumption
            is 0.40 for senior unsecured corporate debt.
        maturity: CDS maturity in years (standard: 1, 3, 5, 7, 10).

    Returns:
        Annualized CDS spread as a decimal fraction. Multiply by
        10,000 for basis points.

    Example:
        >>> spread = cds_spread(0.02, 0.40, 5.0)
        >>> print(f"5Y CDS: {spread*10000:.0f} bps")  # ~120 bps
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
