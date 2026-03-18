"""Value-at-Risk and Conditional VaR (Expected Shortfall) estimation.

Provides the two most important tail-risk measures in quantitative risk
management. VaR is a regulatory standard (Basel II/III); CVaR (Expected
Shortfall) is preferred by Basel IV and is mathematically superior
because it is a *coherent* risk measure (satisfies sub-additivity).

Both measures can be estimated via historical simulation (non-parametric)
or parametric (Gaussian) methods. For heavy-tailed distributions
(equities, credit), historical simulation is generally more accurate;
for smooth risk surfaces (rates), parametric is often sufficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Estimate Value-at-Risk (VaR).

    VaR answers the question: "With X% confidence, what is the maximum
    loss I should expect over one period?" More precisely, VaR is the
    (1 - confidence) quantile of the return distribution, flipped to
    a positive loss number.

    When to use:
        Use VaR for regulatory reporting, margin calculations, and
        setting position limits. Choose historical VaR when you have
        enough data (>500 observations) and want to capture fat tails
        without distributional assumptions. Choose parametric VaR when
        data is scarce or when you need analytical sensitivities (e.g.,
        delta-normal VaR for a derivatives book).

    Mathematical formulation:
        Historical: VaR_alpha = -quantile(returns, 1 - alpha)
        Parametric: VaR_alpha = -(mu + sigma * Phi^{-1}(1 - alpha))

        where alpha is the confidence level, mu and sigma are the
        sample mean and standard deviation, and Phi^{-1} is the
        standard normal inverse CDF.

    How to interpret:
        A 95% daily VaR of 0.02 means: "on 95% of days, the portfolio
        loses less than 2%. On the remaining 5% of days, the loss
        *exceeds* 2%." VaR says nothing about *how much* worse the loss
        can be beyond the threshold -- that is what CVaR captures.

    Parameters:
        returns: Simple return series (e.g., daily percentage changes).
        confidence: Confidence level (e.g., 0.95 for 95%, 0.99 for 99%).
            Basel III uses 0.99; internal risk management often uses 0.95.
        method: Estimation method:
            - ``"historical"`` -- empirical quantile (non-parametric,
              default). No distributional assumption; captures fat tails.
            - ``"parametric"`` -- Gaussian assumption. Smooth but
              underestimates tail risk for leptokurtic returns.

    Returns:
        VaR as a positive float representing the loss threshold. For
        example, 0.025 means a 2.5% loss at the given confidence level.

    Raises:
        ValueError: If *method* is not recognized.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0, 0.01, 1000))
        >>> var_95 = value_at_risk(returns, confidence=0.95)
        >>> var_95 > 0
        True

    Caveats:
        - VaR is *not* sub-additive: the VaR of a portfolio can exceed
          the sum of individual VaRs.  Use ``conditional_var`` for a
          coherent measure.
        - Historical VaR is sensitive to the sample window; recent
          crises dominate short windows.
        - Parametric VaR severely underestimates tail risk for fat-
          tailed distributions (equities, credit).

    See Also:
        conditional_var: Expected loss beyond the VaR threshold (CVaR).
        wraquant.risk.stress.stress_test_returns: Scenario-based analysis.

    References:
        - Jorion (2006), "Value at Risk: The New Benchmark"
        - Basel Committee on Banking Supervision (2019), "Minimum capital
          requirements for market risk"
    """
    clean = returns.dropna().values

    if method == "historical":
        var = float(np.percentile(clean, (1 - confidence) * 100))
    elif method == "parametric":
        mu = clean.mean()
        sigma = clean.std()
        var = float(sp_stats.norm.ppf(1 - confidence, loc=mu, scale=sigma))
    else:
        msg = f"Unknown VaR method: {method!r}"
        raise ValueError(msg)

    return -var


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Estimate Conditional VaR (Expected Shortfall / CVaR).

    CVaR answers: "given that the loss exceeds VaR, what is the
    *expected* loss?" It captures the severity of tail losses, not just
    their threshold. Unlike VaR, CVaR is a *coherent* risk measure
    (Artzner et al. 1999) -- it satisfies sub-additivity, meaning the
    CVaR of a portfolio is at most the sum of individual CVaRs.

    When to use:
        CVaR is preferred over VaR for:
        - Portfolio optimisation (mean-CVaR optimisation is convex).
        - Regulatory capital under Basel IV / FRTB.
        - Any situation where you care about tail *severity*, not just
          tail *frequency*.
        Use historical CVaR with long samples (>1000 obs) and parametric
        CVaR when you need smooth gradients or have short data.

    Mathematical formulation:
        Historical: CVaR_alpha = -mean(returns | returns <= VaR_quantile)
        Parametric: CVaR_alpha = -(mu - sigma * phi(z_alpha) / (1 - alpha))

        where z_alpha = Phi^{-1}(1 - alpha), phi is the standard normal
        PDF, and Phi is the CDF.

    How to interpret:
        A 95% daily CVaR of 0.035 means: "on the worst 5% of days, the
        *average* loss is 3.5%." CVaR is always >= VaR at the same
        confidence level. For normal distributions, 95% CVaR is about
        1.25x the 95% VaR. For fat-tailed distributions, the ratio is
        much larger -- this ratio itself is a useful diagnostic of tail
        heaviness.

    Parameters:
        returns: Simple return series.
        confidence: Confidence level (e.g., 0.95 for 95%).
        method: Estimation method:
            - ``"historical"`` -- mean of returns in the tail
              (default). Non-parametric; captures fat tails.
            - ``"parametric"`` -- Gaussian formula. Smooth but
              underestimates tail risk for heavy-tailed distributions.

    Returns:
        CVaR as a positive float representing the expected tail loss.
        For example, 0.035 means an expected loss of 3.5% in the tail.

    Raises:
        ValueError: If *method* is not recognized.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> returns = pd.Series(np.random.normal(0, 0.01, 1000))
        >>> cvar = conditional_var(returns, confidence=0.95)
        >>> var = value_at_risk(returns, confidence=0.95)
        >>> cvar >= var  # CVaR is always >= VaR
        True

    See Also:
        value_at_risk: The VaR threshold itself.
        wraquant.risk.monte_carlo.importance_sampling_var: Variance-
            reduced tail estimation.

    References:
        - Artzner et al. (1999), "Coherent Measures of Risk"
        - Rockafellar & Uryasev (2000), "Optimization of Conditional
          Value-at-Risk"
    """
    clean = returns.dropna().values

    if method == "historical":
        cutoff = np.percentile(clean, (1 - confidence) * 100)
        tail = clean[clean <= cutoff]
        cvar = float(-tail.mean()) if len(tail) > 0 else 0.0
    elif method == "parametric":
        mu = clean.mean()
        sigma = clean.std()
        alpha = 1 - confidence
        z = sp_stats.norm.ppf(alpha)
        cvar = float(-(mu - sigma * sp_stats.norm.pdf(z) / alpha))
    else:
        msg = f"Unknown CVaR method: {method!r}"
        raise ValueError(msg)

    return cvar
