"""Factor models and attribution for asset pricing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats


def fama_french_regression(
    returns: pd.Series,
    factors_df: pd.DataFrame,
) -> dict:
    """Regress asset returns on Fama-French factors.

    The regression is ``R_i - R_f = alpha + beta_1 * F_1 + ... + eps``.
    The factors DataFrame should contain the factor returns (e.g.,
    Mkt-RF, SMB, HML, and optionally RMW, CMA).  If a column named
    ``RF`` is present it is used to compute excess returns; otherwise
    *returns* are assumed to already be excess returns.

    Parameters:
        returns: Asset return series.
        factors_df: DataFrame of factor returns.  Columns are factor
            names.  An optional ``RF`` column is the risk-free rate.

    Returns:
        Dictionary with ``alpha`` (intercept), ``betas`` (dict mapping
        factor name to coefficient), ``t_stats`` (dict mapping name to
        t-statistic), ``p_values`` (dict), and ``r_squared``.
    """
    common = returns.dropna().index.intersection(factors_df.dropna().index)
    y = returns.loc[common].values.astype(float)

    # Separate RF if present
    if "RF" in factors_df.columns:
        rf = factors_df.loc[common, "RF"].values.astype(float)
        y = y - rf
        factor_cols = [c for c in factors_df.columns if c != "RF"]
    else:
        factor_cols = list(factors_df.columns)

    X = factors_df.loc[common, factor_cols].values.astype(float)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    betas = {name: float(model.params[i + 1]) for i, name in enumerate(factor_cols)}
    t_stats = {name: float(model.tvalues[i + 1]) for i, name in enumerate(factor_cols)}
    p_values = {name: float(model.pvalues[i + 1]) for i, name in enumerate(factor_cols)}

    return {
        "alpha": float(model.params[0]),
        "betas": betas,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": float(model.rsquared),
    }


def factor_attribution(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> dict:
    """Decompose returns into factor contributions and specific return.

    Runs a regression of *returns* on *factor_returns* and attributes
    the mean return to each factor.

    Parameters:
        returns: Asset return series.
        factor_returns: DataFrame of factor return series.

    Returns:
        Dictionary with ``factor_contributions`` (dict mapping factor
        name to its average contribution), ``specific_return`` (mean
        residual return), ``total_return`` (mean of *returns*), and
        ``r_squared``.
    """
    common = returns.dropna().index.intersection(factor_returns.dropna().index)
    y = returns.loc[common].values.astype(float)
    factor_cols = list(factor_returns.columns)
    X = factor_returns.loc[common, factor_cols].values.astype(float)
    X_const = sm.add_constant(X)

    model = sm.OLS(y, X_const).fit()

    # Factor contributions = beta_i * mean(factor_i)
    contributions: dict[str, float] = {}
    for i, name in enumerate(factor_cols):
        beta = model.params[i + 1]
        mean_factor = float(np.mean(X[:, i]))
        contributions[name] = float(beta * mean_factor)

    specific_return = float(model.params[0])  # alpha (intercept)

    return {
        "factor_contributions": contributions,
        "specific_return": specific_return,
        "total_return": float(np.mean(y)),
        "r_squared": float(model.rsquared),
    }


def information_coefficient(
    predictions: pd.Series | np.ndarray,
    returns: pd.Series | np.ndarray,
) -> float:
    """Compute the information coefficient (Spearman rank correlation).

    The IC measures the predictive power of a signal: the rank
    correlation between the cross-sectional predictions and subsequent
    realised returns.

    Parameters:
        predictions: Predicted values or signals.
        returns: Subsequent realised returns.

    Returns:
        Spearman rank correlation coefficient (between -1 and 1).
    """
    pred_arr = np.asarray(predictions, dtype=float)
    ret_arr = np.asarray(returns, dtype=float)

    # Remove NaN pairs
    mask = ~(np.isnan(pred_arr) | np.isnan(ret_arr))
    pred_clean = pred_arr[mask]
    ret_clean = ret_arr[mask]

    if len(pred_clean) < 3:
        return 0.0

    corr, _pval = sp_stats.spearmanr(pred_clean, ret_clean)
    return float(corr)


def quantile_analysis(
    predictions: pd.Series,
    returns: pd.Series,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Analyse returns by prediction quantile.

    Sorts observations into quantiles based on *predictions* and
    computes summary statistics for each quantile bucket.

    Parameters:
        predictions: Predicted values or signals.
        returns: Subsequent realised returns.
        n_quantiles: Number of quantile buckets (default 5 = quintiles).

    Returns:
        DataFrame indexed by quantile (1 = lowest, *n_quantiles* =
        highest) with columns ``mean_return``, ``std_return``,
        ``hit_rate`` (fraction of positive returns), and ``count``.
    """
    combined = pd.DataFrame(
        {
            "prediction": predictions,
            "return": returns,
        }
    ).dropna()

    combined["quantile"] = pd.qcut(
        combined["prediction"],
        q=n_quantiles,
        labels=range(1, n_quantiles + 1),
        duplicates="drop",
    )

    results: list[dict] = []
    for q in sorted(combined["quantile"].unique()):
        group = combined[combined["quantile"] == q]["return"]
        results.append(
            {
                "quantile": int(q),
                "mean_return": float(group.mean()),
                "std_return": float(group.std()),
                "hit_rate": float((group > 0).mean()),
                "count": int(len(group)),
            }
        )

    return pd.DataFrame(results).set_index("quantile")
