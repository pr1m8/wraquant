"""Correlation and covariance estimation for financial data."""

from __future__ import annotations

import pandas as pd
from sklearn.covariance import OAS, LedoitWolf, ShrunkCovariance


def correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute a correlation matrix from asset returns.

    Parameters:
        returns: DataFrame of asset returns (columns = assets).
        method: Correlation method — ``"pearson"``, ``"spearman"``,
            or ``"kendall"``.

    Returns:
        Correlation matrix as a DataFrame.
    """
    return returns.corr(method=method)


def shrunk_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """Compute a shrinkage-estimated covariance matrix.

    Parameters:
        returns: DataFrame of asset returns (columns = assets).
        method: Shrinkage method — ``"ledoit_wolf"`` (default),
            ``"oas"``, or ``"basic"``.

    Returns:
        Shrunk covariance matrix as a DataFrame.

    Raises:
        ValueError: If *method* is not recognized.
    """
    clean = returns.dropna()
    assets = clean.columns

    if method == "ledoit_wolf":
        estimator = LedoitWolf().fit(clean.values)
    elif method == "oas":
        estimator = OAS().fit(clean.values)
    elif method == "basic":
        estimator = ShrunkCovariance().fit(clean.values)
    else:
        msg = f"Unknown shrinkage method: {method!r}"
        raise ValueError(msg)

    return pd.DataFrame(estimator.covariance_, index=assets, columns=assets)


def rolling_correlation(
    x: pd.Series,
    y: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling Pearson correlation between two series.

    Parameters:
        x: First series.
        y: Second series.
        window: Rolling window size.

    Returns:
        Rolling correlation series.
    """
    return x.rolling(window).corr(y)
