"""Advanced risk and portfolio optimisation integrations.

Provides wrappers around PyPortfolioOpt, Riskfolio-Lib, skfolio,
copulas, pyvinecopulib, and pyextremes for portfolio construction,
copula modelling, and extreme value analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "pypfopt_efficient_frontier",
    "riskfolio_portfolio",
    "skfolio_optimize",
    "copulas_fit",
    "copulae_fit",
    "vine_copula",
    "extreme_value_analysis",
]


@requires_extra("risk")
def pypfopt_efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> dict[str, Any]:
    """Compute the efficient frontier using PyPortfolioOpt.

    Solves for the maximum-Sharpe-ratio portfolio and returns the
    optimal weights along with performance metrics.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected annual returns per asset.
    cov_matrix : pd.DataFrame
        Annualised covariance matrix of asset returns.

    Returns
    -------
    dict
        Dictionary containing:

        * **weights** -- dict mapping asset names to optimal weights.
        * **expected_return** -- portfolio expected annual return.
        * **volatility** -- portfolio expected annual volatility.
        * **sharpe_ratio** -- portfolio Sharpe ratio.
    """
    from pypfopt.efficient_frontier import EfficientFrontier

    ef = EfficientFrontier(expected_returns, cov_matrix)
    ef.max_sharpe()
    weights = ef.clean_weights()
    perf = ef.portfolio_performance()

    return {
        "weights": dict(weights),
        "expected_return": float(perf[0]),
        "volatility": float(perf[1]),
        "sharpe_ratio": float(perf[2]),
    }


@requires_extra("risk")
def riskfolio_portfolio(
    returns: pd.DataFrame,
    method: str = "MV",
) -> dict[str, Any]:
    """Optimise a portfolio using Riskfolio-Lib.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical return data. Each column is an asset.
    method : str, default 'MV'
        Risk measure for optimisation:

        * ``'MV'`` -- minimum variance
        * ``'CVaR'`` -- conditional value-at-risk
        * ``'MAD'`` -- mean absolute deviation

    Returns
    -------
    dict
        Dictionary containing:

        * **weights** -- dict mapping asset names to optimal weights.
        * **method** -- risk measure used.
    """
    import riskfolio as rp

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist")

    weights_df = port.optimization(
        model="Classic",
        rm=method,
        obj="Sharpe",
        hist=True,
    )

    weights = {
        asset: float(weights_df.loc[asset, "weights"]) for asset in weights_df.index
    }

    return {
        "weights": weights,
        "method": method,
    }


@requires_extra("risk")
def skfolio_optimize(
    returns: pd.DataFrame,
    objective: str = "min_variance",
) -> dict[str, Any]:
    """Optimise a portfolio using skfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical return data. Each column is an asset.
    objective : str, default 'min_variance'
        Optimisation objective:

        * ``'min_variance'`` -- minimum variance portfolio
        * ``'max_sharpe'`` -- maximum Sharpe ratio portfolio

    Returns
    -------
    dict
        Dictionary containing:

        * **weights** -- dict mapping asset names to optimal weights.
        * **objective** -- objective used.
    """
    from skfolio.optimization import MeanRisk
    from skfolio.prior import EmpiricalPrior

    if objective == "max_sharpe":
        from skfolio.optimization import ObjectiveFunction

        model = MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
            prior_estimator=EmpiricalPrior(),
        )
    else:
        model = MeanRisk(prior_estimator=EmpiricalPrior())

    model.fit(returns)
    weights_arr = model.weights_

    weights = {
        asset: float(w) for asset, w in zip(returns.columns, weights_arr, strict=False)
    }

    return {
        "weights": weights,
        "objective": objective,
    }


@requires_extra("risk")
def copulas_fit(
    data: pd.DataFrame,
    copula_type: str = "gaussian",
) -> dict[str, Any]:
    """Fit a copula model to multivariate data.

    Uses the ``copulas`` library to fit a copula and provides
    methods for sampling and density evaluation.

    Parameters
    ----------
    data : pd.DataFrame
        Multivariate data. Each column is a variable.
    copula_type : str, default 'gaussian'
        Type of copula to fit:

        * ``'gaussian'`` -- Gaussian copula
        * ``'vine'`` -- vine copula (via copulas library)

    Returns
    -------
    dict
        Dictionary containing:

        * **copula** -- fitted copula object.
        * **copula_type** -- type of copula used.
        * **columns** -- list of column names from the input data.
        * **n_samples** -- number of observations used for fitting.
    """
    from copulas.multivariate import GaussianMultivariate, VineCopula

    if copula_type == "gaussian":
        copula = GaussianMultivariate()
    elif copula_type == "vine":
        copula = VineCopula("regular")
    else:
        raise ValueError(
            f"Unknown copula_type: {copula_type!r}. Use 'gaussian' or 'vine'."
        )

    copula.fit(data)

    return {
        "copula": copula,
        "copula_type": copula_type,
        "columns": list(data.columns),
        "n_samples": len(data),
    }


@requires_extra("risk")
def vine_copula(
    data: np.ndarray | pd.DataFrame,
    structure: str = "regular",
) -> dict[str, Any]:
    """Fit a vine copula using pyvinecopulib.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Multivariate data. Each column is a variable. Values should
        ideally be on the unit interval (pseudo-observations); if not,
        the data is rank-transformed automatically.
    structure : str, default 'regular'
        Vine structure type. Currently only ``'regular'`` is supported.

    Returns
    -------
    dict
        Dictionary containing:

        * **vinecop** -- fitted ``pyvinecopulib.Vinecop`` object.
        * **structure** -- vine structure used.
        * **n_vars** -- number of variables.
        * **loglik** -- log-likelihood of the fitted model.
    """
    import pyvinecopulib as pv

    if isinstance(data, pd.DataFrame):
        values = data.values
    else:
        values = np.asarray(data, dtype=np.float64)

    # Convert to pseudo-observations (uniform marginals) if needed
    from scipy.stats import rankdata

    n = values.shape[0]
    u = np.column_stack(
        [rankdata(values[:, j]) / (n + 1) for j in range(values.shape[1])]
    )

    controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.gaussian])
    cop = pv.Vinecop(u.shape[1])
    cop.select(u, controls=controls)

    return {
        "vinecop": cop,
        "structure": structure,
        "n_vars": values.shape[1],
        "loglik": float(cop.loglik(u)),
    }


@requires_extra("risk")
def extreme_value_analysis(
    data: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Perform extreme value analysis using pyextremes.

    Fits a Generalized Extreme Value (GEV) distribution to block
    maxima extracted from the data.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        Univariate time series of observations (e.g. losses or
        negative returns).

    Returns
    -------
    dict
        Dictionary containing:

        * **shape** -- GEV shape parameter (xi).
        * **loc** -- GEV location parameter (mu).
        * **scale** -- GEV scale parameter (sigma).
        * **return_levels** -- dict of return levels for common
          return periods (10, 50, 100 years).
    """
    import pyextremes

    if isinstance(data, np.ndarray):
        data = pd.Series(data, name="values")

    if data.name is None:
        data = data.rename("values")

    pyextremes.get_extremes(data, method="BM", block_size="365.2425D")

    model = pyextremes.EVA(data)
    model.get_extremes(method="BM", block_size="365.2425D")
    model.fit_model()

    summary = model.get_summary(
        return_period=[10, 50, 100],
        alpha=0.95,
    )

    return_levels = {}
    for period in [10, 50, 100]:
        row = summary.loc[summary.index == period]
        if len(row) > 0:
            return_levels[period] = float(row.iloc[0, 0])

    params = model.distribution.mle_parameters
    return {
        "shape": float(params.get("c", params.get("shape", np.nan))),
        "loc": float(params.get("loc", np.nan)),
        "scale": float(params.get("scale", np.nan)),
        "return_levels": return_levels,
    }


# ---------------------------------------------------------------------------
# copulae library
# ---------------------------------------------------------------------------


@requires_extra("risk")
def copulae_fit(
    data: np.ndarray | pd.DataFrame,
    family: str = "gaussian",
) -> dict[str, Any]:
    """Fit a copula using the copulae library.

    Alternative to wraquant's built-in copula fitting (``copulas_fit``)
    and the ``fit_*_copula`` functions in ``wraquant.risk.copulas``.
    The ``copulae`` library provides additional families (Joe, AMH)
    and more robust MLE estimation via IFM (Inference Functions for
    Margins).

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Data of shape ``(n, d)``. Can be raw observations (marginals
        are automatically converted to pseudo-observations) or
        pre-transformed uniform marginals on ``[0, 1]``.
    family : str, default 'gaussian'
        Copula family to fit:

        * ``'gaussian'`` -- Gaussian copula (no tail dependence).
        * ``'student'`` -- Student-t copula (symmetric tail dependence).
        * ``'clayton'`` -- Clayton copula (lower tail dependence).
        * ``'gumbel'`` -- Gumbel copula (upper tail dependence).
        * ``'frank'`` -- Frank copula (symmetric, no tail dependence).

    Returns
    -------
    dict
        Dictionary containing:

        * **params** -- fitted copula parameters (structure depends
          on family).
        * **log_likelihood** -- log-likelihood of the fitted model.
        * **aic** -- Akaike information criterion.
        * **bic** -- Bayesian information criterion (approximate).
        * **fitted_copula** -- the fitted copula object for further
          use (sampling, CDF evaluation, etc.).

    Example
    -------
    >>> import numpy as np
    >>> from wraquant.risk.integrations import copulae_fit
    >>> rng = np.random.default_rng(42)
    >>> data = rng.normal(0, 1, (200, 3))
    >>> result = copulae_fit(data, family="gaussian")
    >>> result["log_likelihood"]  # doctest: +SKIP

    Notes
    -----
    Reference: Joe (2014). *Dependence Modeling with Copulas*.
    Chapman & Hall/CRC.

    See Also
    --------
    copulas_fit : Alternative using the ``copulas`` library.
    fit_gaussian_copula : Built-in Gaussian copula implementation.
    vine_copula : Vine copula fitting via ``pyvinecopulib``.
    """
    from copulae import (
        ClaytonCopula,
        FrankCopula,
        GaussianCopula,
        GumbelCopula,
        StudentCopula,
    )

    families = {
        "gaussian": GaussianCopula,
        "student": StudentCopula,
        "clayton": ClaytonCopula,
        "gumbel": GumbelCopula,
        "frank": FrankCopula,
    }

    copula_cls = families.get(family)
    if copula_cls is None:
        raise ValueError(
            f"Unknown family: {family!r}. Choose from {list(families)}."
        )

    if isinstance(data, pd.DataFrame):
        values = data.values
    else:
        values = np.asarray(data, dtype=np.float64)

    dim = values.shape[1]
    copula = copula_cls(dim=dim)
    copula.fit(values)

    ll = float(copula.log_lik(values))
    n_params = dim * (dim - 1) // 2  # approximate number of params
    n = values.shape[0]
    aic = 2.0 * n_params - 2.0 * ll
    bic = n_params * np.log(n) - 2.0 * ll

    return {
        "params": copula.params,
        "log_likelihood": ll,
        "aic": float(aic),
        "bic": float(bic),
        "fitted_copula": copula,
    }
