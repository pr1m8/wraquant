"""External package wrappers for causal inference.

Functions in this module require the ``causal`` optional dependency group
(DoWhy, EconML, DoubleML) and are guarded by ``@requires_extra('causal')``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "dowhy_causal_model",
    "econml_dml",
    "econml_forest",
    "doubleml_plr",
]


# ---------------------------------------------------------------------------
# DoWhy
# ---------------------------------------------------------------------------


@requires_extra("causal")
def dowhy_causal_model(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    graph: str | None = None,
    common_causes: list[str] | None = None,
    method: str = "backdoor.propensity_score_matching",
) -> dict[str, Any]:
    """Build and estimate a causal model using DoWhy.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data.
    treatment : str
        Name of the treatment column.
    outcome : str
        Name of the outcome column.
    graph : str or None
        Causal graph in GML or DOT format. If None, common_causes must
        be provided and DoWhy will construct a simple graph.
    common_causes : list[str] or None
        List of common cause (confounder) column names. Required if
        graph is not provided.
    method : str
        Estimation method name (e.g., 'backdoor.propensity_score_matching',
        'backdoor.linear_regression', 'iv.instrumental_variable').

    Returns
    -------
    dict
        ``estimate``: float — estimated causal effect,
        ``p_value``: float or None — p-value if available,
        ``method``: str — method used,
        ``model``: DoWhy CausalModel object,
        ``identified_estimand``: the identified estimand,
        ``causal_estimate``: the full DoWhy estimate object.
    """
    import dowhy

    model = dowhy.CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=graph,
        common_causes=common_causes,
    )

    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name=method,
    )

    return {
        "estimate": float(estimate.value),
        "p_value": getattr(estimate, "test_stat_significance", {}).get("p_value"),
        "method": method,
        "model": model,
        "identified_estimand": identified_estimand,
        "causal_estimate": estimate,
    }


# ---------------------------------------------------------------------------
# EconML — Double Machine Learning
# ---------------------------------------------------------------------------


@requires_extra("causal")
def econml_dml(
    outcome: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    covariates: np.ndarray | pd.DataFrame,
    model_y: Any = None,
    model_t: Any = None,
    n_splits: int = 3,
) -> dict[str, Any]:
    """Estimate causal effects using EconML's LinearDML.

    Parameters
    ----------
    outcome : array-like
        Outcome variable.
    treatment : array-like
        Treatment variable (can be continuous).
    covariates : array-like
        Covariate matrix.
    model_y : estimator or None
        Nuisance model for the outcome. Defaults to Lasso.
    model_t : estimator or None
        Nuisance model for the treatment. Defaults to Lasso.
    n_splits : int
        Number of cross-fitting splits.

    Returns
    -------
    dict
        ``ate``: float — average treatment effect,
        ``se``: float — standard error,
        ``ci_lower``: float — lower CI bound,
        ``ci_upper``: float — upper CI bound,
        ``model``: fitted LinearDML object.
    """
    from econml.dml import LinearDML
    from sklearn.linear_model import LassoCV

    Y = np.asarray(outcome).ravel()
    T = np.asarray(treatment).ravel()
    X = np.asarray(covariates)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if model_y is None:
        model_y = LassoCV()
    if model_t is None:
        model_t = LassoCV()

    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=n_splits,
        random_state=42,
    )
    dml.fit(Y, T, X=X)

    ate = float(dml.ate())
    ate_inference = dml.ate_inference()

    return {
        "ate": ate,
        "se": float(ate_inference.stderr),
        "ci_lower": float(ate_inference.conf_int()[0][0]),
        "ci_upper": float(ate_inference.conf_int()[1][0]),
        "model": dml,
    }


# ---------------------------------------------------------------------------
# EconML — Causal Forest
# ---------------------------------------------------------------------------


@requires_extra("causal")
def econml_forest(
    outcome: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    covariates: np.ndarray | pd.DataFrame,
    n_estimators: int = 100,
    min_samples_leaf: int = 5,
) -> dict[str, Any]:
    """Estimate heterogeneous treatment effects using EconML's CausalForestDML.

    Parameters
    ----------
    outcome : array-like
        Outcome variable.
    treatment : array-like
        Treatment variable.
    covariates : array-like
        Covariate matrix.
    n_estimators : int
        Number of trees in the forest.
    min_samples_leaf : int
        Minimum number of samples per leaf.

    Returns
    -------
    dict
        ``ate``: float — average treatment effect,
        ``cate``: np.ndarray — conditional ATE for each observation,
        ``se``: float — standard error of ATE,
        ``model``: fitted CausalForestDML object.
    """
    from econml.dml import CausalForestDML

    Y = np.asarray(outcome).ravel()
    T = np.asarray(treatment).ravel()
    X = np.asarray(covariates)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    forest = CausalForestDML(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    forest.fit(Y, T, X=X)

    cate = forest.effect(X).ravel()
    ate = float(np.mean(cate))
    ate_inference = forest.ate_inference()

    return {
        "ate": ate,
        "cate": cate,
        "se": float(ate_inference.stderr),
        "model": forest,
    }


# ---------------------------------------------------------------------------
# DoubleML — Partially Linear Regression
# ---------------------------------------------------------------------------


@requires_extra("causal")
def doubleml_plr(
    outcome: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    covariates: np.ndarray | pd.DataFrame,
    n_folds: int = 5,
    ml_l: Any = None,
    ml_m: Any = None,
) -> dict[str, Any]:
    """Estimate treatment effect using DoubleML's partially linear regression.

    Parameters
    ----------
    outcome : array-like
        Outcome variable.
    treatment : array-like
        Treatment variable.
    covariates : array-like
        Covariate matrix.
    n_folds : int
        Number of cross-fitting folds.
    ml_l : estimator or None
        Nuisance learner for E[Y|X]. Defaults to Lasso.
    ml_m : estimator or None
        Nuisance learner for E[D|X]. Defaults to Lasso.

    Returns
    -------
    dict
        ``ate``: float — treatment effect estimate (theta),
        ``se``: float — standard error,
        ``ci_lower``: float — lower CI bound,
        ``ci_upper``: float — upper CI bound,
        ``t_stat``: float — t-statistic,
        ``p_value``: float — p-value,
        ``model``: fitted DoubleMLPLR object.
    """
    import doubleml as dml
    from sklearn.linear_model import LassoCV

    Y = np.asarray(outcome).ravel()
    D = np.asarray(treatment).ravel()
    X = np.asarray(covariates)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    col_names = [f"X{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=col_names)
    df["Y"] = Y
    df["D"] = D

    data = dml.DoubleMLData(df, y_col="Y", d_cols="D", x_cols=col_names)

    if ml_l is None:
        ml_l = LassoCV()
    if ml_m is None:
        ml_m = LassoCV()

    plr = dml.DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=n_folds)
    plr.fit()

    summary = plr.summary
    return {
        "ate": float(plr.coef[0]),
        "se": float(plr.se[0]),
        "ci_lower": float(summary["2.5 %"].iloc[0]),
        "ci_upper": float(summary["97.5 %"].iloc[0]),
        "t_stat": float(plr.t_stat[0]),
        "p_value": float(plr.pval[0]),
        "model": plr,
    }
