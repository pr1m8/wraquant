"""Model evaluation utilities for financial machine learning.

Provides both standard classification metrics and finance-specific
performance measures such as Sharpe ratio from predictions and backtesting
with transaction costs.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "classification_metrics",
    "financial_metrics",
    "learning_curve",
    "backtest_predictions",
]


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Use classification metrics to evaluate direction-prediction models
    (e.g., predicting up/down/flat labels).  These metrics assess the
    statistical quality of the classifier independently of PnL; pair
    with ``financial_metrics`` for economic evaluation.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like or None
        Predicted probabilities (for the positive class in binary
        classification).  When provided, log-loss and AUC are included.

    Returns
    -------
    dict[str, float]
        ``accuracy`` : float
            Fraction of correct predictions.
        ``precision`` : float
            Macro-averaged precision (how many predicted positives are
            actually positive).
        ``recall`` : float
            Macro-averaged recall (how many actual positives are
            captured).
        ``f1`` : float
            Macro-averaged F1 score (harmonic mean of precision and
            recall).
        ``log_loss`` : float (only if *y_prob* given)
            Cross-entropy loss.  Lower is better; measures calibration
            quality.
        ``auc`` : float (only if *y_prob* given, binary only)
            Area under the ROC curve.  0.5 = random, 1.0 = perfect.

    Example
    -------
    >>> import numpy as np
    >>> y_true = np.array([1, 0, 1, 1, 0, 1])
    >>> y_pred = np.array([1, 0, 0, 1, 0, 1])
    >>> metrics = classification_metrics(y_true, y_pred)
    >>> metrics['accuracy']
    0.8333333333333334
    >>> metrics['f1'] > 0.5
    True

    See Also
    --------
    financial_metrics : PnL-based evaluation of directional predictions.
    backtest_predictions : Full backtest with transaction costs.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    correct = y_true == y_pred
    accuracy = float(correct.mean())

    # Per-class precision / recall / F1, then macro-average
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    result: dict[str, float] = {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        # Log loss (binary or multi-class safe)
        eps = 1e-15
        if y_prob.ndim == 1:
            # Binary classification
            p = np.clip(y_prob, eps, 1 - eps)
            ll = -float(np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
        else:
            p = np.clip(y_prob, eps, 1 - eps)
            p.shape[1]
            one_hot = np.zeros_like(p)
            for i, c in enumerate(classes):
                one_hot[y_true == c, i] = 1.0
            ll = -float(np.mean(np.sum(one_hot * np.log(p), axis=1)))
        result["log_loss"] = ll

        # AUC (binary only)
        if y_prob.ndim == 1 and len(classes) == 2:
            result["auc"] = _auc_binary(y_true, y_prob, classes)

    return result


def _auc_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
) -> float:
    """Compute AUC for binary classification using the trapezoidal rule."""
    # Map labels to 0/1
    pos_label = classes[1]
    y_bin = (y_true == pos_label).astype(int)

    # Sort by descending probability
    order = np.argsort(-y_prob)
    y_sorted = y_bin[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps

    # Prepend origin
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    return float(np.trapezoid(tpr, fpr))


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------


def financial_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    returns: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute finance-specific evaluation metrics from predictions.

    Use financial metrics to evaluate whether a model's predictions
    translate into actual trading profits.  A model can have high
    accuracy but poor financial performance if it is right on small moves
    and wrong on large moves.  These metrics directly measure economic
    value.

    The predicted labels are interpreted as position signals: ``1`` for
    long, ``-1`` for short, ``0`` for flat.

    Parameters
    ----------
    y_true : array-like
        True directional labels.
    y_pred : array-like
        Predicted directional labels (used as signals).
    returns : array-like
        Actual period returns corresponding to each observation.

    Returns
    -------
    dict[str, float]
        ``strategy_return`` : float
            Cumulative strategy return (sum of signal * return).
        ``sharpe`` : float
            Annualised Sharpe ratio (252 trading days).  Values above
            1.0 are generally considered good; above 2.0 is excellent.
        ``hit_rate`` : float
            Fraction of periods where predicted sign matches actual
            sign.  A hit rate above 0.5 is necessary but not sufficient
            for profitability.
        ``profit_factor`` : float
            Gross profit / gross loss.  Values above 1.0 indicate a
            profitable strategy; above 2.0 is strong.

    Example
    -------
    >>> import numpy as np
    >>> y_true = np.array([1, -1, 1, 1, -1])
    >>> y_pred = np.array([1, -1, -1, 1, 1])
    >>> returns = np.array([0.02, -0.01, 0.015, 0.005, -0.02])
    >>> metrics = financial_metrics(y_true, y_pred, returns)
    >>> metrics['hit_rate']
    0.6
    >>> metrics['sharpe'] != 0
    True

    See Also
    --------
    classification_metrics : Standard ML classification metrics.
    backtest_predictions : Full backtest with transaction costs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    returns = np.asarray(returns, dtype=float)

    strat_returns = y_pred * returns

    cumulative = float(np.nansum(strat_returns))

    # Use canonical Sharpe implementation
    from wraquant.risk.metrics import sharpe_ratio as _sharpe_ratio

    sharpe = _sharpe_ratio(pd.Series(strat_returns)) if len(strat_returns) > 1 else 0.0

    # Hit rate: how often the predicted direction matches the actual
    correct_direction = np.sign(y_pred) == np.sign(y_true)
    hit_rate = float(np.nanmean(correct_direction))

    # Profit factor
    gross_profit = float(np.nansum(strat_returns[strat_returns > 0]))
    gross_loss = float(np.abs(np.nansum(strat_returns[strat_returns < 0])))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "strategy_return": cumulative,
        "sharpe": sharpe,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
    }


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------


@requires_extra("ml")
def learning_curve(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    train_sizes: Sequence[int | float] | np.ndarray | None = None,
    cv: int = 5,
) -> dict[str, np.ndarray]:
    """Generate a learning curve for a model.

    Use learning curves to diagnose whether a model suffers from high
    bias (underfitting) or high variance (overfitting).  If training and
    test scores converge at a low value, the model is too simple.  If
    there is a large gap between training and test scores, the model is
    overfitting and more data or regularisation is needed.

    Parameters
    ----------
    model : estimator
        A scikit-learn-compatible estimator.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector.
    train_sizes : Sequence or None
        Training set sizes (absolute counts or fractions).  Defaults to
        ``np.linspace(0.1, 1.0, 10)``.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    dict
        ``train_sizes`` : np.ndarray
            Absolute number of training samples at each point.
        ``train_scores`` : np.ndarray, shape ``(len(sizes), cv)``
            Training scores at each size/fold.  Plot the mean across
            folds to visualize training performance.
        ``test_scores`` : np.ndarray, shape ``(len(sizes), cv)``
            Test scores at each size/fold.  The gap between train and
            test mean scores indicates overfitting.

    Example
    -------
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> X = np.random.randn(300, 5)
    >>> y = X @ [1, 0.5, 0, 0, 0] + np.random.randn(300) * 0.1
    >>> result = learning_curve(Ridge(), X, y, cv=3)
    >>> result['train_sizes'].shape[0]  # 10 points by default
    10

    See Also
    --------
    classification_metrics : Evaluate classification quality.
    financial_metrics : Evaluate economic value of predictions.
    """
    from sklearn.model_selection import learning_curve as _lc

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    sizes, train_scores, test_scores = _lc(
        model,
        np.asarray(X),
        np.asarray(y),
        train_sizes=np.asarray(train_sizes),
        cv=cv,
        n_jobs=1,
    )

    return {
        "train_sizes": sizes,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }


# ---------------------------------------------------------------------------
# Backtest predictions
# ---------------------------------------------------------------------------


def backtest_predictions(
    predictions: pd.Series | np.ndarray,
    returns: pd.Series | np.ndarray,
    cost_bps: float = 10,
) -> dict[str, Any]:
    """Backtest a prediction signal against actual returns.

    Use backtest_predictions as a quick sanity check of a model's
    economic value before building a full backtest.  It applies
    realistic transaction costs (proportional to position changes)
    and computes key performance metrics including Sharpe, max drawdown,
    and turnover.

    Parameters
    ----------
    predictions : array-like
        Predicted position signals (e.g. 1, 0, -1).  The signal is
        applied as a position: ``signal * return``.
    returns : array-like
        Actual period returns corresponding to each prediction.
    cost_bps : float
        Transaction cost in basis points applied on each position
        change (default 10 bps).  For equities, 5-10 bps is typical;
        for futures, 1-3 bps.

    Returns
    -------
    dict
        ``gross_returns`` : np.ndarray
            Per-period strategy returns before costs.
        ``net_returns`` : np.ndarray
            Per-period strategy returns after costs.
        ``cumulative_return`` : float
            Total cumulative net return.  Positive = profitable.
        ``sharpe`` : float
            Annualised Sharpe ratio of net returns.  Above 1.0 is
            generally good; above 2.0 is excellent.
        ``max_drawdown`` : float
            Maximum peak-to-trough decline in cumulative PnL.
            Always negative or zero.
        ``turnover`` : float
            Mean absolute position change per period.  Higher turnover
            means higher transaction costs.

    Example
    -------
    >>> import numpy as np
    >>> preds = np.array([1, 1, -1, 1, -1, 0, 1])
    >>> rets = np.array([0.01, -0.005, -0.02, 0.015, 0.01, 0.005, 0.008])
    >>> result = backtest_predictions(preds, rets, cost_bps=10)
    >>> result['cumulative_return'] != 0
    True
    >>> result['max_drawdown'] <= 0
    True

    See Also
    --------
    financial_metrics : Quick financial metrics without transaction costs.
    wraquant.ml.pipeline.walk_forward_backtest : Walk-forward backtest.
    """
    preds = np.asarray(predictions, dtype=float)
    rets = np.asarray(returns, dtype=float)

    gross = preds * rets

    # Transaction costs
    position_changes = np.abs(np.diff(preds, prepend=0))
    costs = position_changes * (cost_bps / 10_000.0)
    net = gross - costs

    cumulative = float(np.nansum(net))
    mean_r = np.nanmean(net)
    std_r = np.nanstd(net, ddof=1) if len(net) > 1 else np.nan
    sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r and std_r > 0 else 0.0

    # Max drawdown on cumulative curve
    cum_curve = np.nancumsum(net)
    running_max = np.maximum.accumulate(cum_curve)
    drawdowns = cum_curve - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    turnover = float(np.nanmean(position_changes))

    return {
        "gross_returns": gross,
        "net_returns": net,
        "cumulative_return": cumulative,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "turnover": turnover,
    }
