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
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1``.  If *y_prob* is given, also ``log_loss`` and ``auc``.
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
        ``strategy_return``: cumulative strategy return,
        ``sharpe``: annualised Sharpe ratio (252 trading days),
        ``hit_rate``: fraction of correct direction predictions,
        ``profit_factor``: gross profit / gross loss.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    returns = np.asarray(returns, dtype=float)

    strat_returns = y_pred * returns

    cumulative = float(np.nansum(strat_returns))
    mean_ret = np.nanmean(strat_returns)
    std_ret = np.nanstd(strat_returns, ddof=1) if len(strat_returns) > 1 else np.nan
    sharpe = (
        float(mean_ret / std_ret * np.sqrt(252)) if std_ret and std_ret > 0 else 0.0
    )

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
        ``train_sizes``: np.ndarray of absolute training sizes,
        ``train_scores``: np.ndarray of shape ``(len(sizes), cv)`` with
        training scores,
        ``test_scores``: np.ndarray of shape ``(len(sizes), cv)`` with
        test scores.
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

    Parameters
    ----------
    predictions : array-like
        Predicted position signals (e.g. 1, 0, -1).
    returns : array-like
        Actual period returns.
    cost_bps : float
        Transaction cost in basis points applied on each position
        change.

    Returns
    -------
    dict
        ``gross_returns``: np.ndarray of per-period strategy returns
        before costs,
        ``net_returns``: np.ndarray of per-period strategy returns after
        costs,
        ``cumulative_return``: float,
        ``sharpe``: float (annualised),
        ``max_drawdown``: float,
        ``turnover``: float (mean absolute position change per period).
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
