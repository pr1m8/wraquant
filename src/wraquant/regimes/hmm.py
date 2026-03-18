"""Hidden Markov Model regime detection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from wraquant.core.decorators import requires_extra


@requires_extra("regimes")
def fit_hmm(returns: pd.Series, n_states: int = 2) -> Any:
    """Fit a Gaussian Hidden Markov Model to return data.

    Requires the ``regimes`` optional dependency group (``hmmlearn``).

    Parameters:
        returns: Simple return series.
        n_states: Number of hidden states.

    Returns:
        Fitted ``hmmlearn.hmm.GaussianHMM`` model.
    """
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=100,
        random_state=42,
    )
    X = returns.dropna().values.reshape(-1, 1)
    model.fit(X)
    return model


def predict_regime(model: Any, returns: pd.Series) -> pd.Series:
    """Predict regime states from a fitted HMM.

    Parameters:
        model: A fitted HMM model with a ``predict`` method.
        returns: Return series to classify.

    Returns:
        Series of integer regime labels aligned to *returns*.
    """
    clean = returns.dropna()
    X = clean.values.reshape(-1, 1)
    states = model.predict(X)
    return pd.Series(states, index=clean.index, name="regime")
