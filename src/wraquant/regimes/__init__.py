"""Market regime detection and classification.

Covers Hidden Markov Models, Kalman filtering, Bayesian online
change-point detection, and regime labeling.
"""

from wraquant.regimes.changepoint import online_changepoint
from wraquant.regimes.hmm import fit_hmm, predict_regime
from wraquant.regimes.integrations import (
    dynamax_lgssm,
    filterpy_kalman,
    pomegranate_hmm,
    river_drift_detector,
)
from wraquant.regimes.kalman import kalman_filter
from wraquant.regimes.labels import label_regimes, regime_statistics

__all__ = [
    # hmm
    "fit_hmm",
    "predict_regime",
    # kalman
    "kalman_filter",
    # changepoint
    "online_changepoint",
    # labels
    "label_regimes",
    "regime_statistics",
    # integrations
    "pomegranate_hmm",
    "filterpy_kalman",
    "river_drift_detector",
    "dynamax_lgssm",
]
