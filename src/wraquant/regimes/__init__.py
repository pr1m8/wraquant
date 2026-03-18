"""Market regime detection and classification.

Covers Hidden Markov Models, Markov-switching models, Gaussian mixture
regime detection, Kalman filtering/smoothing, time-varying regression,
Bayesian online change-point detection, and regime labeling.
"""

from wraquant.regimes.changepoint import online_changepoint
from wraquant.regimes.hmm import (
    fit_gaussian_hmm,
    fit_hmm,
    fit_ms_regression,
    gaussian_mixture_regimes,
    predict_regime,
    regime_aware_portfolio,
    regime_statistics,
    regime_transition_analysis,
    rolling_regime_probability,
)
from wraquant.regimes.integrations import (
    dynamax_lgssm,
    filterpy_kalman,
    pomegranate_hmm,
    river_drift_detector,
)
from wraquant.regimes.kalman import (
    kalman_filter,
    kalman_regression,
    kalman_smoother,
    unscented_kalman,
)
from wraquant.regimes.labels import label_regimes

__all__ = [
    # hmm
    "fit_gaussian_hmm",
    "fit_hmm",
    "fit_ms_regression",
    "gaussian_mixture_regimes",
    "predict_regime",
    "regime_aware_portfolio",
    "regime_statistics",
    "regime_transition_analysis",
    "rolling_regime_probability",
    # kalman
    "kalman_filter",
    "kalman_smoother",
    "kalman_regression",
    "unscented_kalman",
    # changepoint
    "online_changepoint",
    # labels
    "label_regimes",
    # integrations
    "pomegranate_hmm",
    "filterpy_kalman",
    "river_drift_detector",
    "dynamax_lgssm",
]
