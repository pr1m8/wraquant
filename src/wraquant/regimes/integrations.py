"""Advanced regime detection integrations using optional packages.

Provides wrappers around pomegranate, filterpy, river, and dynamax for
Hidden Markov Models, Kalman filtering, online drift detection, and
Linear Gaussian State Space Models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "pomegranate_hmm",
    "filterpy_kalman",
    "river_drift_detector",
    "dynamax_lgssm",
]


@requires_extra("regimes")
def pomegranate_hmm(
    data: pd.Series | np.ndarray,
    n_states: int = 2,
) -> dict[str, Any]:
    """Fit a Hidden Markov Model using pomegranate.

    Parameters
    ----------
    data : pd.Series or np.ndarray
        Univariate observations (e.g. returns).
    n_states : int, default 2
        Number of hidden states.

    Returns
    -------
    dict
        Dictionary containing:

        * **states** -- predicted hidden state sequence (1-D int array).
        * **means** -- estimated mean of each state's emission distribution.
        * **model** -- the fitted pomegranate ``DenseHMM`` object.
        * **n_states** -- number of hidden states.
    """
    from pomegranate.distributions import Normal
    from pomegranate.hmm import DenseHMM

    values = np.asarray(data, dtype=np.float64).reshape(-1, 1)

    distributions = [Normal() for _ in range(n_states)]
    model = DenseHMM(distributions=distributions, verbose=False)
    model.fit(values.reshape(1, -1, 1))

    states = model.predict(values.reshape(1, -1, 1))
    states = np.asarray(states).flatten()

    means = []
    for dist in model.distributions:
        means.append(float(dist.means[0]))

    return {
        "states": states.astype(int),
        "means": means,
        "model": model,
        "n_states": n_states,
    }


@requires_extra("regimes")
def filterpy_kalman(
    observations: np.ndarray | pd.Series,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run a Kalman filter using filterpy.

    Parameters
    ----------
    observations : np.ndarray or pd.Series
        Observed measurements. Shape ``(T,)`` for univariate or
        ``(T, dim_z)`` for multivariate observations.
    F : np.ndarray
        State transition matrix. Shape ``(dim_x, dim_x)``.
    H : np.ndarray
        Observation (measurement) matrix. Shape ``(dim_z, dim_x)``.
    Q : np.ndarray
        Process noise covariance. Shape ``(dim_x, dim_x)``.
    R : np.ndarray
        Measurement noise covariance. Shape ``(dim_z, dim_z)``.
    x0 : np.ndarray or None, default None
        Initial state estimate. Shape ``(dim_x,)`` or ``(dim_x, 1)``.
        Defaults to zeros.
    P0 : np.ndarray or None, default None
        Initial covariance estimate. Shape ``(dim_x, dim_x)``.
        Defaults to identity.

    Returns
    -------
    dict
        Dictionary containing:

        * **filtered_states** -- filtered state estimates, shape ``(T, dim_x)``.
        * **filtered_covariances** -- filtered covariance matrices,
          shape ``(T, dim_x, dim_x)``.
        * **log_likelihood** -- total log-likelihood.
        * **residuals** -- measurement residuals, shape ``(T, dim_z)``.
    """
    from filterpy.kalman import KalmanFilter

    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)

    dim_z = obs.shape[1]
    dim_x = F.shape[0]

    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kf.F = F.copy()
    kf.H = H.copy()
    kf.Q = Q.copy()
    kf.R = R.copy()

    if x0 is not None:
        kf.x = x0.reshape(dim_x, 1).copy()
    if P0 is not None:
        kf.P = P0.copy()

    T = len(obs)
    filtered_states = np.zeros((T, dim_x))
    filtered_covs = np.zeros((T, dim_x, dim_x))
    residuals = np.zeros((T, dim_z))
    log_likelihood = 0.0

    for t in range(T):
        kf.predict()
        kf.update(obs[t])
        filtered_states[t] = kf.x.flatten()
        filtered_covs[t] = kf.P.copy()
        residuals[t] = kf.y.flatten()
        log_likelihood += float(kf.log_likelihood)

    return {
        "filtered_states": filtered_states,
        "filtered_covariances": filtered_covs,
        "log_likelihood": log_likelihood,
        "residuals": residuals,
    }


@requires_extra("regimes")
def river_drift_detector(
    stream: np.ndarray | pd.Series | list[float],
    method: str = "adwin",
) -> dict[str, Any]:
    """Detect concept drift in a data stream using river.

    Processes each observation sequentially and records indices where
    a drift is detected.

    Parameters
    ----------
    stream : np.ndarray, pd.Series, or list of float
        Sequential stream of numeric observations.
    method : str, default 'adwin'
        Drift detection method:

        * ``'adwin'`` -- Adaptive Windowing (ADWIN)
        * ``'ddm'`` -- Drift Detection Method
        * ``'eddm'`` -- Early Drift Detection Method
        * ``'page_hinkley'`` -- Page-Hinkley test

    Returns
    -------
    dict
        Dictionary containing:

        * **drift_indices** -- list of indices where drift was detected.
        * **n_drifts** -- total number of drifts detected.
        * **method** -- drift detection method used.
    """
    from river import drift
    from river.drift import binary as _binary

    detectors: dict[str, Any] = {
        "adwin": drift.ADWIN,
        "ddm": _binary.DDM,
        "eddm": _binary.EDDM,
        "page_hinkley": drift.PageHinkley,
    }
    detector_cls = detectors.get(method)
    if detector_cls is None:
        raise ValueError(
            f"Unknown method: {method!r}. Choose from {list(detectors)}."
        )
    detector = detector_cls()

    values = np.asarray(stream, dtype=np.float64)
    drift_indices: list[int] = []

    for i, val in enumerate(values):
        detector.update(float(val))
        if detector.drift_detected:
            drift_indices.append(i)

    return {
        "drift_indices": drift_indices,
        "n_drifts": len(drift_indices),
        "method": method,
    }


# ---------------------------------------------------------------------------
# dynamax Linear Gaussian SSM
# ---------------------------------------------------------------------------


@requires_extra("regimes")
def dynamax_lgssm(
    observations: np.ndarray | pd.Series,
    state_dim: int = 2,
    emission_dim: int | None = None,
    n_iters: int = 100,
) -> dict[str, Any]:
    """Fit a Linear Gaussian State Space Model using dynamax (JAX-based).

    Parameters
    ----------
    observations : np.ndarray or pd.Series
        Observations of shape ``(T, obs_dim)`` or ``(T,)`` for univariate.
    state_dim : int, default 2
        Dimensionality of the latent state.
    emission_dim : int or None
        Observation dimension. Inferred from *observations* if None.
    n_iters : int, default 100
        Number of EM iterations.

    Returns
    -------
    dict
        Dictionary containing:

        * **filtered_means** -- filtered state means, shape ``(T, state_dim)``.
        * **filtered_covs** -- filtered state covariances.
        * **smoothed_means** -- smoothed state means.
        * **params** -- learned model parameters.
        * **log_likelihoods** -- EM log-likelihood trace.
    """
    from dynamax.linear_gaussian_ssm import LinearGaussianSSM
    import jax.numpy as jnp
    import jax.random as jr

    obs = jnp.array(np.asarray(observations, dtype=np.float64))
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    obs_dim = obs.shape[1]
    emission_dim = emission_dim or obs_dim

    model = LinearGaussianSSM(state_dim=state_dim, emission_dim=emission_dim)
    key = jr.PRNGKey(0)
    params, props = model.initialize(key)
    params, log_liks = model.fit_em(params, props, obs, num_iters=n_iters)

    filtered = model.filter(params, obs)
    smoothed = model.smoother(params, obs)

    return {
        "filtered_means": np.asarray(filtered.filtered_means),
        "filtered_covs": np.asarray(filtered.filtered_covariances),
        "smoothed_means": np.asarray(smoothed.smoothed_means),
        "params": params,
        "log_likelihoods": np.asarray(log_liks),
    }
