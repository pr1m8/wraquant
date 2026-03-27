"""Advanced time series integrations using optional packages.

Provides wrappers around tsfresh, stumpy, pywavelets, sktime,
statsforecast, and tslearn for feature extraction, matrix profiles,
wavelet analysis, forecasting, and time series clustering.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.core.decorators import requires_extra

__all__ = [
    "tsfresh_features",
    "stumpy_matrix_profile",
    "wavelet_transform",
    "wavelet_denoise",
    "sktime_forecast",
    "statsforecast_auto",
    "tslearn_dtw",
    "tslearn_kmeans",
    "darts_forecast",
]


@requires_extra("timeseries")
def tsfresh_features(
    df: pd.DataFrame,
    column_id: str = "id",
    column_sort: str = "time",
) -> pd.DataFrame:
    """Extract time series features using tsfresh.

    Wraps ``tsfresh.extract_features`` with efficient defaults and
    returns a cleaned DataFrame with no NaN/infinite columns.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame containing time series data. Must include
        the columns specified by *column_id* and *column_sort*, plus one
        or more value columns.
    column_id : str, default 'id'
        Column identifying distinct time series.
    column_sort : str, default 'time'
        Column used for temporal ordering.

    Returns
    -------
    pd.DataFrame
        Extracted features indexed by *column_id* values. Columns with
        NaN or infinite values are dropped.
    """
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute

    features = extract_features(
        df,
        column_id=column_id,
        column_sort=column_sort,
        disable_progressbar=True,
    )
    features = impute(features)
    # Drop any remaining columns with NaN or infinite values
    features = features.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    return features


@requires_extra("timeseries")
def stumpy_matrix_profile(
    ts: np.ndarray | pd.Series,
    m: int = 50,
) -> dict[str, Any]:
    """Compute the matrix profile of a time series using STUMPY.

    The matrix profile identifies the nearest-neighbour distance for
    every subsequence of length *m*, enabling motif discovery and
    discord (anomaly) detection.

    Parameters
    ----------
    ts : np.ndarray or pd.Series
        Univariate time series.
    m : int, default 50
        Subsequence window length.

    Returns
    -------
    dict
        Dictionary containing:

        * **matrix_profile** -- 1-D array of nearest-neighbour distances.
        * **profile_index** -- 1-D array of indices of the nearest neighbour.
        * **motif_idx** -- index of the top motif (lowest MP value).
        * **discord_idx** -- index of the top discord (highest MP value).
    """
    import stumpy

    values = np.asarray(ts, dtype=np.float64)
    result = stumpy.stump(values, m)
    mp = result[:, 0].astype(float)
    mp_idx = result[:, 1].astype(int)

    return {
        "matrix_profile": mp,
        "profile_index": mp_idx,
        "motif_idx": int(np.argmin(mp)),
        "discord_idx": int(np.argmax(mp)),
    }


@requires_extra("timeseries")
def wavelet_transform(
    data: np.ndarray | pd.Series,
    wavelet: str = "db4",
    level: int | None = None,
) -> dict[str, Any]:
    """Perform discrete wavelet transform (DWT) using PyWavelets.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Univariate signal to decompose.
    wavelet : str, default 'db4'
        Wavelet name (e.g. ``'db4'``, ``'haar'``, ``'sym5'``).
    level : int or None, default None
        Decomposition level.  When *None*, the maximum useful level is
        computed automatically.

    Returns
    -------
    dict
        Dictionary containing:

        * **coeffs** -- list of coefficient arrays ``[cA_n, cD_n, ..., cD_1]``.
        * **wavelet** -- wavelet name used.
        * **level** -- decomposition level used.
    """
    import pywt

    values = np.asarray(data, dtype=np.float64)
    if level is None:
        level = pywt.dwt_max_level(len(values), wavelet)
    coeffs = pywt.wavedec(values, wavelet, level=level)
    return {
        "coeffs": coeffs,
        "wavelet": wavelet,
        "level": level,
    }


@requires_extra("timeseries")
def wavelet_denoise(
    data: np.ndarray | pd.Series,
    wavelet: str = "db4",
    level: int | None = None,
    threshold: str = "soft",
) -> np.ndarray:
    """Denoise a signal using wavelet thresholding.

    Applies universal (VisuShrink) thresholding to the detail
    coefficients after a DWT decomposition.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Noisy univariate signal.
    wavelet : str, default 'db4'
        Wavelet name.
    level : int or None, default None
        Decomposition level (auto-selected when *None*).
    threshold : str, default 'soft'
        Thresholding mode: ``'soft'`` or ``'hard'``.

    Returns
    -------
    np.ndarray
        Denoised signal with the same length as the input.
    """
    import pywt

    values = np.asarray(data, dtype=np.float64)
    if level is None:
        level = pywt.dwt_max_level(len(values), wavelet)
    coeffs = pywt.wavedec(values, wavelet, level=level)

    # Universal threshold (VisuShrink)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(values)))

    denoised_coeffs = [coeffs[0]]  # keep approximation coefficients
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, value=uthresh, mode=threshold))

    reconstructed = pywt.waverec(denoised_coeffs, wavelet)
    # waverec may produce an extra sample; trim to original length
    return reconstructed[: len(values)]


@requires_extra("timeseries")
def sktime_forecast(
    y: pd.Series,
    model: str = "naive",
    horizon: int = 10,
) -> pd.DataFrame:
    """Forecast a time series using sktime's unified interface.

    Parameters
    ----------
    y : pd.Series
        Historical observations indexed by a pandas PeriodIndex or
        integer index.
    model : str, default 'naive'
        Forecasting model. Supported values:

        * ``'naive'`` -- last-value forecast
        * ``'theta'`` -- Theta method
        * ``'ets'`` -- exponential smoothing (AutoETS)
    horizon : int, default 10
        Number of steps ahead to forecast.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``forecast`` and ``index`` covering the
        forecast horizon.
    """
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.theta import ThetaForecaster

    fh = list(range(1, horizon + 1))

    if model == "naive":
        forecaster = NaiveForecaster(strategy="last")
    elif model == "theta":
        forecaster = ThetaForecaster()
    elif model == "ets":
        from sktime.forecasting.ets import AutoETS

        forecaster = AutoETS(auto=True, sp=1)
    else:
        raise ValueError(f"Unknown model: {model!r}. Use 'naive', 'theta', or 'ets'.")

    forecaster.fit(y)
    pred = forecaster.predict(fh=fh)
    return pd.DataFrame({"forecast": pred.values}, index=pred.index)


@requires_extra("timeseries")
def statsforecast_auto(
    y: pd.Series,
    season_length: int = 1,
    horizon: int = 10,
) -> pd.DataFrame:
    """Automatic forecasting using Nixtla's StatsForecast.

    Runs AutoARIMA and returns point forecasts.

    Parameters
    ----------
    y : pd.Series
        Historical observations. Must have a DatetimeIndex or
        integer index.
    season_length : int, default 1
        Seasonal period length (e.g. 12 for monthly data with yearly
        seasonality).
    horizon : int, default 10
        Number of periods to forecast.

    Returns
    -------
    pd.DataFrame
        DataFrame with column ``forecast`` indexed over the forecast
        horizon.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    # StatsForecast expects a DataFrame with columns unique_id, ds, y
    sf_df = pd.DataFrame({
        "unique_id": "series_1",
        "ds": y.index,
        "y": y.values,
    })

    sf = StatsForecast(
        models=[AutoARIMA(season_length=season_length)],
        freq=pd.infer_freq(y.index) or "D",
    )
    sf.fit(sf_df)
    forecast_df = sf.predict(h=horizon)
    return pd.DataFrame(
        {"forecast": forecast_df["AutoARIMA"].values},
        index=forecast_df["ds"].values if "ds" in forecast_df.columns else range(horizon),
    )


@requires_extra("timeseries")
def tslearn_dtw(
    ts1: np.ndarray | pd.Series,
    ts2: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Compute DTW distance between two time series using tslearn.

    Parameters
    ----------
    ts1 : np.ndarray or pd.Series
        First time series.
    ts2 : np.ndarray or pd.Series
        Second time series.

    Returns
    -------
    dict
        Dictionary containing:

        * **distance** -- DTW distance.
        * **path** -- optimal alignment path as a list of ``(i, j)`` tuples.
    """
    from tslearn.metrics import dtw_path

    a = np.asarray(ts1, dtype=np.float64).reshape(-1, 1)
    b = np.asarray(ts2, dtype=np.float64).reshape(-1, 1)
    path, distance = dtw_path(a, b)
    return {
        "distance": float(distance),
        "path": [(int(i), int(j)) for i, j in path],
    }


@requires_extra("timeseries")
def tslearn_kmeans(
    dataset: np.ndarray | list[np.ndarray],
    n_clusters: int = 3,
) -> dict[str, Any]:
    """Cluster time series using DTW-based K-means from tslearn.

    Parameters
    ----------
    dataset : np.ndarray or list of np.ndarray
        Collection of time series. If a 2-D array, each row is treated
        as a separate time series. If a 3-D array, shape should be
        ``(n_series, n_timestamps, n_features)``.
    n_clusters : int, default 3
        Number of clusters.

    Returns
    -------
    dict
        Dictionary containing:

        * **labels** -- cluster assignments (1-D int array).
        * **inertia** -- sum of distances to nearest cluster centre.
        * **cluster_centers** -- array of cluster centre time series.
        * **n_clusters** -- number of clusters used.
    """
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.utils import to_time_series_dataset

    ts_data = to_time_series_dataset(dataset)
    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        max_iter=50,
        random_state=42,
    )
    labels = km.fit_predict(ts_data)
    return {
        "labels": labels,
        "inertia": float(km.inertia_),
        "cluster_centers": km.cluster_centers_,
        "n_clusters": n_clusters,
    }


# ---------------------------------------------------------------------------
# Darts deep learning forecasting
# ---------------------------------------------------------------------------


@requires_extra("timeseries")
def darts_forecast(
    data: np.ndarray | pd.Series,
    model: str = "nbeats",
    horizon: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Forecast using Darts deep learning models.

    Darts provides production-quality implementations of N-BEATS,
    N-HiTS, Temporal Fusion Transformer, and other deep forecasting
    models. Use when you need state-of-the-art accuracy and have
    sufficient data (>1000 observations).

    The model is trained on the provided data and produces a forecast
    of the specified horizon. Default training parameters are tuned
    for quick experimentation; pass additional keyword arguments to
    override them (e.g., ``n_epochs=200``).

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Price or return series. Must be univariate.
    model : str, default 'nbeats'
        Deep learning model to use:

        * ``'nbeats'`` -- N-BEATS (Neural Basis Expansion Analysis).
        * ``'nhits'`` -- N-HiTS (Neural Hierarchical Interpolation).
        * ``'tcn'`` -- Temporal Convolutional Network.
        * ``'rnn'`` -- Simple RNN (LSTM-based).
        * ``'transformer'`` -- Vanilla Transformer.
    horizon : int, default 10
        Number of steps ahead to forecast.
    **kwargs
        Additional keyword arguments passed to the model constructor
        (e.g., ``n_epochs``, ``input_chunk_length``).

    Returns
    -------
    dict
        Dictionary containing:

        * **forecast** -- 1-D numpy array of forecasted values.
        * **model_name** -- name of the model used.
        * **training_loss** -- final training loss (when available).

    Example
    -------
    >>> import numpy as np
    >>> from wraquant.ts.advanced import darts_forecast
    >>> data = np.cumsum(np.random.default_rng(0).normal(0, 1, 500))
    >>> result = darts_forecast(data, model="nbeats", horizon=5, n_epochs=5)
    >>> len(result["forecast"])
    5

    Notes
    -----
    Reference: Oreshkin et al. (2020). "N-BEATS: Neural basis expansion
    analysis for interpretable time series forecasting." *ICLR 2020*.

    See Also
    --------
    sktime_forecast : Simpler statistical forecasting models.
    statsforecast_auto : AutoARIMA and friends.
    auto_forecast : wraquant's built-in automatic forecasting.
    """
    from darts import TimeSeries

    values = np.asarray(data, dtype=np.float64).flatten()

    if isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
        ts = TimeSeries.from_series(data.astype(float))
    else:
        ts = TimeSeries.from_values(values)

    # Default training parameters
    input_chunk = kwargs.pop("input_chunk_length", min(max(horizon * 2, 24), len(values) // 3))
    output_chunk = kwargs.pop("output_chunk_length", horizon)
    n_epochs = kwargs.pop("n_epochs", 50)

    model_map = {}

    def _get_model(name: str):
        if name == "nbeats":
            from darts.models import NBEATSModel
            return NBEATSModel(
                input_chunk_length=input_chunk,
                output_chunk_length=output_chunk,
                n_epochs=n_epochs,
                random_state=42,
                **kwargs,
            )
        elif name == "nhits":
            from darts.models import NHiTSModel
            return NHiTSModel(
                input_chunk_length=input_chunk,
                output_chunk_length=output_chunk,
                n_epochs=n_epochs,
                random_state=42,
                **kwargs,
            )
        elif name == "tcn":
            from darts.models import TCNModel
            return TCNModel(
                input_chunk_length=input_chunk,
                output_chunk_length=output_chunk,
                n_epochs=n_epochs,
                random_state=42,
                **kwargs,
            )
        elif name == "rnn":
            from darts.models import RNNModel
            return RNNModel(
                model="LSTM",
                input_chunk_length=input_chunk,
                output_chunk_length=output_chunk,
                n_epochs=n_epochs,
                random_state=42,
                **kwargs,
            )
        elif name == "transformer":
            from darts.models import TransformerModel
            return TransformerModel(
                input_chunk_length=input_chunk,
                output_chunk_length=output_chunk,
                n_epochs=n_epochs,
                random_state=42,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown model: {name!r}. Choose from "
                "'nbeats', 'nhits', 'tcn', 'rnn', 'transformer'."
            )

    darts_model = _get_model(model)
    darts_model.fit(ts)

    prediction = darts_model.predict(n=horizon)
    forecast_values = prediction.values().flatten()

    # Extract training loss if available
    training_loss = np.nan
    if hasattr(darts_model, "trainer") and darts_model.trainer is not None:
        try:
            training_loss = float(
                darts_model.trainer.callback_metrics.get("train_loss", np.nan)
            )
        except Exception:
            pass

    return {
        "forecast": forecast_values,
        "model_name": model,
        "training_loss": training_loss,
    }
