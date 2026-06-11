"""Change-point detection for time series."""

from __future__ import annotations

import numpy as np
import pandas as pd

def cusum(data: pd.Series, threshold: float = 1.0) -> pd.Series:
    """Cumulative sum (CUSUM) control chart for change detection.

    Returns a CUSUM statistic series. Values exceeding *threshold*
    standard deviations indicate a potential shift.

    Parameters:
        data: Time series.
        threshold: Detection threshold in standard deviation units.

    Returns:
        CUSUM statistic series.
    """
    clean = data.dropna()
    mean = clean.mean()
    std = clean.std()
    if std == 0:
        return pd.Series(0.0, index=clean.index)

    normalised = (clean - mean) / std
    cusum_pos = pd.Series(0.0, index=clean.index)
    cusum_neg = pd.Series(0.0, index=clean.index)

    s_pos = 0.0
    s_neg = 0.0
    for i, val in enumerate(normalised.values):
        s_pos = max(0.0, s_pos + val - threshold / 2)
        s_neg = max(0.0, s_neg - val - threshold / 2)
        cusum_pos.iloc[i] = s_pos
        cusum_neg.iloc[i] = s_neg

    return cusum_pos + cusum_neg


def detect_changepoints(
    data: pd.Series,
    method: str = "pelt",
    penalty: float | None = None,
) -> list[int]:
    """Detect change-points using the ``ruptures`` library.

    Parameters:
        data: Time series.
        method: Algorithm — ``"pelt"`` (default), ``"binseg"``, or
            ``"window"``.
        penalty: Penalty value for the algorithm. When *None*, a
            sensible default is chosen.

    Returns:
        List of change-point indices (positions in the array, excluding
        the final point).

    Raises:
        ValueError: If *method* is not recognized.
    """
    try:
        import ruptures as rpt
    except ModuleNotFoundError:
        values = data.dropna()
        if len(values) < 4:
            return []
        rolling = values.rolling(max(4, min(20, len(values) // 5)), min_periods=2)
        z = ((values - rolling.mean()) / rolling.std()).abs().replace([np.inf], np.nan)
        threshold = float(z.quantile(0.99)) if z.notna().any() else np.inf
        points = np.flatnonzero((z > threshold).to_numpy())
        return [int(point) for point in points if point < len(values) - 1]

    signal = data.dropna().values.reshape(-1, 1)

    if penalty is None:
        penalty = np.log(len(signal)) * data.std() ** 2

    if method == "pelt":
        algo = rpt.Pelt(model="rbf").fit(signal)
        result = algo.predict(pen=penalty)
    elif method == "binseg":
        algo = rpt.Binseg(model="rbf").fit(signal)
        result = algo.predict(pen=penalty)
    elif method == "window":
        algo = rpt.Window(model="rbf").fit(signal)
        result = algo.predict(pen=penalty)
    else:
        msg = f"Unknown changepoint method: {method!r}"
        raise ValueError(msg)

    # ruptures includes the final index; remove it
    return [int(x) for x in result if x < len(signal)]
