"""Tests for advanced time series integrations."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_has_tsfresh = importlib.util.find_spec("tsfresh") is not None
_has_stumpy = importlib.util.find_spec("stumpy") is not None
_has_pywt = importlib.util.find_spec("pywt") is not None
_has_sktime = importlib.util.find_spec("sktime") is not None
_has_statsforecast = importlib.util.find_spec("statsforecast") is not None
_has_tslearn = importlib.util.find_spec("tslearn") is not None


def _make_series(n: int = 200, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(
        np.cumsum(rng.normal(0, 1, n)),
        index=dates,
        name="value",
    )


class TestTsfreshFeatures:
    @pytest.mark.skipif(not _has_tsfresh, reason="tsfresh not installed")
    def test_returns_dataframe(self) -> None:
        from wraquant.ts.advanced import tsfresh_features

        df = pd.DataFrame({
            "id": [1] * 50 + [2] * 50,
            "time": list(range(50)) * 2,
            "value": np.random.default_rng(42).normal(0, 1, 100),
        })
        result = tsfresh_features(df, column_id="id", column_sort="time")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result.isna().sum().sum() == 0

    @pytest.mark.skipif(not _has_tsfresh, reason="tsfresh not installed")
    def test_no_infinite_values(self) -> None:
        from wraquant.ts.advanced import tsfresh_features

        df = pd.DataFrame({
            "id": [1] * 30,
            "time": list(range(30)),
            "value": np.random.default_rng(99).normal(0, 1, 30),
        })
        result = tsfresh_features(df, column_id="id", column_sort="time")
        assert not np.isinf(result.values).any()


class TestStumpyMatrixProfile:
    @pytest.mark.skipif(not _has_stumpy, reason="stumpy not installed")
    def test_returns_dict_keys(self) -> None:
        from wraquant.ts.advanced import stumpy_matrix_profile

        ts = np.random.default_rng(42).normal(0, 1, 200)
        result = stumpy_matrix_profile(ts, m=20)
        assert "matrix_profile" in result
        assert "profile_index" in result
        assert "motif_idx" in result
        assert "discord_idx" in result

    @pytest.mark.skipif(not _has_stumpy, reason="stumpy not installed")
    def test_matrix_profile_length(self) -> None:
        from wraquant.ts.advanced import stumpy_matrix_profile

        ts = np.random.default_rng(42).normal(0, 1, 200)
        m = 20
        result = stumpy_matrix_profile(ts, m=m)
        assert len(result["matrix_profile"]) == len(ts) - m + 1


class TestWaveletTransform:
    @pytest.mark.skipif(not _has_pywt, reason="pywavelets not installed")
    def test_returns_coefficients(self) -> None:
        from wraquant.ts.advanced import wavelet_transform

        data = np.random.default_rng(42).normal(0, 1, 128)
        result = wavelet_transform(data, wavelet="db4")
        assert "coeffs" in result
        assert "wavelet" in result
        assert "level" in result
        assert isinstance(result["coeffs"], list)
        assert len(result["coeffs"]) > 1

    @pytest.mark.skipif(not _has_pywt, reason="pywavelets not installed")
    def test_explicit_level(self) -> None:
        from wraquant.ts.advanced import wavelet_transform

        data = np.random.default_rng(42).normal(0, 1, 128)
        result = wavelet_transform(data, wavelet="haar", level=3)
        assert result["level"] == 3
        assert len(result["coeffs"]) == 4  # cA3, cD3, cD2, cD1


class TestWaveletDenoise:
    @pytest.mark.skipif(not _has_pywt, reason="pywavelets not installed")
    def test_output_length(self) -> None:
        from wraquant.ts.advanced import wavelet_denoise

        data = np.random.default_rng(42).normal(0, 1, 128)
        result = wavelet_denoise(data, wavelet="db4")
        assert len(result) == len(data)

    @pytest.mark.skipif(not _has_pywt, reason="pywavelets not installed")
    def test_denoised_is_smoother(self) -> None:
        from wraquant.ts.advanced import wavelet_denoise

        rng = np.random.default_rng(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 256))
        noisy = signal + rng.normal(0, 0.5, 256)
        denoised = wavelet_denoise(noisy, wavelet="db4")
        # Denoised signal should be closer to true signal
        assert np.std(denoised - signal) < np.std(noisy - signal)


class TestSktimeForecast:
    @pytest.mark.skipif(not _has_sktime, reason="sktime not installed")
    def test_naive_forecast_shape(self) -> None:
        from wraquant.ts.advanced import sktime_forecast

        y = pd.Series(np.random.default_rng(42).normal(100, 10, 50))
        result = sktime_forecast(y, model="naive", horizon=5)
        assert isinstance(result, pd.DataFrame)
        assert "forecast" in result.columns
        assert len(result) == 5

    @pytest.mark.skipif(not _has_sktime, reason="sktime not installed")
    def test_unknown_model_raises(self) -> None:
        from wraquant.ts.advanced import sktime_forecast

        y = pd.Series(np.random.default_rng(42).normal(100, 10, 50))
        with pytest.raises(ValueError, match="Unknown model"):
            sktime_forecast(y, model="nonexistent", horizon=5)


class TestStatsforecastAuto:
    @pytest.mark.skipif(not _has_statsforecast, reason="statsforecast not installed")
    def test_returns_dataframe(self) -> None:
        from wraquant.ts.advanced import statsforecast_auto

        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        y = pd.Series(np.random.default_rng(42).normal(100, 5, 100), index=dates)
        result = statsforecast_auto(y, season_length=7, horizon=5)
        assert isinstance(result, pd.DataFrame)
        assert "forecast" in result.columns
        assert len(result) == 5


class TestTslearnDtw:
    @pytest.mark.skipif(not _has_tslearn, reason="tslearn not installed")
    def test_dtw_distance_nonnegative(self) -> None:
        from wraquant.ts.advanced import tslearn_dtw

        rng = np.random.default_rng(42)
        ts1 = rng.normal(0, 1, 50)
        ts2 = rng.normal(0, 1, 50)
        result = tslearn_dtw(ts1, ts2)
        assert result["distance"] >= 0
        assert isinstance(result["path"], list)

    @pytest.mark.skipif(not _has_tslearn, reason="tslearn not installed")
    def test_dtw_identical_is_zero(self) -> None:
        from wraquant.ts.advanced import tslearn_dtw

        ts = np.ones(30)
        result = tslearn_dtw(ts, ts)
        assert result["distance"] == pytest.approx(0.0, abs=1e-10)


class TestTslearnKmeans:
    @pytest.mark.skipif(not _has_tslearn, reason="tslearn not installed")
    def test_clustering_labels(self) -> None:
        from wraquant.ts.advanced import tslearn_kmeans

        rng = np.random.default_rng(42)
        dataset = [rng.normal(i, 0.5, 30) for i in range(9)]
        result = tslearn_kmeans(dataset, n_clusters=3)
        assert len(result["labels"]) == 9
        assert result["n_clusters"] == 3
        assert result["inertia"] >= 0
