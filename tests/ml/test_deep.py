"""Tests for wraquant.ml.deep — PyTorch deep learning models."""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch")

from wraquant.ml.deep import (
    autoencoder_features,
    gru_forecast,
    lstm_forecast,
    transformer_forecast,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def time_series() -> np.ndarray:
    """Synthetic trending time series with 300 points."""
    np.random.seed(42)
    trend = np.linspace(0, 5, 300)
    noise = np.random.randn(300) * 0.3
    return trend + noise


@pytest.fixture()
def feature_matrix() -> np.ndarray:
    """Synthetic feature matrix (200 samples, 15 features)."""
    np.random.seed(99)
    return np.random.randn(200, 15).astype(np.float32)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------


class TestLSTMForecast:
    def test_output_keys(self, time_series: np.ndarray) -> None:
        result = lstm_forecast(
            time_series, seq_length=10, hidden_dim=16, n_layers=1, n_epochs=3
        )
        assert "predictions" in result
        assert "actuals" in result
        assert "train_losses" in result
        assert "model" in result

    def test_prediction_shape(self, time_series: np.ndarray) -> None:
        result = lstm_forecast(
            time_series,
            seq_length=10,
            hidden_dim=16,
            n_layers=1,
            n_epochs=3,
            train_ratio=0.8,
        )
        # With 300 data points, seq_length=10 => 290 sequences
        # 20% test => ~58 test samples
        assert result["predictions"].shape == result["actuals"].shape
        assert len(result["predictions"]) > 0

    def test_train_losses_decrease_or_exist(self, time_series: np.ndarray) -> None:
        result = lstm_forecast(
            time_series, seq_length=10, hidden_dim=32, n_layers=2, n_epochs=10
        )
        losses = result["train_losses"]
        assert len(losses) == 10
        # At minimum, losses should be finite
        assert all(np.isfinite(l) for l in losses)

    def test_model_is_torch_module(self, time_series: np.ndarray) -> None:
        result = lstm_forecast(
            time_series, seq_length=10, hidden_dim=16, n_layers=1, n_epochs=2
        )
        assert isinstance(result["model"], torch.nn.Module)

    def test_accepts_pandas_series(self) -> None:
        import pandas as pd

        np.random.seed(7)
        s = pd.Series(np.cumsum(np.random.randn(200) * 0.01))
        result = lstm_forecast(s, seq_length=5, hidden_dim=8, n_layers=1, n_epochs=2)
        assert len(result["predictions"]) > 0


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------


class TestGRUForecast:
    def test_output_shape(self, time_series: np.ndarray) -> None:
        result = gru_forecast(
            time_series, seq_length=10, hidden_dim=16, n_layers=1, n_epochs=3
        )
        assert result["predictions"].shape == result["actuals"].shape
        assert len(result["predictions"]) > 0

    def test_model_type(self, time_series: np.ndarray) -> None:
        result = gru_forecast(
            time_series, seq_length=10, hidden_dim=16, n_layers=1, n_epochs=2
        )
        assert isinstance(result["model"], torch.nn.Module)

    def test_losses_finite(self, time_series: np.ndarray) -> None:
        result = gru_forecast(
            time_series, seq_length=10, hidden_dim=16, n_layers=1, n_epochs=5
        )
        assert all(np.isfinite(l) for l in result["train_losses"])


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class TestTransformerForecast:
    def test_output_keys(self, time_series: np.ndarray) -> None:
        result = transformer_forecast(
            time_series,
            seq_length=10,
            d_model=16,
            n_heads=2,
            n_encoder_layers=1,
            n_epochs=3,
        )
        assert "predictions" in result
        assert "actuals" in result
        assert "train_losses" in result
        assert "model" in result

    def test_prediction_shape(self, time_series: np.ndarray) -> None:
        result = transformer_forecast(
            time_series,
            seq_length=10,
            d_model=16,
            n_heads=2,
            n_encoder_layers=1,
            n_epochs=3,
        )
        assert result["predictions"].shape == result["actuals"].shape
        assert len(result["predictions"]) > 0

    def test_model_is_module(self, time_series: np.ndarray) -> None:
        result = transformer_forecast(
            time_series,
            seq_length=10,
            d_model=8,
            n_heads=2,
            n_encoder_layers=1,
            n_epochs=2,
        )
        assert isinstance(result["model"], torch.nn.Module)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------


class TestAutoencoderFeatures:
    def test_output_shape(self, feature_matrix: np.ndarray) -> None:
        result = autoencoder_features(
            feature_matrix, latent_dim=4, hidden_dim=32, n_epochs=5
        )
        assert result["latent_features"].shape == (200, 4)
        assert result["reconstruction_error"].shape == (200,)

    def test_reconstruction_error_non_negative(
        self, feature_matrix: np.ndarray
    ) -> None:
        result = autoencoder_features(
            feature_matrix, latent_dim=4, hidden_dim=32, n_epochs=5
        )
        assert np.all(result["reconstruction_error"] >= 0)

    def test_train_losses_exist(self, feature_matrix: np.ndarray) -> None:
        result = autoencoder_features(
            feature_matrix, latent_dim=4, hidden_dim=32, n_epochs=10
        )
        assert len(result["train_losses"]) == 10
        assert all(np.isfinite(l) for l in result["train_losses"])

    def test_accepts_dataframe(self) -> None:
        import pandas as pd

        np.random.seed(55)
        df = pd.DataFrame(np.random.randn(100, 10))
        result = autoencoder_features(df, latent_dim=3, hidden_dim=16, n_epochs=3)
        assert result["latent_features"].shape == (100, 3)
