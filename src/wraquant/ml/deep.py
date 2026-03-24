"""Deep learning models for quantitative finance.

Provides PyTorch-based neural network architectures tailored for financial
time-series forecasting and feature extraction. All torch imports are guarded
so the rest of the package works without PyTorch installed.

Models included:
- LSTM forecasting
- Transformer-based forecasting
- GRU forecasting
- Variational Autoencoder for feature extraction
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wraquant.core._coerce import coerce_array, coerce_dataframe

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

__all__ = [
    "lstm_forecast",
    "transformer_forecast",
    "autoencoder_features",
    "gru_forecast",
    "multivariate_lstm_forecast",
    "temporal_fusion_transformer",
]


def _check_torch() -> None:
    """Raise a helpful error if PyTorch is not installed."""
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for deep learning models but is not installed. "
            "Install it with: pip install torch  (or see https://pytorch.org "
            "for platform-specific instructions). wraquant does not bundle "
            "torch in any PDM extra group because installation varies by "
            "platform and CUDA version."
        )


# ---------------------------------------------------------------------------
# Sequence creation helper
# ---------------------------------------------------------------------------


def _create_sequences(
    data: np.ndarray,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create input/target sequence pairs from a 1-D time series.

    For a series [x_0, x_1, ..., x_N] with seq_length=k, produces:
        X[i] = [x_i, x_{i+1}, ..., x_{i+k-1}]
        y[i] = x_{i+k}

    Parameters
    ----------
    data : np.ndarray
        1-D array of time-series values.
    seq_length : int
        Number of look-back steps.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``X`` of shape ``(n_samples, seq_length, 1)`` and ``y`` of shape
        ``(n_samples,)``.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length])
    X = np.array(xs, dtype=np.float32).reshape(-1, seq_length, 1)
    y = np.array(ys, dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------


def lstm_forecast(
    series: pd.Series | np.ndarray,
    seq_length: int = 20,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.1,
    n_epochs: int = 50,
    lr: float = 1e-3,
    train_ratio: float = 0.8,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Forecast a financial time series using an LSTM network.

    Long Short-Term Memory networks are recurrent neural networks capable of
    learning long-range dependencies in sequential data. In finance, LSTMs
    are used to capture complex temporal patterns in price, volume, and
    return series that linear models miss.

    The function auto-creates overlapping input/target sequences from the raw
    time series, splits into train/test sets chronologically (no shuffle to
    avoid lookahead bias), trains the model, and returns predictions on the
    test set.

    When to use:
        Use LSTM for multi-step forecasting when you have >1000 observations
        and suspect non-linear temporal dependencies. Works well for return
        prediction, volatility forecasting, and spread modeling.

    Mathematical background:
        At each time step t, the LSTM cell computes:
            f_t = sigma(W_f [h_{t-1}, x_t] + b_f)    (forget gate)
            i_t = sigma(W_i [h_{t-1}, x_t] + b_i)    (input gate)
            o_t = sigma(W_o [h_{t-1}, x_t] + b_o)    (output gate)
            c_t = f_t * c_{t-1} + i_t * tanh(W_c [h_{t-1}, x_t] + b_c)
            h_t = o_t * tanh(c_t)

        The cell state c_t acts as a conveyor belt, allowing gradients to
        flow across many time steps without vanishing.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Univariate time series (e.g., log returns, prices, spreads).
    seq_length : int
        Number of look-back time steps for each input sequence.
    hidden_dim : int
        Number of hidden units in each LSTM layer.
    n_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability between LSTM layers (applied only when
        ``n_layers > 1``).
    n_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the Adam optimizer.
    train_ratio : float
        Fraction of data used for training (the rest is used for testing).
        The split is chronological -- no shuffling.
    batch_size : int
        Mini-batch size for training.

    Returns
    -------
    dict
        ``predictions``: np.ndarray of test-set predictions,
        ``actuals``: np.ndarray of actual test values,
        ``train_losses``: list of per-epoch training losses,
        ``model``: the trained ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    Example
    -------
    >>> import numpy as np
    >>> returns = np.cumsum(np.random.randn(500) * 0.01)
    >>> result = lstm_forecast(returns, seq_length=10, n_epochs=20)
    >>> result["predictions"].shape
    (80,)

    Caveats
    -------
    - Financial time series are notoriously noisy; LSTM is prone to
      overfitting on noise. Use dropout, early stopping, and validation.
    - Chronological train/test split is critical to avoid lookahead bias.
    - Normalisation (handled internally) is essential for gradient stability.

    References
    ----------
    - Hochreiter & Schmidhuber (1997), "Long Short-Term Memory"
    - Fischer & Krauss (2018), "Deep learning with long short-term memory
      networks for financial market predictions"
    """
    _check_torch()

    data = coerce_array(series, name="series")

    # Normalise
    mu, sigma = data.mean(), data.std()
    if sigma == 0:
        sigma = 1.0
    data_norm = ((data - mu) / sigma).astype(np.float32)

    X, y = _create_sequences(data_norm, seq_length)
    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)

    # Build model
    class _LSTMModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = _LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train
    model.train()
    train_losses: list[float] = []
    n_train = len(X_train_t)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

    # Predict
    model.eval()
    with torch.no_grad():
        preds_norm = model(X_test_t).numpy()

    # Denormalise
    preds = preds_norm * sigma + mu
    actuals = y_test * sigma + mu

    return {
        "predictions": preds,
        "actuals": actuals,
        "train_losses": train_losses,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


def transformer_forecast(
    series: pd.Series | np.ndarray,
    seq_length: int = 20,
    d_model: int = 64,
    n_heads: int = 4,
    n_encoder_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    n_epochs: int = 50,
    lr: float = 1e-3,
    train_ratio: float = 0.8,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Forecast a financial time series using a Transformer encoder.

    Transformer models use self-attention to capture dependencies at any
    distance in the input sequence, unlike RNNs which process sequentially.
    This makes them especially effective at discovering long-range patterns
    such as seasonality, lead-lag relationships, and regime persistence in
    financial data.

    When to use:
        Use Transformers when you have sufficient data (>2000 observations)
        and suspect that long-range dependencies matter. They often
        outperform LSTMs on longer sequences but require more data and
        compute.

    Mathematical background:
        Self-attention computes:
            Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

        where Q, K, V are linear projections of the input. Multi-head
        attention runs h parallel attention heads and concatenates:
            MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O

        Positional encoding injects order information:
            PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
            PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Univariate time series.
    seq_length : int
        Number of look-back time steps.
    d_model : int
        Embedding dimension (must be divisible by ``n_heads``).
    n_heads : int
        Number of attention heads.
    n_encoder_layers : int
        Number of Transformer encoder layers.
    dim_feedforward : int
        Hidden dimension in the feedforward sub-layers.
    dropout : float
        Dropout probability.
    n_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam.
    train_ratio : float
        Fraction of data for training.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    dict
        ``predictions``: np.ndarray of test-set predictions,
        ``actuals``: np.ndarray of actual test values,
        ``train_losses``: list of per-epoch training losses,
        ``model``: the trained ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    Example
    -------
    >>> import numpy as np
    >>> prices = np.cumsum(np.random.randn(600) * 0.01) + 100
    >>> result = transformer_forecast(prices, seq_length=15, n_epochs=10)
    >>> len(result["predictions"]) > 0
    True

    Caveats
    -------
    - Transformers are data-hungry; on small datasets (<500 obs) they will
      overfit severely.
    - Quadratic memory in sequence length: keep seq_length reasonable
      (< 256 for typical financial data).
    - No inherent notion of order without positional encoding.

    References
    ----------
    - Vaswani et al. (2017), "Attention Is All You Need"
    - Li et al. (2019), "Enhancing the Locality and Breaking the Memory
      Bottleneck of Transformer on Time Series Forecasting"
    """
    _check_torch()

    data = coerce_array(series, name="series")
    mu, sigma = data.mean(), data.std()
    if sigma == 0:
        sigma = 1.0
    data_norm = ((data - mu) / sigma).astype(np.float32)

    X, y = _create_sequences(data_norm, seq_length)
    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)

    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float()
                * (-np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1), :]

    class _TransformerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.pos_enc = _PositionalEncoding(d_model, max_len=seq_length + 10)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_encoder_layers
            )
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_length, 1)
            x = self.input_proj(x)  # (batch, seq_length, d_model)
            x = self.pos_enc(x)
            x = self.encoder(x)
            # Take the last time step
            return self.fc(x[:, -1, :]).squeeze(-1)

    model = _TransformerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    train_losses: list[float] = []
    n_train = len(X_train_t)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

    model.eval()
    with torch.no_grad():
        preds_norm = model(X_test_t).numpy()

    preds = preds_norm * sigma + mu
    actuals = y_test * sigma + mu

    return {
        "predictions": preds,
        "actuals": actuals,
        "train_losses": train_losses,
        "model": model,
    }


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------


def gru_forecast(
    series: pd.Series | np.ndarray,
    seq_length: int = 20,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.1,
    n_epochs: int = 50,
    lr: float = 1e-3,
    train_ratio: float = 0.8,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Forecast a financial time series using a GRU network.

    Gated Recurrent Units are a simplified variant of LSTMs that merge the
    cell and hidden state, resulting in fewer parameters and faster training
    while achieving comparable performance on many financial forecasting
    tasks.

    When to use:
        Use GRU as a computationally cheaper alternative to LSTM. Preferred
        when you have moderate-sized datasets (500-5000 observations) or
        need faster iteration during model development.

    Mathematical background:
        The GRU update equations at time step t:
            z_t = sigma(W_z [h_{t-1}, x_t])          (update gate)
            r_t = sigma(W_r [h_{t-1}, x_t])          (reset gate)
            h_t_hat = tanh(W [r_t * h_{t-1}, x_t])   (candidate)
            h_t = (1 - z_t) * h_{t-1} + z_t * h_t_hat

        Compared to LSTM, GRU has no separate cell state and uses two gates
        instead of three, giving ~25% fewer parameters.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Univariate time series.
    seq_length : int
        Number of look-back time steps.
    hidden_dim : int
        Number of hidden units per GRU layer.
    n_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout between layers (only when ``n_layers > 1``).
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.
    train_ratio : float
        Fraction of data for training.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    dict
        ``predictions``: np.ndarray of test-set predictions,
        ``actuals``: np.ndarray of actual test values,
        ``train_losses``: list of per-epoch training losses,
        ``model``: the trained ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    Example
    -------
    >>> import numpy as np
    >>> vol = np.abs(np.random.randn(400)) * 0.02
    >>> result = gru_forecast(vol, seq_length=10, n_epochs=15)
    >>> result["predictions"].shape[0] > 0
    True

    Caveats
    -------
    - Same overfitting risks as LSTM; use dropout and validation.
    - On very long sequences (>200 steps), Transformers may outperform GRU.

    References
    ----------
    - Cho et al. (2014), "Learning Phrase Representations using RNN
      Encoder-Decoder for Statistical Machine Translation"
    """
    _check_torch()

    data = coerce_array(series, name="series")
    mu, sigma_val = data.mean(), data.std()
    if sigma_val == 0:
        sigma_val = 1.0
    data_norm = ((data - mu) / sigma_val).astype(np.float32)

    X, y = _create_sequences(data_norm, seq_length)
    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)

    class _GRUModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=1,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = _GRUModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    train_losses: list[float] = []
    n_train = len(X_train_t)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

    model.eval()
    with torch.no_grad():
        preds_norm = model(X_test_t).numpy()

    preds = preds_norm * sigma_val + mu
    actuals = y_test * sigma_val + mu

    return {
        "predictions": preds,
        "actuals": actuals,
        "train_losses": train_losses,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Variational Autoencoder
# ---------------------------------------------------------------------------


def autoencoder_features(
    X: pd.DataFrame | np.ndarray,
    latent_dim: int = 8,
    hidden_dim: int = 64,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    beta: float = 1.0,
) -> dict[str, Any]:
    """Extract latent features using a Variational Autoencoder (VAE).

    A VAE learns a compressed, continuous latent representation of
    high-dimensional input features. In finance, this is valuable for:

    - **Regime detection**: Cluster the latent codes to find market states.
    - **Anomaly detection**: High reconstruction error flags unusual market
      conditions (flash crashes, liquidity crises).
    - **Feature compression**: Reduce hundreds of technical indicators to a
      handful of orthogonal latent factors.

    When to use:
        Use when you have a wide feature matrix (>20 features) and want to
        discover latent structure, detect anomalies, or reduce
        dimensionality in a non-linear way that PCA cannot capture.

    Mathematical background:
        The VAE optimises the Evidence Lower Bound (ELBO):
            L = E_q[log p(x|z)] - beta * KL(q(z|x) || p(z))

        where q(z|x) = N(mu(x), sigma^2(x)) is the encoder, p(x|z) is the
        decoder, and p(z) = N(0, I) is the prior. The KL term regularises
        the latent space to be smooth and continuous.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dim : int
        Hidden layer size in encoder/decoder.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    beta : float
        Weight on the KL divergence term. ``beta=1`` is standard VAE;
        ``beta<1`` gives more reconstruction accuracy; ``beta>1`` forces
        more disentangled representations.

    Returns
    -------
    dict
        ``latent_features``: np.ndarray of shape ``(n_samples, latent_dim)``
        -- the encoded representations,
        ``reconstruction_error``: np.ndarray of per-sample reconstruction
        MSE,
        ``train_losses``: list of per-epoch total losses,
        ``model``: the trained VAE module.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    Example
    -------
    >>> import numpy as np
    >>> X = np.random.randn(500, 30)  # 30 features
    >>> result = autoencoder_features(X, latent_dim=5, n_epochs=20)
    >>> result["latent_features"].shape
    (500, 5)

    Caveats
    -------
    - Normalise your features before encoding; the VAE assumes roughly
      standard-normal inputs for stable training.
    - The latent space is stochastic; for deterministic embeddings, use
      the mean (mu) which is what this function returns.
    - Reconstruction error thresholds for anomaly detection should be
      calibrated on clean training data.

    References
    ----------
    - Kingma & Welling (2014), "Auto-Encoding Variational Bayes"
    - An & Cho (2015), "Variational Autoencoder based Anomaly Detection
      using Reconstruction Probability"
    """
    _check_torch()

    X_df = coerce_dataframe(X, name="X") if hasattr(X, "columns") or isinstance(X, dict) else None
    X_arr = X_df.values.astype(np.float32) if X_df is not None else np.asarray(X, dtype=np.float32)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n_samples, n_features = X_arr.shape

    # Normalise per feature
    mu_X = X_arr.mean(axis=0)
    std_X = X_arr.std(axis=0)
    std_X[std_X == 0] = 1.0
    X_norm = (X_arr - mu_X) / std_X

    X_t = torch.from_numpy(X_norm)

    class _VAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Encoder
            self.enc1 = nn.Linear(n_features, hidden_dim)
            self.enc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
            # Decoder
            self.dec1 = nn.Linear(latent_dim, hidden_dim)
            self.dec2 = nn.Linear(hidden_dim, hidden_dim)
            self.dec_out = nn.Linear(hidden_dim, n_features)

        def encode(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            h = torch.relu(self.enc1(x))
            h = torch.relu(self.enc2(h))
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterise(
            self, mu: torch.Tensor, logvar: torch.Tensor
        ) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.dec1(z))
            h = torch.relu(self.dec2(h))
            return self.dec_out(h)

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(x)
            z = self.reparameterise(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar

    model = _VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def vae_loss(
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        recon = nn.functional.mse_loss(x_recon, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl

    model.train()
    train_losses: list[float] = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_t[idx]

            optimizer.zero_grad()
            x_recon, mu_enc, logvar_enc = model(xb)
            loss = vae_loss(x_recon, xb, mu_enc, logvar_enc)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

    # Extract latent features (use the mean, not a random sample)
    model.eval()
    with torch.no_grad():
        mu_enc, _ = model.encode(X_t)
        latent = mu_enc.numpy()

        # Reconstruction error per sample
        x_recon, _, _ = model(X_t)
        recon_err = (
            (x_recon - X_t).pow(2).mean(dim=1).numpy()
        )

    return {
        "latent_features": latent,
        "reconstruction_error": recon_err,
        "train_losses": train_losses,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Multivariate LSTM
# ---------------------------------------------------------------------------


def _create_multivariate_sequences(
    features: np.ndarray,
    target: np.ndarray,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create input/target sequence pairs from multivariate features.

    Parameters
    ----------
    features : np.ndarray
        2-D array of shape ``(T, n_features)``.
    target : np.ndarray
        1-D array of shape ``(T,)`` -- the variable to predict.
    seq_length : int
        Number of look-back steps.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``X`` of shape ``(n_samples, seq_length, n_features)`` and ``y`` of
        shape ``(n_samples,)``.
    """
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i : i + seq_length])
        ys.append(target[i + seq_length])
    X = np.array(xs, dtype=np.float32)
    y = np.array(ys, dtype=np.float32)
    return X, y


def multivariate_lstm_forecast(
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    seq_length: int = 20,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.1,
    n_epochs: int = 50,
    lr: float = 1e-3,
    train_ratio: float = 0.8,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Forecast a target series using multiple input features via LSTM.

    Multivariate LSTM ingests a DataFrame of features (e.g., returns of
    correlated assets, macro indicators, technical signals) and learns to
    predict a single target variable. This outperforms univariate LSTM when
    cross-asset signals exist -- for example, when sector ETF returns lead
    individual stock returns, when VIX changes anticipate equity moves, or
    when order-flow imbalance across related instruments carries predictive
    information for the target.

    The function normalises each feature column independently (z-score),
    creates multivariate look-back sequences, trains the LSTM with a
    chronological train/test split, and returns predictions on the held-out
    test set along with train and test MSE metrics.

    Mathematical background:
        The LSTM cell equations are the same as in ``lstm_forecast``, but
        the input dimensionality is now n_features rather than 1:
            x_t in R^{n_features}
            f_t = sigma(W_f [h_{t-1}, x_t] + b_f)
            i_t = sigma(W_i [h_{t-1}, x_t] + b_i)
            o_t = sigma(W_o [h_{t-1}, x_t] + b_o)

        The weight matrices W_f, W_i, W_o, W_c have input dimension
        n_features instead of 1, allowing the network to learn cross-feature
        temporal dependencies.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame of shape ``(T, n_features)`` containing the input
        features. All columns are used as inputs to the LSTM.
    target : pd.Series or np.ndarray
        Target variable of length T to predict.
    seq_length : int
        Number of look-back time steps for each input sequence.
    hidden_dim : int
        Number of hidden units in each LSTM layer.
    n_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability between LSTM layers (applied only when
        ``n_layers > 1``).
    n_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the Adam optimizer.
    train_ratio : float
        Fraction of data used for training (chronological split).
    batch_size : int
        Mini-batch size for training.

    Returns
    -------
    dict
        ``predictions``: np.ndarray of test-set predictions,
        ``actuals``: np.ndarray of actual test values,
        ``train_losses``: list of per-epoch training losses,
        ``train_mse``: float MSE on the training set,
        ``test_mse``: float MSE on the test set,
        ``model``: the trained ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    Example
    -------
    >>> import numpy as np, pandas as pd
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'asset_a': np.cumsum(np.random.randn(500) * 0.01),
    ...     'asset_b': np.cumsum(np.random.randn(500) * 0.01),
    ...     'vix': np.abs(np.random.randn(500)) * 15 + 15,
    ... })
    >>> target = pd.Series(np.cumsum(np.random.randn(500) * 0.01))
    >>> result = multivariate_lstm_forecast(df, target, seq_length=10, n_epochs=5)
    >>> result["predictions"].shape[0] > 0
    True

    References
    ----------
    - Hochreiter & Schmidhuber (1997), "Long Short-Term Memory"
    - Fischer & Krauss (2018), "Deep learning with long short-term memory
      networks for financial market predictions"
    """
    _check_torch()

    feat_df = coerce_dataframe(features, name="features")
    feat_arr = feat_df.values.astype(np.float64)
    tgt_arr = coerce_array(target, name="target")

    if feat_arr.ndim == 1:
        feat_arr = feat_arr.reshape(-1, 1)

    n_samples, n_features = feat_arr.shape

    # Normalise features per column
    feat_mu = feat_arr.mean(axis=0)
    feat_std = feat_arr.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    feat_norm = ((feat_arr - feat_mu) / feat_std).astype(np.float32)

    # Normalise target
    tgt_mu, tgt_std = float(tgt_arr.mean()), float(tgt_arr.std())
    if tgt_std == 0:
        tgt_std = 1.0
    tgt_norm = ((tgt_arr - tgt_mu) / tgt_std).astype(np.float32)

    X, y = _create_multivariate_sequences(feat_norm, tgt_norm, seq_length)
    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)

    class _MultivarLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = _MultivarLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train
    model.train()
    train_losses: list[float] = []
    n_train = len(X_train_t)

    for _epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

    # Predict
    model.eval()
    with torch.no_grad():
        train_preds_norm = model(X_train_t).numpy()
        test_preds_norm = model(X_test_t).numpy()

    # Denormalise
    preds = test_preds_norm * tgt_std + tgt_mu
    actuals = y_test * tgt_std + tgt_mu
    train_preds = train_preds_norm * tgt_std + tgt_mu
    train_actuals = y_train * tgt_std + tgt_mu

    train_mse = float(np.mean((train_preds - train_actuals) ** 2))
    test_mse = float(np.mean((preds - actuals) ** 2))

    return {
        "predictions": preds,
        "actuals": actuals,
        "train_losses": train_losses,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Temporal Fusion Transformer (simplified)
# ---------------------------------------------------------------------------


def temporal_fusion_transformer(
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    seq_length: int = 20,
    hidden_dim: int = 64,
    n_heads: int = 4,
    n_lstm_layers: int = 1,
    dropout: float = 0.1,
    n_epochs: int = 50,
    lr: float = 1e-3,
    train_ratio: float = 0.8,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Simplified Temporal Fusion Transformer for interpretable forecasting.

    The most promising architecture for interpretable financial forecasting.
    This implementation provides the core TFT components: a variable
    selection network that learns which input features matter, an LSTM
    encoder for temporal processing, multi-head attention for capturing
    long-range dependencies, and gated residual connections for stable
    gradient flow.

    Unlike black-box models, TFT produces per-feature importance weights
    that reveal *which* inputs drive each prediction -- critical for
    building trust in trading signals and satisfying model governance
    requirements.

    Architecture:
        1. **Variable Selection Network (VSN)**: A soft-attention gate over
           input features. Each feature is projected to ``hidden_dim``,
           then a shared softmax gate selects the most relevant ones.
        2. **LSTM Encoder**: Processes the selected features sequentially
           to capture local temporal patterns.
        3. **Multi-Head Attention**: Attends over the LSTM outputs to
           capture long-range dependencies (e.g., monthly seasonality).
        4. **Gated Residual Network (GRN)**: skip connections with gating
           for stable training on noisy financial data.
        5. **Output layer**: Linear projection to produce the forecast.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame of shape ``(T, n_features)`` containing the input
        features.
    target : pd.Series or np.ndarray
        Target variable of length T.
    seq_length : int
        Number of look-back time steps.
    hidden_dim : int
        Dimensionality of the hidden representations.
    n_heads : int
        Number of attention heads (must divide ``hidden_dim``).
    n_lstm_layers : int
        Number of LSTM layers in the encoder.
    dropout : float
        Dropout probability.
    n_epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam.
    train_ratio : float
        Fraction of data for training (chronological split).
    batch_size : int
        Mini-batch size.

    Returns
    -------
    dict
        ``predictions``: np.ndarray of test-set predictions,
        ``actuals``: np.ndarray of actual test values,
        ``train_losses``: list of per-epoch training losses,
        ``feature_importance``: np.ndarray of shape ``(n_features,)``
        giving the learned importance weight for each input feature
        (higher = more important),
        ``feature_names``: list of feature names from the input DataFrame,
        ``model``: the trained ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    Example
    -------
    >>> import numpy as np, pandas as pd
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'momentum': np.random.randn(500),
    ...     'volume': np.abs(np.random.randn(500)),
    ...     'spread': np.random.randn(500) * 0.1,
    ... })
    >>> target = pd.Series(np.cumsum(np.random.randn(500) * 0.01))
    >>> result = temporal_fusion_transformer(
    ...     df, target, seq_length=10, hidden_dim=16, n_heads=2, n_epochs=5
    ... )
    >>> result["predictions"].shape[0] > 0
    True
    >>> len(result["feature_importance"]) == 3
    True

    References
    ----------
    - Lim et al. (2021), "Temporal Fusion Transformers for Interpretable
      Multi-horizon Time Series Forecasting"
    """
    _check_torch()

    feat_df = coerce_dataframe(features, name="features")
    feature_names = list(feat_df.columns)
    feat_arr = feat_df.values.astype(np.float64)
    tgt_arr = coerce_array(target, name="target")

    if feat_arr.ndim == 1:
        feat_arr = feat_arr.reshape(-1, 1)

    _n_samples, n_features = feat_arr.shape

    # Normalise
    feat_mu = feat_arr.mean(axis=0)
    feat_std = feat_arr.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    feat_norm = ((feat_arr - feat_mu) / feat_std).astype(np.float32)

    tgt_mu, tgt_std = float(tgt_arr.mean()), float(tgt_arr.std())
    if tgt_std == 0:
        tgt_std = 1.0
    tgt_norm = ((tgt_arr - tgt_mu) / tgt_std).astype(np.float32)

    X, y = _create_multivariate_sequences(feat_norm, tgt_norm, seq_length)
    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)

    class _GatedResidualNetwork(nn.Module):
        """Gated Residual Network for stable gradient flow."""

        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, output_dim)
            self.fc2 = nn.Linear(output_dim, output_dim)
            self.gate = nn.Linear(output_dim, output_dim)
            self.layer_norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)
            # Skip connection projection if dims differ
            self.skip = (
                nn.Linear(input_dim, output_dim)
                if input_dim != output_dim
                else nn.Identity()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.skip(x)
            h = torch.nn.functional.elu(self.fc1(x))
            h = self.dropout(self.fc2(h))
            gate = torch.sigmoid(self.gate(h))
            return self.layer_norm(residual + gate * h)

    class _VariableSelectionNetwork(nn.Module):
        """Learns soft attention weights over input features."""

        def __init__(self) -> None:
            super().__init__()
            # Per-feature transformations
            self.feature_transforms = nn.ModuleList(
                [nn.Linear(1, hidden_dim) for _ in range(n_features)]
            )
            # Shared gate
            self.gate = nn.Sequential(
                nn.Linear(n_features * hidden_dim, n_features),
                nn.Softmax(dim=-1),
            )
            self.grn = _GatedResidualNetwork(hidden_dim, hidden_dim)

        def forward(
            self, x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # x: (batch, seq_len, n_features)
            batch_sz, seq_len, _ = x.shape

            # Transform each feature: (batch, seq_len, hidden_dim) each
            transformed = []
            for i in range(n_features):
                transformed.append(
                    torch.relu(
                        self.feature_transforms[i](x[:, :, i : i + 1])
                    )
                )

            # Stack: (batch, seq_len, n_features, hidden_dim)
            stacked = torch.stack(transformed, dim=2)

            # Compute gate weights: flatten features for gate input
            gate_input = stacked.reshape(
                batch_sz, seq_len, n_features * hidden_dim
            )
            weights = self.gate(gate_input)  # (batch, seq_len, n_features)

            # Weighted sum of transformed features
            # weights: (batch, seq_len, n_features, 1)
            weighted = (
                stacked * weights.unsqueeze(-1)
            ).sum(dim=2)  # (batch, seq_len, hidden_dim)

            out = self.grn(weighted)
            # Average weights across batch and time for interpretation
            avg_weights = weights.mean(dim=(0, 1))  # (n_features,)
            return out, avg_weights

    class _TFTModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vsn = _VariableSelectionNetwork()
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_lstm_layers,
                dropout=dropout if n_lstm_layers > 1 else 0.0,
                batch_first=True,
            )
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.grn_post_attn = _GatedResidualNetwork(
                hidden_dim, hidden_dim
            )
            self.fc_out = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self, x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Variable selection
            selected, feat_weights = self.vsn(x)

            # LSTM encoder
            lstm_out, _ = self.lstm(selected)

            # Multi-head self-attention over LSTM outputs
            attn_out, _ = self.attention(
                lstm_out, lstm_out, lstm_out
            )

            # Gated residual connection
            combined = self.grn_post_attn(
                self.dropout(attn_out) + lstm_out
            )

            # Use last time step for prediction
            out = self.fc_out(combined[:, -1, :]).squeeze(-1)
            return out, feat_weights

    model = _TFTModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train
    model.train()
    train_losses: list[float] = []
    n_train = len(X_train_t)

    for _epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

    # Predict and get feature importance
    model.eval()
    with torch.no_grad():
        test_preds_norm, feat_importance = model(X_test_t)
        test_preds_norm = test_preds_norm.numpy()
        feat_importance = feat_importance.numpy()

    preds = test_preds_norm * tgt_std + tgt_mu
    actuals = y_test * tgt_std + tgt_mu

    return {
        "predictions": preds,
        "actuals": actuals,
        "train_losses": train_losses,
        "feature_importance": feat_importance,
        "feature_names": feature_names,
        "model": model,
    }
