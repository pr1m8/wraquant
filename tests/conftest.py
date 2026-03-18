"""Shared test fixtures for wraquant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.Series:
    """Generate sample price series for testing."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    returns = rng.normal(0.0005, 0.02, size=252)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="price")


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, size=252)))
    high = close * (1 + rng.uniform(0, 0.03, size=252))
    low = close * (1 - rng.uniform(0, 0.03, size=252))
    open_ = close * (1 + rng.normal(0, 0.01, size=252))
    volume = rng.integers(1_000_000, 10_000_000, size=252)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def sample_returns(sample_prices: pd.Series) -> pd.Series:
    """Generate sample returns series."""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def sample_multi_asset() -> pd.DataFrame:
    """Generate multi-asset return series."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=252)
    assets = ["SPY", "AGG", "GLD", "EURUSD", "GBPUSD"]
    data = rng.normal(0.0003, 0.015, size=(252, len(assets)))
    return pd.DataFrame(data, index=dates, columns=assets)
