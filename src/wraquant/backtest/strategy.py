"""Strategy abstract base class and common strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Abstract base class for trading strategies.

    Subclasses must implement ``generate_signals`` which maps prices
    to position signals (-1, 0, +1 or fractional).

    Example:
        >>> class MACrossover(Strategy):
        ...     def __init__(self, fast: int = 10, slow: int = 50):
        ...         self.fast = fast
        ...         self.slow = slow
        ...
        ...     def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        ...         fast_ma = prices.rolling(self.fast).mean()
        ...         slow_ma = prices.rolling(self.slow).mean()
        ...         return (fast_ma > slow_ma).astype(float)
    """

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate position signals from price data.

        Parameters:
            prices: DataFrame of asset prices (columns = assets).

        Returns:
            DataFrame of signals with same shape as prices.
            Values represent desired position: 1 = long, -1 = short, 0 = flat.
        """


class BuyAndHold(Strategy):
    """Buy and hold strategy — always fully invested."""

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.ones_like(prices.values),
            index=prices.index,
            columns=prices.columns,
        )


class MomentumStrategy(Strategy):
    """Simple momentum strategy based on lookback returns.

    Parameters:
        lookback: Number of periods for momentum calculation.
        top_n: Number of top assets to go long.
    """

    def __init__(self, lookback: int = 20, top_n: int | None = None) -> None:
        self.lookback = lookback
        self.top_n = top_n

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        mom = prices.pct_change(self.lookback)
        if self.top_n is not None and self.top_n < prices.shape[1]:
            signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            for i in range(len(prices)):
                if i < self.lookback:
                    continue
                row = mom.iloc[i]
                top = row.nlargest(self.top_n).index
                signals.iloc[i][top] = 1.0 / self.top_n
            return signals
        return (mom > 0).astype(float)


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy using z-score.

    Parameters:
        window: Lookback window for mean/std.
        entry_z: Z-score threshold to enter position.
        exit_z: Z-score threshold to exit position.
    """

    def __init__(
        self, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.5
    ) -> None:
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        rolling_mean = prices.rolling(self.window).mean()
        rolling_std = prices.rolling(self.window).std()
        z_scores = (prices - rolling_mean) / rolling_std

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[z_scores < -self.entry_z] = 1.0  # Buy when oversold
        signals[z_scores > self.entry_z] = -1.0  # Sell when overbought
        return signals
