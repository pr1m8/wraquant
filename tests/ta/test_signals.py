"""Tests for wraquant.ta.signals module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.ta.signals import (
    above,
    below,
    crossover,
    crossunder,
    falling,
    highest,
    lowest,
    normalize,
    rising,
)

# ---------------------------------------------------------------------------
# Crossover / Crossunder
# ---------------------------------------------------------------------------


class TestCrossover:
    def test_crossover_with_known_data(self) -> None:
        """Detect crossing above."""
        s1 = pd.Series([1, 2, 3, 4, 3, 2, 1, 2, 3.0])
        s2 = pd.Series([3, 3, 3, 3, 3, 3, 3, 3, 3.0])
        result = crossover(s1, s2)
        # s1 crosses above s2 (value 3) at index 3 (prev was 3<=3, now 4>3)
        assert result.iloc[3] is np.bool_(True)

    def test_crossover_with_constant(self) -> None:
        """Detect crossing above a constant level."""
        s1 = pd.Series([1, 2, 3, 4, 5.0])
        result = crossover(s1, 2.5)
        # At index 3: s1=4 > 2.5 and prev s1=3 was also >2.5 → False
        # At index 2: s1=3 > 2.5 and prev s1=2 <= 2.5 → True
        assert result.iloc[2] is np.bool_(True)

    def test_no_false_positives(self) -> None:
        """No crossover when s1 is always above s2."""
        s1 = pd.Series([10, 11, 12, 13.0])
        s2 = pd.Series([1, 2, 3, 4.0])
        result = crossover(s1, s2)
        # First bar: s1[0]>s2[0] but prev is NaN → no crossover
        assert not result.iloc[1:].any()


class TestCrossunder:
    def test_crossunder_with_known_data(self) -> None:
        """Detect crossing below."""
        s1 = pd.Series([5, 4, 3, 2, 3, 4, 5.0])
        s2 = pd.Series([3, 3, 3, 3, 3, 3, 3.0])
        result = crossunder(s1, s2)
        # s1 crosses below s2 at index 3 (prev was 3>=3, now 2<3)
        assert result.iloc[3] is np.bool_(True)


# ---------------------------------------------------------------------------
# Above / Below
# ---------------------------------------------------------------------------


class TestAboveBelow:
    def test_above_basic(self) -> None:
        s1 = pd.Series([1, 5, 3, 7.0])
        s2 = pd.Series([2, 4, 3, 6.0])
        result = above(s1, s2)
        assert list(result) == [False, True, False, True]

    def test_below_basic(self) -> None:
        s1 = pd.Series([1, 5, 3, 7.0])
        s2 = pd.Series([2, 4, 3, 6.0])
        result = below(s1, s2)
        assert list(result) == [True, False, False, False]


# ---------------------------------------------------------------------------
# Rising / Falling
# ---------------------------------------------------------------------------


class TestRisingFalling:
    def test_rising(self) -> None:
        data = pd.Series([1, 2, 3, 2, 1.0])
        result = rising(data, period=1)
        # Index 1,2 are rising; 3,4 are not
        assert result.iloc[1] is np.bool_(True)
        assert result.iloc[2] is np.bool_(True)
        assert result.iloc[3] is np.bool_(False)

    def test_falling(self) -> None:
        data = pd.Series([5, 4, 3, 4, 5.0])
        result = falling(data, period=1)
        assert result.iloc[1] is np.bool_(True)
        assert result.iloc[2] is np.bool_(True)
        assert result.iloc[3] is np.bool_(False)


# ---------------------------------------------------------------------------
# Highest / Lowest
# ---------------------------------------------------------------------------


class TestHighestLowest:
    def test_highest_with_known_data(self) -> None:
        data = pd.Series([1, 3, 2, 5, 4, 6.0])
        result = highest(data, period=3)
        # Index 2: max(1,3,2) = 3
        assert result.iloc[2] == 3.0
        # Index 3: max(3,2,5) = 5
        assert result.iloc[3] == 5.0
        # Index 5: max(4,4,6)  — wait, let's check: max(5,4,6) = 6
        assert result.iloc[5] == 6.0

    def test_lowest_with_known_data(self) -> None:
        data = pd.Series([5, 3, 4, 1, 2, 6.0])
        result = lowest(data, period=3)
        # Index 2: min(5,3,4) = 3
        assert result.iloc[2] == 3.0
        # Index 3: min(3,4,1) = 1
        assert result.iloc[3] == 1.0

    def test_nan_prefix(self) -> None:
        data = pd.Series([1, 2, 3, 4, 5.0])
        result = highest(data, period=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_full_series_zscore(self) -> None:
        """Full-series normalization should have mean~0 and std~1."""
        np.random.seed(42)
        data = pd.Series(np.random.randn(1000))
        result = normalize(data)
        assert abs(result.mean()) < 0.01
        assert abs(result.std() - 1.0) < 0.05

    def test_rolling_normalization(self) -> None:
        np.random.seed(42)
        data = pd.Series(np.random.randn(100) + 50)
        result = normalize(data, period=20)
        assert len(result) == len(data)
        # First period-1 values should be NaN (or 0 due to fillna)
        # After warm-up, values should be finite
        assert result.iloc[20:].notna().all()
