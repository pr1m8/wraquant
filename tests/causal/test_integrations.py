"""Tests for causal inference external package integrations."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_HAS_DOWHY = importlib.util.find_spec("dowhy") is not None
_HAS_ECONML = importlib.util.find_spec("econml") is not None
_HAS_DOUBLEML = importlib.util.find_spec("doubleml") is not None


# ---------------------------------------------------------------------------
# DoWhy
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_DOWHY, reason="dowhy not installed")
class TestDoWhyCausalModel:
    def test_basic_estimation(self) -> None:
        from wraquant.causal.integrations import dowhy_causal_model

        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        t = (rng.uniform(size=n) < (1 / (1 + np.exp(-x)))).astype(int)
        y = 1.0 + 2.0 * t + x + rng.normal(0, 0.5, n)

        df = pd.DataFrame({"X": x, "T": t, "Y": y})
        result = dowhy_causal_model(
            data=df,
            treatment="T",
            outcome="Y",
            common_causes=["X"],
            method="backdoor.linear_regression",
        )

        assert "estimate" in result
        assert "model" in result
        assert abs(result["estimate"] - 2.0) < 1.0


# ---------------------------------------------------------------------------
# EconML — Double ML
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_ECONML, reason="econml not installed")
class TestEconMLDML:
    def test_basic_dml(self) -> None:
        from wraquant.causal.integrations import econml_dml

        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(0, 1, (n, 3))
        T = X[:, 0] + rng.normal(0, 0.5, n)
        Y = 1.0 + 2.0 * T + X[:, 1] + rng.normal(0, 0.5, n)

        result = econml_dml(Y, T, X)
        assert "ate" in result
        assert "se" in result
        assert abs(result["ate"] - 2.0) < 1.5


# ---------------------------------------------------------------------------
# EconML — Causal Forest
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_ECONML, reason="econml not installed")
class TestEconMLForest:
    def test_basic_forest(self) -> None:
        from wraquant.causal.integrations import econml_forest

        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(0, 1, (n, 2))
        T = (rng.uniform(size=n) > 0.5).astype(float)
        Y = 1.0 + 2.0 * T + X[:, 0] + rng.normal(0, 0.5, n)

        result = econml_forest(Y, T, X, n_estimators=50)
        assert "ate" in result
        assert "cate" in result
        assert result["cate"].shape == (n,)


# ---------------------------------------------------------------------------
# DoubleML — PLR
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_DOUBLEML, reason="doubleml not installed")
class TestDoubleMLPLR:
    def test_basic_plr(self) -> None:
        from wraquant.causal.integrations import doubleml_plr

        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(0, 1, (n, 3))
        D = X[:, 0] + rng.normal(0, 0.5, n)
        Y = 1.0 + 2.0 * D + X[:, 1] + rng.normal(0, 0.5, n)

        result = doubleml_plr(Y, D, X)
        assert "ate" in result
        assert "se" in result
        assert "p_value" in result
        assert abs(result["ate"] - 2.0) < 1.5
