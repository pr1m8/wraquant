"""Tests for Bayesian analysis external package integrations."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

try:
    import pymc  # noqa: F401
    _HAS_PYMC = True
except Exception:
    _HAS_PYMC = False

try:
    import arviz
    _HAS_ARVIZ = hasattr(arviz, "summary")
except Exception:
    _HAS_ARVIZ = False
_HAS_NUMPYRO = importlib.util.find_spec("numpyro") is not None


# ---------------------------------------------------------------------------
# PyMC regression
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PYMC, reason="pymc not installed")
class TestPyMCRegression:
    def test_basic_regression(self) -> None:
        from wraquant.bayes.integrations import pymc_regression

        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, (n, 2))
        true_beta = np.array([1.0, 0.5, -0.3])  # intercept + 2 features
        X_full = np.column_stack([np.ones(n), X])
        y = X_full @ true_beta + rng.normal(0, 0.5, n)

        result = pymc_regression(y, X, samples=500, chains=2)

        assert "trace" in result
        assert "coefficients_mean" in result
        assert "sigma_mean" in result
        assert result["coefficients_mean"].shape == (3,)


# ---------------------------------------------------------------------------
# ArviZ summary
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_ARVIZ, reason="arviz not installed")
class TestArviZSummary:
    def test_summary_from_dict(self) -> None:
        from wraquant.bayes.integrations import arviz_summary

        rng = np.random.default_rng(42)
        trace = {"beta": rng.normal(0, 1, (2, 500, 3))}
        summary = arviz_summary(trace)
        assert isinstance(summary, pd.DataFrame)
        assert "mean" in summary.columns


# ---------------------------------------------------------------------------
# NumPyro regression
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_NUMPYRO, reason="numpyro not installed")
class TestNumPyroRegression:
    def test_basic_regression(self) -> None:
        from wraquant.bayes.integrations import numpyro_regression

        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(0, 1, (n, 2))
        true_beta = np.array([1.0, 0.5, -0.3])
        X_full = np.column_stack([np.ones(n), X])
        y = X_full @ true_beta + rng.normal(0, 0.5, n)

        result = numpyro_regression(y, X, samples=500, warmup=200)

        assert "samples" in result
        assert "coefficients_mean" in result
        assert result["coefficients_mean"].shape == (3,)
