"""Tests for extended Bayesian integration wrappers (Bambi, emcee, BlackJAX)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

_has_bambi = importlib.util.find_spec("bambi") is not None
_has_emcee = importlib.util.find_spec("emcee") is not None
_has_blackjax = importlib.util.find_spec("blackjax") is not None
_has_jax = importlib.util.find_spec("jax") is not None


# ---------------------------------------------------------------------------
# Bambi regression
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_bambi, reason="bambi not installed")
class TestBambiRegression:
    def test_basic_regression(self) -> None:
        from wraquant.bayes.integrations import bambi_regression

        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(0, 0.5, n)
        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        result = bambi_regression("y ~ x1 + x2", data, draws=200, chains=2, seed=42)

        assert "model" in result
        assert "trace" in result
        assert "summary" in result
        assert isinstance(result["summary"], pd.DataFrame)

    def test_poisson_family(self) -> None:
        from wraquant.bayes.integrations import bambi_regression

        rng = np.random.default_rng(42)
        n = 100
        x = rng.normal(0, 0.5, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * x))
        data = pd.DataFrame({"y": y, "x": x})

        result = bambi_regression(
            "y ~ x", data, family="poisson", draws=200, chains=2, seed=42
        )
        assert "model" in result
        assert "summary" in result


# ---------------------------------------------------------------------------
# emcee sampling
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_emcee, reason="emcee not installed")
class TestEmceeSample:
    @staticmethod
    def _log_prob(theta: np.ndarray) -> float:
        """Simple 2D Gaussian log probability."""
        return -0.5 * np.sum(theta**2)

    def test_basic_sampling(self) -> None:
        from wraquant.bayes.integrations import emcee_sample

        n_dim = 2
        n_walkers = 10
        result = emcee_sample(
            self._log_prob,
            n_walkers=n_walkers,
            n_dim=n_dim,
            n_steps=100,
            seed=42,
        )

        assert "samples" in result
        assert "log_prob" in result
        assert "acceptance_fraction" in result
        assert result["samples"].shape == (n_walkers * 100, n_dim)

    def test_custom_initial(self) -> None:
        from wraquant.bayes.integrations import emcee_sample

        n_dim = 2
        n_walkers = 8
        initial = np.random.default_rng(0).normal(0, 0.1, (n_walkers, n_dim))
        result = emcee_sample(
            self._log_prob,
            n_walkers=n_walkers,
            n_dim=n_dim,
            n_steps=50,
            initial=initial,
        )
        assert result["samples"].shape == (n_walkers * 50, n_dim)

    def test_acceptance_fraction_range(self) -> None:
        from wraquant.bayes.integrations import emcee_sample

        result = emcee_sample(
            self._log_prob,
            n_walkers=10,
            n_dim=2,
            n_steps=100,
            seed=42,
        )
        assert 0.0 <= result["acceptance_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# BlackJAX NUTS
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_has_blackjax and _has_jax), reason="blackjax/jax not installed"
)
class TestBlackjaxNuts:
    def test_basic_sampling(self) -> None:
        import jax.numpy as jnp

        from wraquant.bayes.integrations import blackjax_nuts

        def log_prob(x):
            return -0.5 * jnp.sum(x**2)

        initial = jnp.zeros(2)
        result = blackjax_nuts(
            log_prob,
            initial_position=initial,
            n_samples=50,
            step_size=0.1,
            seed=0,
        )

        assert "samples" in result
        assert "divergences" in result
        assert result["samples"].shape == (50, 2)
        assert isinstance(result["divergences"], int)
