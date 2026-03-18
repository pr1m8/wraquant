"""Tests for options pricing models."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.price.options import binomial_tree, black_scholes, monte_carlo_option


class TestBlackScholes:
    """Tests for the Black-Scholes pricing formula."""

    def test_call_put_parity(self) -> None:
        """Call - Put = S - K*exp(-rT) (put-call parity)."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        call = black_scholes(S, K, T, r, sigma, "call")
        put = black_scholes(S, K, T, r, sigma, "put")
        parity = S - K * np.exp(-r * T)
        assert call - put == pytest.approx(parity, abs=1e-10)

    def test_call_put_parity_otm(self) -> None:
        """Put-call parity holds for out-of-the-money options."""
        S, K, T, r, sigma = 100.0, 120.0, 0.5, 0.03, 0.25
        call = black_scholes(S, K, T, r, sigma, "call")
        put = black_scholes(S, K, T, r, sigma, "put")
        parity = S - K * np.exp(-r * T)
        assert call - put == pytest.approx(parity, abs=1e-10)

    def test_call_positive(self) -> None:
        """Call price is non-negative."""
        price = black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
        assert price >= 0.0

    def test_put_positive(self) -> None:
        """Put price is non-negative."""
        price = black_scholes(100, 100, 1.0, 0.05, 0.2, "put")
        assert price >= 0.0

    def test_deep_itm_call(self) -> None:
        """Deep in-the-money call is close to intrinsic value."""
        S, K, T, r, sigma = 200.0, 100.0, 0.01, 0.05, 0.2
        price = black_scholes(S, K, T, r, sigma, "call")
        intrinsic = S - K * np.exp(-r * T)
        assert price == pytest.approx(intrinsic, rel=0.01)

    def test_deep_otm_call(self) -> None:
        """Deep out-of-the-money call is near zero."""
        price = black_scholes(50, 200, 0.01, 0.05, 0.2, "call")
        assert price < 0.01

    def test_deep_itm_put(self) -> None:
        """Deep in-the-money put is close to intrinsic value."""
        S, K, T, r, sigma = 50.0, 200.0, 0.01, 0.05, 0.2
        price = black_scholes(S, K, T, r, sigma, "put")
        intrinsic = K * np.exp(-r * T) - S
        assert price == pytest.approx(intrinsic, rel=0.01)

    def test_deep_otm_put(self) -> None:
        """Deep out-of-the-money put is near zero."""
        price = black_scholes(200, 50, 0.01, 0.05, 0.2, "put")
        assert price < 0.01

    def test_atm_call_known_value(self) -> None:
        """ATM call matches known BS value."""
        # S=100, K=100, T=1, r=0.05, sigma=0.2 -> ~10.4506
        price = black_scholes(100, 100, 1.0, 0.05, 0.2, "call")
        assert price == pytest.approx(10.4506, abs=0.01)

    def test_expired_option(self) -> None:
        """At T=0, option returns intrinsic value."""
        call_itm = black_scholes(110, 100, 0.0, 0.05, 0.2, "call")
        assert call_itm == pytest.approx(10.0, abs=1e-10)
        call_otm = black_scholes(90, 100, 0.0, 0.05, 0.2, "call")
        assert call_otm == pytest.approx(0.0, abs=1e-10)

    def test_accepts_enum(self) -> None:
        """Accepts OptionType enum."""
        from wraquant.core.types import OptionType

        price = black_scholes(100, 100, 1.0, 0.05, 0.2, OptionType.CALL)
        assert price > 0.0


class TestBinomialTree:
    """Tests for the binomial tree pricer."""

    def test_converges_to_bs_call(self) -> None:
        """Binomial tree converges to BS for European calls with many steps."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        bs_price = black_scholes(S, K, T, r, sigma, "call")
        tree_price = binomial_tree(S, K, T, r, sigma, 500, "call", "european")
        assert tree_price == pytest.approx(bs_price, abs=0.05)

    def test_converges_to_bs_put(self) -> None:
        """Binomial tree converges to BS for European puts with many steps."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        bs_price = black_scholes(S, K, T, r, sigma, "put")
        tree_price = binomial_tree(S, K, T, r, sigma, 500, "put", "european")
        assert tree_price == pytest.approx(bs_price, abs=0.05)

    def test_american_put_gte_european(self) -> None:
        """American put is worth at least as much as European put."""
        S, K, T, r, sigma, n = 100.0, 110.0, 1.0, 0.05, 0.3, 200
        eu = binomial_tree(S, K, T, r, sigma, n, "put", "european")
        am = binomial_tree(S, K, T, r, sigma, n, "put", "american")
        assert am >= eu - 1e-10

    def test_american_call_equals_european(self) -> None:
        """American call on non-dividend stock equals European call."""
        S, K, T, r, sigma, n = 100.0, 100.0, 1.0, 0.05, 0.2, 200
        eu = binomial_tree(S, K, T, r, sigma, n, "call", "european")
        am = binomial_tree(S, K, T, r, sigma, n, "call", "american")
        assert am == pytest.approx(eu, abs=0.01)

    def test_positive_price(self) -> None:
        """Binomial tree always gives non-negative prices."""
        price = binomial_tree(100, 100, 1.0, 0.05, 0.2, 100, "call", "european")
        assert price >= 0.0


class TestMonteCarlo:
    """Tests for Monte Carlo option pricing."""

    def test_within_tolerance_of_bs_call(self) -> None:
        """Monte Carlo call price is within tolerance of BS."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        bs_price = black_scholes(S, K, T, r, sigma, "call")
        mc_price = monte_carlo_option(
            S, K, T, r, sigma, n_sims=200_000, n_steps=252, seed=42
        )
        assert mc_price == pytest.approx(bs_price, abs=0.3)

    def test_within_tolerance_of_bs_put(self) -> None:
        """Monte Carlo put price is within tolerance of BS."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        bs_price = black_scholes(S, K, T, r, sigma, "put")
        mc_price = monte_carlo_option(
            S, K, T, r, sigma, n_sims=200_000, n_steps=252, option_type="put", seed=42
        )
        assert mc_price == pytest.approx(bs_price, abs=0.3)

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces same price."""
        args = (100, 100, 1.0, 0.05, 0.2, 10000, 100)
        p1 = monte_carlo_option(*args, seed=123)
        p2 = monte_carlo_option(*args, seed=123)
        assert p1 == p2

    def test_positive_price(self) -> None:
        """Monte Carlo price is non-negative."""
        price = monte_carlo_option(100, 100, 1.0, 0.05, 0.2, 10000, 100, seed=42)
        assert price >= 0.0
