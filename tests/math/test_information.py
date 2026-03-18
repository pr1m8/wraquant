"""Tests for wraquant.math.information."""

from __future__ import annotations

import numpy as np

from wraquant.math.information import (
    conditional_entropy,
    entropy,
    fisher_information,
    kl_divergence,
    mutual_information,
    transfer_entropy,
)


class TestEntropy:
    """Tests for entropy."""

    def test_uniform_is_maximal(self) -> None:
        """Uniform distribution should have higher entropy than peaked."""
        rng = np.random.default_rng(42)
        uniform = rng.uniform(0, 1, 10_000)
        peaked = rng.normal(0.5, 0.01, 10_000)

        h_uniform = entropy(uniform, bins=50)
        h_peaked = entropy(peaked, bins=50)

        assert h_uniform > h_peaked

    def test_entropy_non_negative(self) -> None:
        data = np.random.default_rng(0).standard_normal(500)
        assert entropy(data) >= 0.0


class TestMutualInformation:
    """Tests for mutual_information."""

    def test_identical_series_high_mi(self) -> None:
        """Mutual information of x with itself should be high."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5000)
        mi = mutual_information(x, x, bins=30)
        assert mi > 1.0  # should be equal to H(X)

    def test_independent_series_near_zero(self) -> None:
        """MI of truly independent series should be near zero."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5000)
        y = rng.standard_normal(5000)
        mi = mutual_information(x, y, bins=20)
        assert mi < 0.1

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        assert mutual_information(x, y) >= 0.0


class TestKLDivergence:
    """Tests for kl_divergence."""

    def test_non_negative(self) -> None:
        """KL divergence should always be >= 0."""
        rng = np.random.default_rng(42)
        p = rng.normal(0, 1, 5000)
        q = rng.normal(1, 1, 5000)
        kl = kl_divergence(p, q, bins=30)
        assert kl >= 0.0

    def test_same_distribution_near_zero(self) -> None:
        """KL of identical samples should be near zero."""
        rng = np.random.default_rng(42)
        p = rng.standard_normal(10_000)
        kl = kl_divergence(p, p, bins=30)
        assert kl < 0.05

    def test_different_distributions_positive(self) -> None:
        rng = np.random.default_rng(42)
        p = rng.normal(0, 1, 5000)
        q = rng.normal(3, 1, 5000)
        kl = kl_divergence(p, q, bins=30)
        assert kl > 0.1


class TestConditionalEntropy:
    """Tests for conditional_entropy."""

    def test_less_than_or_equal_to_entropy(self) -> None:
        """H(X|Y) <= H(X) always."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5000)
        y = x + rng.normal(0, 0.1, 5000)  # correlated
        h_x = entropy(x, bins=20)
        h_x_given_y = conditional_entropy(x, y, bins=20)
        assert h_x_given_y <= h_x + 0.1  # small tolerance for discretisation

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        assert conditional_entropy(x, y) >= 0.0


class TestFisherInformation:
    """Tests for fisher_information."""

    def test_quadratic_log_likelihood(self) -> None:
        """For a quadratic log-lik, FIM should be a constant matrix."""
        # log L(theta) = -0.5 * (theta - mu)^T A (theta - mu)
        # FIM = A
        A = np.array([[2.0, 0.5], [0.5, 3.0]])

        def log_lik(params: np.ndarray) -> float:
            d = params - np.array([1.0, 2.0])
            return float(-0.5 * d @ A @ d)

        fim = fisher_information(log_lik, np.array([1.0, 2.0]))
        np.testing.assert_allclose(fim, A, atol=1e-3)


class TestTransferEntropy:
    """Tests for transfer_entropy."""

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        te = transfer_entropy(x, y, lag=1, bins=10)
        assert te >= 0.0

    def test_causal_direction(self) -> None:
        """TE from cause to effect should exceed reverse direction."""
        rng = np.random.default_rng(42)
        n = 5000
        x = rng.standard_normal(n)
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = 0.8 * x[i - 1] + 0.2 * rng.standard_normal()

        te_x_to_y = transfer_entropy(x, y, lag=1, bins=10)
        te_y_to_x = transfer_entropy(y, x, lag=1, bins=10)
        assert te_x_to_y > te_y_to_x
