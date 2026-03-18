"""Tests for causal treatment effect estimation (pure numpy/scipy)."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.causal.treatment import (
    diff_in_diff,
    doubly_robust_ate,
    ipw_ate,
    matching_ate,
    propensity_score,
    regression_discontinuity,
    synthetic_control,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_treatment_data(
    n: int = 500,
    effect: float = 2.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic treatment/outcome data with known ATE."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2))
    # Treatment probability depends on x
    logit = 0.5 * x[:, 0] + 0.3 * x[:, 1]
    prob = 1.0 / (1.0 + np.exp(-logit))
    treatment = (rng.uniform(size=n) < prob).astype(float)
    # Outcome depends on x and treatment
    outcome = 1.0 + x[:, 0] + 0.5 * x[:, 1] + effect * treatment + rng.normal(0, 0.5, n)
    return outcome, treatment, x


# ---------------------------------------------------------------------------
# Propensity score tests
# ---------------------------------------------------------------------------


class TestPropensityScore:
    def test_returns_array_of_correct_length(self) -> None:
        _, treatment, covariates = _make_treatment_data()
        ps = propensity_score(treatment, covariates)
        assert ps.shape == treatment.shape

    def test_scores_between_0_and_1(self) -> None:
        _, treatment, covariates = _make_treatment_data()
        ps = propensity_score(treatment, covariates)
        assert np.all(ps >= 0.01)
        assert np.all(ps <= 0.99)

    def test_scores_correlate_with_treatment(self) -> None:
        """Higher propensity scores should be more common among treated."""
        _, treatment, covariates = _make_treatment_data()
        ps = propensity_score(treatment, covariates)
        mean_treated = np.mean(ps[treatment == 1])
        mean_control = np.mean(ps[treatment == 0])
        assert mean_treated > mean_control

    def test_1d_covariates(self) -> None:
        """Should work with a single covariate."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        t = (x > 0).astype(float)
        ps = propensity_score(t, x)
        assert ps.shape == (100,)


# ---------------------------------------------------------------------------
# IPW ATE tests
# ---------------------------------------------------------------------------


class TestIPWATE:
    def test_recovers_treatment_effect(self) -> None:
        outcome, treatment, covariates = _make_treatment_data(n=1000, effect=2.0)
        ps = propensity_score(treatment, covariates)
        result = ipw_ate(outcome, treatment, ps)
        assert abs(result.ate - 2.0) < 1.0

    def test_output_structure(self) -> None:
        outcome, treatment, covariates = _make_treatment_data()
        ps = propensity_score(treatment, covariates)
        result = ipw_ate(outcome, treatment, ps)
        assert hasattr(result, "ate")
        assert hasattr(result, "se")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "n_treated")
        assert hasattr(result, "n_control")
        assert result.n_treated + result.n_control == len(treatment)

    def test_ci_contains_ate(self) -> None:
        outcome, treatment, covariates = _make_treatment_data(n=1000, effect=2.0)
        ps = propensity_score(treatment, covariates)
        result = ipw_ate(outcome, treatment, ps)
        assert result.ci_lower < result.ci_upper
        # CI should be reasonable
        assert result.ci_lower < result.ate < result.ci_upper


# ---------------------------------------------------------------------------
# Matching ATE tests
# ---------------------------------------------------------------------------


class TestMatchingATE:
    def test_recovers_treatment_effect(self) -> None:
        outcome, treatment, covariates = _make_treatment_data(n=1000, effect=2.0)
        result = matching_ate(outcome, treatment, covariates, n_neighbors=3)
        assert abs(result.ate - 2.0) < 1.0

    def test_output_structure(self) -> None:
        outcome, treatment, covariates = _make_treatment_data()
        result = matching_ate(outcome, treatment, covariates)
        assert hasattr(result, "ate")
        assert hasattr(result, "se")
        assert result.se > 0
        assert result.details["estimator"] == "matching"

    def test_more_neighbors_smoother(self) -> None:
        """More neighbors should generally give a smoother estimate."""
        outcome, treatment, covariates = _make_treatment_data(n=500, seed=99)
        r1 = matching_ate(outcome, treatment, covariates, n_neighbors=1)
        r5 = matching_ate(outcome, treatment, covariates, n_neighbors=5)
        # Both should recover approximately the right effect
        assert abs(r1.ate - 2.0) < 2.0
        assert abs(r5.ate - 2.0) < 2.0


# ---------------------------------------------------------------------------
# Doubly robust ATE tests
# ---------------------------------------------------------------------------


class TestDoublyRobustATE:
    def test_recovers_treatment_effect(self) -> None:
        outcome, treatment, covariates = _make_treatment_data(n=1000, effect=2.0)
        result = doubly_robust_ate(outcome, treatment, covariates)
        assert abs(result.ate - 2.0) < 1.0

    def test_output_structure(self) -> None:
        outcome, treatment, covariates = _make_treatment_data()
        result = doubly_robust_ate(outcome, treatment, covariates)
        assert hasattr(result, "ate")
        assert hasattr(result, "se")
        assert result.se > 0
        assert result.details["estimator"] == "doubly_robust"

    def test_ci_contains_true_effect(self) -> None:
        outcome, treatment, covariates = _make_treatment_data(n=2000, effect=3.0)
        result = doubly_robust_ate(outcome, treatment, covariates)
        assert result.ci_lower < 3.0 < result.ci_upper


# ---------------------------------------------------------------------------
# Regression discontinuity tests
# ---------------------------------------------------------------------------


class TestRegressionDiscontinuity:
    def _make_rd_data(
        self,
        n: int = 1000,
        effect: float = 1.5,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        running_var = rng.uniform(-2, 2, n)
        treated = (running_var >= 0).astype(float)
        outcome = 0.5 * running_var + effect * treated + rng.normal(0, 0.3, n)
        return outcome, running_var

    def test_recovers_treatment_effect(self) -> None:
        outcome, running_var = self._make_rd_data(n=2000, effect=1.5)
        result = regression_discontinuity(outcome, running_var, cutoff=0.0)
        assert abs(result.ate - 1.5) < 0.5

    def test_output_structure(self) -> None:
        outcome, running_var = self._make_rd_data()
        result = regression_discontinuity(outcome, running_var, cutoff=0.0)
        assert hasattr(result, "ate")
        assert hasattr(result, "se")
        assert hasattr(result, "n_left")
        assert hasattr(result, "n_right")
        assert hasattr(result, "bandwidth")
        assert result.n_left > 0
        assert result.n_right > 0

    def test_custom_bandwidth(self) -> None:
        outcome, running_var = self._make_rd_data()
        result = regression_discontinuity(
            outcome, running_var, cutoff=0.0, bandwidth=0.5
        )
        assert result.bandwidth == 0.5

    def test_insufficient_data_raises(self) -> None:
        outcome = np.array([1.0, 2.0, 3.0])
        running_var = np.array([-1.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="Insufficient"):
            regression_discontinuity(outcome, running_var, cutoff=0.0, bandwidth=0.01)


# ---------------------------------------------------------------------------
# Synthetic control tests
# ---------------------------------------------------------------------------


class TestSyntheticControl:
    def _make_sc_data(
        self,
        n_periods: int = 40,
        n_donors: int = 5,
        pre_period: int = 20,
        effect: float = 3.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        rng = np.random.default_rng(seed)
        # Donors follow a common trend
        common = np.cumsum(rng.normal(0, 0.5, n_periods))
        donor_outcomes = np.column_stack(
            [common + rng.normal(0, 0.3, n_periods) for _ in range(n_donors)]
        )
        # Treated unit is a mix of donors pre-treatment, then shifted post
        true_weights = rng.dirichlet(np.ones(n_donors))
        treated = donor_outcomes @ true_weights + rng.normal(0, 0.1, n_periods)
        treated[pre_period:] += effect
        return treated, donor_outcomes, pre_period

    def test_recovers_treatment_effect(self) -> None:
        treated, donors, pre = self._make_sc_data(effect=3.0)
        result = synthetic_control(treated, donors, pre)
        assert abs(result.ate - 3.0) < 1.5

    def test_weights_sum_to_one(self) -> None:
        treated, donors, pre = self._make_sc_data()
        result = synthetic_control(treated, donors, pre)
        np.testing.assert_allclose(np.sum(result.weights), 1.0, atol=1e-6)

    def test_weights_non_negative(self) -> None:
        treated, donors, pre = self._make_sc_data()
        result = synthetic_control(treated, donors, pre)
        assert np.all(result.weights >= -1e-6)

    def test_pre_period_fit(self) -> None:
        treated, donors, pre = self._make_sc_data()
        result = synthetic_control(treated, donors, pre)
        assert result.pre_rmse < 1.0  # should fit well

    def test_invalid_pre_period(self) -> None:
        treated, donors, _ = self._make_sc_data()
        with pytest.raises(ValueError, match="pre_period"):
            synthetic_control(treated, donors, 0)
        with pytest.raises(ValueError, match="pre_period"):
            synthetic_control(treated, donors, len(treated))


# ---------------------------------------------------------------------------
# Difference-in-differences tests
# ---------------------------------------------------------------------------


class TestDiffInDiff:
    def _make_did_data(
        self,
        n_entities: int = 50,
        effect: float = 2.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        # Half entities treated, half control; pre and post observations
        entities = np.repeat(np.arange(n_entities), 2)
        treatment = np.repeat(
            np.concatenate([np.ones(n_entities // 2), np.zeros(n_entities // 2)]), 2
        )
        post = np.tile([0.0, 1.0], n_entities)
        n = len(entities)
        # Entity fixed effects
        entity_fe = rng.normal(0, 1, n_entities)
        outcome = (
            entity_fe[entities]
            + 0.5 * post
            + effect * treatment * post
            + rng.normal(0, 0.3, n)
        )
        return outcome, treatment, post, entities

    def test_recovers_treatment_effect(self) -> None:
        outcome, treatment, post, entity = self._make_did_data(
            n_entities=200, effect=2.0
        )
        result = diff_in_diff(outcome, treatment, post, entity)
        assert abs(result.ate - 2.0) < 0.5

    def test_output_structure(self) -> None:
        outcome, treatment, post, entity = self._make_did_data()
        result = diff_in_diff(outcome, treatment, post, entity)
        assert hasattr(result, "ate")
        assert hasattr(result, "se")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "pre_treatment_mean")
        assert hasattr(result, "post_treatment_mean")
        assert hasattr(result, "pre_control_mean")
        assert hasattr(result, "post_control_mean")

    def test_without_entity_fixed_effects(self) -> None:
        outcome, treatment, post, _ = self._make_did_data(effect=2.0)
        result = diff_in_diff(outcome, treatment, post)
        # Should still get a reasonable estimate, but noisier without FE
        assert abs(result.ate - 2.0) < 2.0

    def test_no_effect(self) -> None:
        outcome, treatment, post, entity = self._make_did_data(
            n_entities=200, effect=0.0
        )
        result = diff_in_diff(outcome, treatment, post, entity)
        assert abs(result.ate) < 0.5
