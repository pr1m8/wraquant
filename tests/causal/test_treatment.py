"""Tests for causal treatment effect estimation (pure numpy/scipy)."""

from __future__ import annotations

import numpy as np
import pytest

from wraquant.causal.treatment import (
    bounds_analysis,
    causal_forest,
    diff_in_diff,
    doubly_robust_ate,
    event_study,
    granger_causality,
    instrumental_variable,
    ipw_ate,
    matching_ate,
    mediation_analysis,
    propensity_score,
    regression_discontinuity,
    regression_discontinuity_robust,
    synthetic_control,
    synthetic_control_weights,
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


# ---------------------------------------------------------------------------
# Granger causality tests
# ---------------------------------------------------------------------------


class TestGrangerCausality:
    def test_detects_known_causal_relationship(self) -> None:
        """x Granger-causes y when y depends on lagged x."""
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(size=n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.5 * y[t - 1] + 0.4 * x[t - 1] + rng.normal(0, 0.3)

        result = granger_causality(x, y, max_lag=5)
        assert result.reject is True
        assert result.p_value < 0.05
        assert result.direction == "x -> y"

    def test_no_causality_independent_series(self) -> None:
        """Two independent stationary series should not show Granger causality."""
        rng = np.random.default_rng(99)
        n = 300
        # Use stationary AR(1) processes (not random walks) to avoid
        # spurious Granger causality from non-stationarity.
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.5 * x[t - 1] + rng.normal(0, 1)
            y[t] = 0.5 * y[t - 1] + rng.normal(0, 1)

        result = granger_causality(x, y, max_lag=5, significance=0.01)
        # Independent stationary series should not reject at 1%.
        assert not result.reject

    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        result = granger_causality(x, y, max_lag=3)

        assert hasattr(result, "f_statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "optimal_lag")
        assert hasattr(result, "direction")
        assert hasattr(result, "all_lags")
        assert result.optimal_lag >= 1
        assert result.optimal_lag <= 3
        assert len(result.all_lags) == 3

    def test_short_series_raises(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        with pytest.raises(ValueError, match="too short"):
            granger_causality(x, y, max_lag=5)

    def test_unequal_length_raises(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0])
        with pytest.raises(ValueError, match="same length"):
            granger_causality(x, y)


# ---------------------------------------------------------------------------
# Instrumental variable tests
# ---------------------------------------------------------------------------


class TestInstrumentalVariable:
    def test_basic_2sls(self) -> None:
        """2SLS should recover the causal effect when OLS is biased."""
        rng = np.random.default_rng(42)
        n = 2000
        z = rng.normal(size=(n, 1))  # instrument
        u = rng.normal(size=n)  # unobserved confounder
        x = z.ravel() + u + rng.normal(0, 0.5, n)  # endogenous
        y = 2.0 * x + u + rng.normal(0, 0.5, n)  # outcome

        result = instrumental_variable(y, x, z)
        # 2SLS should be closer to 2.0 than OLS (which is biased upward)
        assert abs(result.coefficient - 2.0) < 0.5
        assert result.details["coef_ols"] > 2.2  # OLS biased upward

    def test_weak_instrument_detection(self) -> None:
        """Weak instrument should have low first-stage F-statistic."""
        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(size=(n, 1))
        u = rng.normal(size=n)
        # Very weak instrument: barely correlated with endogenous
        x = 0.01 * z.ravel() + u + rng.normal(0, 1, n)
        y = 2.0 * x + u + rng.normal(0, 1, n)

        result = instrumental_variable(y, x, z)
        # First-stage F should be very low
        assert result.first_stage_f < 10

    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(size=(n, 1))
        x = z.ravel() + rng.normal(0, 0.5, n)
        y = 1.5 * x + rng.normal(0, 0.5, n)

        result = instrumental_variable(y, x, z)
        assert hasattr(result, "coefficient")
        assert hasattr(result, "se")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "first_stage_f")
        assert result.se > 0
        assert result.n_obs == n
        assert result.n_instruments == 1

    def test_sargan_with_multiple_instruments(self) -> None:
        """With multiple instruments, Sargan test should be available."""
        rng = np.random.default_rng(42)
        n = 1000
        z = rng.normal(size=(n, 3))
        x = z @ np.array([0.5, 0.3, 0.2]) + rng.normal(0, 0.5, n)
        y = 2.0 * x + rng.normal(0, 0.5, n)

        result = instrumental_variable(y, x, z)
        assert result.sargan_stat is not None
        assert result.sargan_p is not None
        assert result.n_instruments == 3

    def test_with_exogenous_controls(self) -> None:
        rng = np.random.default_rng(42)
        n = 1000
        z = rng.normal(size=(n, 1))
        w = rng.normal(size=(n, 2))
        u = rng.normal(size=n)
        x = z.ravel() + 0.5 * w[:, 0] + u + rng.normal(0, 0.3, n)
        y = 2.0 * x + w[:, 0] + 0.5 * w[:, 1] + u + rng.normal(0, 0.3, n)

        result = instrumental_variable(y, x, z, exogenous=w)
        assert abs(result.coefficient - 2.0) < 0.5


# ---------------------------------------------------------------------------
# Event study tests
# ---------------------------------------------------------------------------


class TestEventStudy:
    def test_detects_positive_abnormal_return(self) -> None:
        """Should detect a positive abnormal return after a synthetic event."""
        rng = np.random.default_rng(42)
        n = 500
        market = rng.normal(0.0005, 0.01, n)
        stock = 0.001 + 1.2 * market + rng.normal(0, 0.005, n)
        # Inject positive event at day 300
        stock[300:306] += 0.02

        result = event_study(
            stock, market, [300],
            estimation_window=120,
            event_window_pre=5,
            event_window_post=5,
            gap=10,
        )
        assert result.car > 0
        assert result.n_events == 1

    def test_car_significance(self) -> None:
        """Large event should produce significant CAR."""
        rng = np.random.default_rng(42)
        n = 500
        market = rng.normal(0.0005, 0.01, n)
        stock = 0.001 + 1.0 * market + rng.normal(0, 0.005, n)
        # Large event
        stock[300:306] += 0.05

        result = event_study(stock, market, [300], event_window_post=5)
        assert result.car_p_value < 0.05

    def test_no_event_insignificant(self) -> None:
        """Without an event, CAR should be near zero."""
        rng = np.random.default_rng(42)
        n = 500
        market = rng.normal(0.0005, 0.01, n)
        stock = 0.001 + 1.0 * market + rng.normal(0, 0.005, n)

        result = event_study(stock, market, [300], event_window_post=5)
        assert abs(result.car) < 0.05  # near zero

    def test_multiple_events(self) -> None:
        """Should handle multiple events and provide cross-sectional test."""
        rng = np.random.default_rng(42)
        n = 800
        market = rng.normal(0.0005, 0.01, n)
        stock = 0.001 + 1.0 * market + rng.normal(0, 0.005, n)
        # Two events
        stock[300:306] += 0.03
        stock[500:506] += 0.03

        result = event_study(
            stock, market, [300, 500],
            event_window_post=5,
        )
        assert result.n_events == 2
        assert result.cross_sectional_t is not None
        assert result.cross_sectional_p is not None

    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        market = rng.normal(0.0005, 0.01, n)
        stock = 0.001 + 1.0 * market + rng.normal(0, 0.005, n)

        result = event_study(stock, market, [300])
        assert hasattr(result, "car")
        assert hasattr(result, "car_se")
        assert hasattr(result, "car_t_stat")
        assert hasattr(result, "car_p_value")
        assert hasattr(result, "abnormal_returns")
        assert hasattr(result, "cumulative_ar")
        assert hasattr(result, "estimation_alpha")
        assert hasattr(result, "estimation_beta")
        # Default event window: 5 + 5 + 1 = 11 days
        assert len(result.abnormal_returns) == 11

    def test_invalid_event_raises(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        market = rng.normal(size=n)
        stock = rng.normal(size=n)
        with pytest.raises(ValueError, match="estimation window"):
            event_study(stock, market, [5], estimation_window=120)


# ---------------------------------------------------------------------------
# Synthetic control (enhanced) tests
# ---------------------------------------------------------------------------


class TestSyntheticControlWeights:
    def _make_sc_data(
        self,
        n_periods: int = 50,
        n_donors: int = 8,
        pre_period: int = 30,
        effect: float = 5.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        rng = np.random.default_rng(seed)
        common = np.cumsum(rng.normal(0, 0.5, n_periods))
        donor_outcomes = np.column_stack(
            [common + rng.normal(0, 0.3, n_periods) for _ in range(n_donors)]
        )
        true_weights = rng.dirichlet(np.ones(n_donors))
        treated = donor_outcomes @ true_weights + rng.normal(0, 0.1, n_periods)
        treated[pre_period:] += effect
        return treated, donor_outcomes, pre_period

    def test_pre_treatment_fit_quality(self) -> None:
        """Pre-treatment RMSPE should be small for well-constructed data."""
        treated, donors, pre = self._make_sc_data()
        result = synthetic_control_weights(treated, donors, pre)
        assert result.pre_rmspe < 1.0

    def test_post_treatment_effect(self) -> None:
        treated, donors, pre = self._make_sc_data(effect=5.0)
        result = synthetic_control_weights(treated, donors, pre)
        assert abs(result.ate - 5.0) < 2.0

    def test_rmspe_ratio(self) -> None:
        treated, donors, pre = self._make_sc_data(effect=5.0)
        result = synthetic_control_weights(treated, donors, pre)
        # Post/pre ratio should be large when there's a treatment effect
        assert result.rmspe_ratio > 1.0

    def test_placebo_inference(self) -> None:
        """Placebo test should produce a p-value."""
        treated, donors, pre = self._make_sc_data(
            n_periods=50, n_donors=5, pre_period=30, effect=5.0
        )
        result = synthetic_control_weights(
            treated, donors, pre, run_placebo=True
        )
        assert result.placebo_p_value is not None
        assert result.placebo_ratios is not None
        assert 0.0 <= result.placebo_p_value <= 1.0

    def test_weights_properties(self) -> None:
        treated, donors, pre = self._make_sc_data()
        result = synthetic_control_weights(treated, donors, pre)
        np.testing.assert_allclose(np.sum(result.weights), 1.0, atol=1e-6)
        assert np.all(result.weights >= -1e-6)

    def test_donor_names(self) -> None:
        treated, donors, pre = self._make_sc_data(n_donors=3)
        names = ["A", "B", "C"]
        result = synthetic_control_weights(
            treated, donors, pre, donor_names=names
        )
        assert result.donor_names == names


# ---------------------------------------------------------------------------
# Causal forest tests
# ---------------------------------------------------------------------------


class TestCausalForest:
    def test_recovers_average_effect(self) -> None:
        """Should recover the average treatment effect."""
        rng = np.random.default_rng(42)
        n = 1000
        X = rng.normal(size=(n, 3))
        T = (rng.uniform(size=n) > 0.5).astype(float)
        # Constant treatment effect of 2.0
        Y = X[:, 1] + 2.0 * T + rng.normal(0, 0.5, n)

        result = causal_forest(Y, T, X, n_estimators=100, honest=False)
        assert abs(result.ate - 2.0) < 1.0

    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(size=(n, 2))
        T = (rng.uniform(size=n) > 0.5).astype(float)
        Y = 1.0 + 2.0 * T + rng.normal(0, 0.5, n)

        result = causal_forest(Y, T, X, n_estimators=50, honest=False)
        assert hasattr(result, "ate")
        assert hasattr(result, "cate")
        assert hasattr(result, "ate_se")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "feature_importances")
        assert result.cate.shape == (n,)
        assert result.feature_importances.shape == (2,)

    def test_honest_estimation(self) -> None:
        """Honest estimation should also give reasonable results."""
        rng = np.random.default_rng(42)
        n = 800
        X = rng.normal(size=(n, 3))
        T = (rng.uniform(size=n) > 0.5).astype(float)
        Y = X[:, 0] + 2.0 * T + rng.normal(0, 0.5, n)

        result = causal_forest(Y, T, X, n_estimators=100, honest=True)
        assert abs(result.ate - 2.0) < 1.5
        assert result.details["honest"] is True


# ---------------------------------------------------------------------------
# Mediation analysis tests
# ---------------------------------------------------------------------------


class TestMediationAnalysis:
    def test_detects_indirect_effect(self) -> None:
        """Should detect the indirect effect through a mediator."""
        rng = np.random.default_rng(42)
        n = 2000
        T = rng.binomial(1, 0.5, n).astype(float)
        M = 0.8 * T + rng.normal(0, 0.5, n)  # path a = 0.8
        Y = 0.5 * T + 0.6 * M + rng.normal(0, 0.5, n)  # c' = 0.5, b = 0.6

        result = mediation_analysis(Y, T, M)
        # Indirect effect should be ~ a * b = 0.48
        assert abs(result.indirect_effect - 0.48) < 0.2
        assert result.sobel_p < 0.05

    def test_total_effect_decomposition(self) -> None:
        """Total effect = direct + indirect."""
        rng = np.random.default_rng(42)
        n = 1000
        T = rng.binomial(1, 0.5, n).astype(float)
        M = 0.5 * T + rng.normal(0, 0.5, n)
        Y = 1.0 * T + 0.7 * M + rng.normal(0, 0.5, n)

        result = mediation_analysis(Y, T, M)
        # total ~ direct + indirect (c ~ c' + a*b)
        assert abs(
            result.total_effect - (result.direct_effect + result.indirect_effect)
        ) < 0.15

    def test_no_mediation(self) -> None:
        """When mediator is independent, indirect effect should be ~0."""
        rng = np.random.default_rng(42)
        n = 1000
        T = rng.binomial(1, 0.5, n).astype(float)
        M = rng.normal(0, 1, n)  # independent mediator
        Y = 2.0 * T + rng.normal(0, 0.5, n)

        result = mediation_analysis(Y, T, M)
        assert abs(result.indirect_effect) < 0.3
        assert result.sobel_p > 0.01

    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        T = rng.binomial(1, 0.5, n).astype(float)
        M = T + rng.normal(0, 1, n)
        Y = T + M + rng.normal(0, 1, n)

        result = mediation_analysis(Y, T, M)
        assert hasattr(result, "total_effect")
        assert hasattr(result, "direct_effect")
        assert hasattr(result, "indirect_effect")
        assert hasattr(result, "sobel_stat")
        assert hasattr(result, "sobel_p")
        assert hasattr(result, "proportion_mediated")
        assert hasattr(result, "path_a")
        assert hasattr(result, "path_b")
        assert 0.0 <= result.proportion_mediated <= 1.0

    def test_with_covariates(self) -> None:
        """Should work with additional control variables."""
        rng = np.random.default_rng(42)
        n = 1000
        W = rng.normal(size=(n, 2))
        T = rng.binomial(1, 0.5, n).astype(float)
        M = 0.6 * T + 0.3 * W[:, 0] + rng.normal(0, 0.5, n)
        Y = 0.5 * T + 0.7 * M + W[:, 1] + rng.normal(0, 0.5, n)

        result = mediation_analysis(Y, T, M, covariates=W)
        assert abs(result.indirect_effect - 0.42) < 0.3


# ---------------------------------------------------------------------------
# Robust regression discontinuity tests
# ---------------------------------------------------------------------------


class TestRDRobust:
    def test_recovers_effect_with_ik_bandwidth(self) -> None:
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(-2, 2, n)
        y = 0.5 * x + 1.5 * (x >= 0) + rng.normal(0, 0.3, n)

        result = regression_discontinuity_robust(y, x, cutoff=0.0)
        assert abs(result.ate - 1.5) < 0.5
        assert result.bandwidth_method == "imbens_kalyanaraman"

    def test_user_specified_bandwidth(self) -> None:
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(-2, 2, n)
        y = 0.5 * x + 1.5 * (x >= 0) + rng.normal(0, 0.3, n)

        result = regression_discontinuity_robust(
            y, x, cutoff=0.0, bandwidth=1.0
        )
        assert result.bandwidth == 1.0
        assert result.bandwidth_method == "user_specified"

    def test_mccrary_test_runs(self) -> None:
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(-2, 2, n)
        y = x + rng.normal(0, 0.5, n)

        result = regression_discontinuity_robust(
            y, x, cutoff=0.0, run_mccrary=True
        )
        assert result.mccrary_stat is not None
        assert result.mccrary_p is not None

    def test_quadratic_polynomial(self) -> None:
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.uniform(-2, 2, n)
        y = 0.5 * x + 0.2 * x**2 + 2.0 * (x >= 0) + rng.normal(0, 0.3, n)

        result = regression_discontinuity_robust(
            y, x, cutoff=0.0, poly_order=2
        )
        assert result.poly_order == 2
        assert abs(result.ate - 2.0) < 1.0

    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(-2, 2, n)
        y = x + 1.0 * (x >= 0) + rng.normal(0, 0.5, n)

        result = regression_discontinuity_robust(y, x)
        assert hasattr(result, "ate")
        assert hasattr(result, "se")
        assert hasattr(result, "bandwidth")
        assert hasattr(result, "bandwidth_method")
        assert hasattr(result, "n_left")
        assert hasattr(result, "n_right")
        assert hasattr(result, "poly_order")
        assert hasattr(result, "rd_type")
        assert result.rd_type == "sharp"
        assert result.se > 0


# ---------------------------------------------------------------------------
# Bounds analysis tests
# ---------------------------------------------------------------------------


class TestBoundsAnalysis:
    def test_manski_bounds_contain_true_effect(self) -> None:
        """Manski bounds should contain the true treatment effect."""
        rng = np.random.default_rng(42)
        n = 1000
        T = rng.binomial(1, 0.5, n).astype(float)
        Y = 2.0 * T + rng.normal(0, 1, n)

        result = bounds_analysis(Y, T, outcome_bounds=(-5, 10))
        assert result.lower_bound < 2.0 < result.upper_bound
        assert result.bound_type == "manski"

    def test_manski_bounds_ordering(self) -> None:
        """Lower bound should always be <= upper bound."""
        rng = np.random.default_rng(42)
        n = 1000
        T = rng.binomial(1, 0.5, n).astype(float)
        Y = 3.0 * T + rng.normal(0, 0.5, n)

        result = bounds_analysis(Y, T, outcome_bounds=(-5, 10))
        assert result.lower_bound <= result.upper_bound
        # With bounded support, the gap between upper and lower should
        # be at most (y_max - y_min) = 15
        assert result.upper_bound - result.lower_bound <= 15.0 + 0.1

    def test_manski_wide_bounds(self) -> None:
        """With wide outcome support, bounds should be wide."""
        rng = np.random.default_rng(42)
        n = 500
        T = rng.binomial(1, 0.5, n).astype(float)
        Y = 1.0 * T + rng.normal(0, 1, n)

        result = bounds_analysis(Y, T, outcome_bounds=(-100, 100))
        # Bounds should be very wide
        assert result.upper_bound - result.lower_bound > 50

    def test_lee_bounds(self) -> None:
        """Lee bounds with differential selection."""
        rng = np.random.default_rng(42)
        n = 2000
        T = rng.binomial(1, 0.5, n).astype(float)
        # Selection more likely when treated
        S = rng.binomial(1, 0.9 * T + 0.7 * (1 - T), n).astype(float)
        Y = 2.0 * T + rng.normal(0, 1, n)

        result = bounds_analysis(Y, T, selection=S, method="lee")
        assert result.bound_type == "lee"
        assert result.lower_bound <= result.upper_bound

    def test_output_structure(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        T = rng.binomial(1, 0.5, n).astype(float)
        Y = T + rng.normal(0, 1, n)

        result = bounds_analysis(Y, T)
        assert hasattr(result, "lower_bound")
        assert hasattr(result, "upper_bound")
        assert hasattr(result, "bound_type")
        assert hasattr(result, "identified")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert result.ci_lower <= result.lower_bound
        assert result.ci_upper >= result.upper_bound

    def test_lee_without_selection_raises(self) -> None:
        T = np.array([0.0, 1.0, 0.0, 1.0])
        Y = np.array([1.0, 2.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="selection"):
            bounds_analysis(Y, T, method="lee")
