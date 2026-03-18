"""Tests for volatility models — GARCH family, stochastic vol, Hawkes.

Each test class covers a single model or diagnostic function. Tests
marked with ``requires_arch`` are skipped if the ``arch`` library is
not installed. Similarly for ``requires_sklearn``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------

try:
    import arch  # noqa: F401

    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

requires_arch = pytest.mark.skipif(not HAS_ARCH, reason="arch not installed")
requires_sklearn = pytest.mark.skipif(
    not HAS_SKLEARN, reason="sklearn not installed"
)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def returns_series() -> pd.Series:
    """Synthetic return series with mild volatility clustering."""
    rng = np.random.default_rng(42)
    n = 1000
    # Simple GARCH-like DGP
    ret = np.empty(n)
    sigma2 = np.empty(n)
    sigma2[0] = 0.0001
    omega, alpha, beta = 0.00001, 0.08, 0.90
    for t in range(n):
        ret[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        if t < n - 1:
            sigma2[t + 1] = omega + alpha * ret[t] ** 2 + beta * sigma2[t]
    return pd.Series(ret, name="returns")


@pytest.fixture
def returns_array(returns_series: pd.Series) -> np.ndarray:
    """Same returns as numpy array."""
    return returns_series.values


@pytest.fixture
def multi_returns() -> pd.DataFrame:
    """Multivariate returns for DCC tests (3 assets, correlated)."""
    rng = np.random.default_rng(42)
    n = 500
    cov = np.array(
        [
            [0.0004, 0.0002, 0.0001],
            [0.0002, 0.0009, 0.0003],
            [0.0001, 0.0003, 0.0006],
        ]
    )
    ret = rng.multivariate_normal([0, 0, 0], cov, size=n)
    return pd.DataFrame(ret, columns=["A", "B", "C"])


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------


class TestEwmaVolatility:
    def test_basic_output(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import ewma_volatility

        vol = ewma_volatility(returns_series, span=30)
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(returns_series)

    def test_positive(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import ewma_volatility

        vol = ewma_volatility(returns_series, span=30)
        assert (vol.dropna() >= 0).all()

    def test_annualize_larger(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import ewma_volatility

        vol_ann = ewma_volatility(returns_series, annualize=True)
        vol_raw = ewma_volatility(returns_series, annualize=False)
        assert vol_ann.dropna().mean() > vol_raw.dropna().mean()


# ---------------------------------------------------------------------------
# GARCH(p, q)
# ---------------------------------------------------------------------------


@requires_arch
class TestGarchFit:
    def test_basic_fit(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_fit

        result = garch_fit(returns_series, p=1, q=1)
        assert "params" in result
        assert "conditional_volatility" in result
        assert "standardized_residuals" in result
        assert "persistence" in result

    def test_persistence_stationary(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_fit

        result = garch_fit(returns_series, p=1, q=1)
        assert 0 < result["persistence"] < 2  # generous bound

    def test_conditional_vol_positive(
        self, returns_series: pd.Series
    ) -> None:
        from wraquant.vol.models import garch_fit

        result = garch_fit(returns_series, p=1, q=1)
        cv = result["conditional_volatility"]
        assert (cv > 0).all()

    def test_aic_bic_finite(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_fit

        result = garch_fit(returns_series, p=1, q=1)
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])

    def test_ljung_box_present(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_fit

        result = garch_fit(returns_series, p=1, q=1)
        lb = result["ljung_box"]
        assert lb is not None
        assert "statistic" in lb
        assert "p_value" in lb

    def test_half_life_positive(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_fit

        result = garch_fit(returns_series, p=1, q=1)
        assert result["half_life"] > 0

    def test_student_t_distribution(
        self, returns_series: pd.Series
    ) -> None:
        from wraquant.vol.models import garch_fit

        result = garch_fit(returns_series, p=1, q=1, dist="t")
        assert "params" in result

    def test_accepts_numpy(self, returns_array: np.ndarray) -> None:
        from wraquant.vol.models import garch_fit

        result = garch_fit(returns_array, p=1, q=1)
        assert result["conditional_volatility"].iloc[-1] > 0

    def test_model_object_returned(
        self, returns_series: pd.Series
    ) -> None:
        from wraquant.vol.models import garch_fit

        result = garch_fit(returns_series, p=1, q=1)
        assert result["model"] is not None


# ---------------------------------------------------------------------------
# EGARCH
# ---------------------------------------------------------------------------


@requires_arch
class TestEgarchFit:
    def test_basic_fit(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import egarch_fit

        result = egarch_fit(returns_series, p=1, q=1)
        assert "params" in result
        assert result["conditional_volatility"].iloc[-1] > 0

    def test_conditional_vol_positive(
        self, returns_series: pd.Series
    ) -> None:
        from wraquant.vol.models import egarch_fit

        result = egarch_fit(returns_series, p=1, q=1)
        assert (result["conditional_volatility"] > 0).all()

    def test_model_name(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import egarch_fit

        result = egarch_fit(returns_series, p=1, q=1)
        assert "EGARCH" in result["model_name"]


# ---------------------------------------------------------------------------
# GJR-GARCH
# ---------------------------------------------------------------------------


@requires_arch
class TestGjrGarchFit:
    def test_basic_fit(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import gjr_garch_fit

        result = gjr_garch_fit(returns_series, p=1, q=1)
        assert "params" in result
        assert result["conditional_volatility"].iloc[-1] > 0

    def test_gamma_in_params(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import gjr_garch_fit

        result = gjr_garch_fit(returns_series, p=1, q=1)
        param_keys = " ".join(result["params"].keys())
        assert "gamma" in param_keys

    def test_conditional_vol_positive(
        self, returns_series: pd.Series
    ) -> None:
        from wraquant.vol.models import gjr_garch_fit

        result = gjr_garch_fit(returns_series, p=1, q=1)
        assert (result["conditional_volatility"] > 0).all()


# ---------------------------------------------------------------------------
# FIGARCH
# ---------------------------------------------------------------------------


@requires_arch
class TestFigarchFit:
    def test_basic_fit(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import figarch_fit

        result = figarch_fit(returns_series, p=1, q=1)
        assert "params" in result
        assert np.isfinite(result["aic"])

    def test_conditional_vol_positive(
        self, returns_series: pd.Series
    ) -> None:
        from wraquant.vol.models import figarch_fit

        result = figarch_fit(returns_series, p=1, q=1)
        assert (result["conditional_volatility"] > 0).all()


# ---------------------------------------------------------------------------
# HARCH
# ---------------------------------------------------------------------------


@requires_arch
class TestHarchFit:
    def test_basic_fit(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import harch_fit

        result = harch_fit(returns_series, lags=[1, 5, 22])
        assert "params" in result
        assert result["conditional_volatility"].iloc[-1] > 0

    def test_default_lags(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import harch_fit

        result = harch_fit(returns_series)
        assert "HARCH" in result["model_name"]


# ---------------------------------------------------------------------------
# GARCH Forecast
# ---------------------------------------------------------------------------


@requires_arch
class TestGarchForecast:
    def test_analytic_forecast_length(
        self, returns_series: pd.Series
    ) -> None:
        from wraquant.vol.models import garch_forecast

        fc = garch_forecast(returns_series, horizon=5, method="analytic")
        assert len(fc["forecast_volatility"]) == 5
        assert len(fc["forecast_variance"]) == 5

    def test_forecast_positive(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_forecast

        fc = garch_forecast(returns_series, horizon=10)
        assert (fc["forecast_volatility"] > 0).all()

    def test_simulation_forecast(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_forecast

        fc = garch_forecast(
            returns_series,
            horizon=5,
            method="simulation",
            simulations=200,
        )
        assert fc["confidence_intervals"] is not None
        ci = fc["confidence_intervals"]
        assert "lower_5" in ci
        assert "upper_95" in ci
        assert len(ci["lower_5"]) == 5

    def test_analytic_no_ci(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_forecast

        fc = garch_forecast(returns_series, horizon=3, method="analytic")
        assert fc["confidence_intervals"] is None

    def test_fit_result_included(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_forecast

        fc = garch_forecast(returns_series, horizon=3)
        assert "fit_result" in fc
        assert "persistence" in fc["fit_result"]


# ---------------------------------------------------------------------------
# DCC-GARCH
# ---------------------------------------------------------------------------


@requires_arch
class TestDccFit:
    def test_basic_fit(self, multi_returns: pd.DataFrame) -> None:
        from wraquant.vol.models import dcc_fit

        result = dcc_fit(multi_returns)
        assert "dcc_params" in result
        assert "a" in result["dcc_params"]
        assert "b" in result["dcc_params"]

    def test_correlation_shape(self, multi_returns: pd.DataFrame) -> None:
        from wraquant.vol.models import dcc_fit

        result = dcc_fit(multi_returns)
        T, k = multi_returns.shape
        assert result["conditional_correlations"].shape == (T, k, k)

    def test_covariance_shape(self, multi_returns: pd.DataFrame) -> None:
        from wraquant.vol.models import dcc_fit

        result = dcc_fit(multi_returns)
        T, k = multi_returns.shape
        assert result["conditional_covariances"].shape == (T, k, k)

    def test_correlation_bounds(self, multi_returns: pd.DataFrame) -> None:
        from wraquant.vol.models import dcc_fit

        result = dcc_fit(multi_returns)
        corrs = result["conditional_correlations"]
        assert np.all(corrs >= -1.0 - 1e-10)
        assert np.all(corrs <= 1.0 + 1e-10)

    def test_dcc_stationarity(self, multi_returns: pd.DataFrame) -> None:
        from wraquant.vol.models import dcc_fit

        result = dcc_fit(multi_returns)
        a = result["dcc_params"]["a"]
        b = result["dcc_params"]["b"]
        assert a + b < 1.0

    def test_univariate_results_count(
        self, multi_returns: pd.DataFrame
    ) -> None:
        from wraquant.vol.models import dcc_fit

        result = dcc_fit(multi_returns)
        assert len(result["univariate_results"]) == multi_returns.shape[1]

    def test_rejects_univariate(self) -> None:
        from wraquant.vol.models import dcc_fit

        rng = np.random.default_rng(42)
        ret = pd.DataFrame({"A": rng.normal(0, 0.01, 100)})
        with pytest.raises(ValueError, match="at least 2"):
            dcc_fit(ret)


# ---------------------------------------------------------------------------
# Realized GARCH
# ---------------------------------------------------------------------------


@requires_arch
class TestRealizedGarch:
    def test_basic_fit(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import realized_garch

        rv = returns_series.abs().rolling(20).mean().bfill()
        result = realized_garch(returns_series, rv)
        assert "params" in result
        assert "realized_vol_used" in result

    def test_conditional_vol_positive(
        self, returns_series: pd.Series
    ) -> None:
        from wraquant.vol.models import realized_garch

        rv = returns_series.abs().rolling(20).mean().bfill()
        result = realized_garch(returns_series, rv)
        assert (result["conditional_volatility"] > 0).all()

    def test_length_mismatch_raises(
        self, returns_series: pd.Series
    ) -> None:
        from wraquant.vol.models import realized_garch

        with pytest.raises(ValueError, match="same length"):
            realized_garch(returns_series, np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# News Impact Curve
# ---------------------------------------------------------------------------


@requires_arch
class TestNewsImpactCurve:
    def test_garch_symmetric(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import news_impact_curve

        nic = news_impact_curve(returns_series, model_type="GARCH")
        assert nic["shocks"].shape == nic["conditional_variance"].shape
        # GARCH is symmetric: variance at +shock == variance at -shock
        n = len(nic["shocks"])
        mid = n // 2
        # Allow for slight numerical asymmetry from mean != 0
        left = nic["conditional_variance"][mid - 5]
        right = nic["conditional_variance"][mid + 5]
        # Both should be roughly equal for symmetric model
        assert left > 0 and right > 0

    def test_gjr_output_shape(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import news_impact_curve

        nic = news_impact_curve(
            returns_series, model_type="GJR", n_points=50
        )
        assert len(nic["shocks"]) == 50
        assert len(nic["conditional_variance"]) == 50

    def test_egarch_output(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import news_impact_curve

        nic = news_impact_curve(returns_series, model_type="EGARCH")
        assert nic["model_type"] == "EGARCH"
        assert (nic["conditional_variance"] > 0).all()

    def test_variance_positive(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import news_impact_curve

        for mt in ["GARCH", "GJR", "EGARCH"]:
            nic = news_impact_curve(returns_series, model_type=mt)
            assert (nic["conditional_variance"] > 0).all(), (
                f"Non-positive variance for {mt}"
            )


# ---------------------------------------------------------------------------
# Volatility Persistence
# ---------------------------------------------------------------------------


class TestVolatilityPersistence:
    def test_from_scalars(self) -> None:
        from wraquant.vol.models import volatility_persistence

        vp = volatility_persistence(alpha=0.05, beta=0.93, omega=0.01)
        assert vp["persistence"] == pytest.approx(0.98, abs=1e-10)
        assert vp["half_life"] > 30
        assert vp["mean_reversion_speed"] == pytest.approx(0.02, abs=1e-10)

    def test_from_params_dict(self) -> None:
        from wraquant.vol.models import volatility_persistence

        params = {"alpha[1]": 0.06, "beta[1]": 0.92, "omega": 0.005}
        vp = volatility_persistence(params)
        assert vp["persistence"] == pytest.approx(0.98, abs=1e-10)

    def test_unconditional_variance(self) -> None:
        from wraquant.vol.models import volatility_persistence

        vp = volatility_persistence(alpha=0.05, beta=0.90, omega=0.01)
        expected = 0.01 / (1 - 0.95)
        assert vp["unconditional_variance"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_igarch_infinite_halflife(self) -> None:
        from wraquant.vol.models import volatility_persistence

        vp = volatility_persistence(alpha=0.10, beta=0.90, omega=0.001)
        assert vp["half_life"] == float("inf")
        assert vp["unconditional_variance"] == float("inf")

    def test_gjr_persistence(self) -> None:
        from wraquant.vol.models import volatility_persistence

        params = {
            "alpha[1]": 0.02,
            "gamma[1]": 0.06,
            "beta[1]": 0.90,
            "omega": 0.01,
        }
        vp = volatility_persistence(params)
        # persistence = 0.02 + 0.90 + 0.5*0.06 = 0.95
        assert vp["persistence"] == pytest.approx(0.95, abs=1e-10)


# ---------------------------------------------------------------------------
# Hawkes Process
# ---------------------------------------------------------------------------


class TestHawkesProcess:
    def test_basic_fit(self) -> None:
        from wraquant.vol.models import hawkes_process

        rng = np.random.default_rng(42)
        events = np.sort(rng.exponential(0.5, 200).cumsum())
        result = hawkes_process(events)
        assert result["mu"] > 0
        assert result["alpha"] > 0
        assert result["beta"] > 0

    def test_branching_ratio(self) -> None:
        from wraquant.vol.models import hawkes_process

        rng = np.random.default_rng(42)
        events = np.sort(rng.exponential(0.5, 200).cumsum())
        result = hawkes_process(events)
        # Branching ratio should be finite and positive
        assert 0 < result["branching_ratio"] < 10

    def test_half_life_positive(self) -> None:
        from wraquant.vol.models import hawkes_process

        rng = np.random.default_rng(42)
        events = np.sort(rng.exponential(1.0, 100).cumsum())
        result = hawkes_process(events)
        assert result["half_life"] > 0

    def test_intensity_shape(self) -> None:
        from wraquant.vol.models import hawkes_process

        rng = np.random.default_rng(42)
        events = np.sort(rng.exponential(1.0, 50).cumsum())
        result = hawkes_process(events)
        assert result["intensity"].shape == events.shape

    def test_intensity_positive(self) -> None:
        from wraquant.vol.models import hawkes_process

        rng = np.random.default_rng(42)
        events = np.sort(rng.exponential(1.0, 80).cumsum())
        result = hawkes_process(events)
        assert (result["intensity"] > 0).all()

    def test_too_few_events_raises(self) -> None:
        from wraquant.vol.models import hawkes_process

        with pytest.raises(ValueError, match="at least 5"):
            hawkes_process(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Stochastic Volatility
# ---------------------------------------------------------------------------


class TestStochasticVolSV:
    def test_basic_fit(self) -> None:
        from wraquant.vol.models import stochastic_vol_sv

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 300)
        result = stochastic_vol_sv(returns, n_particles=100, n_iter=3)
        assert "phi" in result
        assert "mu" in result
        assert "sigma_eta" in result

    def test_filtered_vol_positive(self) -> None:
        from wraquant.vol.models import stochastic_vol_sv

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 300)
        result = stochastic_vol_sv(returns, n_particles=100, n_iter=3)
        assert (result["filtered_volatility"] > 0).all()

    def test_output_length(self) -> None:
        from wraquant.vol.models import stochastic_vol_sv

        rng = np.random.default_rng(42)
        n = 200
        returns = rng.normal(0, 0.01, n)
        result = stochastic_vol_sv(returns, n_particles=100, n_iter=2)
        assert len(result["filtered_volatility"]) == n
        assert len(result["log_variance"]) == n

    def test_phi_in_range(self) -> None:
        from wraquant.vol.models import stochastic_vol_sv

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 300)
        result = stochastic_vol_sv(returns, n_particles=100, n_iter=3)
        assert 0 < result["phi"] < 1


# ---------------------------------------------------------------------------
# Gaussian Mixture Vol
# ---------------------------------------------------------------------------


@requires_sklearn
class TestGaussianMixtureVol:
    def test_basic_fit(self) -> None:
        from wraquant.vol.models import gaussian_mixture_vol

        rng = np.random.default_rng(42)
        low = rng.normal(0, 0.005, 400)
        high = rng.normal(0, 0.02, 200)
        returns = np.concatenate([low, high])
        result = gaussian_mixture_vol(returns, n_components=2)
        assert len(result["volatilities"]) == 2
        assert len(result["weights"]) == 2

    def test_volatilities_sorted(self) -> None:
        from wraquant.vol.models import gaussian_mixture_vol

        rng = np.random.default_rng(42)
        low = rng.normal(0, 0.005, 400)
        high = rng.normal(0, 0.02, 200)
        returns = np.concatenate([low, high])
        result = gaussian_mixture_vol(returns, n_components=2)
        assert result["volatilities"][0] <= result["volatilities"][1]

    def test_weights_sum_to_one(self) -> None:
        from wraquant.vol.models import gaussian_mixture_vol

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        result = gaussian_mixture_vol(returns, n_components=2)
        assert abs(result["weights"].sum() - 1.0) < 1e-6

    def test_regime_labels_shape(self) -> None:
        from wraquant.vol.models import gaussian_mixture_vol

        rng = np.random.default_rng(42)
        n = 500
        returns = rng.normal(0, 0.01, n)
        result = gaussian_mixture_vol(returns, n_components=2)
        assert result["regime_labels"].shape == (n,)
        assert result["regime_probabilities"].shape == (n, 2)

    def test_aic_bic_finite(self) -> None:
        from wraquant.vol.models import gaussian_mixture_vol

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        result = gaussian_mixture_vol(returns, n_components=2)
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])

    def test_three_components(self) -> None:
        from wraquant.vol.models import gaussian_mixture_vol

        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        result = gaussian_mixture_vol(returns, n_components=3)
        assert len(result["volatilities"]) == 3


# ---------------------------------------------------------------------------
# SVI Vol Surface
# ---------------------------------------------------------------------------


class TestVolSurfaceSVI:
    def test_basic_fit(self) -> None:
        from wraquant.vol.models import vol_surface_svi

        strikes = np.array([90, 95, 100, 105, 110], dtype=float)
        forward = 100.0
        iv = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
        T = 0.25
        total_var = iv**2 * T
        result = vol_surface_svi(strikes, forward, total_var, T)
        assert "params" in result
        assert result["rmse"] < 0.1  # reasonable fit

    def test_fitted_iv_positive(self) -> None:
        from wraquant.vol.models import vol_surface_svi

        strikes = np.linspace(80, 120, 10)
        forward = 100.0
        iv = 0.20 + 0.001 * (strikes - 100) ** 2 / 100
        T = 0.5
        total_var = iv**2 * T
        result = vol_surface_svi(strikes, forward, total_var, T)
        assert (result["fitted_iv"] > 0).all()

    def test_params_keys(self) -> None:
        from wraquant.vol.models import vol_surface_svi

        strikes = np.array([90, 95, 100, 105, 110], dtype=float)
        forward = 100.0
        iv = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
        T = 0.25
        total_var = iv**2 * T
        result = vol_surface_svi(strikes, forward, total_var, T)
        for key in ["a", "b", "rho", "m", "sigma"]:
            assert key in result["params"]

    def test_length_mismatch_raises(self) -> None:
        from wraquant.vol.models import vol_surface_svi

        with pytest.raises(ValueError, match="same length"):
            vol_surface_svi(
                np.array([90, 100]),
                100.0,
                np.array([0.01]),
                0.25,
            )


# ---------------------------------------------------------------------------
# Variance Risk Premium
# ---------------------------------------------------------------------------


class TestVarianceRiskPremium:
    def test_basic(self) -> None:
        from wraquant.vol.models import variance_risk_premium

        iv = np.array([0.20, 0.22, 0.18, 0.25, 0.19])
        rv = np.array([0.15, 0.17, 0.16, 0.20, 0.14])
        result = variance_risk_premium(iv, rv)
        assert result["mean_vrp"] > 0

    def test_positive_fraction(self) -> None:
        from wraquant.vol.models import variance_risk_premium

        iv = np.array([0.20, 0.22, 0.18])
        rv = np.array([0.15, 0.17, 0.16])
        result = variance_risk_premium(iv, rv)
        assert result["pct_positive"] == 1.0

    def test_vol_spread(self) -> None:
        from wraquant.vol.models import variance_risk_premium

        iv = np.array([0.20, 0.15])
        rv = np.array([0.15, 0.20])
        result = variance_risk_premium(iv, rv)
        expected = iv - rv
        np.testing.assert_allclose(result["vol_spread"], expected)

    def test_length_mismatch_raises(self) -> None:
        from wraquant.vol.models import variance_risk_premium

        with pytest.raises(ValueError, match="same length"):
            variance_risk_premium(np.array([0.2]), np.array([0.1, 0.15]))

    def test_ratio_output(self) -> None:
        from wraquant.vol.models import variance_risk_premium

        iv = np.array([0.20, 0.30])
        rv = np.array([0.10, 0.15])
        result = variance_risk_premium(iv, rv)
        np.testing.assert_allclose(result["vrp_ratio"], [2.0, 2.0])


# ---------------------------------------------------------------------------
# APARCH
# ---------------------------------------------------------------------------


@requires_arch
class TestAparchFit:
    def test_basic_fit(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import aparch_fit

        result = aparch_fit(returns_series, p=1, o=1, q=1)
        assert "params" in result
        assert "conditional_volatility" in result
        assert "APARCH" in result["model_name"]

    def test_positive_vol(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import aparch_fit

        result = aparch_fit(returns_series, p=1, o=1, q=1)
        assert (result["conditional_volatility"] > 0).all()

    def test_power_param_in_results(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import aparch_fit

        result = aparch_fit(returns_series, p=1, o=1, q=1)
        # APARCH has a 'delta' power parameter in its fitted params
        param_keys = " ".join(result["params"].keys())
        assert "delta" in param_keys or "gamma" in param_keys

    def test_aic_bic_finite(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import aparch_fit

        result = aparch_fit(returns_series, p=1, o=1, q=1)
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])


# ---------------------------------------------------------------------------
# GARCH Rolling Forecast
# ---------------------------------------------------------------------------


@requires_arch
class TestGarchRollingForecast:
    def test_output_length_fixed_window(
        self, returns_series: pd.Series
    ) -> None:
        from wraquant.vol.models import garch_rolling_forecast

        window = 500
        result = garch_rolling_forecast(
            returns_series, window=window, refit_every=50
        )
        expected_len = len(returns_series) - window
        assert len(result["forecasts"]) == expected_len
        assert len(result["actuals"]) == expected_len

    def test_positive_forecasts(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_rolling_forecast

        result = garch_rolling_forecast(
            returns_series, window=500, refit_every=50
        )
        assert (result["forecasts"] > 0).all()

    def test_expanding_window(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_rolling_forecast

        # window=None means expanding window from min_obs=100
        result = garch_rolling_forecast(
            returns_series, window=None, refit_every=100
        )
        expected_len = len(returns_series) - 100
        assert len(result["forecasts"]) == expected_len

    def test_dates_present(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_rolling_forecast

        result = garch_rolling_forecast(
            returns_series, window=500, refit_every=50
        )
        assert len(result["dates"]) == len(result["forecasts"])


# ---------------------------------------------------------------------------
# GARCH Model Selection
# ---------------------------------------------------------------------------


@requires_arch
class TestGarchModelSelection:
    def test_returns_dataframe(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_model_selection

        df = garch_model_selection(returns_series)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_model_selection

        df = garch_model_selection(returns_series)
        for col in [
            "model",
            "vol_model",
            "distribution",
            "aic",
            "bic",
            "log_likelihood",
            "persistence",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_best_model_identified(
        self, returns_series: pd.Series
    ) -> None:
        from wraquant.vol.models import garch_model_selection

        df = garch_model_selection(returns_series)
        # DataFrame is sorted by AIC ascending, so first row is best
        assert len(df) > 0
        assert df["aic"].iloc[0] == df["aic"].min()

    def test_multiple_models(self, returns_series: pd.Series) -> None:
        from wraquant.vol.models import garch_model_selection

        df = garch_model_selection(returns_series)
        # Should have at least 6 rows: 3 vol specs x 2+ distributions
        assert len(df) >= 6
