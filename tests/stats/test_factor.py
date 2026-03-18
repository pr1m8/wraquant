"""Tests for factor models and attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.stats.factor import (
    factor_attribution,
    fama_french_regression,
    information_coefficient,
    quantile_analysis,
)


def _make_factor_data(n: int = 252, seed: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    """Create synthetic asset returns and Fama-French-style factor returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)

    mkt = rng.normal(0.0004, 0.01, size=n)
    smb = rng.normal(0.0001, 0.005, size=n)
    hml = rng.normal(0.0001, 0.005, size=n)
    rf = np.full(n, 0.0001)

    # Asset return = alpha + beta_mkt * mkt + beta_smb * smb + noise
    returns = 0.0002 + 1.2 * mkt + 0.3 * smb + rng.normal(0, 0.005, size=n)

    factors_df = pd.DataFrame(
        {"Mkt-RF": mkt, "SMB": smb, "HML": hml, "RF": rf},
        index=dates,
    )
    return pd.Series(returns, index=dates, name="asset"), factors_df


class TestFamaFrenchRegression:
    def test_output_keys(self) -> None:
        returns, factors_df = _make_factor_data()
        result = fama_french_regression(returns, factors_df)
        assert "alpha" in result
        assert "betas" in result
        assert "t_stats" in result
        assert "p_values" in result
        assert "r_squared" in result

    def test_betas_are_dict(self) -> None:
        returns, factors_df = _make_factor_data()
        result = fama_french_regression(returns, factors_df)
        assert isinstance(result["betas"], dict)
        assert "Mkt-RF" in result["betas"]
        assert "SMB" in result["betas"]
        assert "HML" in result["betas"]

    def test_r_squared_valid(self) -> None:
        returns, factors_df = _make_factor_data()
        result = fama_french_regression(returns, factors_df)
        assert 0 <= result["r_squared"] <= 1

    def test_market_beta_close_to_true(self) -> None:
        returns, factors_df = _make_factor_data()
        result = fama_french_regression(returns, factors_df)
        # True beta_mkt = 1.2
        assert abs(result["betas"]["Mkt-RF"] - 1.2) < 0.3

    def test_without_rf_column(self) -> None:
        returns, factors_df = _make_factor_data()
        # Drop RF column — returns treated as already excess
        factors_no_rf = factors_df.drop(columns=["RF"])
        result = fama_french_regression(returns, factors_no_rf)
        assert "alpha" in result
        assert "r_squared" in result


class TestFactorAttribution:
    def test_decomposition_sums_approximately(self) -> None:
        returns, factors_df = _make_factor_data()
        factor_returns = factors_df.drop(columns=["RF"])
        result = factor_attribution(returns, factor_returns)

        # Sum of factor contributions + specific return ~ total return
        total_from_factors = sum(result["factor_contributions"].values())
        reconstructed = total_from_factors + result["specific_return"]
        assert abs(reconstructed - result["total_return"]) < 0.01

    def test_output_keys(self) -> None:
        returns, factors_df = _make_factor_data()
        factor_returns = factors_df.drop(columns=["RF"])
        result = factor_attribution(returns, factor_returns)
        assert "factor_contributions" in result
        assert "specific_return" in result
        assert "total_return" in result
        assert "r_squared" in result

    def test_r_squared_valid(self) -> None:
        returns, factors_df = _make_factor_data()
        factor_returns = factors_df.drop(columns=["RF"])
        result = factor_attribution(returns, factor_returns)
        assert 0 <= result["r_squared"] <= 1


class TestInformationCoefficient:
    def test_bounds(self) -> None:
        rng = np.random.default_rng(42)
        predictions = pd.Series(rng.normal(0, 1, size=100))
        returns = pd.Series(rng.normal(0, 1, size=100))
        ic = information_coefficient(predictions, returns)
        assert -1 <= ic <= 1

    def test_perfect_positive_correlation(self) -> None:
        values = pd.Series(np.arange(100, dtype=float))
        ic = information_coefficient(values, values)
        assert abs(ic - 1.0) < 1e-10

    def test_perfect_negative_correlation(self) -> None:
        values = pd.Series(np.arange(100, dtype=float))
        ic = information_coefficient(values, -values)
        assert abs(ic - (-1.0)) < 1e-10

    def test_handles_nan(self) -> None:
        rng = np.random.default_rng(42)
        predictions = pd.Series(rng.normal(0, 1, size=100))
        returns = pd.Series(rng.normal(0, 1, size=100))
        predictions.iloc[0] = np.nan
        returns.iloc[5] = np.nan
        ic = information_coefficient(predictions, returns)
        assert -1 <= ic <= 1


class TestQuantileAnalysis:
    def test_correct_number_of_quantiles(self) -> None:
        rng = np.random.default_rng(42)
        predictions = pd.Series(rng.normal(0, 1, size=200))
        returns = pd.Series(rng.normal(0, 0.02, size=200))
        result = quantile_analysis(predictions, returns, n_quantiles=5)
        assert len(result) == 5

    def test_output_columns(self) -> None:
        rng = np.random.default_rng(42)
        predictions = pd.Series(rng.normal(0, 1, size=200))
        returns = pd.Series(rng.normal(0, 0.02, size=200))
        result = quantile_analysis(predictions, returns)
        assert "mean_return" in result.columns
        assert "std_return" in result.columns
        assert "hit_rate" in result.columns
        assert "count" in result.columns

    def test_counts_sum_to_total(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        predictions = pd.Series(rng.normal(0, 1, size=n))
        returns = pd.Series(rng.normal(0, 0.02, size=n))
        result = quantile_analysis(predictions, returns, n_quantiles=4)
        assert result["count"].sum() == n

    def test_hit_rate_between_0_and_1(self) -> None:
        rng = np.random.default_rng(42)
        predictions = pd.Series(rng.normal(0, 1, size=200))
        returns = pd.Series(rng.normal(0, 0.02, size=200))
        result = quantile_analysis(predictions, returns)
        assert (result["hit_rate"] >= 0).all()
        assert (result["hit_rate"] <= 1).all()

    def test_three_quantiles(self) -> None:
        rng = np.random.default_rng(42)
        predictions = pd.Series(rng.normal(0, 1, size=300))
        returns = pd.Series(rng.normal(0, 0.02, size=300))
        result = quantile_analysis(predictions, returns, n_quantiles=3)
        assert len(result) == 3
