"""Tests for extended stress testing functions (correlation_stress, liquidity_stress, scenario_library)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.risk.stress import (
    correlation_stress,
    liquidity_stress,
    scenario_library,
)


@pytest.fixture()
def multi_asset_returns():
    """Generate synthetic multi-asset returns."""
    np.random.seed(42)
    n = 252
    return pd.DataFrame(
        {
            "SPY": np.random.normal(0.0005, 0.01, n),
            "TLT": np.random.normal(0.0002, 0.005, n),
            "GLD": np.random.normal(0.0003, 0.008, n),
        }
    )


class TestCorrelationStress:
    """Tests for correlation_stress."""

    def test_returns_dict(self, multi_asset_returns):
        result = correlation_stress(multi_asset_returns)
        assert "results" in result
        assert "base_vol" in result

    def test_shock_levels(self, multi_asset_returns):
        levels = [0.0, 0.5, 1.0]
        result = correlation_stress(multi_asset_returns, shock_levels=levels)
        assert len(result["results"]) == 3

    def test_higher_correlation_higher_vol(self, multi_asset_returns):
        result = correlation_stress(multi_asset_returns, shock_levels=[0.0, 0.5, 1.0])
        vol_0 = result["results"][0.0]["portfolio_vol"]
        vol_1 = result["results"][1.0]["portfolio_vol"]
        assert vol_1 >= vol_0

    def test_base_vol_matches_zero_shock(self, multi_asset_returns):
        result = correlation_stress(multi_asset_returns, shock_levels=[0.0, 0.5])
        assert result["base_vol"] == pytest.approx(
            result["results"][0.0]["portfolio_vol"], rel=1e-10
        )

    def test_perfect_correlation_max_vol(self, multi_asset_returns):
        result = correlation_stress(multi_asset_returns, shock_levels=[1.0])
        # Average correlation should be 1.0
        assert result["results"][1.0]["avg_correlation"] == pytest.approx(1.0, abs=0.01)


class TestLiquidityStress:
    """Tests for liquidity_stress."""

    def test_returns_dict(self, multi_asset_returns):
        result = liquidity_stress(multi_asset_returns)
        assert "total_cost" in result
        assert "total_cost_pct" in result
        assert "asset_costs" in result

    def test_positive_cost(self, multi_asset_returns):
        result = liquidity_stress(multi_asset_returns)
        assert result["total_cost"] > 0
        assert result["total_cost_pct"] > 0

    def test_with_haircuts(self, multi_asset_returns):
        haircuts = {"SPY": 0.001, "TLT": 0.002, "GLD": 0.005}
        result = liquidity_stress(
            multi_asset_returns,
            liquidity_haircuts=haircuts,
            portfolio_value=1_000_000,
        )
        assert result["total_cost"] > 0
        assert len(result["asset_costs"]) == 3

    def test_with_volumes(self, multi_asset_returns):
        np.random.seed(42)
        volumes = pd.DataFrame(
            {
                "SPY": np.random.uniform(50_000_000, 100_000_000, 252),
                "TLT": np.random.uniform(10_000_000, 30_000_000, 252),
                "GLD": np.random.uniform(5_000_000, 15_000_000, 252),
            },
            index=multi_asset_returns.index,
        )
        result = liquidity_stress(
            multi_asset_returns,
            volumes=volumes,
            portfolio_value=1_000_000,
        )
        assert result["total_cost"] > 0
        assert "days_to_liquidate" in result

    def test_portfolio_value_scales(self, multi_asset_returns):
        r1 = liquidity_stress(multi_asset_returns, portfolio_value=1_000_000)
        r2 = liquidity_stress(multi_asset_returns, portfolio_value=2_000_000)
        assert r2["total_cost"] > r1["total_cost"]


class TestScenarioLibrary:
    """Tests for scenario_library."""

    def test_returns_dict(self, multi_asset_returns):
        result = scenario_library(multi_asset_returns)
        assert "scenario_results" in result
        assert "available_scenarios" in result

    def test_all_scenarios(self, multi_asset_returns):
        result = scenario_library(multi_asset_returns)
        assert len(result["scenario_results"]) == len(result["available_scenarios"])

    def test_specific_scenarios(self, multi_asset_returns):
        result = scenario_library(
            multi_asset_returns,
            scenarios=["gfc_2008", "covid_2020"],
        )
        assert "gfc_2008" in result["scenario_results"]
        assert "covid_2020" in result["scenario_results"]
        assert len(result["scenario_results"]) == 2

    def test_scenario_params(self, multi_asset_returns):
        result = scenario_library(
            multi_asset_returns,
            scenarios=["gfc_2008"],
        )
        gfc = result["scenario_results"]["gfc_2008"]
        assert "stressed_portfolio_return" in gfc
        assert "stressed_vol" in gfc
        assert "scenario_params" in gfc

    def test_stressed_vol_higher(self, multi_asset_returns):
        """Scenarios with vol_multiplier > 1 should have higher stressed vol."""
        result = scenario_library(
            multi_asset_returns,
            scenarios=["gfc_2008"],
        )
        # GFC has vol_multiplier=3.0, so stressed vol should be higher
        base_vol = float(
            np.sqrt(np.ones(3) / 3 @ multi_asset_returns.cov().values @ np.ones(3) / 3)
        )
        assert result["scenario_results"]["gfc_2008"]["stressed_vol"] > base_vol

    def test_available_scenarios_list(self, multi_asset_returns):
        result = scenario_library(multi_asset_returns)
        expected = [
            "gfc_2008",
            "covid_2020",
            "dot_com_2000",
            "rate_hike_2022",
            "stagflation",
            "flash_crash",
            "em_crisis",
        ]
        for s in expected:
            assert s in result["available_scenarios"]

    def test_unknown_scenario_skipped(self, multi_asset_returns):
        result = scenario_library(
            multi_asset_returns,
            scenarios=["gfc_2008", "nonexistent"],
        )
        assert "nonexistent" not in result["scenario_results"]
        assert len(result["scenario_results"]) == 1
