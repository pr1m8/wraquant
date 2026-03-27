"""Tests for advanced pricing integrations."""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta

import numpy as np
import pytest

_has_quantlib = importlib.util.find_spec("QuantLib") is not None
_has_financepy = importlib.util.find_spec("financepy") is not None
_has_rateslib = importlib.util.find_spec("rateslib") is not None
_has_sdeint = importlib.util.find_spec("sdeint") is not None
try:
    from py_vollib.black_scholes.implied_volatility import implied_volatility as _iv  # noqa: F401
    _has_vollib = True
except Exception:
    _has_vollib = False


class TestQuantlibBond:
    @pytest.mark.skipif(not _has_quantlib, reason="QuantLib not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.price.integrations import quantlib_bond

        today = date.today()
        maturity = today + timedelta(days=365 * 5)
        yield_curve = [
            (today, 0.03),
            (today + timedelta(days=365), 0.035),
            (today + timedelta(days=365 * 2), 0.04),
            (today + timedelta(days=365 * 5), 0.045),
        ]
        result = quantlib_bond(
            face=1000.0,
            coupon=0.05,
            maturity=maturity,
            yield_curve=yield_curve,
            settlement=today,
        )
        assert "clean_price" in result
        assert "dirty_price" in result
        assert "ytm" in result
        assert "duration" in result
        assert "convexity" in result

    @pytest.mark.skipif(not _has_quantlib, reason="QuantLib not installed")
    def test_price_is_positive(self) -> None:
        from wraquant.price.integrations import quantlib_bond

        today = date.today()
        maturity = today + timedelta(days=365 * 5)
        yield_curve = [
            (today, 0.03),
            (maturity, 0.04),
        ]
        result = quantlib_bond(
            face=1000.0,
            coupon=0.05,
            maturity=maturity,
            yield_curve=yield_curve,
            settlement=today,
        )
        assert result["clean_price"] > 0
        assert result["dirty_price"] > 0


class TestQuantlibOption:
    @pytest.mark.skipif(not _has_quantlib, reason="QuantLib not installed")
    def test_call_price_positive(self) -> None:
        from wraquant.price.integrations import quantlib_option

        result = quantlib_option(
            spot=100.0, strike=100.0, vol=0.2, rf=0.05, maturity=1.0, option_type="call"
        )
        assert result["price"] > 0
        assert "delta" in result
        assert "gamma" in result
        assert "theta" in result
        assert "vega" in result
        assert "rho" in result

    @pytest.mark.skipif(not _has_quantlib, reason="QuantLib not installed")
    def test_put_price_positive(self) -> None:
        from wraquant.price.integrations import quantlib_option

        result = quantlib_option(
            spot=100.0, strike=100.0, vol=0.2, rf=0.05, maturity=1.0, option_type="put"
        )
        assert result["price"] > 0

    @pytest.mark.skipif(not _has_quantlib, reason="QuantLib not installed")
    def test_call_delta_positive(self) -> None:
        from wraquant.price.integrations import quantlib_option

        result = quantlib_option(
            spot=100.0, strike=100.0, vol=0.2, rf=0.05, maturity=1.0, option_type="call"
        )
        assert result["delta"] > 0


class TestQuantlibYieldCurve:
    @pytest.mark.skipif(not _has_quantlib, reason="QuantLib not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.price.integrations import quantlib_yield_curve

        today = date.today()
        dates = [
            today,
            today + timedelta(days=90),
            today + timedelta(days=180),
            today + timedelta(days=365),
        ]
        rates = [0.02, 0.025, 0.03, 0.035]
        result = quantlib_yield_curve(dates, rates)
        assert "reference_date" in result
        assert "max_date" in result
        assert "discount_factors" in result
        assert "forward_rates" in result
        assert "curve" in result
        assert len(result["discount_factors"]) == 4


class TestFinancepyOption:
    @pytest.mark.skipif(not _has_financepy, reason="financepy not installed")
    def test_returns_call_and_put(self) -> None:
        from wraquant.price.integrations import financepy_option

        result = financepy_option(
            spot=100.0, strike=100.0, vol=0.2, rf=0.05, maturity=1.0
        )
        assert "call_price" in result
        assert "put_price" in result
        assert "call_delta" in result
        assert "put_delta" in result
        assert result["call_price"] > 0
        assert result["put_price"] > 0


class TestRateslibSwap:
    @pytest.mark.skipif(not _has_rateslib, reason="rateslib not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.price.integrations import rateslib_swap

        result = rateslib_swap(
            notional=1_000_000.0,
            fixed_rate=0.03,
            float_spread=10.0,
            maturity="5Y",
        )
        assert "npv" in result
        assert "fixed_rate" in result
        assert "float_spread" in result
        assert "maturity" in result
        assert "notional" in result
        assert isinstance(result["npv"], float)


class TestVollib:
    @pytest.mark.skipif(not _has_vollib, reason="py-vollib not installed")
    def test_implied_vol_positive(self) -> None:
        from wraquant.price.integrations import vollib_implied_vol

        result = vollib_implied_vol(
            price=10.0, spot=100.0, strike=100.0, rf=0.05, maturity=1.0, option_type="call"
        )
        assert result["implied_vol"] > 0
        assert result["option_type"] == "call"
        assert result["price"] == 10.0

    @pytest.mark.skipif(not _has_vollib, reason="py-vollib not installed")
    def test_put_implied_vol(self) -> None:
        from wraquant.price.integrations import vollib_implied_vol

        result = vollib_implied_vol(
            price=5.0, spot=100.0, strike=100.0, rf=0.05, maturity=1.0, option_type="put"
        )
        assert result["implied_vol"] > 0
        assert result["option_type"] == "put"


# ---------------------------------------------------------------------------
# sdeint SDE solver
# ---------------------------------------------------------------------------


class TestSdeintSolve:
    @pytest.mark.skipif(not _has_sdeint, reason="sdeint not installed")
    def test_returns_expected_keys(self) -> None:
        from wraquant.price.integrations import sdeint_solve

        result = sdeint_solve(
            drift_fn=lambda y, t: 0.05 * y,
            diffusion_fn=lambda y, t: 0.2 * y,
            y0=100.0,
            tspan=(0.0, 1.0),
            dt=0.01,
        )
        assert "times" in result
        assert "paths" in result
        assert "final_values" in result

    @pytest.mark.skipif(not _has_sdeint, reason="sdeint not installed")
    def test_paths_shape(self) -> None:
        from wraquant.price.integrations import sdeint_solve

        result = sdeint_solve(
            drift_fn=lambda y, t: 0.05 * y,
            diffusion_fn=lambda y, t: 0.2 * y,
            y0=100.0,
            tspan=(0.0, 1.0),
            dt=0.01,
        )
        assert result["paths"].shape[0] == len(result["times"])
        assert result["paths"].shape[1] == 1  # scalar SDE

    @pytest.mark.skipif(not _has_sdeint, reason="sdeint not installed")
    def test_sri2_method(self) -> None:
        from wraquant.price.integrations import sdeint_solve

        result = sdeint_solve(
            drift_fn=lambda y, t: 0.05 * y,
            diffusion_fn=lambda y, t: 0.2 * y,
            y0=100.0,
            tspan=(0.0, 0.5),
            dt=0.01,
            method="sri2",
        )
        assert result["final_values"].shape == (1,)

    @pytest.mark.skipif(not _has_sdeint, reason="sdeint not installed")
    def test_unknown_method_raises(self) -> None:
        from wraquant.price.integrations import sdeint_solve

        with pytest.raises(ValueError, match="Unknown method"):
            sdeint_solve(
                drift_fn=lambda y, t: y,
                diffusion_fn=lambda y, t: y,
                y0=1.0,
                tspan=(0.0, 0.1),
                method="invalid",
            )

    @pytest.mark.skipif(not _has_sdeint, reason="sdeint not installed")
    def test_final_value_positive_for_gbm(self) -> None:
        from wraquant.price.integrations import sdeint_solve

        # GBM stays positive (almost surely)
        result = sdeint_solve(
            drift_fn=lambda y, t: 0.1 * y,
            diffusion_fn=lambda y, t: 0.01 * y,  # low vol for reliability
            y0=100.0,
            tspan=(0.0, 0.5),
            dt=0.01,
        )
        assert result["final_values"][0] > 0
