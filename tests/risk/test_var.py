"""Tests for VaR and CVaR calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wraquant.risk.var import conditional_var, garch_var, value_at_risk


def _make_returns(n: int = 1000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.02, size=n), name="returns")


class TestValueAtRisk:
    def test_positive(self) -> None:
        ret = _make_returns()
        var = value_at_risk(ret, confidence=0.95)
        assert var > 0

    def test_higher_confidence_higher_var(self) -> None:
        ret = _make_returns()
        var_95 = value_at_risk(ret, confidence=0.95)
        var_99 = value_at_risk(ret, confidence=0.99)
        assert var_99 > var_95

    def test_parametric(self) -> None:
        ret = _make_returns()
        var = value_at_risk(ret, confidence=0.95, method="parametric")
        assert var > 0

    def test_historical_vs_parametric(self) -> None:
        ret = _make_returns()
        hist = value_at_risk(ret, confidence=0.95, method="historical")
        para = value_at_risk(ret, confidence=0.95, method="parametric")
        # Both should be in the same ballpark for normal data
        assert abs(hist - para) / hist < 0.5


class TestConditionalVar:
    def test_positive(self) -> None:
        ret = _make_returns()
        cvar = conditional_var(ret, confidence=0.95)
        assert cvar > 0

    def test_cvar_exceeds_var(self) -> None:
        ret = _make_returns()
        var = value_at_risk(ret, confidence=0.95)
        cvar = conditional_var(ret, confidence=0.95)
        assert cvar >= var

    def test_parametric(self) -> None:
        ret = _make_returns()
        cvar = conditional_var(ret, confidence=0.95, method="parametric")
        assert cvar > 0


class TestGarchVar:
    """Tests for GARCH-informed VaR."""

    def test_basic_shape(self) -> None:
        ret = _make_returns(n=500, seed=42)
        result = garch_var(ret, alpha=0.05, vol_model="GARCH", dist="normal")
        assert "var" in result
        assert "cvar" in result
        assert "conditional_vol" in result
        assert "breaches" in result
        assert "breach_rate" in result
        assert "garch_params" in result
        # VaR and CVaR should be Series with same length as conditional_vol
        assert len(result["var"]) == len(result["conditional_vol"])
        assert len(result["cvar"]) == len(result["conditional_vol"])

    def test_var_positive(self) -> None:
        ret = _make_returns(n=500, seed=42)
        result = garch_var(ret, alpha=0.05)
        # VaR values should be mostly positive (loss thresholds)
        assert float(result["var"].mean()) > 0

    def test_cvar_exceeds_var(self) -> None:
        ret = _make_returns(n=500, seed=42)
        result = garch_var(ret, alpha=0.05)
        # CVaR should on average be >= VaR
        assert float(result["cvar"].mean()) >= float(result["var"].mean())

    def test_breach_rate_reasonable(self) -> None:
        ret = _make_returns(n=1000, seed=42)
        result = garch_var(ret, alpha=0.05)
        # Breach rate should be in a reasonable range around alpha
        # We allow a wide tolerance since this is random data
        assert 0.0 <= result["breach_rate"] <= 0.30

    def test_invalid_vol_model_raises(self) -> None:
        ret = _make_returns(n=500, seed=42)
        with pytest.raises(ValueError, match="Unknown vol_model"):
            garch_var(ret, vol_model="INVALID")
