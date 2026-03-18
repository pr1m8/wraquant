"""Tests for VaR and CVaR calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wraquant.risk.var import conditional_var, value_at_risk


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
