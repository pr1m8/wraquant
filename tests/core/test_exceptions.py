"""Tests for wraquant.core.exceptions."""

from __future__ import annotations

import pytest

from wraquant.core.exceptions import (
    DataFetchError,
    MissingDependencyError,
    OptimizationError,
    ValidationError,
    WQError,
)


class TestExceptionHierarchy:
    def test_base_exception(self) -> None:
        with pytest.raises(WQError):
            raise WQError("test")

    def test_missing_dep_is_import_error(self) -> None:
        err = MissingDependencyError("yfinance", "market-data")
        assert isinstance(err, ImportError)
        assert isinstance(err, WQError)
        assert "yfinance" in str(err)
        assert "pdm install -G market-data" in str(err)

    def test_data_fetch_error(self) -> None:
        err = DataFetchError("yahoo", "AAPL", "rate limited")
        assert "AAPL" in str(err)
        assert "yahoo" in str(err)
        assert "rate limited" in str(err)

    def test_validation_error_is_value_error(self) -> None:
        err = ValidationError("bad input")
        assert isinstance(err, ValueError)
        assert isinstance(err, WQError)

    def test_optimization_error(self) -> None:
        err = OptimizationError("OSQP", "infeasible")
        assert "OSQP" in str(err)
        assert "infeasible" in str(err)
