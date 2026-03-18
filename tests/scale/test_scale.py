"""Tests for wraquant.scale module."""

from __future__ import annotations

import pandas as pd
import pytest

from wraquant.scale import parallel_backtest


def _dummy_strategy(prices: pd.DataFrame, **params) -> dict:
    """Dummy strategy for testing parallel_backtest."""
    window = params.get("window", 10)
    return {
        "window": window,
        "mean_return": float(prices["close"].pct_change().dropna().mean()),
        "n_rows": len(prices),
    }


class TestParallelBacktest:
    """Tests for parallel_backtest with joblib backend (always available)."""

    @pytest.fixture()
    def sample_prices(self) -> pd.DataFrame:
        import numpy as np

        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({"close": close})

    def test_joblib_backend_basic(self, sample_prices):
        grid = [{"window": 5}, {"window": 10}, {"window": 20}]
        results = parallel_backtest(
            _dummy_strategy, grid, sample_prices, backend="joblib", n_jobs=1
        )
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert [r["window"] for r in results] == [5, 10, 20]

    def test_joblib_backend_single_param(self, sample_prices):
        grid = [{"window": 15}]
        results = parallel_backtest(
            _dummy_strategy, grid, sample_prices, backend="joblib", n_jobs=1
        )
        assert len(results) == 1
        assert results[0]["window"] == 15

    def test_joblib_backend_empty_grid(self, sample_prices):
        results = parallel_backtest(
            _dummy_strategy, [], sample_prices, backend="joblib", n_jobs=1
        )
        assert results == []

    def test_invalid_backend_raises(self, sample_prices):
        with pytest.raises(ValueError, match="Unknown backend"):
            parallel_backtest(
                _dummy_strategy, [{}], sample_prices, backend="invalid"
            )

    def test_results_contain_expected_keys(self, sample_prices):
        grid = [{"window": 10}]
        results = parallel_backtest(
            _dummy_strategy, grid, sample_prices, backend="joblib", n_jobs=1
        )
        assert "mean_return" in results[0]
        assert "n_rows" in results[0]
        assert results[0]["n_rows"] == 100


def _dask_available() -> bool:
    try:
        import dask  # noqa: F401
        from dask.distributed import Client  # noqa: F401

        return True
    except ImportError:
        return False


def _ray_available() -> bool:
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _dask_available(), reason="dask not installed")
class TestDaskMap:
    """Tests for dask_map (requires dask)."""

    def test_dask_map_import(self):
        from wraquant.scale import dask_map

        assert callable(dask_map)


@pytest.mark.skipif(not _ray_available(), reason="ray not installed")
class TestRayMap:
    """Tests for ray_map (requires ray)."""

    def test_ray_map_import(self):
        from wraquant.scale import ray_map

        assert callable(ray_map)
