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


# ---------------------------------------------------------------------------
# Helper fixtures and functions for new tests
# ---------------------------------------------------------------------------

import numpy as np

from wraquant.scale import (
    chunk_apply,
    distributed_backtest,
    parallel_feature_compute,
    parallel_monte_carlo,
    parallel_optimize,
    parallel_walk_forward,
)


@pytest.fixture()
def multi_asset_returns() -> pd.DataFrame:
    """252 days of returns for 4 assets."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(252, 4) * 0.01,
        columns=["SPY", "TLT", "GLD", "VWO"],
    )


@pytest.fixture()
def sample_prices_close() -> pd.DataFrame:
    """100-row price DataFrame with a 'close' column."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame({"close": close})


# ---------------------------------------------------------------------------
# TestParallelOptimize
# ---------------------------------------------------------------------------


class TestParallelOptimize:
    """Tests for parallel_optimize with joblib backend."""

    def test_returns_correct_length(self, multi_asset_returns):
        constraint_sets = [
            {},
            {},
            {},
        ]
        results = parallel_optimize(
            multi_asset_returns,
            constraint_sets,
            method="equal_weight",
            backend="joblib",
            n_jobs=1,
        )
        assert len(results) == 3

    def test_each_result_has_weights(self, multi_asset_returns):
        constraint_sets = [{}]
        results = parallel_optimize(
            multi_asset_returns,
            constraint_sets,
            method="equal_weight",
            backend="joblib",
            n_jobs=1,
        )
        assert len(results) == 1
        r = results[0]
        assert "weights" in r
        assert isinstance(r["weights"], np.ndarray)
        assert r["error"] is None

    def test_unknown_method_raises(self, multi_asset_returns):
        with pytest.raises(ValueError, match="Unknown method"):
            parallel_optimize(
                multi_asset_returns, [{}], method="bogus", backend="joblib"
            )

    def test_empty_constraint_sets_raises(self, multi_asset_returns):
        with pytest.raises(ValueError, match="non-empty"):
            parallel_optimize(multi_asset_returns, [], method="equal_weight")

    def test_multiple_constraint_sets(self, multi_asset_returns):
        constraint_sets = [{}, {}, {}, {}, {}]
        results = parallel_optimize(
            multi_asset_returns,
            constraint_sets,
            method="equal_weight",
            backend="joblib",
            n_jobs=1,
        )
        assert len(results) == 5
        for r in results:
            assert r["error"] is None
            assert len(r["weights"]) == 4  # 4 assets


# ---------------------------------------------------------------------------
# TestParallelWalkForward
# ---------------------------------------------------------------------------


class TestParallelWalkForward:
    """Tests for parallel_walk_forward with joblib backend."""

    @staticmethod
    def _mean_model(train: pd.DataFrame, test: pd.DataFrame) -> dict:
        """Predict mean of training set for each test row."""
        mean_val = train.iloc[:, 0].mean()
        preds = np.full(len(test), mean_val)
        return {"predictions": preds, "metrics": {"rmse": 0.01}}

    def test_predictions_shape_matches(self):
        np.random.seed(42)
        data = pd.DataFrame({"value": np.random.randn(100)})
        n_windows = 4
        result = parallel_walk_forward(
            self._mean_model, data, n_windows=n_windows, backend="joblib", n_jobs=1
        )
        # Total OOS predictions should equal total test rows
        assert len(result["predictions"]) == len(result["actuals"])
        assert len(result["predictions"]) > 0

    def test_window_count_matches(self):
        np.random.seed(42)
        data = pd.DataFrame({"value": np.random.randn(200)})
        n_windows = 5
        result = parallel_walk_forward(
            self._mean_model, data, n_windows=n_windows, backend="joblib", n_jobs=1
        )
        assert len(result["metrics_per_window"]) == n_windows
        assert len(result["window_indices"]) == n_windows

    def test_metrics_per_window_present(self):
        np.random.seed(42)
        data = pd.DataFrame({"value": np.random.randn(100)})
        result = parallel_walk_forward(
            self._mean_model, data, n_windows=3, backend="joblib", n_jobs=1
        )
        for m in result["metrics_per_window"]:
            assert "rmse" in m

    def test_too_small_data_raises(self):
        data = pd.DataFrame({"value": [1.0]})
        with pytest.raises(ValueError, match="too small"):
            parallel_walk_forward(
                self._mean_model, data, n_windows=5, backend="joblib"
            )

    def test_single_window(self):
        np.random.seed(42)
        data = pd.DataFrame({"value": np.random.randn(50)})
        result = parallel_walk_forward(
            self._mean_model, data, n_windows=1, backend="joblib", n_jobs=1
        )
        assert len(result["metrics_per_window"]) == 1
        assert len(result["predictions"]) > 0


# ---------------------------------------------------------------------------
# TestParallelMonteCarlo
# ---------------------------------------------------------------------------


class TestParallelMonteCarlo:
    """Tests for parallel_monte_carlo with joblib backend."""

    @staticmethod
    def _simple_sim(n: int) -> np.ndarray:
        """Return (n, 10) random array simulating paths."""
        return np.random.randn(n, 10)

    def test_result_length_equals_n_simulations(self):
        n_sims = 100
        result = parallel_monte_carlo(
            self._simple_sim,
            n_simulations=n_sims,
            n_workers=2,
            backend="joblib",
        )
        assert result.shape[0] == n_sims

    def test_result_shape_second_axis(self):
        result = parallel_monte_carlo(
            self._simple_sim,
            n_simulations=50,
            n_workers=2,
            backend="joblib",
        )
        assert result.shape == (50, 10)

    def test_single_simulation(self):
        result = parallel_monte_carlo(
            self._simple_sim,
            n_simulations=1,
            n_workers=1,
            backend="joblib",
        )
        assert result.shape[0] == 1

    def test_more_workers_than_sims(self):
        result = parallel_monte_carlo(
            self._simple_sim,
            n_simulations=3,
            n_workers=10,
            backend="joblib",
        )
        assert result.shape[0] == 3

    def test_invalid_n_simulations_raises(self):
        with pytest.raises(ValueError, match="n_simulations must be >= 1"):
            parallel_monte_carlo(self._simple_sim, n_simulations=0)

    def test_large_batch(self):
        result = parallel_monte_carlo(
            self._simple_sim,
            n_simulations=1000,
            n_workers=4,
            backend="joblib",
        )
        assert result.shape[0] == 1000


# ---------------------------------------------------------------------------
# TestChunkApply
# ---------------------------------------------------------------------------


class TestChunkApply:
    """Tests for chunk_apply with joblib backend."""

    def test_output_matches_sequential(self):
        np.random.seed(42)
        df = pd.DataFrame({"x": np.arange(200), "y": np.random.randn(200)})

        def double_y(chunk: pd.DataFrame) -> pd.DataFrame:
            chunk = chunk.copy()
            chunk["y"] = chunk["y"] * 2
            return chunk

        sequential = double_y(df)
        parallel = chunk_apply(df, double_y, n_chunks=4, backend="joblib", n_jobs=1)

        assert len(parallel) == len(df)
        np.testing.assert_array_almost_equal(
            parallel["y"].values, sequential["y"].values
        )

    def test_row_based_preserves_length(self):
        np.random.seed(42)
        df = pd.DataFrame({"a": np.arange(500)})

        result = chunk_apply(
            df, lambda c: c, n_chunks=8, backend="joblib", n_jobs=1
        )
        assert len(result) == 500

    def test_group_based_chunking(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "symbol": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            "price": np.random.randn(150),
        })

        def add_rank(chunk: pd.DataFrame) -> pd.DataFrame:
            chunk = chunk.copy()
            chunk["rank"] = range(len(chunk))
            return chunk

        result = chunk_apply(
            df, add_rank, by="symbol", backend="joblib", n_jobs=1
        )
        assert len(result) == 150
        assert "rank" in result.columns

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            chunk_apply(pd.DataFrame(), lambda c: c, n_chunks=2)

    def test_single_chunk(self):
        df = pd.DataFrame({"v": [1, 2, 3]})
        result = chunk_apply(
            df, lambda c: c, n_chunks=1, backend="joblib", n_jobs=1
        )
        assert len(result) == 3


# ---------------------------------------------------------------------------
# TestParallelFeatureCompute
# ---------------------------------------------------------------------------


class TestParallelFeatureCompute:
    """Tests for parallel_feature_compute with joblib backend."""

    def test_returns_dict_per_asset(self):
        np.random.seed(42)
        prices = pd.DataFrame(
            np.cumsum(np.random.randn(100, 3), axis=0) + 100,
            columns=["A", "B", "C"],
        )

        def feats(s: pd.Series) -> pd.DataFrame:
            return pd.DataFrame({"sma": s.rolling(5).mean()})

        result = parallel_feature_compute(
            prices, feats, backend="joblib", n_jobs=1
        )
        assert sorted(result.keys()) == ["A", "B", "C"]
        for v in result.values():
            assert isinstance(v, pd.DataFrame)
            assert "sma" in v.columns
            assert len(v) == 100


# ---------------------------------------------------------------------------
# TestDistributedBacktest
# ---------------------------------------------------------------------------


class TestDistributedBacktest:
    """Tests for distributed_backtest with joblib backend."""

    @staticmethod
    def _strat(prices: pd.DataFrame, window: int = 10) -> dict:
        ret = prices["close"].pct_change().dropna().rolling(window).mean().iloc[-1]
        return {"sharpe": float(ret / 0.01), "mean_ret": float(ret)}

    def test_results_df_shape(self, sample_prices_close):
        grid = [{"window": w} for w in [5, 10, 20]]
        out = distributed_backtest(
            self._strat, grid, sample_prices_close, backend="joblib", n_jobs=1
        )
        assert isinstance(out["results_df"], pd.DataFrame)
        assert len(out["results_df"]) == 3
        assert out["n_succeeded"] == 3
        assert out["n_failed"] == 0

    def test_best_params_present(self, sample_prices_close):
        grid = [{"window": w} for w in [5, 10, 20]]
        out = distributed_backtest(
            self._strat, grid, sample_prices_close, backend="joblib", n_jobs=1
        )
        assert "window" in out["best_params"]
        assert "sharpe" in out["best_metrics"]

    def test_auto_backend_selects_joblib(self, sample_prices_close):
        """With a small grid and no dask/ray preference, should use joblib."""
        grid = [{"window": 10}]
        out = distributed_backtest(
            self._strat, grid, sample_prices_close, backend=None, n_jobs=1
        )
        assert out["n_succeeded"] == 1
