"""Tests for wraquant.flow module."""

from __future__ import annotations

import logging
import tempfile
import time

import pytest

from wraquant.flow import (
    DAG,
    Pipeline,
    cache_result,
    dag,
    log_step,
    parallel_pipeline,
    pipeline,
    retry,
)


def _prefect_available() -> bool:
    try:
        import prefect  # noqa: F401

        return True
    except ImportError:
        return False


def _apscheduler_available() -> bool:
    try:
        import apscheduler  # noqa: F401

        return True
    except ImportError:
        return False


class TestPipeline:
    """Tests for the Pipeline class (no external deps needed)."""

    def test_pipeline_single_step(self):
        pipe = pipeline(lambda x: x * 2)
        assert pipe.run(5) == 10

    def test_pipeline_multiple_steps(self):
        pipe = pipeline(
            lambda x: x + 1,
            lambda x: x * 3,
            lambda x: x - 2,
        )
        # (5 + 1) * 3 - 2 = 16
        assert pipe.run(5) == 16

    def test_pipeline_with_dataframe(self):
        import pandas as pd

        prices = pd.Series([100, 102, 101, 105, 103], dtype=float)
        pipe = pipeline(
            lambda s: s.pct_change().dropna(),
            lambda r: {"mean": r.mean(), "std": r.std()},
        )
        result = pipe.run(prices)
        assert "mean" in result
        assert "std" in result

    def test_pipeline_rshift_callable(self):
        pipe = pipeline(lambda x: x + 1)
        extended = pipe >> (lambda x: x * 2)
        assert isinstance(extended, Pipeline)
        # (5 + 1) * 2 = 12
        assert extended.run(5) == 12

    def test_pipeline_rshift_pipeline(self):
        pipe1 = pipeline(lambda x: x + 1)
        pipe2 = pipeline(lambda x: x * 2)
        combined = pipe1 >> pipe2
        assert isinstance(combined, Pipeline)
        assert len(combined.steps) == 2
        # (5 + 1) * 2 = 12
        assert combined.run(5) == 12

    def test_pipeline_rshift_invalid(self):
        pipe = pipeline(lambda x: x + 1)
        with pytest.raises(TypeError, match="Cannot compose Pipeline"):
            pipe >> 42  # noqa: B018

    def test_pipeline_empty(self):
        pipe = pipeline()
        assert pipe.run(42) == 42

    def test_pipeline_steps_attribute(self):
        steps = [lambda x: x + 1, lambda x: x * 2]
        pipe = Pipeline(steps)
        assert pipe.steps is steps


class TestPipelineFactory:
    """Tests for the pipeline() factory function."""

    def test_returns_pipeline_instance(self):
        result = pipeline(lambda x: x)
        assert isinstance(result, Pipeline)

    def test_no_args_returns_empty_pipeline(self):
        result = pipeline()
        assert isinstance(result, Pipeline)
        assert len(result.steps) == 0


class TestDAG:
    """Tests for the DAG class."""

    def test_simple_dag(self):
        d = dag({
            "double": (lambda data: data * 2, []),
            "add_one": (lambda data: data + 1, ["double"]),
        })
        results = d.run(initial_data=5)
        assert results["double"] == 10
        assert results["add_one"] == 11

    def test_multiple_roots(self):
        d = dag({
            "a": (lambda data: data + 1, []),
            "b": (lambda data: data * 2, []),
            "c": (lambda data: data["a"] + data["b"], ["a", "b"]),
        })
        results = d.run(initial_data=5)
        assert results["a"] == 6
        assert results["b"] == 10
        assert results["c"] == 16

    def test_diamond_dag(self):
        d = dag({
            "root": (lambda data: data, []),
            "left": (lambda data: data + 10, ["root"]),
            "right": (lambda data: data * 3, ["root"]),
            "merge": (lambda data: data["left"] + data["right"], ["left", "right"]),
        })
        results = d.run(initial_data=5)
        assert results["root"] == 5
        assert results["left"] == 15
        assert results["right"] == 15
        assert results["merge"] == 30

    def test_cycle_detection(self):
        with pytest.raises(ValueError, match="cycle"):
            dag({
                "a": (lambda x: x, ["b"]),
                "b": (lambda x: x, ["a"]),
            })

    def test_missing_dependency(self):
        with pytest.raises(ValueError, match="undefined"):
            dag({
                "a": (lambda x: x, ["nonexistent"]),
            })

    def test_single_step(self):
        d = dag({"only": (lambda x: x * 10, [])})
        results = d.run(initial_data=3)
        assert results["only"] == 30

    def test_returns_all_results(self):
        d = dag({
            "a": (lambda x: x, []),
            "b": (lambda x: x + 1, ["a"]),
            "c": (lambda x: x + 2, ["b"]),
        })
        results = d.run(initial_data=0)
        assert set(results.keys()) == {"a", "b", "c"}

    def test_dag_is_dag_type(self):
        d = dag({"a": (lambda x: x, [])})
        assert isinstance(d, DAG)


class TestRetry:
    """Tests for the retry decorator."""

    def test_succeeds_first_try(self):
        @retry(max_retries=3, delay=0.01)
        def always_works():
            return 42

        assert always_works() == 42

    def test_retries_on_failure(self):
        call_count = [0]

        @retry(max_retries=3, delay=0.01)
        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("fail")
            return "ok"

        assert flaky() == "ok"
        assert call_count[0] == 3

    def test_raises_after_max_retries(self):
        @retry(max_retries=2, delay=0.01)
        def always_fails():
            raise ValueError("always")

        with pytest.raises(ValueError, match="always"):
            always_fails()

    def test_no_args_decorator(self):
        @retry
        def simple():
            return 1

        assert simple() == 1

    def test_specific_exceptions(self):
        call_count = [0]

        @retry(max_retries=3, delay=0.01, exceptions=(ConnectionError,))
        def specific_fail():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("temp")
            return "done"

        assert specific_fail() == "done"

    def test_non_matching_exception_not_retried(self):
        @retry(max_retries=3, delay=0.01, exceptions=(ConnectionError,))
        def wrong_exception():
            raise TypeError("wrong")

        with pytest.raises(TypeError):
            wrong_exception()


class TestCacheResult:
    """Tests for the cache_result decorator."""

    def test_caches_result(self):
        call_count = [0]
        tmp = tempfile.mkdtemp()

        @cache_result(cache_dir=tmp, ttl_hours=1)
        def expensive(x):
            call_count[0] += 1
            return x ** 2

        assert expensive(5) == 25
        assert expensive(5) == 25  # from cache
        assert call_count[0] == 1  # only called once

    def test_different_args_separate_cache(self):
        call_count = [0]
        tmp = tempfile.mkdtemp()

        @cache_result(cache_dir=tmp, ttl_hours=1)
        def add(x, y):
            call_count[0] += 1
            return x + y

        assert add(1, 2) == 3
        assert add(3, 4) == 7
        assert call_count[0] == 2

    def test_expired_cache_recomputes(self):
        call_count = [0]
        tmp = tempfile.mkdtemp()

        @cache_result(cache_dir=tmp, ttl_hours=0.0001)  # ~0.36 seconds
        def compute(x):
            call_count[0] += 1
            return x * 2

        assert compute(5) == 10
        time.sleep(0.5)  # wait for TTL to expire
        assert compute(5) == 10
        assert call_count[0] == 2

    def test_no_args_decorator(self):
        tmp = tempfile.mkdtemp()

        @cache_result(cache_dir=tmp)
        def simple():
            return 42

        assert simple() == 42


class TestLogStep:
    """Tests for the log_step decorator."""

    def test_returns_result(self):
        @log_step
        def compute(x):
            return x * 2

        assert compute(5) == 10

    def test_with_logger(self):
        logger = logging.getLogger("test.log_step")

        @log_step(logger=logger)
        def compute(x):
            return x + 1

        assert compute(5) == 6

    def test_exception_propagated(self):
        @log_step
        def fails():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            fails()

    def test_no_args_decorator(self):
        @log_step
        def simple():
            return 1

        assert simple() == 1


class TestParallelPipeline:
    """Tests for parallel_pipeline."""

    def test_basic_parallel(self):
        pipe1 = pipeline(lambda x: 10)
        pipe2 = pipeline(lambda x: 20)
        results = parallel_pipeline(pipe1, pipe2)
        assert sorted(results) == [10, 20]

    def test_order_preserved(self):
        pipes = [pipeline(lambda x, i=i: i) for i in range(5)]
        results = parallel_pipeline(*pipes)
        assert results == [0, 1, 2, 3, 4]

    def test_empty_input(self):
        results = parallel_pipeline()
        assert results == []

    def test_single_pipeline(self):
        pipe = pipeline(lambda x: 42)
        results = parallel_pipeline(pipe)
        assert results == [42]

    def test_max_workers(self):
        pipes = [pipeline(lambda x, i=i: i * 2) for i in range(3)]
        results = parallel_pipeline(*pipes, max_workers=2)
        assert results == [0, 2, 4]


@pytest.mark.skipif(
    not _prefect_available(),
    reason="prefect not installed",
)
class TestPrefectBacktestFlow:
    """Tests for prefect_backtest_flow (requires prefect)."""

    def test_prefect_flow_import(self):
        from wraquant.flow import prefect_backtest_flow

        assert callable(prefect_backtest_flow)


@pytest.mark.skipif(
    not _apscheduler_available(),
    reason="apscheduler not installed",
)
class TestScheduleDataRefresh:
    """Tests for schedule_data_refresh (requires apscheduler)."""

    def test_schedule_import(self):
        from wraquant.flow import schedule_data_refresh

        assert callable(schedule_data_refresh)
