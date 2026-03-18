"""Tests for wraquant.flow module."""

from __future__ import annotations

import pytest

from wraquant.flow import Pipeline, pipeline


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
