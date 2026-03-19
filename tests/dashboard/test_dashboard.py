"""Smoke tests for the wraquant dashboard module.

These tests verify importability and basic API presence without
requiring Streamlit to be installed.  When Streamlit is available,
additional checks validate that page render functions exist.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Core importability
# ---------------------------------------------------------------------------


class TestLaunchFunction:
    """Tests for the ``launch()`` entry-point."""

    def test_launch_importable(self):
        from wraquant.dashboard import launch

        assert callable(launch)

    def test_launch_in_all(self):
        import wraquant.dashboard as mod

        assert "launch" in mod.__all__

    def test_main_module_exists(self):
        """``__main__.py`` should be importable for ``python -m`` usage."""
        import importlib

        spec = importlib.util.find_spec("wraquant.dashboard.__main__")
        assert spec is not None


# ---------------------------------------------------------------------------
# Component imports
# ---------------------------------------------------------------------------


class TestComponents:
    """Verify that component modules are importable."""

    def test_import_charts(self):
        from wraquant.dashboard.components import charts

        assert hasattr(charts, "line_chart")
        assert hasattr(charts, "equity_curve_chart")
        assert hasattr(charts, "histogram_chart")

    def test_import_metrics(self):
        from wraquant.dashboard.components import metrics

        assert hasattr(metrics, "metric_card")
        assert hasattr(metrics, "metrics_row")

    def test_import_sidebar(self):
        from wraquant.dashboard.components import sidebar

        assert hasattr(sidebar, "date_range_selector")
        assert hasattr(sidebar, "file_uploader_returns")

    def test_components_init_exports(self):
        from wraquant.dashboard.components import __all__ as exports

        expected = {
            "equity_curve_chart",
            "histogram_chart",
            "line_chart",
            "metric_card",
            "metrics_row",
            "date_range_selector",
            "file_uploader_returns",
        }
        assert expected.issubset(set(exports))


# ---------------------------------------------------------------------------
# Page modules
# ---------------------------------------------------------------------------


class TestPages:
    """Verify that page modules expose a ``render()`` callable."""

    @pytest.mark.parametrize(
        "page_module",
        [
            "wraquant.dashboard.pages.experiment_browser",
            "wraquant.dashboard.pages.strategy_analysis",
            "wraquant.dashboard.pages.risk_monitor",
            "wraquant.dashboard.pages.regime_viewer",
            "wraquant.dashboard.pages.portfolio_optimizer",
            "wraquant.dashboard.pages.ta_screener",
        ],
    )
    def test_page_has_render(self, page_module: str):
        import importlib

        mod = importlib.import_module(page_module)
        assert hasattr(mod, "render")
        assert callable(mod.render)


# ---------------------------------------------------------------------------
# Chart components (no Streamlit needed)
# ---------------------------------------------------------------------------


class TestChartHelpers:
    """Test chart component functions that don't need Streamlit."""

    def test_dark_layout(self):
        from wraquant.dashboard.components.charts import _dark_layout

        layout = _dark_layout(title="Test")
        assert layout["template"] == "plotly_dark"
        assert layout["title"] == "Test"

    def test_line_chart_without_plotly(self):
        """Should return None gracefully when plotly is missing."""
        # This test runs regardless -- if plotly IS installed it returns
        # a Figure; if not, it returns None. Both are acceptable.
        import pandas as pd

        from wraquant.dashboard.components.charts import line_chart

        s = pd.Series([1, 2, 3], name="test")
        result = line_chart(s, title="Test")
        # Either a Figure or None is fine
        assert result is None or hasattr(result, "update_layout")


# ---------------------------------------------------------------------------
# TA Screener indicator registry
# ---------------------------------------------------------------------------


class TestTAScreenerRegistry:
    """Verify the indicator registry is well-formed."""

    def test_registry_not_empty(self):
        from wraquant.dashboard.pages.ta_screener import _INDICATOR_REGISTRY

        assert len(_INDICATOR_REGISTRY) > 10

    def test_registry_entries_have_three_fields(self):
        from wraquant.dashboard.pages.ta_screener import _INDICATOR_REGISTRY

        for name, entry in _INDICATOR_REGISTRY.items():
            assert len(entry) == 3, f"{name} should have (module, func, input_type)"
            module_path, func_name, input_type = entry
            assert module_path.startswith("wraquant.ta.")
            assert isinstance(func_name, str)
            assert input_type in ("close", "ohlcv", "ohlc", "hlc", "hl", "cv", "hlcv", "hlv")
