"""Reusable dashboard UI components.

Provides chart helpers, metric display cards, and common sidebar
elements used across multiple dashboard pages.
"""

from __future__ import annotations

from wraquant.dashboard.components.charts import (
    equity_curve_chart,
    histogram_chart,
    line_chart,
)
from wraquant.dashboard.components.metrics import metric_card, metrics_row
from wraquant.dashboard.components.sidebar import (
    date_range_selector,
    file_uploader_returns,
)

__all__ = [
    "equity_curve_chart",
    "histogram_chart",
    "line_chart",
    "metric_card",
    "metrics_row",
    "date_range_selector",
    "file_uploader_returns",
]
