"""Reusable dashboard UI components.

Provides chart helpers, metric display cards, and common sidebar
elements used across multiple dashboard pages.
"""

from __future__ import annotations

from wraquant.dashboard.components.charts import (
    COLORS,
    SERIES_COLORS,
    dark_layout,
    equity_curve_chart,
    histogram_chart,
    line_chart,
)
from wraquant.dashboard.components.metrics import (
    fmt_currency,
    fmt_number,
    fmt_pct,
    metric_card,
    metrics_row,
)
from wraquant.dashboard.components.sidebar import (
    check_api_key,
    date_range_selector,
    file_uploader_returns,
    ticker_input,
)

__all__ = [
    "COLORS",
    "SERIES_COLORS",
    "check_api_key",
    "dark_layout",
    "date_range_selector",
    "equity_curve_chart",
    "file_uploader_returns",
    "fmt_currency",
    "fmt_number",
    "fmt_pct",
    "histogram_chart",
    "line_chart",
    "metric_card",
    "metrics_row",
    "ticker_input",
]
