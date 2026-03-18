"""Financial visualizations for wraquant.

Provides publication-quality plots for returns analysis, portfolio
diagnostics, time series exploration, and risk reporting.  All
functions require the ``viz`` optional dependency group::

    pdm install -G viz
"""

from wraquant.viz.portfolio import (
    plot_correlation_matrix,
    plot_efficient_frontier,
    plot_risk_contribution,
    plot_weights,
)
from wraquant.viz.returns import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_monthly_heatmap,
    plot_return_distribution,
    plot_rolling_returns,
)
from wraquant.viz.risk import (
    plot_rolling_volatility,
    plot_tail_distribution,
    plot_var_backtest,
)
from wraquant.viz.themes import COLORS, apply_theme, set_wraquant_style
from wraquant.viz.timeseries import (
    plot_decomposition,
    plot_regime_overlay,
    plot_series,
)

__all__ = [
    # Themes
    "set_wraquant_style",
    "apply_theme",
    "COLORS",
    # Returns
    "plot_cumulative_returns",
    "plot_drawdowns",
    "plot_return_distribution",
    "plot_rolling_returns",
    "plot_monthly_heatmap",
    # Portfolio
    "plot_weights",
    "plot_efficient_frontier",
    "plot_risk_contribution",
    "plot_correlation_matrix",
    # Time series
    "plot_series",
    "plot_regime_overlay",
    "plot_decomposition",
    # Risk
    "plot_var_backtest",
    "plot_rolling_volatility",
    "plot_tail_distribution",
]
