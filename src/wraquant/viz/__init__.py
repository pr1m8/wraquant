"""Financial visualizations for wraquant.

Provides publication-quality plots for returns analysis, portfolio
diagnostics, time series exploration, and risk reporting.  All
functions require the ``viz`` optional dependency group::

    pdm install -G viz

Interactive Plotly-based visualizations are available alongside the
original matplotlib charts.
"""

from wraquant.viz.advanced import (
    plotly_copula_scatter,
    plotly_network_graph,
    plotly_radar,
    plotly_regime_overlay,
    plotly_sankey_flow,
    plotly_term_structure,
    plotly_treemap,
    plotly_vol_surface,
)
from wraquant.viz.candlestick import (
    plotly_candlestick,
    plotly_heikin_ashi,
    plotly_market_profile,
    plotly_renko,
)
from wraquant.viz.interactive import (
    plotly_correlation_heatmap,
    plotly_distribution,
    plotly_drawdown,
    plotly_efficient_frontier,
    plotly_returns,
    plotly_risk_return_scatter,
    plotly_rolling_stats,
)
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
    # Returns (matplotlib)
    "plot_cumulative_returns",
    "plot_drawdowns",
    "plot_return_distribution",
    "plot_rolling_returns",
    "plot_monthly_heatmap",
    # Portfolio (matplotlib)
    "plot_weights",
    "plot_efficient_frontier",
    "plot_risk_contribution",
    "plot_correlation_matrix",
    # Time series (matplotlib)
    "plot_series",
    "plot_regime_overlay",
    "plot_decomposition",
    # Risk (matplotlib)
    "plot_var_backtest",
    "plot_rolling_volatility",
    "plot_tail_distribution",
    # Interactive (Plotly) — core charts
    "plotly_returns",
    "plotly_drawdown",
    "plotly_rolling_stats",
    "plotly_distribution",
    "plotly_correlation_heatmap",
    "plotly_efficient_frontier",
    "plotly_risk_return_scatter",
    # Interactive (Plotly) — advanced / wacky
    "plotly_regime_overlay",
    "plotly_vol_surface",
    "plotly_term_structure",
    "plotly_copula_scatter",
    "plotly_network_graph",
    "plotly_sankey_flow",
    "plotly_treemap",
    "plotly_radar",
    # Interactive (Plotly) — candlestick / OHLCV
    "plotly_candlestick",
    "plotly_market_profile",
    "plotly_renko",
    "plotly_heikin_ashi",
]
