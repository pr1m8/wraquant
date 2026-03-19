"""Financial visualizations for wraquant.

Provides publication-quality plots for returns analysis, portfolio
diagnostics, time series exploration, and risk reporting.  All
functions require the ``viz`` optional dependency group::

    pdm install -G viz

Interactive Plotly-based visualizations are available alongside the
original matplotlib charts.  Rich multi-panel dashboards and standalone
chart functions offer comprehensive analysis views with dark-themed
Plotly figures.
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
from wraquant.viz.charts import (
    plot_backtest_tearsheet,
    plot_correlation_network,
    plot_distribution_analysis,
    plot_multi_asset,
    plot_regime_overlay as plot_regime_overlay_probs,
    plot_vol_surface,
)
from wraquant.viz.dashboard import (
    portfolio_dashboard,
    regime_dashboard,
    risk_dashboard,
    technical_dashboard,
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

def auto_plot(data, kind=None, **kwargs):
    """Automatically choose the best visualization for the data.

    Detects the data type and calls the appropriate viz function,
    bridging ``viz`` with ``risk``, ``regimes``, and ``backtest``
    without requiring the caller to know which plot function to use.

    Detection logic:

    - **pd.Series of returns** (values in [-1, 1] range, name contains
      'return' or 'ret') -- distribution plot + cumulative returns.
    - **pd.DataFrame of returns** -- correlation heatmap + multi-asset
      cumulative returns.
    - **RegimeResult** (from ``wraquant.regimes.base``) -- regime
      overlay dashboard showing states, probabilities, and statistics.
    - **dict with 'equity_curve' key** (backtest result) -- backtest
      tearsheet with drawdowns, rolling Sharpe, etc.
    - **kind='distribution'** -- force distribution analysis.
    - **kind='correlation'** -- force correlation heatmap.

    Parameters:
        data: The data to visualize.  Can be a pd.Series, pd.DataFrame,
            RegimeResult, or backtest result dict.
        kind: Force a specific visualization type.  Options:
            ``'distribution'``, ``'correlation'``, ``'cumulative'``,
            ``'regime'``, ``'tearsheet'``.  If *None*, auto-detects
            from the data type.
        **kwargs: Additional keyword arguments forwarded to the
            underlying plot function.

    Returns:
        The plot object (matplotlib Figure or Plotly Figure) from the
        underlying visualization function.

    Example:
        >>> import pandas as pd, numpy as np
        >>> returns = pd.Series(np.random.randn(252) * 0.01, name='returns')
        >>> fig = auto_plot(returns)  # auto-detects returns, plots distribution

    See Also:
        plot_return_distribution: Returns distribution plot.
        plot_cumulative_returns: Cumulative returns chart.
        plotly_correlation_heatmap: Correlation heatmap.
        regime_dashboard: Regime analysis dashboard.
        plot_backtest_tearsheet: Backtest tearsheet.
    """
    import pandas as pd

    # Check for RegimeResult
    try:
        from wraquant.regimes.base import RegimeResult
        is_regime = isinstance(data, RegimeResult)
    except ImportError:
        is_regime = False

    # Explicit kind override
    if kind == "distribution":
        if isinstance(data, pd.DataFrame):
            return plot_distribution_analysis(data.iloc[:, 0], **kwargs)
        return plot_distribution_analysis(data, **kwargs)

    if kind == "correlation":
        if isinstance(data, pd.DataFrame):
            return plotly_correlation_heatmap(data, **kwargs)
        raise TypeError("Correlation heatmap requires a DataFrame")

    if kind == "cumulative":
        return plot_cumulative_returns(data, **kwargs)

    if kind == "regime" or is_regime:
        if is_regime:
            return regime_dashboard(data, **kwargs)
        raise TypeError("Regime dashboard requires a RegimeResult")

    if kind == "tearsheet":
        if isinstance(data, dict) and "equity_curve" in data:
            return plot_backtest_tearsheet(data, **kwargs)
        raise TypeError("Tearsheet requires a backtest result dict with 'equity_curve'")

    # Auto-detection
    if isinstance(data, dict) and "equity_curve" in data:
        return plot_backtest_tearsheet(data, **kwargs)

    if isinstance(data, pd.DataFrame):
        return plotly_correlation_heatmap(data, **kwargs)

    if isinstance(data, pd.Series):
        return plot_return_distribution(data, **kwargs)

    raise TypeError(
        f"Cannot auto-detect visualization for type {type(data).__name__}. "
        f"Pass kind= explicitly."
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
    # Dashboards (Plotly, dark theme)
    "portfolio_dashboard",
    "regime_dashboard",
    "risk_dashboard",
    "technical_dashboard",
    # Rich standalone charts (Plotly, dark theme)
    "plot_multi_asset",
    "plot_vol_surface",
    "plot_regime_overlay_probs",
    "plot_distribution_analysis",
    "plot_correlation_network",
    "plot_backtest_tearsheet",
    # Auto-detection
    "auto_plot",
]
