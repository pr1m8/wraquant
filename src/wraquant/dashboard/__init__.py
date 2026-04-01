"""Interactive Streamlit dashboard for wraquant.

Provides a multi-page Streamlit web application for interactive
exploration of wraquant's analytical capabilities.  The dashboard
offers point-and-click access to experiment browsing, strategy
analysis, risk monitoring, regime detection, portfolio optimization,
and technical analysis screening -- without writing any code.

Key features:

- **Experiment browser** -- Browse and compare results from
  ``wraquant.experiment`` runs, with parameter sensitivity heatmaps
  and performance comparison tables.
- **Strategy analysis** -- Run backtests interactively, view equity
  curves, drawdown charts, and performance metrics.
- **Risk monitoring** -- Real-time VaR, CVaR, and stress test
  dashboards for portfolio surveillance.
- **Regime detection** -- Fit HMM/GMM models to market data and
  visualize regime probabilities and transition matrices.
- **Portfolio optimization** -- Interactive efficient frontier, risk
  parity, and Black-Litterman allocation with constraint sliders.
- **Technical analysis** -- Screen assets with configurable indicator
  overlays and signal detection.

Launch with:

    >>> from wraquant.dashboard import launch
    >>> launch()

Or from the command line::

    $ python -m wraquant.dashboard

Use ``wraquant.dashboard`` for interactive exploration and presentation.
For programmatic visualization (embedding charts in notebooks or
reports), use ``wraquant.viz`` directly.
"""

from __future__ import annotations

__all__ = ["launch"]


def launch(port: int = 8501, **kwargs: object) -> None:
    """Launch the wraquant Streamlit dashboard.

    Starts a local Streamlit server serving the wraquant dashboard
    application.  The dashboard provides interactive pages for
    experiment browsing, strategy analysis, risk monitoring, regime
    detection, portfolio optimization, and technical analysis screening.

    Parameters:
        port: Port to run on (default 8501).
        **kwargs: Additional keyword arguments passed to
            ``subprocess.run`` (e.g. ``check=True``).
    """
    import subprocess
    import sys
    from pathlib import Path

    app_path = str(Path(__file__).parent / "app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.port", str(port)],
        **kwargs,  # type: ignore[arg-type]
    )
