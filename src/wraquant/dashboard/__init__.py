"""Interactive Streamlit dashboard for wraquant.

Provides a multi-page Streamlit web application for interactive
exploration of wraquant's analytical capabilities.  The dashboard
offers point-and-click access to fundamental analysis, valuation
models, technical analysis, risk monitoring, news sentiment,
and stock screening -- without writing any code.

Key features:

- **Company overview** -- Profile, key metrics, and sentiment snapshot.
- **Fundamental analysis** -- Income, balance sheet, and cash flow
  trends with financial health scoring and DuPont decomposition.
- **Valuation** -- DCF with adjustable inputs, Graham Number, Peter
  Lynch value, relative valuation vs peers, margin of safety.
- **Technical analysis** -- Interactive candlestick charts with SMA,
  EMA, Bollinger Bands overlays, RSI and MACD subplots, TA summary.
- **Risk & regimes** -- Risk metrics (Sharpe, Sortino, VaR, CVaR),
  drawdown analysis, regime detection, rolling volatility.
- **News & events** -- Sentiment-scored headlines, earnings surprise
  history, insider trading activity, institutional ownership.
- **Screener** -- Value, Growth, Quality, Piotroski, and Magic Formula
  preset screens with custom criteria builder.

Launch with:

    >>> from wraquant.dashboard import launch
    >>> launch()

Or from the command line::

    $ python -m wraquant.dashboard
    $ streamlit run src/wraquant/dashboard/app.py

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
    fundamental analysis, valuation, technical analysis, risk
    monitoring, news sentiment, and stock screening.

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
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            app_path,
            "--server.port",
            str(port),
        ],
        **kwargs,  # type: ignore[arg-type]
    )
