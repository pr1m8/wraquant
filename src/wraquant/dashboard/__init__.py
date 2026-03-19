"""Interactive Streamlit dashboard for wraquant.

Launch with:
    >>> from wraquant.dashboard import launch
    >>> launch()

Or from CLI:
    $ python -m wraquant.dashboard
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
