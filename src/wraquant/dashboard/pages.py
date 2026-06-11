"""Compatibility modules for legacy dashboard page import paths.

The Streamlit app routes through :mod:`wraquant.dashboard.views`.  Older smoke
tests and integrations import ``wraquant.dashboard.pages.<name>`` though, so
this module exposes virtual submodules without creating a physical ``pages/``
directory that Streamlit treats as a multipage app.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable

_PAGE_TARGETS: dict[str, str] = {
    "experiment_browser": "wraquant.dashboard.views.quant_lab",
    "portfolio_optimizer": "wraquant.dashboard.views.portfolio_risk",
    "regime_viewer": "wraquant.dashboard.views.regime_analysis",
    "risk_monitor": "wraquant.dashboard.views.risk_regimes",
    "strategy_analysis": "wraquant.dashboard.views.backtest_lab",
    "ta_screener": "wraquant.dashboard.views.technical_analysis",
}

_INDICATOR_REGISTRY: dict[str, tuple[str, str, str]] = {
    "SMA": ("wraquant.ta.overlap", "sma", "close"),
    "EMA": ("wraquant.ta.overlap", "ema", "close"),
    "WMA": ("wraquant.ta.overlap", "wma", "close"),
    "RSI": ("wraquant.ta.momentum", "rsi", "close"),
    "MACD": ("wraquant.ta.momentum", "macd", "close"),
    "ROC": ("wraquant.ta.momentum", "roc", "close"),
    "ATR": ("wraquant.ta.volatility", "atr", "hlc"),
    "Bollinger Bands": ("wraquant.ta.overlap", "bollinger_bands", "close"),
    "Donchian Channel": ("wraquant.ta.overlap", "donchian_channel", "hl"),
    "OBV": ("wraquant.ta.volume", "obv", "cv"),
    "VWAP": ("wraquant.ta.overlap", "vwap", "hlcv"),
    "ADX": ("wraquant.ta.trend", "adx", "hlc"),
    "CCI": ("wraquant.ta.momentum", "cci", "hlc"),
    "Williams R": ("wraquant.ta.momentum", "williams_r", "hlc"),
    "Aroon": ("wraquant.ta.trend", "aroon", "hl"),
}

__path__: list[str] = []
__all__ = list(_PAGE_TARGETS)


def _render_for(target: str) -> Callable[[], None]:
    def render() -> None:
        """Render the dashboard page."""
        from importlib import import_module

        import_module(target).render()

    return render


def _register_virtual_page(name: str, target: str) -> None:
    module_name = f"{__name__}.{name}"
    module = types.ModuleType(module_name)
    module.__doc__ = f"Compatibility wrapper for {target}."
    module.render = _render_for(target)  # type: ignore[attr-defined]
    module.__all__ = ["render"]  # type: ignore[attr-defined]

    if name == "ta_screener":
        module._INDICATOR_REGISTRY = _INDICATOR_REGISTRY  # type: ignore[attr-defined]
        module.__all__ = ["_INDICATOR_REGISTRY", "render"]  # type: ignore[attr-defined]

    sys.modules[module_name] = module
    setattr(sys.modules[__name__], name, module)


for _name, _target in _PAGE_TARGETS.items():
    _register_virtual_page(_name, _target)

del _name, _target
