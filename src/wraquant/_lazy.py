"""Lazy import infrastructure for optional dependencies.

Provides utilities to defer importing heavy optional packages until they
are actually accessed, and to check whether optional dependency groups
are installed.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


def lazy_import(module_name: str) -> Any:
    """Lazily import a module, deferring the actual import until first attribute access.

    Parameters:
        module_name: Fully qualified module name (e.g., 'yfinance').

    Returns:
        A lazy module proxy that imports on first access.

    Raises:
        ModuleNotFoundError: When the module is accessed but not installed.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return _MissingModule(module_name)

    loader = importlib.util.LazyLoader(spec.loader)  # type: ignore[arg-type]
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def is_available(module_name: str) -> bool:
    """Check if a module is installed without importing it.

    Parameters:
        module_name: Fully qualified module name.

    Returns:
        True if the module can be found, False otherwise.
    """
    return importlib.util.find_spec(module_name) is not None


class _MissingModule:
    """Proxy for a missing optional module that raises on any attribute access."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        raise ModuleNotFoundError(
            f"Optional dependency '{self._name}' is not installed. "
            f"Install it with: pdm install -G <group>"
        )

    def __repr__(self) -> str:
        return f"<MissingModule: {self._name}>"


# Mapping from optional dependency group names to their key module
_EXTRA_TO_MODULES: dict[str, list[str]] = {
    "accelerate": ["polars", "numba"],
    "symbolic": ["sympy"],
    "logging": ["loguru", "rich"],
    "ml": ["sklearn"],
    "market-data": ["yfinance", "fredapi", "nasdaqdatalink"],
    "timeseries": ["pmdarima", "arch", "ruptures", "darts"],
    "cleaning": ["janitor", "rapidfuzz"],
    "validation": ["pandera"],
    "etl": ["sqlalchemy", "connectorx"],
    "ingestion": ["httpx", "aiohttp"],
    "workflow": ["prefect", "dagster", "apscheduler"],
    "optimization": ["cvxpy", "pymoo"],
    "regimes": ["hmmlearn", "pykalman", "dynamax", "river"],
    "backtesting": ["vectorbt", "quantstats"],
    "risk": ["pypfopt", "riskfolio", "copulas", "copulae", "pyextremes"],
    "pricing": ["QuantLib", "financepy"],
    "stochastic": ["sdepy", "sdeint"],
    "viz": ["matplotlib", "plotly"],
    "bayes": ["pymc", "arviz", "emcee", "blackjax"],
    "quant-math": ["jax"],
    "scale": ["dask", "ray"],
    "causal": ["dowhy", "econml"],
}


def check_extra(group: str) -> bool:
    """Check if all key modules for an optional dependency group are available.

    Parameters:
        group: Name of the optional dependency group (e.g., 'market-data').

    Returns:
        True if the key modules for the group are installed.
    """
    modules = _EXTRA_TO_MODULES.get(group)
    if modules is None:
        return False
    return any(is_available(m) for m in modules)
