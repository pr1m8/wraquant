"""Composable workflow system for wraquant.

Build end-to-end quant workflows by chaining steps that automatically
pass data between wraquant modules.  Each step is a function that
receives a shared context dict and enriches it with new results.
Steps auto-chain so users never have to write manual glue code for
data alignment, type conversion, or module-to-module wiring.

Example:
    >>> from wraquant.compose import Workflow, steps
    >>> wf = (
    ...     Workflow("momentum_research")
    ...     .add(steps.returns())
    ...     .add(steps.regime_detect(n_regimes=2))
    ...     .add(steps.garch_vol())
    ...     .add(steps.risk_metrics())
    ... )
    >>> result = wf.run(prices)
    >>> print(result.risk)
    >>> print(result.regimes)

Pre-built workflows are available for common patterns::

    >>> from wraquant.compose import quick_analysis_workflow
    >>> result = quick_analysis_workflow().run(prices)

See Also:
    wraquant.flow: Lower-level Pipeline/DAG orchestration.
    wraquant.recipes: Monolithic analysis functions (pre-compose era).
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any, Callable

import numpy as np
import pandas as pd

__all__ = [
    "Workflow",
    "WorkflowResult",
    "steps",
    "quick_analysis_workflow",
    "risk_workflow",
    "ml_workflow",
    "portfolio_workflow",
]

_log = logging.getLogger("wraquant.compose")


def _get_returns(ctx: dict[str, Any]) -> pd.Series | None:
    """Get the best available return series from context.

    Prefers ``strategy_returns`` over ``returns``, using an explicit
    None check to avoid pandas truth-value ambiguity.
    """
    r = ctx.get("strategy_returns")
    if r is not None:
        return r
    return ctx.get("returns")


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------


class WorkflowResult:
    """Container for workflow output with attribute access.

    Wraps the raw context dict produced by a :class:`Workflow` run so
    that each computed output is accessible as an attribute.  Provides
    dict-like inspection (``keys``, ``to_dict``) and an auto-``plot``
    method.

    Parameters:
        context: The raw context dict from a workflow run.

    Example:
        >>> result = WorkflowResult({"sharpe": 1.2, "returns": [0.01, -0.02]})
        >>> result.sharpe
        1.2
        >>> "returns" in result.keys()
        True
    """

    def __init__(self, context: dict[str, Any]) -> None:
        self._ctx = context

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._ctx[name]
        except KeyError:
            raise AttributeError(
                f"WorkflowResult has no attribute {name!r}. "
                f"Available: {list(self._ctx.keys())}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._ctx

    def keys(self) -> list[str]:
        """Return the names of all computed outputs."""
        return list(self._ctx.keys())

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of all computed outputs."""
        return dict(self._ctx)

    def plot(self) -> Any:
        """Auto-plot the most relevant visualization.

        Chooses strategy returns if available, otherwise raw returns.
        Delegates to ``wraquant.viz.auto_plot``.
        """
        from wraquant.viz import auto_plot

        if "strategy_returns" in self._ctx:
            return auto_plot(self._ctx["strategy_returns"])
        if "returns" in self._ctx:
            return auto_plot(self._ctx["returns"])
        return None

    def __repr__(self) -> str:
        keys = list(self._ctx.keys())
        return f"WorkflowResult(keys={keys})"


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class Workflow:
    """Composable workflow that chains wraquant module steps.

    Each step receives the full context dict and can read/write to it.
    Steps execute in order, with each step enriching the context.
    Failed steps log a warning and continue (fail-soft) unless
    ``strict=True`` is passed to :meth:`run`.

    Parameters:
        name: Workflow name for logging and tracking.

    Example:
        >>> wf = (
        ...     Workflow("demo")
        ...     .add(steps.returns())
        ...     .add(steps.risk_metrics())
        ... )
        >>> result = wf.run(prices)
    """

    def __init__(self, name: str = "workflow") -> None:
        self.name = name
        self._steps: list[Callable[[dict], dict]] = []

    def add(self, step: Callable[[dict], dict]) -> Workflow:
        """Add a step to the workflow.  Returns *self* for chaining."""
        self._steps.append(step)
        return self

    def run(
        self,
        data: pd.Series | pd.DataFrame | np.ndarray | None = None,
        *,
        strict: bool = False,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute the workflow on input data.

        Parameters:
            data: Input data.  Accepted shapes:

                - ``pd.Series`` -- treated as a single-asset price series.
                - ``pd.DataFrame`` -- treated as multi-asset prices (or
                  OHLCV if columns include high/low/close).
                - ``np.ndarray`` -- wrapped in a pd.Series.
                - *None* -- useful when the context is pre-populated
                  via *kwargs*.
            strict: If True, raise on step failure instead of skipping.
            **kwargs: Additional context variables (e.g.,
                ``benchmark=bench_returns``).

        Returns:
            WorkflowResult with all computed outputs as attributes.
        """
        ctx: dict[str, Any] = {**kwargs}

        # Auto-detect data type and seed the context
        if data is not None:
            ctx["input_data"] = data
            if isinstance(data, pd.Series):
                ctx["prices"] = data
                ctx["is_multivariate"] = False
            elif isinstance(data, pd.DataFrame):
                ctx["prices_df"] = data
                ctx["is_multivariate"] = True
                # Detect OHLCV structure
                cols_lower = {c.lower() for c in data.columns}
                if {"high", "low", "close"}.issubset(cols_lower):
                    ctx["is_ohlcv"] = True
                    # Normalise column access
                    col_map = {c.lower(): c for c in data.columns}
                    ctx["close"] = data[col_map["close"]]
                    ctx["high"] = data[col_map["high"]]
                    ctx["low"] = data[col_map["low"]]
                    if "volume" in cols_lower:
                        ctx["volume"] = data[col_map["volume"]]
                    # Default prices to close
                    ctx["prices"] = ctx["close"]
            elif isinstance(data, np.ndarray):
                ctx["prices"] = pd.Series(data, name="prices")
                ctx["is_multivariate"] = False

        for step in self._steps:
            step_name = getattr(step, "__name__", repr(step))
            t0 = _time.perf_counter()
            try:
                ctx = step(ctx)
                elapsed = _time.perf_counter() - t0
                _log.debug("step %s completed in %.3fs", step_name, elapsed)
            except Exception:
                elapsed = _time.perf_counter() - t0
                _log.warning(
                    "step %s failed after %.3fs",
                    step_name,
                    elapsed,
                    exc_info=True,
                )
                if strict:
                    raise

        return WorkflowResult(ctx)

    def __repr__(self) -> str:
        step_names = [getattr(s, "__name__", str(s)) for s in self._steps]
        return f"Workflow({self.name!r}, steps={step_names})"


# ---------------------------------------------------------------------------
# Built-in steps
# ---------------------------------------------------------------------------


class steps:
    """Pre-built workflow steps that chain wraquant modules.

    Each static method returns a callable that takes and returns a
    context dict.  Steps read what they need from the context and
    write their results back.  Missing inputs are silently skipped
    so that workflows degrade gracefully.
    """

    @staticmethod
    def returns() -> Callable[[dict], dict]:
        """Compute returns from prices.

        Reads ``prices`` (Series) or ``prices_df`` (DataFrame) from
        the context and writes ``returns`` and optionally ``returns_df``.
        """

        def _step(ctx: dict) -> dict:
            if "prices" in ctx and "returns" not in ctx:
                ctx["returns"] = ctx["prices"].pct_change().dropna()
            if "prices_df" in ctx and "returns_df" not in ctx:
                ctx["returns_df"] = ctx["prices_df"].pct_change().dropna()
                # If no single-asset returns yet, take first column
                if "returns" not in ctx:
                    ctx["returns"] = ctx["returns_df"].iloc[:, 0]
            return ctx

        _step.__name__ = "returns"
        return _step

    @staticmethod
    def regime_detect(
        method: str = "hmm",
        n_regimes: int = 2,
        **kwargs: Any,
    ) -> Callable[[dict], dict]:
        """Detect market regimes.

        Reads ``returns`` from the context.  Writes ``regimes``
        (RegimeResult) and ``regime_stats``.
        """

        def _step(ctx: dict) -> dict:
            from wraquant.regimes.base import detect_regimes

            r = ctx.get("returns")
            if r is not None and len(r) >= 50:
                result = detect_regimes(
                    r.values, method=method, n_regimes=n_regimes, **kwargs
                )
                ctx["regimes"] = result
                ctx["regime_stats"] = result.statistics
            return ctx

        _step.__name__ = "regime_detect"
        return _step

    @staticmethod
    def garch_vol(
        model: str = "GARCH",
        dist: str = "t",
        **kwargs: Any,
    ) -> Callable[[dict], dict]:
        """Fit GARCH volatility model.

        Reads ``returns`` from the context.  Writes ``garch`` (result
        dict) and ``conditional_vol`` (pd.Series, in decimal form).
        Requires at least 200 observations.
        """

        def _step(ctx: dict) -> dict:
            from wraquant.vol.models import garch_fit

            r = ctx.get("returns")
            if r is not None and len(r) >= 200:
                # arch expects percentage returns
                result = garch_fit(r * 100, dist=dist, **kwargs)
                ctx["garch"] = result
                ctx["conditional_vol"] = result["conditional_volatility"] / 100
            return ctx

        _step.__name__ = "garch_vol"
        return _step

    @staticmethod
    def ta_features(
        indicators: list[str] | None = None,
    ) -> Callable[[dict], dict]:
        """Compute technical analysis features.

        Reads ``high``, ``low``, ``close``, and optionally ``volume``
        from the context (populated when OHLCV data is provided).
        Falls back to constructing synthetic high/low from ``prices``
        if OHLCV columns are not available.

        Writes ``ta_features`` (DataFrame).
        """

        def _step(ctx: dict) -> dict:
            from wraquant.ml.features import ta_features as _ta_features

            kw: dict[str, Any] = {}
            if indicators is not None:
                kw["include"] = indicators

            if ctx.get("is_ohlcv"):
                result = _ta_features(
                    high=ctx["high"],
                    low=ctx["low"],
                    close=ctx["close"],
                    volume=ctx.get("volume"),
                    **kw,
                )
                ctx["ta_features"] = result
            elif "prices" in ctx:
                # Synthesise high/low from close prices
                prices = ctx["prices"]
                result = _ta_features(
                    high=prices,
                    low=prices,
                    close=prices,
                    **kw,
                )
                ctx["ta_features"] = result
            return ctx

        _step.__name__ = "ta_features"
        return _step

    @staticmethod
    def ml_features() -> Callable[[dict], dict]:
        """Compute ML feature set (returns + vol + optional TA).

        Reads ``returns`` and optionally ``ta_features`` from the
        context.  Writes ``features`` (DataFrame).
        """

        def _step(ctx: dict) -> dict:
            from wraquant.ml.features import return_features, volatility_features

            r = ctx.get("returns")
            if r is not None:
                parts = [return_features(r), volatility_features(r)]
                if "ta_features" in ctx:
                    parts.append(ctx["ta_features"])
                ctx["features"] = pd.concat(parts, axis=1).dropna()
            return ctx

        _step.__name__ = "ml_features"
        return _step

    @staticmethod
    def risk_metrics(risk_free: float = 0.0) -> Callable[[dict], dict]:
        """Compute core risk metrics (Sharpe, Sortino, max drawdown).

        Reads ``strategy_returns`` or ``returns`` from the context.
        Writes ``risk`` (dict).  If a ``benchmark`` Series is in the
        context, also computes information ratio.
        """

        def _step(ctx: dict) -> dict:
            from wraquant.risk.metrics import (
                information_ratio,
                max_drawdown,
                sharpe_ratio,
                sortino_ratio,
            )

            r = _get_returns(ctx)
            if r is not None:
                prices = (1 + r).cumprod()
                risk_dict: dict[str, Any] = {
                    "sharpe": sharpe_ratio(r, risk_free),
                    "sortino": sortino_ratio(r, risk_free),
                    "max_drawdown": max_drawdown(prices),
                }
                benchmark = ctx.get("benchmark")
                if benchmark is not None:
                    n = min(len(r), len(benchmark))
                    risk_dict["information_ratio"] = information_ratio(
                        r.iloc[-n:], benchmark.iloc[-n:]
                    )
                ctx["risk"] = risk_dict
            return ctx

        _step.__name__ = "risk_metrics"
        return _step

    @staticmethod
    def var_analysis(
        confidence: float = 0.95,
        method: str = "historical",
    ) -> Callable[[dict], dict]:
        """Compute Value at Risk and Conditional VaR.

        Reads ``strategy_returns`` or ``returns``.  Writes ``var``
        (float) and ``cvar`` (float).
        """

        def _step(ctx: dict) -> dict:
            from wraquant.risk.var import conditional_var, value_at_risk

            r = _get_returns(ctx)
            if r is not None:
                ctx["var"] = value_at_risk(r, confidence=confidence, method=method)
                ctx["cvar"] = conditional_var(r, confidence=confidence, method=method)
            return ctx

        _step.__name__ = "var_analysis"
        return _step

    @staticmethod
    def garch_var(
        confidence: float = 0.95,
        vol_model: str = "GJR",
        dist: str = "t",
    ) -> Callable[[dict], dict]:
        """GARCH-based time-varying VaR.

        Reads ``returns``.  Writes ``garch_var`` (dict with time-varying
        VaR, CVaR, conditional vol, and breach diagnostics).
        Requires at least 200 observations.
        """

        def _step(ctx: dict) -> dict:
            from wraquant.risk.var import garch_var as _garch_var

            r = ctx.get("returns")
            if r is not None and len(r) >= 200:
                alpha = 1 - confidence
                ctx["garch_var"] = _garch_var(
                    r, alpha=alpha, vol_model=vol_model, dist=dist
                )
            return ctx

        _step.__name__ = "garch_var"
        return _step

    @staticmethod
    def stationarity_test() -> Callable[[dict], dict]:
        """Test stationarity (ADF + KPSS).

        Reads ``returns``.  Writes ``stationarity`` dict with
        ``adf`` and ``kpss`` sub-dicts.
        """

        def _step(ctx: dict) -> dict:
            from wraquant.ts.stationarity import adf_test, kpss_test

            r = ctx.get("returns")
            if r is not None:
                ctx["stationarity"] = {
                    "adf": adf_test(r),
                    "kpss": kpss_test(r),
                }
            return ctx

        _step.__name__ = "stationarity_test"
        return _step

    @staticmethod
    def forecast(
        horizon: int = 10,
    ) -> Callable[[dict], dict]:
        """Forecast returns.

        Reads ``returns``.  Writes ``forecast`` (dict from
        ``auto_forecast``).
        """

        def _step(ctx: dict) -> dict:
            from wraquant.ts.forecasting import auto_forecast

            r = ctx.get("returns")
            if r is not None and len(r) >= 30:
                ctx["forecast"] = auto_forecast(r, h=horizon)
            return ctx

        _step.__name__ = "forecast"
        return _step

    @staticmethod
    def tearsheet() -> Callable[[dict], dict]:
        """Generate comprehensive tearsheet.

        Reads ``strategy_returns`` or ``returns``.  Writes
        ``tearsheet`` (dict).
        """

        def _step(ctx: dict) -> dict:
            from wraquant.backtest.tearsheet import comprehensive_tearsheet

            r = _get_returns(ctx)
            if r is not None:
                ctx["tearsheet"] = comprehensive_tearsheet(r)
            return ctx

        _step.__name__ = "tearsheet"
        return _step

    @staticmethod
    def optimize(
        method: str = "risk_parity",
    ) -> Callable[[dict], dict]:
        """Optimise portfolio weights.

        Reads ``returns_df`` (multi-asset).  Writes ``optimization``
        (OptimizationResult) and ``weights`` (np.ndarray).
        """

        def _step(ctx: dict) -> dict:
            from wraquant.opt.portfolio import mean_variance, risk_parity

            returns_df = ctx.get("returns_df")
            if returns_df is not None and returns_df.shape[1] >= 2:
                if method == "risk_parity":
                    opt = risk_parity(returns_df)
                else:
                    opt = mean_variance(returns_df)
                ctx["optimization"] = opt
                ctx["weights"] = opt.weights
            return ctx

        _step.__name__ = "optimize"
        return _step

    @staticmethod
    def backtest_signals(
        signal_fn: Callable[[dict], Any] | None = None,
    ) -> Callable[[dict], dict]:
        """Generate and backtest signals.

        The *signal_fn* receives the full context and must return an
        array-like of position signals (+1, 0, -1).  The step
        multiplies signals by returns to produce ``strategy_returns``.
        """

        def _step(ctx: dict) -> dict:
            r = ctx.get("returns")
            if r is not None and signal_fn is not None:
                signals = signal_fn(ctx)
                n = min(len(r), len(signals))
                ctx["signals"] = np.asarray(signals[-n:])
                ctx["strategy_returns"] = pd.Series(
                    r.values[-n:] * np.asarray(signals[-n:]),
                    index=r.index[-n:],
                    name="strategy",
                )
            return ctx

        _step.__name__ = "backtest_signals"
        return _step

    @staticmethod
    def stress_test(
        scenarios: dict[str, float] | None = None,
    ) -> Callable[[dict], dict]:
        """Run stress testing.

        Reads ``returns`` or ``strategy_returns``.  Writes ``stress``
        (dict) with per-scenario stressed metrics.

        Parameters:
            scenarios: Mapping of scenario name to additive return shock.
                If None, uses a default set of shocks.
        """

        def _step(ctx: dict) -> dict:
            from wraquant.risk.stress import stress_test_returns

            r = _get_returns(ctx)
            if r is not None:
                sc = scenarios or {
                    "mild_shock": -0.01,
                    "moderate_shock": -0.03,
                    "severe_shock": -0.05,
                    "extreme_shock": -0.10,
                }
                ctx["stress"] = stress_test_returns(r, scenarios=sc)
            return ctx

        _step.__name__ = "stress_test"
        return _step

    @staticmethod
    def beta_analysis(
        benchmark: pd.Series | None = None,
    ) -> Callable[[dict], dict]:
        """Compute rolling and conditional beta.

        Reads ``returns`` and ``benchmark`` (from context or argument).
        Writes ``beta`` (dict with rolling and conditional sub-dicts).
        """

        def _step(ctx: dict) -> dict:
            from wraquant.risk.beta import conditional_beta, rolling_beta

            r = ctx.get("returns")
            b = benchmark if benchmark is not None else ctx.get("benchmark")
            if r is not None and b is not None:
                n = min(len(r), len(b))
                r_aligned = r.iloc[-n:]
                b_aligned = b.iloc[-n:]
                ctx["beta"] = {
                    "rolling": rolling_beta(r_aligned, b_aligned),
                    "conditional": conditional_beta(r_aligned, b_aligned),
                }
            return ctx

        _step.__name__ = "beta_analysis"
        return _step

    @staticmethod
    def custom(
        fn: Callable[[dict], dict],
        name: str = "custom",
    ) -> Callable[[dict], dict]:
        """Wrap a user-provided function as a workflow step.

        Parameters:
            fn: Callable that takes and returns a context dict.
            name: Display name for logging.
        """
        fn.__name__ = name
        return fn


# ---------------------------------------------------------------------------
# Pre-built workflow templates
# ---------------------------------------------------------------------------


def quick_analysis_workflow() -> Workflow:
    """Pre-built: returns -> risk -> stationarity -> regimes -> vol.

    The "just give me everything" workflow.  Runs the most common
    analyses on a price or return series and produces a comprehensive
    result object.

    Example:
        >>> result = quick_analysis_workflow().run(prices)
        >>> print(result.risk)
    """
    return (
        Workflow("quick_analysis")
        .add(steps.returns())
        .add(steps.risk_metrics())
        .add(steps.stationarity_test())
        .add(steps.regime_detect())
        .add(steps.garch_vol())
        .add(steps.tearsheet())
    )


def risk_workflow() -> Workflow:
    """Pre-built: returns -> risk -> VaR -> GARCH VaR -> stress.

    Focused risk analysis workflow that computes static and
    time-varying risk measures plus stress scenarios.

    Example:
        >>> result = risk_workflow().run(prices)
        >>> print(result.var, result.cvar)
    """
    return (
        Workflow("risk_analysis")
        .add(steps.returns())
        .add(steps.risk_metrics())
        .add(steps.var_analysis())
        .add(steps.garch_var())
        .add(steps.stress_test())
    )


def ml_workflow(
    signal_fn: Callable[[dict], Any] | None = None,
) -> Workflow:
    """Pre-built: returns -> features -> backtest signals -> risk -> tearsheet.

    ML alpha research workflow.  Optionally pass a *signal_fn* that
    takes the context (containing ``features``, ``returns``, etc.) and
    returns an array of position signals.

    Example:
        >>> def my_signal(ctx):
        ...     # Simple momentum signal
        ...     return (ctx["returns"].rolling(20).mean() > 0).astype(int)
        >>> result = ml_workflow(signal_fn=my_signal).run(prices)
    """
    return (
        Workflow("ml_alpha")
        .add(steps.returns())
        .add(steps.ta_features())
        .add(steps.ml_features())
        .add(steps.backtest_signals(signal_fn))
        .add(steps.risk_metrics())
        .add(steps.tearsheet())
    )


def portfolio_workflow() -> Workflow:
    """Pre-built: returns -> optimize -> risk -> regimes -> tearsheet.

    Multi-asset portfolio construction workflow.  Expects a DataFrame
    of multi-asset prices.

    Example:
        >>> result = portfolio_workflow().run(multi_asset_prices_df)
        >>> print(result.weights)
    """
    return (
        Workflow("portfolio")
        .add(steps.returns())
        .add(steps.optimize())
        .add(steps.risk_metrics())
        .add(steps.regime_detect())
        .add(steps.tearsheet())
    )
