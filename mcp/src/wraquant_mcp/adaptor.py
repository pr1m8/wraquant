"""ToolAdaptor — auto-wraps wraquant functions as MCP tools.

The adaptor handles the translation between MCP tool calls and
wraquant function calls:

1. INPUT: Resolve dataset_id → DataFrame from DuckDB
2. INPUT: Coerce inline data (JSON) → numpy/pandas
3. CALL: Call the wraquant function
4. OUTPUT: Store DataFrame results in DuckDB, return ID
5. OUTPUT: Store model results in models/, return ID + summary
6. OUTPUT: Return scalar/dict results directly (float-coerced)
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


class ToolAdaptor:
    """Auto-generates MCP tool wrappers from wraquant functions.

    Each wrapped function:
    - Resolves dataset IDs to DataFrames from the shared DuckDB
    - Calls the wraquant function with resolved data
    - Stores results back in DuckDB (DataFrames) or models/ (fitted objects)
    - Returns JSON-safe metadata, never raw DataFrames

    Parameters:
        ctx: The shared AnalysisContext for the MCP session.
    """

    def __init__(self, ctx: AnalysisContext) -> None:
        self.ctx = ctx

    def wrap(
        self,
        func: Callable,
        module_name: str,
        data_params: list[str] | None = None,
        model_params: list[str] | None = None,
    ) -> Callable:
        """Wrap a wraquant function for MCP tool use.

        Parameters:
            func: The wraquant function to wrap.
            module_name: Module name for logging (e.g., "risk").
            data_params: Parameter names that should be resolved from DuckDB.
                If None, tries to auto-detect ("returns", "data", "prices", etc.)
            model_params: Parameter names that should be resolved from model store.
        """
        if data_params is None:
            data_params = _detect_data_params(func)
        if model_params is None:
            model_params = []

        ctx = self.ctx

        @wraps(func)
        def tool_fn(**kwargs: Any) -> dict[str, Any]:
            # 1. Resolve dataset references
            resolved = {}
            for param, value in kwargs.items():
                if param in data_params and isinstance(value, str):
                    if ctx.dataset_exists(value):
                        resolved[param] = ctx.get_dataset(value)
                    else:
                        resolved[param] = value
                elif param in model_params and isinstance(value, str):
                    try:
                        resolved[param] = ctx.get_model(value)
                    except KeyError:
                        resolved[param] = value
                else:
                    resolved[param] = value

            # 2. Call wraquant function
            try:
                result = func(**resolved)
            except Exception as e:
                return {
                    "error": type(e).__name__,
                    "message": str(e),
                    "tool": f"{module_name}.{func.__name__}",
                }

            # 3. Store and return based on result type
            return _handle_result(
                ctx,
                result,
                func_name=func.__name__,
                module_name=module_name,
                input_params=kwargs,
            )

        tool_fn.__name__ = f"{module_name}_{func.__name__}"
        tool_fn.__doc__ = func.__doc__
        return tool_fn


def _detect_data_params(func: Callable) -> list[str]:
    """Auto-detect which parameters are data (need DuckDB resolution).

    Looks at parameter names to identify common data patterns.
    """
    import inspect

    sig = inspect.signature(func)
    data_names = {
        "returns", "data", "prices", "benchmark", "factors",
        "factors_df", "returns_df", "prices_df",
        "high", "low", "close", "open", "volume",
        "y", "X", "observations",
    }
    return [
        name
        for name in sig.parameters
        if name in data_names
    ]


def _handle_result(
    ctx: AnalysisContext,
    result: Any,
    func_name: str,
    module_name: str,
    input_params: dict[str, Any],
) -> dict[str, Any]:
    """Route a wraquant function result to the right storage.

    - pd.Series → store as DuckDB table (1 column)
    - pd.DataFrame → store as DuckDB table
    - dict with pd.Series values → store as DuckDB table
    - float/int → return directly
    - dict with scalar values → return directly
    - Dataclass with .params/.aic → store as model + return summary
    """
    tool_id = f"{module_name}.{func_name}"

    # Float/int scalar
    if isinstance(result, (int, float, np.integer, np.floating)):
        return {
            "tool": tool_id,
            "result": float(result),
        }

    # pd.Series → store as dataset
    if isinstance(result, pd.Series):
        name = f"{func_name}_result"
        df = result.to_frame(name=result.name or func_name)
        stored = ctx.store_dataset(name, df, source_op=tool_id)
        return {
            "tool": tool_id,
            **stored,
            "summary": {
                "mean": float(result.mean()) if len(result) > 0 else None,
                "std": float(result.std()) if len(result) > 0 else None,
                "min": float(result.min()) if len(result) > 0 else None,
                "max": float(result.max()) if len(result) > 0 else None,
            },
        }

    # pd.DataFrame → store as dataset
    if isinstance(result, pd.DataFrame):
        name = f"{func_name}_result"
        stored = ctx.store_dataset(name, result, source_op=tool_id)
        return {
            "tool": tool_id,
            **stored,
        }

    # Dataclass with model-like attributes (GARCHResult, RegimeResult, etc.)
    if hasattr(result, "persistence") or hasattr(result, "n_regimes"):
        # It's a model result — store it
        name = f"{func_name}_model"
        model_type = type(result).__name__
        metrics = {}
        for attr in ("aic", "bic", "persistence", "half_life", "log_likelihood"):
            if hasattr(result, attr):
                val = getattr(result, attr)
                if isinstance(val, (int, float, np.integer, np.floating)):
                    metrics[attr] = float(val)

        stored = ctx.store_model(
            name, result, model_type=model_type, metrics=metrics,
        )

        # Also store conditional_volatility / states as datasets if present
        for series_attr in ("conditional_volatility", "states", "probabilities"):
            if hasattr(result, series_attr):
                val = getattr(result, series_attr)
                if isinstance(val, (pd.Series, np.ndarray)):
                    df = pd.DataFrame({series_attr: np.asarray(val).ravel()})
                    ctx.store_dataset(
                        f"{name}_{series_attr}", df, source_op=tool_id,
                    )

        return {
            "tool": tool_id,
            **stored,
        }

    # Dict → check if values are Series (multi-output) or scalars
    if isinstance(result, dict):
        # Check if any values are Series/arrays (multi-output indicator)
        series_values = {
            k: v for k, v in result.items() if isinstance(v, (pd.Series, np.ndarray))
        }
        if series_values:
            # Store as DataFrame
            df_data = {}
            for k, v in series_values.items():
                if isinstance(v, pd.Series):
                    df_data[k] = v.values
                elif isinstance(v, np.ndarray) and v.ndim == 1:
                    df_data[k] = v
            if df_data:
                max_len = max(len(v) for v in df_data.values())
                # Pad shorter arrays
                for k in df_data:
                    if len(df_data[k]) < max_len:
                        df_data[k] = np.pad(
                            df_data[k], (0, max_len - len(df_data[k])),
                            constant_values=np.nan,
                        )
                df = pd.DataFrame(df_data)
                name = f"{func_name}_result"
                stored = ctx.store_dataset(name, df, source_op=tool_id)
                # Also include scalar values
                scalar_values = {
                    k: _sanitize_for_json(v)
                    for k, v in result.items()
                    if not isinstance(v, (pd.Series, np.ndarray))
                }
                return {
                    "tool": tool_id,
                    **stored,
                    **scalar_values,
                }

        # All scalar dict — return directly
        return {
            "tool": tool_id,
            "result": _sanitize_for_json(result),
        }

    # Fallback — try to serialize
    return {
        "tool": tool_id,
        "result": str(result),
    }
