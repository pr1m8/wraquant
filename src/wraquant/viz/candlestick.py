"""Interactive OHLCV candlestick and alternative chart types.

Full-featured candlestick with overlays, market/volume profile,
Renko charts, and Heikin-Ashi candlesticks --- all as interactive Plotly
figures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wraquant.core.decorators import requires_extra
from wraquant.viz.themes import COLORS

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

__all__ = [
    "plotly_candlestick",
    "plotly_market_profile",
    "plotly_renko",
    "plotly_heikin_ashi",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLOTLY_TEMPLATE = "plotly_white"


def _base_layout(**overrides: object) -> dict:
    """Return a base Plotly layout dict with wraquant styling."""
    defaults: dict = dict(
        template=_PLOTLY_TEMPLATE,
        font=dict(family="sans-serif", size=12, color="#333333"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=60, r=30, t=50, b=50),
    )
    defaults.update(overrides)
    return defaults


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase column names."""
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@requires_extra("viz")
def plotly_candlestick(
    ohlcv_df: pd.DataFrame,
    overlays: list[str] | None = None,
    indicators: list[str] | None = None,
) -> go.Figure:
    """Full-featured interactive candlestick chart.

    Supports optional overlays (moving averages, Bollinger Bands) and a
    secondary volume bar chart.

    Parameters:
        ohlcv_df: DataFrame with columns ``open, high, low, close`` and
            optionally ``volume``.  Column names are case-insensitive.
        overlays: List of overlay names to draw.  Supported values:
            ``"sma20"``, ``"sma50"``, ``"sma200"``, ``"ema20"``,
            ``"bb"`` (Bollinger Bands, 20-period, 2 std).
        indicators: Reserved for future sub-chart indicators.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = _normalise_columns(ohlcv_df)
    has_volume = "volume" in df.columns

    rows = 2 if has_volume else 1
    row_heights = [0.75, 0.25] if has_volume else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=COLORS["positive"],
            decreasing_line_color=COLORS["negative"],
            name="OHLC",
        ),
        row=1, col=1,
    )

    # Overlays
    overlay_colors = [COLORS["secondary"], COLORS["accent"], COLORS["info"],
                      COLORS["warning"]]
    if overlays:
        color_idx = 0
        for ov in overlays:
            ov_lower = ov.lower().strip()
            if ov_lower.startswith("sma"):
                period = int(ov_lower.replace("sma", ""))
                sma = df["close"].rolling(period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=sma,
                        mode="lines",
                        name=f"SMA {period}",
                        line=dict(
                            color=overlay_colors[color_idx % len(overlay_colors)],
                            width=1.3,
                        ),
                    ),
                    row=1, col=1,
                )
                color_idx += 1

            elif ov_lower.startswith("ema"):
                period = int(ov_lower.replace("ema", ""))
                ema = df["close"].ewm(span=period, adjust=False).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=ema,
                        mode="lines",
                        name=f"EMA {period}",
                        line=dict(
                            color=overlay_colors[color_idx % len(overlay_colors)],
                            width=1.3,
                            dash="dot",
                        ),
                    ),
                    row=1, col=1,
                )
                color_idx += 1

            elif ov_lower == "bb":
                sma20 = df["close"].rolling(20).mean()
                std20 = df["close"].rolling(20).std()
                upper = sma20 + 2 * std20
                lower = sma20 - 2 * std20

                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=upper,
                        mode="lines",
                        name="BB Upper",
                        line=dict(color=COLORS["neutral"], width=1, dash="dash"),
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=lower,
                        mode="lines",
                        name="BB Lower",
                        line=dict(color=COLORS["neutral"], width=1, dash="dash"),
                        fill="tonexty",
                        fillcolor="rgba(127, 127, 127, 0.08)",
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=sma20,
                        mode="lines",
                        name="BB Mid",
                        line=dict(color=COLORS["neutral"], width=1),
                    ),
                    row=1, col=1,
                )

    # Volume bars
    if has_volume:
        colors = [
            COLORS["positive"] if c >= o else COLORS["negative"]
            for c, o in zip(df["close"], df["open"], strict=False)
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                marker_color=colors,
                opacity=0.55,
                name="Volume",
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_layout(
        **_base_layout(
            title="Candlestick Chart",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=600 if has_volume else 450,
        )
    )
    return fig


@requires_extra("viz")
def plotly_market_profile(
    ohlcv_df: pd.DataFrame,
) -> go.Figure:
    """Market / volume profile chart.

    Shows a horizontal histogram of volume at each price level alongside
    a candlestick chart.

    Parameters:
        ohlcv_df: DataFrame with ``open, high, low, close, volume`` columns.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = _normalise_columns(ohlcv_df)

    # Build volume-at-price histogram
    price_min = df["low"].min()
    price_max = df["high"].max()
    n_bins = 50
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_at_price = np.zeros(n_bins)

    for _, row in df.iterrows():
        # Distribute volume across the bar's range
        mask = (bin_centers >= row["low"]) & (bin_centers <= row["high"])
        count = mask.sum()
        if count > 0:
            vol_at_price[mask] += row["volume"] / count

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.75, 0.25],
        shared_yaxes=True,
        horizontal_spacing=0.02,
    )

    # Candlestick on left
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=COLORS["positive"],
            decreasing_line_color=COLORS["negative"],
            name="OHLC",
        ),
        row=1, col=1,
    )

    # Volume profile on right (horizontal bar)
    fig.add_trace(
        go.Bar(
            x=vol_at_price,
            y=bin_centers,
            orientation="h",
            marker_color=COLORS["primary"],
            opacity=0.6,
            name="Volume Profile",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        **_base_layout(
            title="Market Profile (Volume at Price)",
            xaxis_rangeslider_visible=False,
            height=550,
            showlegend=False,
        )
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_xaxes(title_text="Volume", row=1, col=2)
    return fig


@requires_extra("viz")
def plotly_renko(
    prices: pd.Series,
    brick_size: float | None = None,
) -> go.Figure:
    """Renko chart built from a price series.

    Parameters:
        prices: Close price series.
        brick_size: Fixed brick size.  If *None*, uses the ATR(14) of
            daily price changes as a heuristic.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import numpy as np
    import plotly.graph_objects as go

    vals = prices.values.astype(float)

    if brick_size is None:
        daily_range = np.abs(np.diff(vals))
        brick_size = float(np.mean(daily_range[-min(14, len(daily_range)):]))
        if brick_size == 0:
            brick_size = 1.0

    # Build Renko bricks
    bricks_open: list[float] = []
    bricks_close: list[float] = []
    bricks_color: list[str] = []

    base = vals[0]
    for price in vals[1:]:
        while price >= base + brick_size:
            bricks_open.append(base)
            base += brick_size
            bricks_close.append(base)
            bricks_color.append(COLORS["positive"])
        while price <= base - brick_size:
            bricks_open.append(base)
            base -= brick_size
            bricks_close.append(base)
            bricks_color.append(COLORS["negative"])

    if not bricks_open:
        # Not enough movement for any bricks; place a single neutral brick
        bricks_open.append(vals[0])
        bricks_close.append(vals[0] + brick_size)
        bricks_color.append(COLORS["neutral"])

    n = len(bricks_open)
    x_indices = list(range(n))

    # Use OHLC-like representation
    highs = [max(o, c) for o, c in zip(bricks_open, bricks_close, strict=False)]
    lows = [min(o, c) for o, c in zip(bricks_open, bricks_close, strict=False)]

    fig = go.Figure()
    # Draw each brick as a filled rectangle via bar
    for i in range(n):
        fig.add_trace(
            go.Bar(
                x=[i],
                y=[abs(bricks_close[i] - bricks_open[i])],
                base=lows[i],
                marker_color=bricks_color[i],
                width=0.8,
                showlegend=False,
                hovertemplate=(
                    f"Brick {i + 1}<br>"
                    f"Open: {bricks_open[i]:.2f}<br>"
                    f"Close: {bricks_close[i]:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **_base_layout(
            title=f"Renko Chart (brick={brick_size:.2f})",
            xaxis_title="Brick #",
            yaxis_title="Price",
            showlegend=False,
            barmode="stack",
            height=450,
        )
    )
    return fig


@requires_extra("viz")
def plotly_heikin_ashi(
    ohlcv_df: pd.DataFrame,
) -> go.Figure:
    """Heikin-Ashi candlestick chart.

    Computes Heikin-Ashi OHLC values from the raw data and plots them
    as an interactive candlestick chart.

    Parameters:
        ohlcv_df: DataFrame with ``open, high, low, close`` columns.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    df = _normalise_columns(ohlcv_df)

    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    # First bar
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)

    fig = go.Figure(
        data=go.Candlestick(
            x=df.index,
            open=ha_open,
            high=ha_high,
            low=ha_low,
            close=ha_close,
            increasing_line_color=COLORS["positive"],
            decreasing_line_color=COLORS["negative"],
            name="Heikin-Ashi",
        )
    )

    fig.update_layout(
        **_base_layout(
            title="Heikin-Ashi Candlestick",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            height=500,
        )
    )
    return fig
