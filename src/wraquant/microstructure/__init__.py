"""Market microstructure analytics.

Provides quantitative measures of market quality, liquidity, and order
flow toxicity for both high-frequency tick data and daily-frequency
OHLCV data.  These metrics are essential for execution cost analysis,
informed-trading detection, and understanding the micro-level dynamics
that drive price formation.

Key sub-modules:

- **Liquidity** (``liquidity``) -- Bid-ask spread estimation and
  liquidity measurement: ``amihud_illiquidity`` (the standard illiquidity
  proxy for daily data), ``roll_spread`` (implied spread from
  autocovariance), ``corwin_schultz_spread`` (high-low spread estimator),
  ``kyle_lambda`` (price impact coefficient), ``effective_spread``,
  ``realized_spread``, ``spread_decomposition``, and
  ``liquidity_commonality``.
- **Toxicity** (``toxicity``) -- Informed trading and order flow toxicity:
  ``vpin`` (Volume-Synchronized Probability of Informed Trading -- the
  real-time toxicity metric that predicted the Flash Crash),
  ``pin_model`` (classical PIN model), ``adjusted_pin``,
  ``order_flow_imbalance``, ``bulk_volume_classification``,
  ``trade_classification`` (Lee-Ready tick rule), and
  ``informed_trading_intensity``.
- **Market Quality** (``market_quality``) -- Structural market efficiency:
  ``variance_ratio`` (tests random walk hypothesis),
  ``market_efficiency_ratio``, ``hasbrouck_information_share``,
  ``gonzalo_granger_component``, ``intraday_volatility_pattern``,
  ``price_impact_regression``, ``depth``, and ``resiliency``.

Example:
    >>> from wraquant.microstructure import amihud_illiquidity, vpin
    >>> illiq = amihud_illiquidity(returns, volume)
    >>> toxicity = vpin(trades, volume_bucket_size=50_000)

Use ``wraquant.microstructure`` when analyzing execution quality, detecting
informed trading, or building features for ML models that capture market
structure.  For execution algorithms that consume these metrics, see
``wraquant.execution``.  For generating microstructure-based ML features,
see ``wraquant.ml.features.microstructure_features``.
"""

from __future__ import annotations

from wraquant.microstructure.liquidity import (
    amihud_illiquidity,
    amihud_rolling,
    closing_quoted_spread,
    corwin_schultz_spread,
    depth_imbalance,
    effective_spread,
    kyle_lambda,
    lambda_kyle_rolling,
    liquidity_commonality,
    price_impact,
    realized_spread,
    roll_spread,
    spread_decomposition,
    turnover_ratio,
)
from wraquant.microstructure.market_quality import (
    depth,
    gonzalo_granger_component,
    hasbrouck_information_share,
    intraday_volatility_pattern,
    market_efficiency_ratio,
    price_impact_regression,
    quoted_spread,
    relative_spread,
    resiliency,
    variance_ratio,
)
from wraquant.microstructure.toxicity import (
    adjusted_pin,
    bulk_volume_classification,
    information_share,
    informed_trading_intensity,
    order_flow_imbalance,
    pin_model,
    toxicity_index,
    trade_classification,
    vpin,
)

__all__ = [
    # Liquidity
    "amihud_illiquidity",
    "amihud_rolling",
    "closing_quoted_spread",
    "corwin_schultz_spread",
    "depth_imbalance",
    "effective_spread",
    "kyle_lambda",
    "lambda_kyle_rolling",
    "liquidity_commonality",
    "price_impact",
    "realized_spread",
    "roll_spread",
    "spread_decomposition",
    "turnover_ratio",
    # Toxicity
    "adjusted_pin",
    "bulk_volume_classification",
    "information_share",
    "informed_trading_intensity",
    "order_flow_imbalance",
    "pin_model",
    "toxicity_index",
    "trade_classification",
    "vpin",
    # Market quality
    "depth",
    "gonzalo_granger_component",
    "hasbrouck_information_share",
    "intraday_volatility_pattern",
    "market_efficiency_ratio",
    "price_impact_regression",
    "quoted_spread",
    "relative_spread",
    "resiliency",
    "variance_ratio",
]
