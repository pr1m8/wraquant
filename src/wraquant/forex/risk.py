"""Forex-specific risk management.

Bridges the forex and risk modules by providing FX-adjusted portfolio
risk calculations that account for currency exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fx_portfolio_risk(
    positions: dict[str, float],
    exchange_rates: dict[str, float],
    base_currency: str = "USD",
    returns: pd.DataFrame | None = None,
    fx_returns: pd.DataFrame | None = None,
) -> dict[str, float | dict]:
    """Compute FX-adjusted portfolio risk.

    Accounts for currency exposure in portfolio risk calculation.
    Without FX adjustment, a portfolio denominated in multiple
    currencies has hidden risk from exchange rate movements.  This
    function bridges ``forex`` and ``risk`` by computing:

    1. **Base-currency positions**: converts all positions to the
       base currency using current exchange rates.
    2. **Currency exposure**: the net exposure to each currency as a
       fraction of total portfolio value.
    3. **FX-adjusted volatility**: if asset and FX return data are
       provided, computes the portfolio volatility including currency
       risk.

    When to use:
        Use this for any multi-currency portfolio to understand how
        much of your total risk comes from FX movements vs asset
        returns.  Essential for international equity, fixed income,
        and carry trade portfolios.

    Parameters:
        positions: Dictionary mapping asset names to position values
            in their local currency (e.g., ``{'AAPL': 100_000,
            'Toyota': 5_000_000}``).
        exchange_rates: Dictionary mapping currency codes to their
            value in base currency (e.g., ``{'USD': 1.0, 'JPY': 0.0067,
            'EUR': 1.10}``).  Each asset's currency should be present.
            If an asset name contains a known currency code, it is
            auto-detected.
        base_currency: The base (reporting) currency (default ``'USD'``).
        returns: Optional DataFrame of asset returns (columns = asset
            names).  If provided along with *fx_returns*, enables
            full FX-adjusted volatility calculation.
        fx_returns: Optional DataFrame of FX returns (columns =
            currency codes).  Required for volatility decomposition.

    Returns:
        Dictionary containing:

        - ``'total_value_base'`` (*float*) -- Total portfolio value
          in base currency.
        - ``'positions_base'`` (*dict*) -- Each position converted to
          base currency.
        - ``'currency_exposure'`` (*dict*) -- Net exposure to each
          currency as a fraction of total value.
        - ``'fx_adjusted_vol'`` (*float or None*) -- Annualised
          portfolio volatility including FX risk (only if returns and
          fx_returns are provided).
        - ``'asset_vol'`` (*float or None*) -- Annualised portfolio
          volatility from asset returns only.
        - ``'fx_vol_contribution'`` (*float or None*) -- Additional
          volatility from FX movements.

    Example:
        >>> positions = {'US_stock': 100_000, 'EU_stock': 80_000}
        >>> rates = {'USD': 1.0, 'EUR': 1.10}
        >>> # Map assets to currencies
        >>> result = fx_portfolio_risk(
        ...     positions, rates, base_currency='USD',
        ... )
        >>> result['total_value_base'] > 0
        True

    See Also:
        wraquant.risk.portfolio.portfolio_volatility: Asset-only vol.
        wraquant.forex.carry.carry_return: Carry trade returns.
    """
    # Convert positions to base currency
    positions_base: dict[str, float] = {}

    # Try to auto-detect currency from asset name or use first matching key
    for asset, value in positions.items():
        # Look for a currency code in the exchange_rates that appears in the asset name
        matched_rate = 1.0
        for ccy, rate in exchange_rates.items():
            if ccy.upper() in asset.upper():
                matched_rate = rate
                break
        else:
            # If no match found and base_currency is in exchange_rates, assume local
            # currency = base currency
            if base_currency in exchange_rates:
                matched_rate = 1.0
            # Otherwise use the first non-base rate as a guess, or 1.0
        positions_base[asset] = value * matched_rate

    total_value_base = sum(positions_base.values())

    # Currency exposure
    currency_exposure: dict[str, float] = {}
    for asset, value in positions.items():
        matched_ccy = base_currency
        for ccy in exchange_rates:
            if ccy.upper() in asset.upper():
                matched_ccy = ccy
                break
        if matched_ccy not in currency_exposure:
            currency_exposure[matched_ccy] = 0.0
        currency_exposure[matched_ccy] += positions_base[asset]

    # Normalise to fractions
    if total_value_base > 0:
        currency_exposure = {
            ccy: val / total_value_base for ccy, val in currency_exposure.items()
        }

    # FX-adjusted volatility (if return data is provided)
    fx_adjusted_vol = None
    asset_vol = None
    fx_vol_contribution = None

    if returns is not None and fx_returns is not None:
        # Compute weights
        if total_value_base > 0:
            weights = np.array([positions_base.get(c, 0.0) for c in returns.columns])
            weights = weights / total_value_base
        else:
            weights = np.ones(len(returns.columns)) / len(returns.columns)

        # Asset-only volatility
        asset_cov = returns.cov().values
        asset_vol_sq = float(weights @ asset_cov @ weights)
        asset_vol = float(np.sqrt(asset_vol_sq * 252))

        # Combined returns = asset returns + FX returns
        # For each asset, find matching FX column
        combined_returns = returns.copy()
        for col in returns.columns:
            for ccy in fx_returns.columns:
                if ccy.upper() in col.upper() and ccy.upper() != base_currency.upper():
                    # Add FX return to asset return
                    aligned_fx = fx_returns[ccy].reindex(returns.index).fillna(0)
                    combined_returns[col] = returns[col] + aligned_fx
                    break

        combined_cov = combined_returns.cov().values
        combined_vol_sq = float(weights @ combined_cov @ weights)
        fx_adjusted_vol = float(np.sqrt(combined_vol_sq * 252))
        fx_vol_contribution = fx_adjusted_vol - asset_vol

    return {
        "total_value_base": float(total_value_base),
        "positions_base": positions_base,
        "currency_exposure": currency_exposure,
        "fx_adjusted_vol": fx_adjusted_vol,
        "asset_vol": asset_vol,
        "fx_vol_contribution": fx_vol_contribution,
    }
