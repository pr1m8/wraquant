"""Pre-built quantitative finance workflows that chain wraquant modules.

These recipes show how the library's modules work together as a
cohesive framework rather than independent tools.  Each recipe is a
complete pipeline that wires data through several wraquant subsystems
and returns a consolidated result dictionary.

Recipes are intentionally *thin* orchestration layers.  The real logic
lives in the individual modules; recipes just sequence the calls,
align data, and assemble the outputs.

Example:
    >>> import wraquant as wq
    >>> result = wq.analyze(daily_returns)
    >>> print(result["risk"]["sharpe"])
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# analyze -- "just give me everything"
# ---------------------------------------------------------------------------


def analyze(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | None = None,
) -> dict[str, Any]:
    """Quick comprehensive analysis of a return series.

    The "just give me everything" function.  Runs relevant analyses
    from stats, risk, vol, regimes, and ts modules and returns
    a comprehensive report.

    Pipeline: returns -> descriptive stats -> risk metrics ->
    distribution fit -> stationarity test -> (optional) regime
    detection -> (optional) GARCH volatility -> (optional)
    benchmark-relative metrics.

    Chains: stats -> risk -> ts -> regimes -> vol.

    Parameters:
        returns: Return series or multi-asset DataFrame.  If a
            DataFrame is provided, the first column is used as the
            primary series.
        benchmark: Optional benchmark return series for relative
            metrics (information ratio, beta).

    Returns:
        Dictionary with sections:

        - **descriptive** -- mean, std, skew, kurtosis, min, max, count.
        - **risk** -- sharpe, sortino, max_drawdown.
        - **distribution** -- fitted normal params + KS test.
        - **stationarity** -- ADF test statistic, p-value, is_stationary.
        - **regime** *(optional, requires >= 100 obs)* -- current regime,
          probabilities, n_regimes.
        - **volatility** *(optional, requires >= 200 obs)* -- GARCH
          persistence, half-life, current conditional vol.
        - **relative** *(only when benchmark provided)* -- information
          ratio, beta.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> rets = pd.Series(np.random.normal(0.0005, 0.01, 500))
        >>> report = analyze(rets)
        >>> sorted(report.keys())  # doctest: +NORMALIZE_WHITESPACE
        ['descriptive', 'distribution', 'risk', 'stationarity']
    """
    from wraquant.risk.metrics import max_drawdown, sharpe_ratio, sortino_ratio
    from wraquant.stats.descriptive import summary_stats
    from wraquant.stats.distributions import fit_distribution
    from wraquant.ts.stationarity import adf_test

    if isinstance(returns, pd.DataFrame):
        primary = returns.iloc[:, 0].dropna()
    else:
        primary = returns.dropna()

    # Ensure we have a pd.Series for the functions that expect one
    r_series = pd.Series(primary.values, index=primary.index, name="returns")

    result: dict[str, Any] = {
        "descriptive": summary_stats(r_series),
        "risk": {
            "sharpe": sharpe_ratio(r_series),
            "sortino": sortino_ratio(r_series),
            "max_drawdown": max_drawdown(
                (1 + r_series).cumprod(),
            ),
        },
        "distribution": fit_distribution(r_series, dist="norm"),
        "stationarity": adf_test(r_series),
    }

    # Optional: regime detection (may fail for short series or
    # if hmmlearn is not installed)
    try:
        from wraquant.regimes.base import detect_regimes

        if len(r_series) >= 100:
            regime = detect_regimes(r_series.values, method="hmm", n_regimes=2)
            result["regime"] = {
                "current": regime.current_regime,
                "probabilities": regime.current_probabilities.tolist(),
                "n_regimes": regime.n_regimes,
            }
    except Exception:  # noqa: BLE001
        pass

    # Optional: GARCH vol (requires arch)
    try:
        from wraquant.vol.models import garch_fit

        if len(r_series) >= 200:
            # arch expects percentage returns
            garch = garch_fit(r_series * 100)
            result["volatility"] = {
                "persistence": garch["persistence"],
                "half_life": garch["half_life"],
                "current_vol": float(garch["conditional_volatility"].iloc[-1]) / 100,
            }
    except Exception:  # noqa: BLE001
        pass

    # Benchmark-relative metrics
    if benchmark is not None:
        from wraquant.risk.metrics import information_ratio

        b = benchmark.dropna()
        n = min(len(r_series), len(b))
        r_aligned = r_series.iloc[-n:].reset_index(drop=True)
        b_aligned = b.iloc[-n:].reset_index(drop=True)
        result["relative"] = {
            "information_ratio": information_ratio(r_aligned, b_aligned),
            "beta": float(
                np.cov(r_aligned.values, b_aligned.values)[0, 1]
                / np.var(b_aligned.values)
            ),
        }

    return result


# ---------------------------------------------------------------------------
# regime_aware_backtest
# ---------------------------------------------------------------------------


def regime_aware_backtest(
    prices: pd.Series,
    n_regimes: int = 2,
    bull_weight: float = 1.0,
    bear_weight: float = 0.0,
    vol_target: float = 0.15,
) -> dict[str, Any]:
    """Full regime-aware backtest pipeline.

    Pipeline: prices -> returns -> detect regimes -> compute regime
    stats -> size positions by regime -> generate strategy returns ->
    tearsheet + risk metrics.

    Chains: data -> regimes -> backtest -> risk.

    Parameters:
        prices: Price series (e.g. adjusted close).
        n_regimes: Number of market regimes (default 2: bull/bear).
        bull_weight: Portfolio weight in the low-volatility (bull)
            regime.  1.0 = fully invested.
        bear_weight: Portfolio weight in the high-volatility (bear)
            regime.  0.0 = flat.
        vol_target: Annual volatility target for position sizing
            (informational; not used for scaling in this recipe).

    Returns:
        Dictionary with:

        - **regime_result** -- ``RegimeResult`` dataclass from
          ``wraquant.regimes.base``.
        - **strategy_returns** -- pd.Series of regime-weighted strategy
          returns.
        - **tearsheet** -- comprehensive tearsheet dict from
          ``wraquant.backtest.tearsheet``.
        - **regime_stats** -- per-regime summary DataFrame.
        - **risk_metrics** -- dict with sharpe, sortino, max_drawdown.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> prices = pd.Series(
        ...     np.cumprod(1 + np.random.normal(0.0003, 0.01, 500)),
        ...     index=pd.bdate_range("2020-01-01", periods=500),
        ... )
        >>> result = regime_aware_backtest(prices)
        >>> "strategy_returns" in result
        True
    """
    from wraquant.backtest.position import regime_conditional_sizing
    from wraquant.backtest.tearsheet import comprehensive_tearsheet
    from wraquant.regimes.base import detect_regimes
    from wraquant.risk.metrics import max_drawdown, sharpe_ratio, sortino_ratio

    returns = prices.pct_change().dropna()

    # 1. Detect regimes
    regime_result = detect_regimes(returns.values, method="hmm", n_regimes=n_regimes)

    # 2. Size positions by regime
    # regime_conditional_sizing expects {str: float} dicts for both
    # probabilities and multipliers.
    base_weights = np.array([1.0])
    regime_multipliers = {
        f"regime_{k}": (bull_weight if k == 0 else bear_weight)
        for k in range(n_regimes)
    }

    n = min(len(returns), len(regime_result.probabilities))
    positions = np.empty(n)
    for t in range(n):
        probs_dict = {
            f"regime_{k}": float(regime_result.probabilities[t, k])
            for k in range(n_regimes)
        }
        sized = regime_conditional_sizing(
            base_weights,
            probs_dict,
            regime_multipliers,
        )
        positions[t] = float(sized[0])

    # 3. Generate strategy returns
    strategy_returns = pd.Series(
        returns.values[-n:] * positions,
        index=returns.index[-n:],
        name="regime_strategy",
    )

    # 4. Tearsheet + risk
    tearsheet = comprehensive_tearsheet(strategy_returns)

    equity_curve = (1 + strategy_returns).cumprod()

    return {
        "regime_result": regime_result,
        "strategy_returns": strategy_returns,
        "tearsheet": tearsheet,
        "regime_stats": regime_result.statistics,
        "risk_metrics": {
            "sharpe": sharpe_ratio(strategy_returns),
            "sortino": sortino_ratio(strategy_returns),
            "max_drawdown": max_drawdown(equity_curve),
        },
    }


# ---------------------------------------------------------------------------
# garch_risk_pipeline
# ---------------------------------------------------------------------------


def garch_risk_pipeline(
    returns: pd.Series,
    vol_model: str = "GJR",
    dist: str = "t",
    var_alpha: float = 0.05,
) -> dict[str, Any]:
    """GARCH volatility -> VaR/CVaR -> stress testing pipeline.

    Pipeline: returns -> fit GARCH -> conditional vol -> time-varying
    VaR -> news impact curve -> stress scenarios -> risk report.

    Chains: vol -> risk -> stress.

    Parameters:
        returns: Simple return series (daily).
        vol_model: GARCH variant -- ``"GARCH"``, ``"GJR"``, or
            ``"EGARCH"``.
        dist: Error distribution for the GARCH model.  ``"normal"``,
            ``"t"`` (Student-t), or ``"skewt"`` (skewed Student-t).
        var_alpha: Significance level for VaR (0.05 = 95% VaR).

    Returns:
        Dictionary with:

        - **garch** -- fitted GARCH result dict (params, conditional
          vol, diagnostics).
        - **var** -- time-varying VaR/CVaR result dict.
        - **news_impact** -- news impact curve dict.
        - **diagnostics** -- summary dict with persistence, half_life,
          current_vol, breach_rate.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> rets = pd.Series(np.random.normal(0.0003, 0.01, 500))
        >>> result = garch_risk_pipeline(rets)
        >>> "garch" in result and "var" in result
        True
    """
    from wraquant.risk.var import garch_var
    from wraquant.vol.models import (
        egarch_fit,
        garch_fit,
        gjr_garch_fit,
        news_impact_curve,
    )

    # 1. Fit GARCH
    fit_fns = {"GARCH": garch_fit, "GJR": gjr_garch_fit, "EGARCH": egarch_fit}
    fit_fn = fit_fns.get(vol_model.upper(), garch_fit)
    garch_result = fit_fn(returns, dist=dist)

    # 2. Time-varying VaR
    var_result = garch_var(returns, vol_model=vol_model, dist=dist, alpha=var_alpha)

    # 3. News impact curve
    nic = news_impact_curve(returns.values, model_type=vol_model.lower())

    return {
        "garch": garch_result,
        "var": var_result,
        "news_impact": nic,
        "diagnostics": {
            "persistence": garch_result["persistence"],
            "half_life": garch_result["half_life"],
            "current_vol": float(garch_result["conditional_volatility"].iloc[-1]),
            "breach_rate": var_result["breach_rate"],
        },
    }


# ---------------------------------------------------------------------------
# ml_alpha_pipeline
# ---------------------------------------------------------------------------


def ml_alpha_pipeline(
    prices_df: pd.DataFrame,
    target_col: str,
    model: str = "gradient_boost",
    walk_forward_windows: int = 5,
) -> dict[str, Any]:
    """ML alpha research pipeline.

    Pipeline: prices -> features -> walk-forward train/predict ->
    evaluate -> feature importance.

    Chains: ml/features -> ml/pipeline -> ml/advanced -> risk.

    Parameters:
        prices_df: Multi-asset price DataFrame (columns = tickers).
        target_col: Column name of the target asset to predict.
        model: Model type.  Currently uses sklearn's
            ``GradientBoostingClassifier`` under the hood.
        walk_forward_windows: Not used in current implementation
            (walk-forward uses fixed train/test sizes).

    Returns:
        Dictionary with:

        - **walk_forward** -- walk-forward backtest result dict
          (predictions, actuals, pnl, sharpe, hit_rate, equity_curve).
        - **feature_importance** -- random forest importance ranking
          (or None if too few samples).
        - **hit_rate** -- out-of-sample directional accuracy.
        - **sharpe** -- out-of-sample Sharpe ratio.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> prices = pd.DataFrame({
        ...     "SPY": np.cumprod(1 + np.random.normal(0.0003, 0.01, 600)),
        ...     "TLT": np.cumprod(1 + np.random.normal(0.0001, 0.005, 600)),
        ... })
        >>> result = ml_alpha_pipeline(prices, target_col="SPY")
        >>> "walk_forward" in result
        True
    """
    from sklearn.ensemble import GradientBoostingClassifier

    from wraquant.ml.advanced import random_forest_importance
    from wraquant.ml.features import return_features, volatility_features
    from wraquant.ml.pipeline import walk_forward_backtest

    returns = prices_df.pct_change().dropna()

    # 1. Build features (return_features expects a price series)
    features = return_features(prices_df[target_col])
    vol_feats = volatility_features(returns[target_col])
    all_features = pd.concat([features, vol_feats], axis=1).dropna()

    # 2. Align target -- binary classification: up/down
    target = returns[target_col].reindex(all_features.index)
    valid = all_features.index.intersection(target.dropna().index)
    X = all_features.loc[valid]
    y_binary = (target.loc[valid] > 0).astype(int)

    # 3. Walk-forward backtest (expects a sklearn-compatible estimator)
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42,
    )
    wf_result = walk_forward_backtest(
        model=clf,
        X=X,
        y=y_binary,
        train_size=min(252, len(X) // 2),
        test_size=21,
        step_size=21,
    )

    # 4. Feature importance (only if enough data)
    importance = None
    if len(X) > 50:
        importance = random_forest_importance(X, y_binary)

    return {
        "walk_forward": wf_result,
        "feature_importance": importance,
        "hit_rate": wf_result.get("hit_rate", 0),
        "sharpe": wf_result.get("sharpe", 0),
    }


# ---------------------------------------------------------------------------
# portfolio_construction_pipeline
# ---------------------------------------------------------------------------


def portfolio_construction_pipeline(
    returns_df: pd.DataFrame,
    method: str = "risk_parity",
    regime_aware: bool = True,
    n_regimes: int = 2,
) -> dict[str, Any]:
    """Full portfolio construction pipeline.

    Pipeline: returns -> covariance estimation -> optimize ->
    (optional) regime adjust -> risk decomposition -> betas.

    Chains: stats -> opt -> regimes -> risk/portfolio_analytics.

    Parameters:
        returns_df: Multi-asset return DataFrame (columns = tickers,
            rows = daily returns).
        method: Optimization method.  ``"risk_parity"`` (default) or
            ``"mean_variance"``.
        regime_aware: If True, adjust weights by current regime
            probability (scales down in high-vol regimes).
        n_regimes: Number of regimes for the optional regime adjustment.

    Returns:
        Dictionary with:

        - **weights** -- dict mapping asset name to weight.
        - **optimization** -- ``OptimizationResult`` dataclass.
        - **component_var** -- per-asset VaR contribution (pd.Series).
        - **diversification_ratio** -- portfolio diversification ratio.
        - **betas** -- dict mapping asset name to rolling beta vs
          first asset.
        - **regime_adjusted** -- bool indicating whether regime
          scaling was applied.

    Example:
        >>> import pandas as pd, numpy as np
        >>> np.random.seed(42)
        >>> rets = pd.DataFrame(
        ...     np.random.randn(252, 3) * np.array([0.01, 0.02, 0.005]),
        ...     columns=["Bonds", "Equity", "Gold"],
        ... )
        >>> result = portfolio_construction_pipeline(rets, regime_aware=False)
        >>> sum(result["weights"].values())  # doctest: +ELLIPSIS
        1.0...
    """
    from wraquant.opt.portfolio import mean_variance, risk_parity
    from wraquant.risk.beta import rolling_beta
    from wraquant.risk.portfolio_analytics import component_var, diversification_ratio

    # 1. Optimize
    if method == "risk_parity":
        opt_result = risk_parity(returns_df)
    else:
        opt_result = mean_variance(returns_df)

    weights = opt_result.weights.copy()

    # 2. Regime adjustment (optional)
    if regime_aware:
        try:
            from wraquant.regimes.base import detect_regimes

            # Use first asset as market proxy
            market = returns_df.iloc[:, 0].values
            regime = detect_regimes(market, method="hmm", n_regimes=n_regimes)
            current_prob = regime.current_probabilities
            # Scale down in high-vol regime: probability of low-vol regime
            vol_scale = float(current_prob[0])
            weights = weights * (0.5 + 0.5 * vol_scale)
            weights = weights / weights.sum()  # renormalize
        except Exception:  # noqa: BLE001
            # If regime detection fails, proceed without adjustment
            regime_aware = False

    # 3. Risk decomposition
    cov = np.cov(returns_df.values, rowvar=False) * 252
    comp_var = component_var(weights, returns_df)
    div_ratio = diversification_ratio(weights, cov)

    # 4. Rolling betas vs first asset
    market_returns = returns_df.iloc[:, 0]
    betas: dict[str, float] = {}
    for col in returns_df.columns:
        b = rolling_beta(returns_df[col], market_returns, window=60)
        valid = b.dropna()
        betas[col] = float(valid.iloc[-1]) if len(valid) > 0 else 1.0

    return {
        "weights": dict(zip(returns_df.columns, weights.tolist(), strict=False)),
        "optimization": opt_result,
        "component_var": comp_var,
        "diversification_ratio": div_ratio,
        "betas": betas,
        "regime_adjusted": regime_aware,
    }
