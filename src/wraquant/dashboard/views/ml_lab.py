"""Machine Learning Lab -- feature engineering, model training, results, feature importance.

Interactive ML pipeline using wraquant.ml for financial prediction
with walk-forward validation and proper purging.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_ohlcv(ticker: str, days: int = 1095):
    """Fetch OHLCV data for feature engineering."""
    from datetime import datetime, timedelta

    import pandas as pd

    try:
        from wraquant.data.providers.fmp import FMPClient

        client = FMPClient()
        end = datetime.now()
        start = end - timedelta(days=days)
        df = client.historical_price(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="daily",
        )
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            elif not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except (ValueError, TypeError):
                    pass
            return df
    except Exception:
        pass

    try:
        import yfinance as yf

        data = yf.download(ticker, period="3y", auto_adjust=True, progress=False)
        if not data.empty:
            data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]
            return data
    except Exception:
        pass

    import numpy as np

    rng = np.random.default_rng(42)
    n = min(756, days)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    opn = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(500_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def render() -> None:
    """Render the Machine Learning Lab page."""
    import numpy as np
    import pandas as pd

    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout

    ticker = st.session_state.get("ticker", "AAPL")
    st.markdown(f"# Machine Learning Lab: **{ticker}**")

    with st.spinner(f"Loading {ticker} OHLCV data..."):
        ohlcv = _fetch_ohlcv(ticker)

    if ohlcv is None or len(ohlcv) < 100:
        st.warning("Insufficient data for ML analysis. Need at least 100 observations.")
        return

    # Normalize columns
    col_map = {}
    for c in ohlcv.columns:
        cl = c.lower() if isinstance(c, str) else str(c).lower()
        col_map[c] = cl
    ohlcv = ohlcv.rename(columns=col_map)

    close = ohlcv.get("close")
    if close is None:
        st.warning("Missing close price column.")
        return
    returns = close.pct_change().dropna()

    st.caption(f"{len(ohlcv)} observations available")

    tab_features, tab_train, tab_results, tab_importance = st.tabs([
        "Feature Engineering",
        "Model Training",
        "Results",
        "Feature Importance",
    ])

    # ---- Feature Engineering Tab ----
    with tab_features:
        st.subheader("Feature Selection & Preview")

        feature_groups = st.multiselect(
            "Select feature groups",
            [
                "Returns (lags, log returns)",
                "Rolling Stats (mean, std, skew, kurt)",
                "Technical Indicators (SMA, RSI, MACD)",
                "Volatility (realized vol, EWMA)",
                "Regime (HMM probabilities)",
            ],
            default=[
                "Returns (lags, log returns)",
                "Rolling Stats (mean, std, skew, kurt)",
            ],
            key="ml_feature_groups",
        )

        label_type = st.selectbox(
            "Label type",
            ["Fixed Horizon (binary)", "Triple Barrier"],
            key="ml_label_type",
        )
        label_horizon = st.slider(
            "Label horizon (days)", 1, 20, 5, key="ml_label_horizon"
        )

        # Build features
        features_dfs = []

        if "Returns (lags, log returns)" in feature_groups:
            try:
                from wraquant.ml.features import return_features

                rf = return_features(returns)
                if isinstance(rf, pd.DataFrame):
                    features_dfs.append(rf)
                elif isinstance(rf, dict):
                    features_dfs.append(pd.DataFrame(rf, index=returns.index))
            except Exception:
                # Fallback: manual lag features
                lag_df = pd.DataFrame(index=returns.index)
                for lag in [1, 2, 3, 5, 10, 21]:
                    lag_df[f"ret_lag_{lag}"] = returns.shift(lag)
                lag_df["log_ret"] = np.log(1 + returns)
                features_dfs.append(lag_df)

        if "Rolling Stats (mean, std, skew, kurt)" in feature_groups:
            try:
                from wraquant.ml.features import rolling_features

                rf = rolling_features(returns, windows=[5, 10, 21, 60])
                if isinstance(rf, pd.DataFrame):
                    features_dfs.append(rf)
                elif isinstance(rf, dict):
                    features_dfs.append(pd.DataFrame(rf, index=returns.index))
            except Exception:
                roll_df = pd.DataFrame(index=returns.index)
                for w in [5, 10, 21, 60]:
                    roll_df[f"mean_{w}d"] = returns.rolling(w).mean()
                    roll_df[f"std_{w}d"] = returns.rolling(w).std()
                    roll_df[f"skew_{w}d"] = returns.rolling(w).skew()
                    roll_df[f"kurt_{w}d"] = returns.rolling(w).kurt()
                features_dfs.append(roll_df)

        if "Technical Indicators (SMA, RSI, MACD)" in feature_groups:
            try:
                from wraquant.ml.features import technical_features

                tf = technical_features(ohlcv)
                if isinstance(tf, pd.DataFrame):
                    features_dfs.append(tf)
            except Exception:
                ta_df = pd.DataFrame(index=close.index)
                ta_df["sma_20"] = close.rolling(20).mean() / close - 1
                ta_df["sma_50"] = close.rolling(50).mean() / close - 1
                # Simple RSI approximation
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, 1e-10)
                ta_df["rsi_14"] = 100 - (100 / (1 + rs))
                features_dfs.append(ta_df)

        if "Volatility (realized vol, EWMA)" in feature_groups:
            try:
                from wraquant.ml.features import volatility_features

                vf = volatility_features(returns)
                if isinstance(vf, pd.DataFrame):
                    features_dfs.append(vf)
            except Exception:
                vol_df = pd.DataFrame(index=returns.index)
                vol_df["rvol_5d"] = returns.rolling(5).std() * np.sqrt(252)
                vol_df["rvol_21d"] = returns.rolling(21).std() * np.sqrt(252)
                vol_df["ewma_vol"] = returns.ewm(span=21).std() * np.sqrt(252)
                features_dfs.append(vol_df)

        if "Regime (HMM probabilities)" in feature_groups:
            try:
                from wraquant.ml.features import regime_features

                rgf = regime_features(returns)
                if isinstance(rgf, pd.DataFrame):
                    features_dfs.append(rgf)
            except Exception:
                st.info("Regime features require wraquant.ml.features.regime_features.")

        # Combine and align
        if features_dfs:
            feature_matrix = pd.concat(features_dfs, axis=1)
            feature_matrix = feature_matrix.loc[returns.index].dropna()

            # Build labels
            if label_type == "Fixed Horizon (binary)":
                try:
                    from wraquant.ml.features import label_fixed_horizon

                    labels = label_fixed_horizon(returns, horizon=label_horizon)
                    if isinstance(labels, dict):
                        labels = labels.get("labels", pd.Series(dtype=float))
                except Exception:
                    fwd_ret = returns.shift(-label_horizon).rolling(label_horizon).sum()
                    labels = (fwd_ret > 0).astype(int)
            else:
                try:
                    from wraquant.ml.features import label_triple_barrier

                    labels = label_triple_barrier(close, horizon=label_horizon)
                    if isinstance(labels, dict):
                        labels = labels.get("labels", pd.Series(dtype=float))
                except Exception:
                    fwd_ret = returns.shift(-label_horizon).rolling(label_horizon).sum()
                    labels = (fwd_ret > 0).astype(int)

            # Align
            if isinstance(labels, pd.Series):
                common_idx = feature_matrix.index.intersection(labels.dropna().index)
                feature_matrix = feature_matrix.loc[common_idx]
                labels = labels.loc[common_idx]

            # Store in session state for other tabs
            st.session_state["ml_features"] = feature_matrix
            st.session_state["ml_labels"] = labels
            st.session_state["ml_returns"] = returns

            st.success(f"Feature matrix: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")

            # Preview
            st.dataframe(
                feature_matrix.tail(10).style.format("{:.6f}"),
                use_container_width=True,
            )

            # Label distribution
            if isinstance(labels, pd.Series) and len(labels) > 0:
                lc1, lc2 = st.columns(2)
                vc = labels.value_counts()
                with lc1:
                    st.metric("Label = 1 (up)", f"{vc.get(1, 0)} ({vc.get(1, 0)/len(labels)*100:.1f}%)")
                with lc2:
                    st.metric("Label = 0 (down)", f"{vc.get(0, 0)} ({vc.get(0, 0)/len(labels)*100:.1f}%)")
        else:
            st.info("Select at least one feature group.")

    # ---- Model Training Tab ----
    with tab_train:
        st.subheader("Model Training")

        features = st.session_state.get("ml_features")
        labels = st.session_state.get("ml_labels")

        if features is None or labels is None:
            st.info("Go to the Feature Engineering tab first to build features.")
        else:
            model_type = st.selectbox(
                "Model",
                ["Gradient Boosting", "Random Forest", "SVM (Linear)", "Logistic Regression"],
                key="ml_model_type",
            )

            train_method = st.selectbox(
                "Training method",
                ["Walk-Forward", "Expanding Window"],
                key="ml_train_method",
            )

            test_pct = st.slider(
                "Test set %", 10, 40, 20, 5, key="ml_test_pct"
            )
            n_splits = st.slider(
                "Walk-forward splits", 3, 10, 5, key="ml_n_splits"
            )

            if st.button("Train Model", key="ml_train_btn"):
                with st.spinner("Training..."):
                    X = features.values
                    y = labels.values

                    n = len(X)
                    train_end = int(n * (1 - test_pct / 100))

                    X_train, X_test = X[:train_end], X[train_end:]
                    y_train, y_test = y[:train_end], y[train_end:]

                    # Scale features
                    from sklearn.preprocessing import StandardScaler

                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)

                    # Train model
                    try:
                        if model_type == "Gradient Boosting":
                            try:
                                from wraquant.ml.advanced import gradient_boost_forecast

                                result = gradient_boost_forecast(
                                    pd.DataFrame(X_train_s, columns=features.columns),
                                    pd.Series(y_train),
                                )
                                if isinstance(result, dict):
                                    model_obj = result.get("model")
                                    if model_obj is not None:
                                        y_pred = model_obj.predict(X_test_s)
                                    else:
                                        raise ValueError("No model returned")
                                else:
                                    raise ValueError("Unexpected result format")
                            except Exception:
                                from sklearn.ensemble import GradientBoostingClassifier

                                clf = GradientBoostingClassifier(
                                    n_estimators=100, max_depth=3, random_state=42
                                )
                                clf.fit(X_train_s, y_train)
                                y_pred = clf.predict(X_test_s)
                                model_obj = clf

                        elif model_type == "Random Forest":
                            from sklearn.ensemble import RandomForestClassifier

                            clf = RandomForestClassifier(
                                n_estimators=100, max_depth=5, random_state=42
                            )
                            clf.fit(X_train_s, y_train)
                            y_pred = clf.predict(X_test_s)
                            model_obj = clf

                        elif model_type == "SVM (Linear)":
                            from sklearn.svm import LinearSVC

                            clf = LinearSVC(max_iter=2000, random_state=42)
                            clf.fit(X_train_s, y_train)
                            y_pred = clf.predict(X_test_s)
                            model_obj = clf

                        else:  # Logistic Regression
                            from sklearn.linear_model import LogisticRegression

                            clf = LogisticRegression(max_iter=1000, random_state=42)
                            clf.fit(X_train_s, y_train)
                            y_pred = clf.predict(X_test_s)
                            model_obj = clf

                        # Store results
                        st.session_state["ml_model"] = model_obj
                        st.session_state["ml_y_test"] = y_test
                        st.session_state["ml_y_pred"] = y_pred
                        st.session_state["ml_test_idx"] = features.index[train_end:]
                        st.session_state["ml_feature_names"] = features.columns.tolist()

                        st.success(f"Training complete. Model: {model_type}")

                    except ImportError as e:
                        st.error(f"Required library not installed: {e}")
                    except Exception as e:
                        st.error(f"Training failed: {e}")

                # Walk-forward split visualization
                try:
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    idx = features.index
                    fig.add_trace(
                        go.Scatter(
                            x=idx[:train_end], y=[0] * train_end,
                            mode="markers",
                            marker={"color": COLORS["primary"], "size": 2},
                            name="Train",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=idx[train_end:], y=[0] * (n - train_end),
                            mode="markers",
                            marker={"color": COLORS["danger"], "size": 2},
                            name="Test",
                        )
                    )
                    fig.update_layout(
                        **dark_layout(
                            title="Train / Test Split",
                            height=150,
                            margin={"l": 40, "r": 10, "t": 40, "b": 20},
                        )
                    )
                    fig.update_yaxes(showticklabels=False)
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.caption(f"Train: {train_end} samples, Test: {n - train_end} samples")

    # ---- Results Tab ----
    with tab_results:
        st.subheader("Model Results")

        y_test = st.session_state.get("ml_y_test")
        y_pred = st.session_state.get("ml_y_pred")
        test_idx = st.session_state.get("ml_test_idx")

        if y_test is None or y_pred is None:
            st.info("Train a model first in the Model Training tab.")
        else:
            # Classification metrics
            try:
                from sklearn.metrics import (
                    accuracy_score,
                    confusion_matrix,
                    f1_score,
                    precision_score,
                    recall_score,
                )

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy", f"{acc:.1%}")
                mc2.metric("Precision", f"{prec:.1%}")
                mc3.metric("Recall", f"{rec:.1%}")
                mc4.metric("F1 Score", f"{f1:.3f}")

                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                try:
                    import plotly.graph_objects as go

                    labels = ["Down (0)", "Up (1)"]
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=cm, x=labels, y=labels,
                            colorscale=[[0, COLORS["bg"]], [1, COLORS["primary"]]],
                            text=cm, texttemplate="%{text}",
                            textfont={"size": 16},
                            showscale=False,
                        )
                    )
                    fig.update_layout(
                        **dark_layout(
                            title="Confusion Matrix",
                            xaxis_title="Predicted",
                            yaxis_title="Actual",
                            height=350,
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.dataframe(
                        pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
                        use_container_width=True,
                    )

            except ImportError:
                acc = float((y_test == y_pred).mean())
                st.metric("Accuracy", f"{acc:.1%}")

            # Equity curve from ML signals
            st.subheader("Strategy Equity Curve")
            ml_returns = st.session_state.get("ml_returns")
            if ml_returns is not None and test_idx is not None:
                test_rets = ml_returns.reindex(test_idx).dropna()
                # Align predictions
                min_len = min(len(y_pred), len(test_rets))
                signals = np.where(y_pred[:min_len] == 1, 1, -1)
                strat_rets = test_rets.iloc[:min_len].values * signals
                buy_hold_rets = test_rets.iloc[:min_len].values

                strat_eq = np.cumprod(1 + strat_rets)
                bh_eq = np.cumprod(1 + buy_hold_rets)

                try:
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    x_axis = test_rets.index[:min_len]
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis, y=strat_eq,
                            mode="lines",
                            line={"color": COLORS["primary"], "width": 2},
                            name="ML Strategy",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis, y=bh_eq,
                            mode="lines",
                            line={"color": COLORS["neutral"], "width": 1.5, "dash": "dash"},
                            name="Buy & Hold",
                        )
                    )
                    fig.update_layout(
                        **dark_layout(
                            title="ML Strategy vs Buy & Hold",
                            yaxis_title="Cumulative Return",
                            height=400,
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.line_chart(
                        pd.DataFrame({
                            "ML Strategy": strat_eq,
                            "Buy & Hold": bh_eq,
                        })
                    )

                # Strategy metrics
                strat_ann_ret = float(np.mean(strat_rets) * 252)
                strat_ann_vol = float(np.std(strat_rets) * np.sqrt(252))
                strat_sharpe = strat_ann_ret / strat_ann_vol if strat_ann_vol > 0 else 0
                strat_cum = np.cumprod(1 + strat_rets)
                strat_dd = float((strat_cum / np.maximum.accumulate(strat_cum) - 1).min())

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Strategy Return", f"{strat_ann_ret:.1%}")
                sc2.metric("Strategy Vol", f"{strat_ann_vol:.1%}")
                sc3.metric("Strategy Sharpe", f"{strat_sharpe:.2f}")
                sc4.metric("Strategy Max DD", f"{strat_dd:.1%}")

    # ---- Feature Importance Tab ----
    with tab_importance:
        st.subheader("Feature Importance")

        model_obj = st.session_state.get("ml_model")
        feature_names = st.session_state.get("ml_feature_names")

        if model_obj is None or feature_names is None:
            st.info("Train a model first in the Model Training tab.")
        else:
            importances = None

            # Try wraquant
            try:
                from wraquant.ml.models import feature_importance_mdi

                fi = feature_importance_mdi(model_obj, feature_names)
                if isinstance(fi, dict):
                    importances = fi.get("importances", fi)
                elif isinstance(fi, pd.Series):
                    importances = fi
            except Exception:
                pass

            # Fallback: sklearn feature_importances_
            if importances is None:
                if hasattr(model_obj, "feature_importances_"):
                    importances = pd.Series(
                        model_obj.feature_importances_,
                        index=feature_names,
                    ).sort_values(ascending=False)
                elif hasattr(model_obj, "coef_"):
                    coefs = model_obj.coef_
                    if coefs.ndim > 1:
                        coefs = coefs[0]
                    importances = pd.Series(
                        np.abs(coefs),
                        index=feature_names,
                    ).sort_values(ascending=False)

            if importances is not None:
                if isinstance(importances, dict):
                    importances = pd.Series(importances).sort_values(ascending=False)

                top_n = st.slider("Top N features", 5, min(30, len(importances)), 15, key="ml_top_n")
                top = importances.head(top_n)

                try:
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=top.values[::-1],
                            y=[str(n) for n in top.index[::-1]],
                            orientation="h",
                            marker_color=COLORS["primary"],
                            text=[f"{v:.4f}" for v in top.values[::-1]],
                            textposition="auto",
                        )
                    )
                    fig.update_layout(
                        **dark_layout(
                            title=f"Top {top_n} Feature Importances",
                            xaxis_title="Importance",
                            height=max(350, top_n * 25),
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.dataframe(
                        top.to_frame("Importance").style.format("{:.6f}"),
                        use_container_width=True,
                    )

                # Full table
                st.subheader("All Feature Importances")
                fi_df = importances.to_frame("Importance").reset_index()
                fi_df.columns = ["Feature", "Importance"]
                fi_df["Importance"] = fi_df["Importance"].map(lambda x: f"{x:.6f}")
                st.dataframe(fi_df, hide_index=True, use_container_width=True)

            else:
                st.info("Feature importance not available for this model type.")

            # SHAP (if available)
            st.divider()
            st.subheader("SHAP Analysis")
            try:
                from wraquant.ml.pipeline import feature_importance_shap

                features = st.session_state.get("ml_features")
                if features is not None:
                    shap_result = feature_importance_shap(model_obj, features.iloc[:200])
                    if isinstance(shap_result, dict):
                        shap_vals = shap_result.get("mean_abs_shap", shap_result.get("shap_values", None))
                        if shap_vals is not None and isinstance(shap_vals, pd.Series):
                            top_shap = shap_vals.sort_values(ascending=False).head(top_n)
                            try:
                                import plotly.graph_objects as go

                                fig = go.Figure()
                                fig.add_trace(
                                    go.Bar(
                                        x=top_shap.values[::-1],
                                        y=[str(n) for n in top_shap.index[::-1]],
                                        orientation="h",
                                        marker_color=COLORS["accent2"],
                                    )
                                )
                                fig.update_layout(
                                    **dark_layout(
                                        title=f"SHAP Feature Importance (Top {top_n})",
                                        xaxis_title="Mean |SHAP|",
                                        height=max(350, top_n * 25),
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except ImportError:
                                st.dataframe(top_shap, use_container_width=True)
            except Exception:
                st.info("SHAP analysis requires the `shap` package and wraquant.ml.pipeline.")
