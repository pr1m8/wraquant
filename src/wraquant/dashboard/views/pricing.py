"""Options Pricing & Fixed Income dashboard view.

Provides an interactive pricing laboratory with six tabs:

- **Options Calculator**: Black-Scholes pricer with full Greeks table
  and put-call parity verification.
- **Greeks Visualizer**: Plot delta/gamma/theta/vega across a range of
  spot prices with interactive sliders for strike, volatility, and time.
- **Vol Surface**: 3D implied-volatility surface (strike x maturity x IV)
  using synthetic option grids.
- **Payoff Diagram**: Strategy payoff at expiry for calls, puts,
  straddles, strangles, and vertical spreads.
- **Bond Analysis**: Fixed-income calculator for price, yield, duration,
  modified duration, and convexity.
- **Yield Curve**: Treasury-style yield curve with bootstrapped zero
  rates and forward rates.
"""

from __future__ import annotations

import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _compute_greeks_surface(
    S_range: tuple[float, float, int],
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> dict:
    """Compute Greeks across a range of spot prices."""
    try:
        from wraquant.price.greeks import all_greeks

        spots = np.linspace(S_range[0], S_range[1], S_range[2])
        results: dict[str, list[float]] = {
            "spot": spots.tolist(),
            "delta_call": [],
            "delta_put": [],
            "gamma": [],
            "theta_call": [],
            "theta_put": [],
            "vega": [],
        }
        for s in spots:
            gc = all_greeks(float(s), K, T, r, sigma, "call")
            gp = all_greeks(float(s), K, T, r, sigma, "put")
            results["delta_call"].append(gc["delta"])
            results["delta_put"].append(gp["delta"])
            results["gamma"].append(gc["gamma"])
            results["theta_call"].append(gc["theta"])
            results["theta_put"].append(gp["theta"])
            results["vega"].append(gc["vega"])
        return results
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _compute_vol_surface(
    S: float,
    r: float,
    base_sigma: float,
) -> dict:
    """Build a synthetic implied-volatility surface."""
    try:
        from wraquant.price.volatility import vol_surface

        strikes = np.linspace(S * 0.80, S * 1.20, 15)
        expiries = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

        # Generate synthetic BS prices with a smile/skew baked in
        from wraquant.price.options import black_scholes

        prices = np.zeros((len(expiries), len(strikes)))
        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                # Skew: OTM puts have higher vol, OTM calls lower
                moneyness = np.log(K / S)
                skew_adj = -0.15 * moneyness + 0.03 * moneyness**2
                term_adj = 0.02 * (1.0 / max(T, 0.05) - 1.0)
                local_sigma = max(base_sigma + skew_adj + term_adj, 0.05)
                prices[i, j] = float(
                    black_scholes(S, float(K), float(T), r, local_sigma, "call")
                )

        result = vol_surface(
            strikes.tolist(),
            expiries.tolist(),
            prices.tolist(),
            S,
            r,
            "call",
        )
        return {
            "strikes": result["strikes"].tolist(),
            "expiries": result["expiries"].tolist(),
            "ivs": result["ivs"].tolist(),
        }
    except Exception:
        # Fallback: return the synthetic vols directly
        strikes = np.linspace(S * 0.80, S * 1.20, 15)
        expiries = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        ivs = np.zeros((len(expiries), len(strikes)))
        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                moneyness = np.log(K / S)
                skew_adj = -0.15 * moneyness + 0.03 * moneyness**2
                term_adj = 0.02 * (1.0 / max(T, 0.05) - 1.0)
                ivs[i, j] = max(base_sigma + skew_adj + term_adj, 0.05)
        return {
            "strikes": strikes.tolist(),
            "expiries": expiries.tolist(),
            "ivs": ivs.tolist(),
        }


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def _tab_options_calculator() -> None:
    """Black-Scholes options calculator with Greeks table and put-call parity."""
    from wraquant.dashboard.components.charts import COLORS, dark_layout

    st.markdown("### Black-Scholes Options Calculator")

    col_inputs, col_results = st.columns([1, 2])

    with col_inputs:
        S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, step=1.0)
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=1.0)
        T = st.number_input(
            "Time to Expiry (years)", value=0.25, min_value=0.01, max_value=10.0, step=0.01
        )
        r = st.number_input(
            "Risk-Free Rate", value=0.05, min_value=0.0, max_value=0.5, step=0.005, format="%.4f"
        )
        sigma = st.number_input(
            "Volatility (sigma)", value=0.20, min_value=0.01, max_value=3.0, step=0.01, format="%.4f"
        )

    with col_results:
        try:
            from wraquant.price.greeks import all_greeks
            from wraquant.price.options import black_scholes

            call_price = float(black_scholes(S, K, T, r, sigma, "call"))
            put_price = float(black_scholes(S, K, T, r, sigma, "put"))
            greeks_call = all_greeks(S, K, T, r, sigma, "call")
            greeks_put = all_greeks(S, K, T, r, sigma, "put")

            # -- Prices --
            p1, p2 = st.columns(2)
            p1.metric("Call Price", f"${call_price:.4f}")
            p2.metric("Put Price", f"${put_price:.4f}")

            # -- Greeks table --
            st.markdown("#### Greeks")
            greek_names = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
            greek_keys = ["delta", "gamma", "theta", "vega", "rho"]
            import pandas as pd

            df_greeks = pd.DataFrame(
                {
                    "Greek": greek_names,
                    "Call": [greeks_call[k] for k in greek_keys],
                    "Put": [greeks_put[k] for k in greek_keys],
                }
            )
            df_greeks["Call"] = df_greeks["Call"].map(lambda x: f"{x:.6f}")
            df_greeks["Put"] = df_greeks["Put"].map(lambda x: f"{x:.6f}")
            st.dataframe(df_greeks, use_container_width=True, hide_index=True)

            # -- Put-call parity --
            st.markdown("#### Put-Call Parity Check")
            lhs = call_price - put_price
            rhs = S - K * np.exp(-r * T)
            parity_diff = abs(lhs - rhs)
            parity_ok = parity_diff < 1e-6

            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("C - P", f"${lhs:.6f}")
            pc2.metric("S - K*exp(-rT)", f"${rhs:.6f}")
            pc3.metric(
                "Parity Error",
                f"${parity_diff:.2e}",
                delta="PASS" if parity_ok else "FAIL",
                delta_color="normal" if parity_ok else "inverse",
            )
        except Exception as exc:
            st.warning(f"Pricing computation failed: {exc}")


def _tab_greeks_visualizer() -> None:
    """Plot Greeks as a function of spot price with interactive controls."""
    from wraquant.dashboard.components.charts import COLORS, SERIES_COLORS, dark_layout

    st.markdown("### Greeks vs Spot Price")

    c1, c2, c3 = st.columns(3)
    with c1:
        K = st.slider("Strike Price", 50.0, 200.0, 100.0, 1.0, key="greeks_K")
    with c2:
        sigma = st.slider("Volatility", 0.05, 1.0, 0.20, 0.01, key="greeks_sigma")
    with c3:
        T = st.slider("Time to Expiry (yrs)", 0.01, 3.0, 0.25, 0.01, key="greeks_T")

    r = 0.05
    S_min, S_max = max(K * 0.5, 1.0), K * 1.5
    n_pts = 200

    data = _compute_greeks_surface((S_min, S_max, n_pts), K, T, r, sigma)
    if not data:
        st.warning("Could not compute Greeks. Check that wraquant.price is available.")
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        spots = data["spot"]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Delta", "Gamma", "Theta", "Vega"],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )

        # Delta
        fig.add_trace(
            go.Scatter(x=spots, y=data["delta_call"], name="Call Delta",
                       line={"color": COLORS["success"], "width": 2}),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=spots, y=data["delta_put"], name="Put Delta",
                       line={"color": COLORS["danger"], "width": 2}),
            row=1, col=1,
        )

        # Gamma
        fig.add_trace(
            go.Scatter(x=spots, y=data["gamma"], name="Gamma",
                       line={"color": COLORS["primary"], "width": 2}),
            row=1, col=2,
        )

        # Theta
        fig.add_trace(
            go.Scatter(x=spots, y=data["theta_call"], name="Call Theta",
                       line={"color": COLORS["warning"], "width": 2}),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=spots, y=data["theta_put"], name="Put Theta",
                       line={"color": COLORS["accent1"], "width": 2}),
            row=2, col=1,
        )

        # Vega
        fig.add_trace(
            go.Scatter(x=spots, y=data["vega"], name="Vega",
                       line={"color": COLORS["info"], "width": 2}),
            row=2, col=2,
        )

        # Add strike line to each subplot
        for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            fig.add_vline(
                x=K, line_dash="dash", line_color=COLORS["text_muted"],
                opacity=0.5, row=row, col=col,
            )

        layout = dark_layout(
            title=f"Greeks | K={K:.0f}  sigma={sigma:.0%}  T={T:.2f}y",
            height=550,
            showlegend=True,
        )
        fig.update_layout(**layout)

        # Axis labels
        for i in range(1, 5):
            fig.update_xaxes(
                title_text="Spot Price" if i > 2 else "",
                gridcolor="rgba(255,255,255,0.06)", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1,
            )
            fig.update_yaxes(
                gridcolor="rgba(255,255,255,0.06)", row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1,
            )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Chart rendering failed: {exc}")


def _tab_vol_surface() -> None:
    """3D implied-volatility surface."""
    from wraquant.dashboard.components.charts import COLORS, dark_layout

    st.markdown("### Implied Volatility Surface")
    st.caption(
        "Synthetic vol surface showing strike-maturity-IV relationship "
        "with realistic skew and term structure."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        S = st.number_input("Underlying Price", value=100.0, min_value=1.0, step=1.0, key="vs_S")
    with c2:
        base_sigma = st.slider(
            "Base Volatility", 0.10, 0.80, 0.25, 0.01, key="vs_sigma"
        )
    with c3:
        r = st.number_input(
            "Risk-Free Rate", value=0.05, min_value=0.0, max_value=0.5, step=0.005, key="vs_r",
            format="%.4f",
        )

    surface = _compute_vol_surface(S, r, base_sigma)

    try:
        import plotly.graph_objects as go

        strikes = np.array(surface["strikes"])
        expiries = np.array(surface["expiries"])
        ivs = np.array(surface["ivs"])

        fig = go.Figure(
            data=[
                go.Surface(
                    x=strikes,
                    y=expiries,
                    z=ivs * 100,  # percentage
                    colorscale="Viridis",
                    colorbar={"title": "IV (%)", "ticksuffix": "%"},
                    hovertemplate=(
                        "Strike: %{x:.0f}<br>"
                        "Expiry: %{y:.2f}y<br>"
                        "IV: %{z:.1f}%<extra></extra>"
                    ),
                )
            ]
        )

        layout = dark_layout(
            title=f"Vol Surface | S={S:.0f}  base IV={base_sigma:.0%}",
            height=550,
        )
        layout["scene"] = {
            "xaxis": {"title": "Strike", "backgroundcolor": COLORS["bg"]},
            "yaxis": {"title": "Expiry (years)", "backgroundcolor": COLORS["bg"]},
            "zaxis": {"title": "IV (%)", "backgroundcolor": COLORS["bg"]},
            "bgcolor": COLORS["bg"],
        }
        layout.pop("hovermode", None)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Vol surface rendering failed: {exc}")


def _tab_payoff_diagram() -> None:
    """Strategy payoff diagram at expiry."""
    from wraquant.dashboard.components.charts import COLORS, dark_layout

    st.markdown("### Options Payoff at Expiry")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        strategy = st.selectbox(
            "Strategy",
            ["Long Call", "Long Put", "Straddle", "Strangle", "Bull Call Spread", "Bear Put Spread"],
            key="payoff_strat",
        )
    with c2:
        S0 = st.number_input("Current Spot", value=100.0, min_value=0.01, step=1.0, key="payoff_S0")
    with c3:
        K1 = st.number_input("Strike 1", value=100.0, min_value=0.01, step=1.0, key="payoff_K1")
    with c4:
        K2 = st.number_input(
            "Strike 2 (spreads/strangle)",
            value=110.0, min_value=0.01, step=1.0, key="payoff_K2",
        )

    premium_col1, premium_col2 = st.columns(2)
    with premium_col1:
        prem1 = st.number_input("Premium 1", value=5.0, min_value=0.0, step=0.25, key="payoff_p1")
    with premium_col2:
        prem2 = st.number_input("Premium 2", value=3.0, min_value=0.0, step=0.25, key="payoff_p2")

    # Compute payoff across spot range
    S_range = np.linspace(max(K1 * 0.5, 1.0), K1 * 1.5, 500)

    if strategy == "Long Call":
        payoff = np.maximum(S_range - K1, 0) - prem1
        desc = f"Long Call K={K1:.0f}, premium={prem1:.2f}"
    elif strategy == "Long Put":
        payoff = np.maximum(K1 - S_range, 0) - prem1
        desc = f"Long Put K={K1:.0f}, premium={prem1:.2f}"
    elif strategy == "Straddle":
        payoff = np.maximum(S_range - K1, 0) + np.maximum(K1 - S_range, 0) - prem1 - prem2
        desc = f"Straddle K={K1:.0f}, total premium={prem1 + prem2:.2f}"
    elif strategy == "Strangle":
        K_low, K_high = min(K1, K2), max(K1, K2)
        payoff = np.maximum(S_range - K_high, 0) + np.maximum(K_low - S_range, 0) - prem1 - prem2
        desc = f"Strangle K_low={K_low:.0f}, K_high={K_high:.0f}"
    elif strategy == "Bull Call Spread":
        K_low, K_high = min(K1, K2), max(K1, K2)
        payoff = np.maximum(S_range - K_low, 0) - np.maximum(S_range - K_high, 0) - (prem1 - prem2)
        desc = f"Bull Call Spread {K_low:.0f}/{K_high:.0f}"
    elif strategy == "Bear Put Spread":
        K_low, K_high = min(K1, K2), max(K1, K2)
        payoff = np.maximum(K_high - S_range, 0) - np.maximum(K_low - S_range, 0) - (prem1 - prem2)
        desc = f"Bear Put Spread {K_low:.0f}/{K_high:.0f}"
    else:
        payoff = np.zeros_like(S_range)
        desc = ""

    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Profit/loss regions
        profit_mask = payoff >= 0
        loss_mask = payoff < 0

        fig.add_trace(
            go.Scatter(
                x=S_range, y=payoff,
                mode="lines",
                name="P&L",
                line={"color": COLORS["primary"], "width": 2.5},
            )
        )

        # Fill profit green, loss red
        fig.add_trace(
            go.Scatter(
                x=S_range, y=np.where(profit_mask, payoff, 0),
                fill="tozeroy",
                fillcolor="rgba(34,197,94,0.15)",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=S_range, y=np.where(loss_mask, payoff, 0),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.15)",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Zero line + current spot
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_muted"], opacity=0.5)
        fig.add_vline(x=S0, line_dash="dot", line_color=COLORS["warning"], opacity=0.7)

        fig.update_layout(
            **dark_layout(
                title=desc,
                xaxis_title="Spot Price at Expiry",
                yaxis_title="Profit / Loss ($)",
                height=450,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Key metrics
        max_profit = float(np.max(payoff))
        max_loss = float(np.min(payoff))
        breakeven_indices = np.where(np.diff(np.sign(payoff)))[0]
        breakevens = [float(S_range[i]) for i in breakeven_indices]

        m1, m2, m3 = st.columns(3)
        m1.metric("Max Profit", f"${max_profit:.2f}" if max_profit < 1e6 else "Unlimited")
        m2.metric("Max Loss", f"${max_loss:.2f}")
        m3.metric(
            "Breakeven(s)",
            ", ".join(f"${b:.2f}" for b in breakevens) if breakevens else "N/A",
        )
    except Exception as exc:
        st.warning(f"Payoff chart failed: {exc}")


def _tab_bond_analysis() -> None:
    """Bond price, yield, duration, and convexity calculator."""
    st.markdown("### Bond Analytics Calculator")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Inputs")
        face = st.number_input("Face Value ($)", value=1000.0, min_value=1.0, step=100.0, key="bond_face")
        coupon_rate = st.number_input(
            "Coupon Rate (annual)", value=0.05, min_value=0.0, max_value=0.30,
            step=0.005, format="%.4f", key="bond_coupon",
        )
        maturity_years = st.number_input(
            "Maturity (years)", value=10.0, min_value=0.5, max_value=50.0,
            step=0.5, key="bond_mat",
        )
        freq = st.selectbox("Coupon Frequency", [1, 2, 4, 12], index=1, key="bond_freq")
        mode = st.radio(
            "Compute",
            ["Price from Yield", "Yield from Price"],
            key="bond_mode",
        )
        if mode == "Price from Yield":
            ytm_input = st.number_input(
                "Yield to Maturity", value=0.05, min_value=0.0, max_value=0.50,
                step=0.005, format="%.4f", key="bond_ytm",
            )
        else:
            price_input = st.number_input(
                "Market Price ($)", value=950.0, min_value=1.0, step=1.0, key="bond_price_in",
            )

    with c2:
        st.markdown("#### Results")
        periods = int(maturity_years * freq)
        try:
            from wraquant.price.fixed_income import (
                bond_price,
                bond_yield,
                convexity,
                duration,
                modified_duration,
            )

            if mode == "Price from Yield":
                ytm = ytm_input
                price = float(bond_price(face, coupon_rate, ytm, periods, freq))
                st.metric("Bond Price", f"${price:,.4f}")
            else:
                price = price_input
                ytm = float(bond_yield(price, face, coupon_rate, periods, freq))
                st.metric("Yield to Maturity", f"{ytm:.4%}")

            dur = float(duration(face, coupon_rate, ytm, periods, freq))
            mod_dur = float(modified_duration(face, coupon_rate, ytm, periods, freq))
            conv = float(convexity(face, coupon_rate, ytm, periods, freq))

            r1, r2 = st.columns(2)
            r1.metric("Macaulay Duration", f"{dur:.4f} yrs")
            r2.metric("Modified Duration", f"{mod_dur:.4f}")
            st.metric("Convexity", f"{conv:.4f}")

            # Price sensitivity estimate
            delta_y = 0.01  # 100 bps
            approx_pct_change = -mod_dur * delta_y + 0.5 * conv * delta_y**2
            st.caption(
                f"Approx. price change for +100bps yield shift: "
                f"{approx_pct_change:.4%} (${price * approx_pct_change:+,.2f})"
            )

            # Premium / discount / par
            if abs(price - face) < 0.01:
                st.info("Bond trades at **par**.")
            elif price > face:
                st.success(f"Bond trades at a **premium** (+${price - face:,.2f}).")
            else:
                st.error(f"Bond trades at a **discount** (-${face - price:,.2f}).")

        except Exception as exc:
            st.warning(f"Bond calculation failed: {exc}")


def _tab_yield_curve() -> None:
    """Plot a yield curve with zero rates and forward rates."""
    from wraquant.dashboard.components.charts import COLORS, dark_layout

    st.markdown("### Yield Curve Analysis")
    st.caption(
        "Synthetic or user-defined par yield curve with bootstrapped "
        "zero rates and implied forward rates."
    )

    curve_preset = st.selectbox(
        "Curve Preset",
        ["Normal (upward sloping)", "Flat", "Inverted", "Humped", "Custom"],
        key="yc_preset",
    )

    maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]

    if curve_preset == "Normal (upward sloping)":
        par_rates = [4.5, 4.6, 4.7, 4.75, 4.8, 4.9, 5.0, 5.1, 5.3, 5.4]
    elif curve_preset == "Flat":
        par_rates = [5.0] * 10
    elif curve_preset == "Inverted":
        par_rates = [5.5, 5.4, 5.3, 5.1, 5.0, 4.8, 4.7, 4.6, 4.5, 4.5]
    elif curve_preset == "Humped":
        par_rates = [4.5, 4.8, 5.1, 5.3, 5.2, 5.0, 4.9, 4.8, 4.7, 4.6]
    else:
        st.markdown("Enter par rates (%) for each maturity:")
        par_rates = []
        cols = st.columns(5)
        for i, m in enumerate(maturities):
            with cols[i % 5]:
                val = st.number_input(
                    f"{m}Y", value=5.0, min_value=0.0, max_value=20.0,
                    step=0.1, key=f"yc_rate_{i}",
                )
                par_rates.append(val)

    # Convert to decimals
    par_rates_dec = [r / 100.0 for r in par_rates]

    try:
        from wraquant.price.curves import bootstrap_zero_curve, forward_rate

        # Bootstrap zero rates — needs evenly spaced maturities at 1/freq
        # Use synthetic even spacing for bootstrapping
        even_mats = np.arange(0.5, 30.5, 0.5)
        even_pars = np.interp(even_mats, maturities, par_rates_dec)
        zeros = bootstrap_zero_curve(even_mats.tolist(), even_pars.tolist(), freq=2)

        # Get zero rates at our target maturities
        zero_at_target = np.interp(maturities, even_mats, zeros)

        # Forward rates between consecutive maturities
        fwd_rates = []
        for i in range(len(maturities) - 1):
            try:
                fwd = float(forward_rate(
                    zeros.tolist(), even_mats.tolist(),
                    maturities[i], maturities[i + 1],
                ))
                fwd_rates.append(fwd)
            except Exception:
                fwd_rates.append(np.nan)
        fwd_maturities = [
            (maturities[i] + maturities[i + 1]) / 2 for i in range(len(maturities) - 1)
        ]

        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=maturities, y=[r * 100 for r in par_rates_dec],
                name="Par Rates",
                mode="lines+markers",
                line={"color": COLORS["primary"], "width": 2.5},
                marker={"size": 6},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=maturities, y=[float(z) * 100 for z in zero_at_target],
                name="Zero Rates",
                mode="lines+markers",
                line={"color": COLORS["success"], "width": 2, "dash": "dash"},
                marker={"size": 5},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fwd_maturities, y=[f * 100 for f in fwd_rates],
                name="Forward Rates",
                mode="lines+markers",
                line={"color": COLORS["warning"], "width": 2, "dash": "dot"},
                marker={"size": 5},
            )
        )

        fig.update_layout(
            **dark_layout(
                title="Yield Curve: Par, Zero, and Forward Rates",
                xaxis_title="Maturity (years)",
                yaxis_title="Rate (%)",
                height=450,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        import pandas as pd

        df = pd.DataFrame(
            {
                "Maturity": maturities,
                "Par Rate (%)": [f"{r:.3f}" for r in par_rates],
                "Zero Rate (%)": [f"{float(z) * 100:.3f}" for z in zero_at_target],
            }
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Yield curve computation failed: {exc}")

        # Fallback: just plot the par rates
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=maturities, y=par_rates,
                    name="Par Rates",
                    mode="lines+markers",
                    line={"color": COLORS["primary"], "width": 2.5},
                    marker={"size": 6},
                )
            )
            fig.update_layout(
                **dark_layout(
                    title="Par Yield Curve",
                    xaxis_title="Maturity (years)",
                    yaxis_title="Rate (%)",
                    height=400,
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Options Pricing & Fixed Income dashboard page."""
    ticker = st.session_state.get("ticker", "AAPL")

    st.markdown("# Options Pricing & Fixed Income")
    st.caption(
        f"Interactive pricing laboratory | Context: **{ticker}** | "
        "Black-Scholes, Greeks, vol surfaces, payoff diagrams, and bond analytics"
    )

    tab_calc, tab_greeks, tab_vol, tab_payoff, tab_bond, tab_yc = st.tabs(
        [
            "Options Calculator",
            "Greeks Visualizer",
            "Vol Surface",
            "Payoff Diagram",
            "Bond Analysis",
            "Yield Curve",
        ]
    )

    with tab_calc:
        _tab_options_calculator()

    with tab_greeks:
        _tab_greeks_visualizer()

    with tab_vol:
        _tab_vol_surface()

    with tab_payoff:
        _tab_payoff_diagram()

    with tab_bond:
        _tab_bond_analysis()

    with tab_yc:
        _tab_yield_curve()
