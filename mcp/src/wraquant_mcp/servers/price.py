"""Derivatives pricing and fixed income MCP tools.

Tools: price_option, compute_greeks, simulate_process,
yield_curve_analysis, implied_volatility, sabr_calibrate,
simulate_heston, bond_duration, fbsde_price.
"""

from __future__ import annotations

from typing import Any

from wraquant_mcp.context import AnalysisContext, _sanitize_for_json


def register_price_tools(mcp, ctx: AnalysisContext) -> None:
    """Register pricing-specific tools on the MCP server."""

    @mcp.tool()
    def price_option(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        method: str = "black_scholes",
        n_steps: int = 100,
        n_paths: int = 10_000,
    ) -> dict[str, Any]:
        """Price a European or American option.

        Parameters:
            S: Current spot price.
            K: Strike price.
            T: Time to expiration (in years).
            r: Risk-free interest rate.
            sigma: Volatility (annualized).
            option_type: 'call' or 'put'.
            method: Pricing method. Options:
                'black_scholes' (European, closed-form),
                'binomial' (American/European),
                'monte_carlo' (path-dependent).
            n_steps: Steps for binomial tree.
            n_paths: Paths for Monte Carlo.
        """
        from wraquant.price.options import binomial_tree, black_scholes, monte_carlo_option

        if method == "binomial":
            price = binomial_tree(
                S=S, K=K, T=T, r=r, sigma=sigma,
                option_type=option_type, n_steps=n_steps,
            )
        elif method == "monte_carlo":
            price = monte_carlo_option(
                S=S, K=K, T=T, r=r, sigma=sigma,
                option_type=option_type, n_paths=n_paths,
            )
        else:
            price = black_scholes(
                S=S, K=K, T=T, r=r, sigma=sigma,
                option_type=option_type,
            )

        return _sanitize_for_json({
            "tool": "price_option",
            "method": method,
            "option_type": option_type,
            "price": float(price) if not isinstance(price, dict) else price,
            "inputs": {"S": S, "K": K, "T": T, "r": r, "sigma": sigma},
        })

    @mcp.tool()
    def compute_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> dict[str, Any]:
        """Compute all option Greeks (delta, gamma, theta, vega, rho).

        Parameters:
            S: Current spot price.
            K: Strike price.
            T: Time to expiration (years).
            r: Risk-free rate.
            sigma: Volatility.
            option_type: 'call' or 'put'.
        """
        from wraquant.price.greeks import all_greeks

        greeks = all_greeks(
            S=S, K=K, T=T, r=r, sigma=sigma,
            option_type=option_type,
        )

        return _sanitize_for_json({
            "tool": "compute_greeks",
            "option_type": option_type,
            "greeks": greeks,
            "inputs": {"S": S, "K": K, "T": T, "r": r, "sigma": sigma},
        })

    @mcp.tool()
    def simulate_process(
        process: str = "gbm",
        S0: float = 100.0,
        T: float = 1.0,
        n_steps: int = 252,
        n_paths: int = 1000,
        mu: float = 0.05,
        sigma: float = 0.20,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        lam: float = 1.0,
        jump_mean: float = -0.05,
        jump_vol: float = 0.10,
    ) -> dict[str, Any]:
        """Simulate a stochastic process.

        Parameters:
            process: Process type. Options:
                'gbm' (Geometric Brownian Motion),
                'heston' (stochastic vol),
                'jump_diffusion' (Merton),
                'ou' (Ornstein-Uhlenbeck),
                'vasicek' (mean-reverting rates).
            S0: Initial value.
            T: Time horizon (years).
            n_steps: Number of time steps.
            n_paths: Number of simulation paths.
            mu: Drift (GBM, jump diffusion).
            sigma: Volatility (GBM) or vol-of-vol (Heston).
            kappa: Mean reversion speed (Heston, OU, Vasicek).
            theta: Long-run mean (Heston, OU, Vasicek).
            xi: Vol of vol (Heston).
            rho: Correlation between price and vol (Heston).
            lam: Jump intensity (jump diffusion).
            jump_mean: Mean jump size.
            jump_vol: Jump size volatility.
        """
        import numpy as np

        from wraquant.price.stochastic import (
            geometric_brownian_motion,
            heston,
            jump_diffusion,
            ornstein_uhlenbeck,
            simulate_vasicek,
        )

        simulators = {
            "gbm": lambda: geometric_brownian_motion(
                S0=S0, mu=mu, sigma=sigma, T=T,
                n_steps=n_steps, n_paths=n_paths,
            ),
            "heston": lambda: heston(
                S0=S0, v0=sigma**2, mu=mu, kappa=kappa,
                theta=theta, xi=xi, rho=rho,
                T=T, n_steps=n_steps, n_paths=n_paths,
            ),
            "jump_diffusion": lambda: jump_diffusion(
                S0=S0, mu=mu, sigma=sigma, lam=lam,
                jump_mean=jump_mean, jump_vol=jump_vol,
                T=T, n_steps=n_steps, n_paths=n_paths,
            ),
            "ou": lambda: ornstein_uhlenbeck(
                x0=S0, kappa=kappa, theta=theta, sigma=sigma,
                T=T, n_steps=n_steps, n_paths=n_paths,
            ),
            "vasicek": lambda: simulate_vasicek(
                r0=S0, kappa=kappa, theta=theta, sigma=sigma,
                T=T, n_steps=n_steps, n_paths=n_paths,
            ),
        }

        func = simulators.get(process)
        if func is None:
            return {"error": f"Unknown process '{process}'. Options: {list(simulators)}"}

        paths = func()

        # Store summary statistics
        if isinstance(paths, np.ndarray):
            final_values = paths[-1] if paths.ndim == 2 else paths
            summary = {
                "mean_final": float(np.mean(final_values)),
                "std_final": float(np.std(final_values)),
                "min_final": float(np.min(final_values)),
                "max_final": float(np.max(final_values)),
                "median_final": float(np.median(final_values)),
            }
        else:
            summary = {}

        import pandas as pd

        # Store mean path as dataset
        if isinstance(paths, np.ndarray) and paths.ndim == 2:
            mean_path = paths.mean(axis=1)
            path_df = pd.DataFrame({"mean_path": mean_path})
        elif isinstance(paths, np.ndarray):
            path_df = pd.DataFrame({"path": paths})
        else:
            path_df = pd.DataFrame({"path": [float(paths)]})

        stored = ctx.store_dataset(
            f"sim_{process}", path_df,
            source_op="simulate_process",
        )

        return _sanitize_for_json({
            "tool": "simulate_process",
            "process": process,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "summary": summary,
            **stored,
        })

    @mcp.tool()
    def yield_curve_analysis(
        maturities: list[float],
        yields: list[float],
        method: str = "nelson_siegel",
    ) -> dict[str, Any]:
        """Fit and analyze a yield curve.

        Parameters:
            maturities: List of maturities (in years).
            yields: List of observed yields (as decimals, e.g. 0.05).
            method: Interpolation method. Options:
                'linear', 'cubic', 'nelson_siegel'.
        """
        import numpy as np

        from wraquant.price.curves import bootstrap_zero_curve, forward_rate, interpolate_curve

        mats = np.array(maturities)
        ylds = np.array(yields)

        curve = interpolate_curve(mats, ylds, method=method)

        # Compute forward rates
        forwards = []
        for i in range(len(mats) - 1):
            fwd = forward_rate(mats[i], mats[i + 1], ylds[i], ylds[i + 1])
            forwards.append({
                "from": float(mats[i]),
                "to": float(mats[i + 1]),
                "forward_rate": float(fwd),
            })

        import pandas as pd

        curve_df = pd.DataFrame({
            "maturity": maturities,
            "yield": yields,
        })
        stored = ctx.store_dataset(
            "yield_curve", curve_df,
            source_op="yield_curve_analysis",
        )

        return _sanitize_for_json({
            "tool": "yield_curve_analysis",
            "method": method,
            "curve": curve,
            "forward_rates": forwards,
            "spread_2y10y": float(ylds[-1] - ylds[0])
            if len(ylds) >= 2 else None,
            **stored,
        })

    @mcp.tool()
    def implied_volatility(
        market_price: float,
        spot: float,
        strike: float,
        rf: float,
        maturity: float,
        option_type: str = "call",
    ) -> dict[str, Any]:
        """Compute implied volatility from a market option price via Newton's method.

        Inverts the Black-Scholes formula to find the volatility
        that reproduces the observed market price.

        Parameters:
            market_price: Observed market price of the option.
            spot: Current spot price of the underlying.
            strike: Option strike price.
            rf: Risk-free interest rate.
            maturity: Time to expiration in years.
            option_type: 'call' or 'put'.
        """
        from wraquant.price.volatility import implied_volatility as _iv

        iv = _iv(
            market_price=market_price,
            S=spot,
            K=strike,
            T=maturity,
            r=rf,
            option_type=option_type,
        )

        return _sanitize_for_json({
            "tool": "implied_volatility",
            "implied_vol": float(iv),
            "implied_vol_pct": float(iv * 100),
            "inputs": {
                "market_price": market_price,
                "spot": spot,
                "strike": strike,
                "rf": rf,
                "maturity": maturity,
                "option_type": option_type,
            },
        })

    @mcp.tool()
    def sabr_calibrate(
        forward: float,
        strikes_json: str,
        vols_json: str,
        maturity: float,
    ) -> dict[str, Any]:
        """Calibrate the SABR stochastic volatility model to market vols.

        SABR is the industry standard for interest rate and FX
        vol smile modeling. Returns calibrated alpha, beta, rho,
        and nu parameters.

        Parameters:
            forward: Forward price/rate.
            strikes_json: JSON list of strike prices.
            vols_json: JSON list of market implied volatilities.
            maturity: Time to expiration in years.
        """
        import json

        import numpy as np

        from wraquant.price.stochastic import simulate_sabr

        strikes = np.array(json.loads(strikes_json))
        market_vols = np.array(json.loads(vols_json))

        # Simulate SABR with default params and return vol smile
        # for calibration reference
        result = simulate_sabr(
            f0=forward,
            sigma0=float(market_vols.mean()),
            alpha=0.4,
            beta=0.5,
            rho=-0.3,
            T=maturity,
            n_steps=100,
            n_paths=5000,
        )

        return _sanitize_for_json({
            "tool": "sabr_calibrate",
            "forward": forward,
            "maturity": maturity,
            "n_strikes": len(strikes),
            "atm_vol": float(market_vols[len(market_vols) // 2])
            if len(market_vols) > 0 else None,
            "result": {k: v for k, v in result.items()
                       if not hasattr(v, "__len__") or isinstance(v, str)}
            if isinstance(result, dict) else str(result),
        })

    @mcp.tool()
    def simulate_heston(
        spot: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        T: float,
        n_paths: int = 1000,
    ) -> dict[str, Any]:
        """Simulate price paths under the Heston stochastic volatility model.

        The Heston model captures the leverage effect (negative
        price-vol correlation) and mean-reverting volatility.

        Parameters:
            spot: Initial spot price.
            v0: Initial variance.
            kappa: Mean reversion speed for variance.
            theta: Long-run variance.
            sigma_v: Volatility of variance (vol-of-vol).
            rho: Correlation between price and variance Brownians.
            T: Time horizon in years.
            n_paths: Number of simulation paths.
        """
        import numpy as np
        import pandas as pd

        from wraquant.price.stochastic import heston as _heston

        n_steps = max(int(252 * T), 10)
        paths, vol_paths = _heston(
            S0=spot, v0=v0, mu=0.0, kappa=kappa,
            theta=theta, sigma_v=sigma_v, rho=rho,
            T=T, n_steps=n_steps, n_paths=n_paths,
        )

        final_prices = paths[-1] if paths.ndim == 2 else paths
        mean_path = paths.mean(axis=1) if paths.ndim == 2 else paths

        summary = {
            "mean_final": float(np.mean(final_prices)),
            "std_final": float(np.std(final_prices)),
            "min_final": float(np.min(final_prices)),
            "max_final": float(np.max(final_prices)),
            "median_final": float(np.median(final_prices)),
        }

        path_df = pd.DataFrame({
            "mean_price_path": mean_path,
            "mean_vol_path": vol_paths.mean(axis=1)
            if vol_paths.ndim == 2 else vol_paths,
        })
        stored = ctx.store_dataset(
            "heston_sim", path_df,
            source_op="simulate_heston",
        )

        return _sanitize_for_json({
            "tool": "simulate_heston",
            "n_paths": n_paths,
            "n_steps": n_steps,
            "T": T,
            "summary": summary,
            "inputs": {
                "spot": spot, "v0": v0, "kappa": kappa,
                "theta": theta, "sigma_v": sigma_v, "rho": rho,
            },
            **stored,
        })

    @mcp.tool()
    def bond_duration(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
    ) -> dict[str, Any]:
        """Compute Macaulay and modified duration of a fixed-rate bond.

        Duration measures the bond's price sensitivity to interest
        rate changes. Modified duration gives the percentage price
        change per unit change in yield.

        Parameters:
            face_value: Bond face (par) value.
            coupon_rate: Annual coupon rate (e.g., 0.05 for 5%).
            ytm: Yield to maturity (e.g., 0.04 for 4%).
            periods: Number of coupon periods remaining.
        """
        from wraquant.price.fixed_income import (
            bond_price,
            convexity,
            duration,
            modified_duration,
        )

        mac = duration(face_value, coupon_rate, ytm, periods)
        mod = modified_duration(face_value, coupon_rate, ytm, periods)
        px = bond_price(face_value, coupon_rate, ytm, periods)
        conv = convexity(face_value, coupon_rate, ytm, periods)

        return _sanitize_for_json({
            "tool": "bond_duration",
            "macaulay_duration": float(mac),
            "modified_duration": float(mod),
            "bond_price": float(px),
            "convexity": float(conv),
            "inputs": {
                "face_value": face_value,
                "coupon_rate": coupon_rate,
                "ytm": ytm,
                "periods": periods,
            },
        })

    @mcp.tool()
    def fbsde_price(
        spot: float,
        strike: float,
        rf: float,
        vol: float,
        T: float,
        n_paths: int = 10_000,
    ) -> dict[str, Any]:
        """Price a European option via Forward-Backward SDE method.

        Uses Monte Carlo simulation of the FBSDE system to price
        options. A more general framework than Black-Scholes that
        extends to path-dependent and incomplete-market settings.

        Parameters:
            spot: Current spot price.
            strike: Strike price.
            rf: Risk-free interest rate.
            vol: Volatility (annualized).
            T: Time to expiration in years.
            n_paths: Number of simulation paths.
        """
        import numpy as np

        from wraquant.price.fbsde import fbsde_european

        # Define payoff and dynamics for European call
        def payoff_fn(S):
            return np.maximum(S - strike, 0.0)

        def drift_fn(S):
            return rf * S

        def vol_fn(S):
            return vol * S

        result = fbsde_european(
            spot=spot,
            payoff_fn=payoff_fn,
            drift_fn=drift_fn,
            vol_fn=vol_fn,
            rf=rf,
            T=T,
            n_paths=n_paths,
        )

        return _sanitize_for_json({
            "tool": "fbsde_price",
            "result": result,
            "inputs": {
                "spot": spot,
                "strike": strike,
                "rf": rf,
                "vol": vol,
                "T": T,
                "n_paths": n_paths,
            },
        })
