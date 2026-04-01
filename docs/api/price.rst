Pricing (``wraquant.price``)
============================

The pricing module provides 50+ functions for derivatives pricing, fixed
income analytics, stochastic process simulation, and advanced numerical
methods including FBSDE solvers and characteristic function pricing.

**Submodules:**

- **Options** -- Black-Scholes, binomial tree, Monte Carlo
- **Greeks** -- analytical delta, gamma, theta, vega, rho
- **Volatility surfaces** -- implied vol, smile, SVI parameterization
- **Fixed income** -- bond pricing, duration, convexity, yield curves
- **Levy pricing** -- FFT and COS-method for VG, NIG, CGMY
- **Characteristic functions** -- Heston, VG, NIG, CGMY constructors
- **FBSDE solvers** -- Forward-Backward SDE for European/American derivatives
- **Stochastic processes** -- GBM, Heston, jump-diffusion, SABR, rough Bergomi, CIR, Vasicek

Quick Example
-------------

.. code-block:: python

   from wraquant.price import black_scholes, all_greeks, implied_volatility

   # Black-Scholes European call
   price = black_scholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="call")
   print(f"Call price: ${price:.4f}")

   # All Greeks at once
   greeks = all_greeks(S=100, K=105, T=0.25, r=0.05, sigma=0.20)
   print(f"Delta: {greeks['delta']:.4f}")
   print(f"Gamma: {greeks['gamma']:.4f}")
   print(f"Theta: {greeks['theta']:.4f}")
   print(f"Vega:  {greeks['vega']:.4f}")

   # Implied volatility from market price
   iv = implied_volatility(market_price=3.50, S=100, K=105, T=0.25, r=0.05)
   print(f"Implied vol: {iv:.2%}")

Stochastic Models
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.price import heston, geometric_brownian_motion, simulate_sabr

   # Simulate Heston stochastic volatility paths
   paths = heston(S0=100, v0=0.04, mu=0.05, kappa=2.0,
                  theta=0.04, sigma=0.3, rho=-0.7, T=1.0, n_paths=10000)
   print(f"Heston terminal mean: {paths['terminal'].mean():.2f}")

   # SABR for interest rate vol
   sabr_paths = simulate_sabr(f0=0.03, alpha=0.2, beta=0.5,
                              rho=-0.3, vol_vol=0.4, T=1.0)

Characteristic Function Pricing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.price import heston_characteristic, characteristic_function_price

   # Price using Heston characteristic function (FFT-based)
   cf = heston_characteristic(v0=0.04, kappa=2.0, theta=0.04,
                              sigma=0.3, rho=-0.7)
   price = characteristic_function_price(cf, S=100, K=105, T=0.25, r=0.05)
   print(f"Heston price: ${price:.4f}")

Fixed Income
^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.price import bond_price, duration, convexity

   price = bond_price(face=1000, coupon_rate=0.05, ytm=0.04, maturity=10)
   dur = duration(face=1000, coupon_rate=0.05, ytm=0.04, maturity=10)
   conv = convexity(face=1000, coupon_rate=0.05, ytm=0.04, maturity=10)
   print(f"Bond price: ${price:.2f}")
   print(f"Duration: {dur:.4f} years")
   print(f"Convexity: {conv:.4f}")

.. seealso::

   - :doc:`vol` -- Volatility models for pricing inputs
   - :doc:`risk` -- VaR using pricing models
   - :doc:`math` -- Levy processes and numerical methods

API Reference
-------------

.. automodule:: wraquant.price
   :members:
   :undoc-members:
   :show-inheritance:

Options Pricing
~~~~~~~~~~~~~~~

.. automodule:: wraquant.price.options
   :members:

Greeks
~~~~~~

.. automodule:: wraquant.price.greeks
   :members:

Volatility Surfaces
~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.price.volatility
   :members:

Fixed Income
~~~~~~~~~~~~

.. automodule:: wraquant.price.fixed_income
   :members:

Yield Curves
~~~~~~~~~~~~

.. automodule:: wraquant.price.curves
   :members:

Stochastic Processes
~~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.price.stochastic
   :members:

Levy Process Pricing
~~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.price.levy_pricing
   :members:

Characteristic Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.price.characteristic
   :members:

FBSDE Methods
~~~~~~~~~~~~~

.. automodule:: wraquant.price.fbsde
   :members:

Integrations
~~~~~~~~~~~~

.. automodule:: wraquant.price.integrations
   :members:
