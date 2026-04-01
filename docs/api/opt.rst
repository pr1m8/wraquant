Optimization (``wraquant.opt``)
===============================

Portfolio and mathematical optimization: mean-variance optimization, risk
parity, Black-Litterman, Hierarchical Risk Parity, convex/nonlinear solvers,
and multi-objective optimization.

**Portfolio methods:**

- ``max_sharpe`` -- maximum Sharpe ratio portfolio
- ``min_volatility`` -- minimum variance portfolio
- ``risk_parity`` -- equal risk contribution
- ``black_litterman`` -- equilibrium + views blending
- ``hierarchical_risk_parity`` -- clustering-based diversification (Lopez de Prado)
- ``equal_weight`` / ``inverse_volatility`` -- naive diversification baselines

**Optimization solvers:**

- Convex: QP, SOCP, SDP
- Linear: LP, MILP
- Nonlinear: local and global optimization
- Multi-objective: Pareto front, NSGA-II, epsilon-constraint

Quick Example
-------------

.. code-block:: python

   from wraquant.opt import max_sharpe, risk_parity, black_litterman

   # Max Sharpe portfolio
   result = max_sharpe(returns_df)
   print(f"Weights: {result['weights']}")
   print(f"Sharpe:  {result['sharpe_ratio']:.4f}")

   # Risk parity (equal risk contribution)
   rp = risk_parity(returns_df)
   print(f"Risk parity weights: {rp['weights']}")

   # Black-Litterman with views
   bl = black_litterman(returns_df, market_caps, views={"AAPL": 0.12})
   print(f"BL weights: {bl['weights']}")

   # HRP (no covariance inversion needed)
   from wraquant.opt import hierarchical_risk_parity
   hrp = hierarchical_risk_parity(returns_df)

Constraints
^^^^^^^^^^^^

.. code-block:: python

   from wraquant.opt import weight_constraint, sector_constraints, turnover_constraint

   # Weight bounds
   w_bounds = weight_constraint(lower=0.02, upper=0.30)

   # Sector constraints
   sectors = {"Tech": ["AAPL", "MSFT"], "Finance": ["JPM", "GS"]}
   s_bounds = sector_constraints(sectors, max_sector=0.40)

   # Turnover constraint (limit rebalancing costs)
   t_bound = turnover_constraint(max_turnover=0.20, current_weights=old_weights)

.. seealso::

   - :doc:`/tutorials/portfolio_construction` -- Full portfolio construction tutorial
   - :doc:`risk` -- Risk decomposition for optimized portfolios
   - :doc:`regimes` -- Regime-conditional optimization

API Reference
-------------

.. automodule:: wraquant.opt
   :members:
   :undoc-members:
   :show-inheritance:

Portfolio Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.opt.portfolio
   :members:

Convex Optimization
~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.opt.convex
   :members:

Linear Programming
~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.opt.linear
   :members:

Nonlinear Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.opt.nonlinear
   :members:

Multi-Objective Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wraquant.opt.multi_objective
   :members:

Base Classes
~~~~~~~~~~~~

.. automodule:: wraquant.opt.base
   :members:

Utilities
~~~~~~~~~

.. automodule:: wraquant.opt.utils
   :members:
