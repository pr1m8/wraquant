Fundamental Analysis (``wraquant.fundamental``)
================================================

The fundamental module provides 28 functions for financial ratio analysis,
intrinsic valuation, financial statement analysis, and stock screening --
the building blocks of fundamental-driven quant strategies.

**Four areas of coverage:**

1. **Financial ratios** -- Profitability, liquidity, leverage, efficiency,
   valuation, and growth ratios. Includes 3-way and 5-way DuPont
   decomposition and a convenience function that computes all ratios at once.
2. **Valuation models** -- DCF, relative valuation, Graham Number, Peter Lynch
   fair value, Dividend Discount Model, Residual Income Model, Piotroski
   F-Score, and margin of safety analysis.
3. **Financial statement analysis** -- Income, balance sheet, and cash flow
   trend analysis. Composite financial health scoring (0--100 with letter
   grades), earnings quality assessment (accruals analysis), and common-size
   statements.
4. **Stock screening** -- Pre-built screens for value, growth, quality,
   Piotroski F-Score, and Greenblatt's Magic Formula strategies, plus
   a flexible custom screener.

All functions accept a ticker symbol and optionally an ``fmp_client``
parameter to reuse a single FMP provider instance. Requires the
``market-data`` extra and an ``FMP_API_KEY`` environment variable.


Quick Example
-------------

.. code-block:: python

   from wraquant.fundamental import (
       profitability_ratios,
       dcf_valuation,
       financial_health_score,
       value_screen,
   )

   # Profitability ratios for Apple
   prof = profitability_ratios("AAPL")
   print(f"ROE:        {prof['roe']:.2%}")
   print(f"ROIC:       {prof['roic']:.2%}")
   print(f"Net margin: {prof['net_margin']:.2%}")

   # DCF intrinsic value
   dcf = dcf_valuation("AAPL")
   print(f"Intrinsic value: ${dcf['intrinsic_value_per_share']:.2f}")
   print(f"Current price:   ${dcf['current_price']:.2f}")
   print(f"Margin of safety: {dcf['margin_of_safety']:.1%}")

   # Financial health composite score (0-100)
   health = financial_health_score("AAPL")
   print(f"Score: {health['score']:.0f}/100 ({health['grade']})")

   # Screen for value stocks
   stocks = value_screen(max_pe=15, min_dividend_yield=0.03)
   print(f"Found {len(stocks)} value stocks")

DuPont Decomposition
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.fundamental import dupont_decomposition

   # 5-way DuPont: decompose ROE into five drivers
   dupont = dupont_decomposition("MSFT", method="5-way")
   print(f"Tax burden:       {dupont['tax_burden']:.4f}")
   print(f"Interest burden:  {dupont['interest_burden']:.4f}")
   print(f"Operating margin: {dupont['operating_margin']:.4f}")
   print(f"Asset turnover:   {dupont['asset_turnover']:.4f}")
   print(f"Equity multiplier:{dupont['equity_multiplier']:.4f}")
   print(f"ROE (product):    {dupont['roe']:.4f}")

Valuation Models
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.fundamental import (
       graham_number,
       dividend_discount_model,
       relative_valuation,
       piotroski_f_score,
   )

   # Graham Number: conservative intrinsic value
   gn = graham_number("JNJ")
   print(f"Graham Number: ${gn['graham_number']:.2f}")

   # Dividend Discount Model (Gordon growth)
   ddm = dividend_discount_model("KO", cost_of_equity=0.08)
   print(f"DDM value: ${ddm['fair_value']:.2f}")

   # Relative valuation vs. peer group
   rv = relative_valuation("AAPL", peers=["MSFT", "GOOGL", "META"])
   print(f"P/E relative: {rv['pe_relative']:.2f}x")

   # Piotroski F-Score (0-9, higher = better)
   f_score = piotroski_f_score("AAPL")
   print(f"F-Score: {f_score['score']}/9")

Stock Screening
^^^^^^^^^^^^^^^^

.. code-block:: python

   from wraquant.fundamental import (
       value_screen,
       growth_screen,
       quality_factor_screen,
       piotroski_screen,
       magic_formula_screen,
       custom_screen,
   )

   # Pre-built screens
   value_stocks = value_screen(max_pe=12, min_dividend_yield=0.04)
   growth_stocks = growth_screen(min_revenue_growth=0.20)
   quality_stocks = quality_factor_screen(min_roe=0.20, max_debt_equity=0.5)

   # Piotroski: filter by F-Score >= 7
   strong = piotroski_screen(min_score=7)

   # Magic Formula: rank by ROIC + earnings yield
   magic = magic_formula_screen(top_n=30)

   # Custom: arbitrary criteria
   custom = custom_screen(criteria={
       "pe_ratio": {"max": 15},
       "roe": {"min": 0.15},
       "market_cap": {"min": 1e9},
   })

.. seealso::

   - :doc:`/getting_started` -- First analysis walkthrough
   - :doc:`news` -- News sentiment and event-driven analysis
   - :doc:`risk` -- Risk metrics for fundamental-driven portfolios
   - :doc:`ml` -- ML features from fundamental data


API Reference
-------------

.. automodule:: wraquant.fundamental
   :members:
   :undoc-members:
   :show-inheritance:

Ratios
^^^^^^

.. automodule:: wraquant.fundamental.ratios
   :members:

Valuation
^^^^^^^^^

.. automodule:: wraquant.fundamental.valuation
   :members:

Financial Statements
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: wraquant.fundamental.financials
   :members:

Screening
^^^^^^^^^

.. automodule:: wraquant.fundamental.screening
   :members:
