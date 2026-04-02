Changelog
=========

All notable changes to wraquant are documented here.


v1.0.0 (2026-03-28)
--------------------

The **AI-native quant research lab** release. wraquant-mcp turns wraquant
into a tool suite that AI agents can operate directly.

MCP Server (wraquant-mcp)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- 218 hand-crafted tools across 22 module servers
- 327 guided workflow prompt templates
- Shared DuckDB workspace with automatic data resolution and result storage
- Three-tier tool architecture: discovery (5), curated analysis (205),
  auto-registered functions (~900+)
- Supervisor tools: ``recommend_workflow``, ``module_guide``
- Workspace management: create, open, snapshot, restore, delete
- Supports stdio (Claude Desktop) and HTTP (LangChain) transports
- Composable with OpenBB MCP, DuckDB MCP, Jupyter MCP, and Alpaca MCP

Fundamental Analysis Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Ratios** -- ``profitability_ratios``, ``liquidity_ratios``,
  ``leverage_ratios``, ``efficiency_ratios``, ``valuation_ratios``,
  ``growth_ratios``, ``dupont_decomposition`` (3-way and 5-way),
  ``comprehensive_ratios``
- **Valuation** -- ``dcf_valuation``, ``relative_valuation``,
  ``graham_number``, ``peter_lynch_value``, ``dividend_discount_model``,
  ``residual_income_model``, ``margin_of_safety``, ``piotroski_f_score``,
  ``quality_screen``
- **Financials** -- ``income_analysis``, ``balance_sheet_analysis``,
  ``cash_flow_analysis``, ``financial_health_score``,
  ``earnings_quality``, ``common_size_analysis``
- **Screening** -- ``value_screen``, ``growth_screen``,
  ``quality_factor_screen``, ``piotroski_screen``,
  ``magic_formula_screen``, ``custom_screen``
- All backed by FMP (Financial Modeling Prep) data provider

News & Sentiment Module
^^^^^^^^^^^^^^^^^^^^^^^^^

- **Sentiment** -- ``news_sentiment``, ``sentiment_timeseries``,
  ``sentiment_signal``, ``sentiment_score``, ``news_impact``
- **Events** -- ``earnings_calendar``, ``earnings_surprises``,
  ``upcoming_earnings``, ``earnings_history``, ``dividend_history``,
  ``insider_activity``, ``institutional_ownership``
- **Filings** -- ``recent_filings``, ``annual_reports``,
  ``quarterly_reports``, ``material_events``, ``filing_search``
- Built-in Loughran-McDonald keyword lexicon; optional VADER/TextBlob

Other Highlights
^^^^^^^^^^^^^^^^^

- 1,097 exported functions across 27 modules
- 3,630+ tests (2,400+ unit + 1,200+ MCP integration)
- 263 technical analysis indicators across 19 sub-modules
- 100K+ lines of code
- Comprehensive Sphinx documentation with tutorials
- PDM-managed dependency system with 25+ optional extras


v0.9.0
------

- Initial public release
- Core modules: risk, vol, regimes, ta, stats, ts, opt, backtest, price,
  ml, data, viz, econometrics, microstructure, execution, causal, forex,
  bayes, math, core, io, experiment
- Composable workflow system (``wraquant.compose``)
- Pre-built recipes (``wraquant.recipes``)
- Furo-based Sphinx documentation
