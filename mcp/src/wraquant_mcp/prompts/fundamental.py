"""Fundamental analysis workflow prompt templates.

Multi-step prompts that guide AI agents through FMP-backed fundamental
analysis: deep dives, value screening, earnings analysis, and credit
assessment using the wraquant fundamental and news modules.
"""

from __future__ import annotations

from typing import Any


def register_fundamental_prompts(mcp: Any) -> None:
    """Register fundamental analysis prompt templates on the MCP server."""

    @mcp.prompt()
    def fundamental_deep_dive(symbol: str = "AAPL") -> list[dict]:
        """Complete fundamental analysis: profile, financials, ratios, valuation, health, quality."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a comprehensive fundamental deep dive on {symbol}. This is a multi-phase
workflow that uses the fundamental/ and news/ MCP tools backed by FMP data. The goal
is a complete picture: business profile, financial trends, valuation, quality, and
health — synthesized into an actionable investment assessment.

---

## Phase 1: Company Overview

1. **Company profile**: Run company_profile("{symbol}").
   Extract: sector, industry, market cap, description, key metrics.
   This sets the context — what business are we analyzing?

2. **Recent news**: Run stock_news("{symbol}", limit=10).
   Scan headlines for material events: earnings, M&A, regulatory, executive changes.
   Flag anything that could change the fundamental picture.

---

## Phase 2: Financial Statement Analysis

3. **Income analysis**: Run income_analysis("{symbol}", period="annual").
   Report: revenue growth trajectory (3-5 year CAGR), gross/operating/net margin trends.
   Key question: Is this business growing, stable, or declining?

4. **Balance sheet**: Run balance_sheet_analysis("{symbol}", period="annual").
   Report: asset composition, D/E ratio, working capital trend, goodwill exposure.
   Key question: Is the balance sheet strong enough to support the business?

5. **Cash flow**: Run cash_flow_analysis("{symbol}", period="annual").
   Report: FCF margin, OCF/NI ratio, capex intensity, FCF growth trajectory.
   Key question: Is the company generating real cash from its operations?

---

## Phase 3: Ratio Analysis & Quality

6. **Financial ratios**: Run financial_ratios("{symbol}").
   Report comprehensive ratios across all categories: profitability, liquidity,
   leverage, efficiency, valuation, growth. Compare to sector norms.

7. **DuPont decomposition**: Run dupont_analysis("{symbol}").
   Decompose ROE into its three drivers: margin, turnover, leverage.
   Is high ROE coming from quality (margin) or risk (leverage)?

8. **Earnings quality**: Run earnings_quality("{symbol}").
   Check accruals ratio, cash conversion, earnings persistence.
   Are the reported numbers trustworthy?

---

## Phase 4: Valuation

9. **DCF valuation**: Run dcf_valuation("{symbol}").
   Also run with discount_rate=0.08 and discount_rate=0.12 for sensitivity.
   Report intrinsic value range and margin of safety at current price.

10. **Relative valuation**: Run relative_valuation("{symbol}").
    Compare P/E, P/B, EV/EBITDA, P/S to peers.
    Is the stock cheap or expensive relative to comparable companies?

---

## Phase 5: Health & Risk Assessment

11. **Financial health**: Run financial_health("{symbol}").
    Report composite score and letter grade. Flag any sub-scores below 50.

12. **Piotroski F-Score**: Run piotroski_score("{symbol}").
    Score 0-9. >= 7 = strong fundamentals. <= 2 = weak.

13. **Altman Z-Score**: Run altman_z("{symbol}").
    Z > 2.99 = safe. 1.81-2.99 = gray zone. < 1.81 = distress.

---

## Phase 6: Synthesis

Compile all findings into a structured assessment:
- **Business quality**: Revenue growth, margin stability, competitive position.
- **Financial strength**: Leverage, liquidity, cash generation.
- **Earnings trust**: Quality score, accruals, cash backing.
- **Valuation**: DCF intrinsic value vs market price vs peer multiples.
- **Overall grade**: Buy / Hold / Sell with confidence level and key risks.
""",
                },
            }
        ]

    @mcp.prompt()
    def value_investing_screen() -> list[dict]:
        """Screen for undervalued stocks, then deep-dive the top picks."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": """
First load the wraquant_system_context prompt for full module context.

Run a value investing screen using stock_screener, then perform deep-dive
analysis on the top candidates. This workflow finds stocks that are cheap
on fundamentals AND have quality characteristics.

---

## Phase 1: Initial Screen

1. **Run the screener**: stock_screener with criteria:
   '{"max_pe": 20, "min_roe": 0.12, "max_debt_to_equity": 1.5, "min_market_cap": 1e9}'

   This finds: reasonable valuation + profitable + not over-leveraged + liquid.

2. **Review results**: The screener_results dataset has the top matches.
   Sort by P/E ascending. Note sector distribution — avoid concentrating in one sector.

---

## Phase 2: Quality Filter

3. For each of the top 5 results, run:
   a. financial_health(symbol) — composite health score.
   b. piotroski_score(symbol) — fundamental strength.
   c. earnings_quality(symbol) — verify earnings are real.

4. **Eliminate**: Remove any stock with:
   - Financial health score < 50
   - Piotroski < 5
   - Earnings quality red flags (accruals > 10% of assets, OCF/NI < 0.8)

---

## Phase 3: Deep Dive on Survivors

5. For each surviving candidate (top 3), run:
   a. company_profile(symbol) — understand the business.
   b. income_analysis(symbol) — revenue/margin trends.
   c. cash_flow_analysis(symbol) — FCF generation.
   d. dcf_valuation(symbol) — intrinsic value estimate.
   e. relative_valuation(symbol) — peer comparison.
   f. insider_activity(symbol) — are insiders buying?
   g. news_sentiment(symbol) — current sentiment context.

---

## Phase 4: Final Ranking

6. Rank the final candidates by:
   - Margin of safety (DCF intrinsic value vs market price)
   - Financial health score
   - Insider buying activity
   - Sentiment direction (improving = positive)

7. Compile a report with the top 3 picks, including:
   - Investment thesis for each (why it's undervalued)
   - Key risks to monitor
   - Suggested entry price range (based on DCF and peer multiples)
""",
                },
            }
        ]

    @mcp.prompt()
    def earnings_analysis(symbol: str = "AAPL") -> list[dict]:
        """Comprehensive earnings analysis: history, surprises, quality, upcoming."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a comprehensive earnings analysis of {symbol}. This workflow examines
earnings history, surprise patterns, quality metrics, and upcoming events to
assess earnings trajectory and identify trading opportunities around earnings.

---

## Phase 1: Earnings History & Surprises

1. **Earnings data**: Run earnings_data("{symbol}").
   Report: historical EPS trajectory, upcoming earnings date.

2. **Earnings surprises**: Run earnings_surprises("{symbol}", limit=20).
   Report: beat rate, average surprise magnitude, consistency.
   - Beat rate > 80% + large avg surprise = analysts consistently underestimate.
   - Recent misses after long beat streak = potential inflection point.

---

## Phase 2: Earnings Quality Deep Dive

3. **Income analysis**: Run income_analysis("{symbol}", period="quarter").
   Use QUARTERLY data to see the most recent trajectory.
   Report: sequential revenue growth, margin trends, operating leverage.

4. **Earnings quality**: Run earnings_quality("{symbol}").
   Check: accruals ratio, cash conversion, revenue quality.
   Are earnings backed by cash flow or driven by accounting?

5. **Cash flow backing**: Run cash_flow_analysis("{symbol}").
   Compare OCF trajectory to earnings trajectory.
   Divergence = red flag for future earnings disappointment.

---

## Phase 3: Context & Catalysts

6. **News sentiment**: Run news_sentiment("{symbol}").
   Is sentiment bullish or bearish heading into earnings?
   Sentiment alignment with surprise direction amplifies the move.

7. **Insider activity**: Run insider_activity("{symbol}").
   Heavy insider buying before earnings = confidence signal.
   Heavy insider selling = caution, even if public guidance is positive.

8. **Analyst context**: Run financial_ratios("{symbol}") for P/E and forward P/E.
   High P/E = market expects beats. Any miss will be severely punished.
   Low P/E = low expectations. Even a modest beat could spark a rally.

---

## Phase 4: Synthesis

Compile an earnings assessment:
- **Earnings momentum**: Improving, stable, or deteriorating?
- **Surprise probability**: Likely beat, meet, or miss based on patterns?
- **Quality verdict**: Are earnings trustworthy and sustainable?
- **Event risk**: Is the stock priced for perfection or pessimism?
- **Trading implications**: Positioning into the next earnings event.
""",
                },
            }
        ]

    @mcp.prompt()
    def credit_assessment(symbol: str = "AAPL") -> list[dict]:
        """Credit/solvency assessment: Altman Z, Piotroski, leverage, cash flow quality."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a comprehensive credit and solvency assessment of {symbol}. This workflow
evaluates bankruptcy risk, financial strength, leverage sustainability, and cash flow
quality to determine the company's creditworthiness and solvency trajectory.

---

## Phase 1: Distress Risk Indicators

1. **Altman Z-Score**: Run altman_z("{symbol}").
   - Z > 2.99 = safe zone (< 1% default probability within 2 years).
   - 1.81 < Z < 2.99 = gray zone (watch closely, 10-30% distress probability).
   - Z < 1.81 = distress zone (high default probability, immediate concern).

2. **Piotroski F-Score**: Run piotroski_score("{symbol}").
   - F >= 7 = strong. Improving profitability, leverage, and efficiency.
   - F <= 3 = weak. Deteriorating fundamentals across multiple dimensions.
   - Focus on the leverage and liquidity sub-components specifically.

3. **Financial health score**: Run financial_health("{symbol}").
   Report the leverage and liquidity sub-scores specifically.
   These components matter most for credit assessment.

---

## Phase 2: Leverage & Coverage Analysis

4. **Balance sheet analysis**: Run balance_sheet_analysis("{symbol}").
   Report:
   - Total debt / total assets: > 0.6 = high leverage.
   - Net debt / EBITDA: > 4x = heavy debt load, refinancing risk.
   - Current ratio: < 1 = may struggle to meet short-term obligations.
   - Interest coverage (from ratios): < 2x = debt servicing stress.

5. **Financial ratios**: Run financial_ratios("{symbol}").
   Focus on: D/E ratio, interest coverage, current ratio, quick ratio.
   Trend over time matters more than a single snapshot.

---

## Phase 3: Cash Flow Quality

6. **Cash flow analysis**: Run cash_flow_analysis("{symbol}").
   Report:
   - OCF / total debt: > 0.2 = can service debt from operations.
   - FCF / debt service: Can the company pay interest AND principal?
   - Capex flexibility: Can capex be deferred if cash gets tight?
   - Working capital trend: Increasing WC needs = cash drain.

7. **Earnings quality**: Run earnings_quality("{symbol}").
   High accruals + low OCF/NI = earnings are overstated, real cash position worse.
   This is critical for credit — lenders care about cash, not accounting income.

---

## Phase 4: External Indicators

8. **Recent SEC filings**: Run sec_filings("{symbol}", form_type="8-K").
   Look for: covenant violations, credit facility amendments, going concern opinions,
   debt restructuring, asset sales. These are early warning signals.

9. **News sentiment**: Run news_sentiment("{symbol}").
   Negative sentiment + deteriorating fundamentals = compounding risk.
   Credit downgrades often follow sustained negative press.

---

## Phase 5: Credit Synthesis

Compile a credit assessment report:
- **Distress probability**: Based on Altman Z and Piotroski.
- **Leverage sustainability**: Can the company service and refinance its debt?
- **Cash flow adequacy**: Is operating cash flow sufficient for obligations?
- **Quality of reported numbers**: Are fundamentals as good as they appear?
- **Overall credit grade**: Investment grade / Speculative / Distressed.
- **Key watch items**: What would trigger a downgrade?
""",
                },
            }
        ]
