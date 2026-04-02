"""Data workflow prompt templates."""

from __future__ import annotations

from typing import Any


def register_data_prompts(mcp: Any) -> None:

    @mcp.prompt()
    def data_pipeline(
        source: str = "yahoo",
        tickers: str = "AAPL,MSFT,GOOGL",
    ) -> list[dict]:
        """Full data pipeline: fetch, clean, validate, compute returns, store."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Build a complete data pipeline for {tickers} from {source}. This uses the data/
tools to fetch, clean, validate, and prepare data for downstream analysis.

---

## Phase 1: Data Acquisition

1. **Workspace check**: Run workspace_status to see existing datasets.
   Avoid re-fetching data that already exists. Check date ranges of existing data.

2. **Fetch data**: For each ticker in [{tickers}]:
   - Run fetch_yahoo with the ticker and desired date range.
     Request at least 3 years of history for GARCH/HMM convergence.
     5+ years is ideal for stress testing and regime analysis.
   - Verify columns: open, high, low, close, volume are all present.
   - If fetch fails: check ticker symbol, try alternative source.

3. **Alternative sources**: If {source} is not yahoo:
   - CSV: Use load_csv with the file path.
   - JSON: Use load_json with the file path.
   - Manual: Use store_data with column names and values.
   Ensure consistent column naming (lowercase: open, high, low, close, volume).

---

## Phase 2: Data Cleaning

4. **Initial inspection**: Run describe_dataset on each loaded dataset.
   Check:
   - Date range: start and end dates as expected?
   - Row count: ~252 trading days per year. Missing days = gaps.
   - Column types: All numeric except date index?
   - NaN count: Any missing values?

5. **Clean dataset**: Run clean_dataset on each dataset.
   This handles:
   - Forward-fill small gaps (1-2 days, holidays).
   - Drop rows with all-NaN.
   - Handle stock splits (if adjusted close differs from close).
   - Remove duplicate timestamps.

6. **Outlier detection**: Use query_data to find extreme values:
   - |daily return| > 20%: Likely data error or stock split. Investigate.
   - Volume = 0: Non-trading day or data gap. Forward-fill or drop.
   - Price <= 0: Data error. Drop or fix.

---

## Phase 3: Validation

7. **Validate returns**: Run compute_returns on each cleaned dataset.
   Then run validate_returns_tool on the returns:
   - Checks for: NaN, inf, returns > threshold, stationarity.
   - Reports any issues found.

8. **Cross-validation**: If multiple tickers loaded:
   - Run align_datasets to verify date alignment across all assets.
   - Missing dates in one but not others = trading halt or data issue.
   - Use merge_datasets to create a combined multi-asset dataset.

9. **Statistical sanity check**: Run analyze() on each return series.
   - Mean: Should be near 0 for daily returns (slightly positive for equities).
   - Std: 1-4% daily is typical for single stocks. < 0.5% may indicate stale data.
   - ADF p-value: Returns must be stationary (p < 0.05). If not, data problem.

---

## Phase 4: Derived Data

10. **Log returns**: Run compute_log_returns if log returns are needed
    (preferred for multi-period analysis, GARCH fitting, portfolio optimization).

11. **OHLCV resampling**: If you need weekly or monthly data:
    Run resample_ohlcv with the desired frequency (W, M, Q).
    This correctly aggregates OHLCV: open=first, high=max, low=min, close=last, volume=sum.

12. **Custom columns**: Use add_column to add derived features:
    - Dollar volume: close * volume
    - Intraday range: (high - low) / close
    - Gap: open / prev_close - 1

---

## Phase 5: Final Dataset Catalog

13. **Workspace summary**: Run workspace_status to see all stored datasets.
    For each dataset, run dataset_info to get metadata:
    - Columns, dtypes, row count, date range.
    - Source operation and parent dataset (provenance).

14. **Export if needed**: Run export_dataset to save to CSV/JSON for external use.

15. **Pipeline summary**:
    - Datasets created: list each with row count and date range.
    - Data quality: any issues found and how they were resolved.
    - Derived datasets: returns, log returns, resampled data.
    - Ready for: stats, vol, regimes, ta, risk, backtest analysis.

**Related prompts**: Use data_quality_audit for deeper quality checks,
multi_asset_setup for portfolio-level data preparation.
""",
                },
            }
        ]

    @mcp.prompt()
    def data_quality_audit(dataset: str = "prices") -> list[dict]:
        """Comprehensive data quality check: gaps, outliers, consistency, stationarity."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Perform a thorough data quality audit on {dataset}. This ensures the data is
suitable for quantitative analysis before running any models.

---

## Phase 1: Structure & Completeness

1. **Dataset overview**: Run describe_dataset on {dataset}.
   Report: shape, columns, dtypes, memory usage.
   - Expected columns for OHLCV: open, high, low, close, volume.
   - Expected index: DatetimeIndex (sorted ascending, no duplicates).

2. **Missing value analysis**: Use query_data to count NaN per column:
   - NaN in close: Critical. Must fix before any analysis.
   - NaN in volume: Less critical but affects microstructure analysis.
   - NaN pattern: Random (MCAR) or systematic (MNAR)?
     Systematic gaps (e.g., all columns missing on same days) = trading halts.
     Random gaps = data provider issues.

3. **Date continuity**: Check for gaps in the date index.
   - Weekend gaps: Expected (no trading).
   - Holiday gaps: Expected (market closed).
   - Multi-day gaps on weekdays: Data issue or trading halt.
   - Use query_data to find the longest gap and its dates.

4. **Duplicate detection**: Check for duplicate timestamps.
   Duplicates cause errors in time-series analysis. Remove or average them.

---

## Phase 2: Value Quality

5. **Range checks**: Use query_data to find values outside expected ranges:
   - Price <= 0: Data error.
   - Volume < 0: Data error.
   - High < Low: OHLC consistency violation. Data error.
   - Close > High or Close < Low: OHLC consistency violation.
   - Open > High or Open < Low: OHLC consistency violation.
   Report all violations with dates.

6. **Outlier detection**: Run compute_returns on {dataset}, then analyze().
   - Flag returns > 3 standard deviations (potential outliers).
   - Flag returns > 20% in a single day (likely split or error).
   - Check if outliers correspond to known events (earnings, splits, crashes).
   - Volume outliers: > 5x average volume = event (OK) or data error (check).

7. **Split/dividend detection**: Look for large price jumps that revert next day.
   - Price drops ~50% and volume is normal = stock split.
   - Price drops 1-5% on ex-dividend date = dividend adjustment.
   - Verify that adjusted close accounts for these correctly.

---

## Phase 3: Statistical Quality

8. **Stationarity**: Run analyze() on returns.
   - ADF p < 0.05: Returns are stationary (required for most models).
   - ADF p > 0.05: Returns may not be stationary. Check for level shifts.

9. **Autocorrelation**: Check Ljung-Box p-value from analyze().
   - p < 0.05 at lag 1: Significant first-order autocorrelation.
     Could be real (momentum/mean-reversion) or artifact (stale prices).
   - For illiquid stocks, autocorrelation is often an artifact of stale prices.

10. **Distribution check**: Run distribution_fit on returns.
    - Compare AIC across normal, t, and skewed-t fits.
    - Very low degrees of freedom (< 3) = extremely fat tails. Check for data errors.
    - Extreme skew (|skew| > 2) may indicate data issues rather than real distribution.

---

## Phase 4: Cross-Validation

11. **Price-volume consistency**: Do high-volume days have larger price moves?
    Run correlation_analysis on |returns| vs volume. Positive correlation expected.
    Near-zero or negative correlation = potential data quality issue.

12. **Benchmark comparison** (if available): Compare {dataset} returns to a
    known benchmark source. Large discrepancies indicate data errors.
    - Use merge_datasets to align dates.
    - Compute return differences. Mean should be near 0, std should be small.

---

## Phase 5: Audit Report

13. **Data quality scorecard**:

    | Check | Status | Details |
    |-------|--------|---------|
    | Completeness | PASS/FAIL | N missing values, M date gaps |
    | OHLC consistency | PASS/FAIL | N violations |
    | Outliers | WARN/PASS | N outlier returns |
    | Stationarity | PASS/FAIL | ADF p-value |
    | Distribution | INFO | Best fit, df parameter |

14. **Recommendations**:
    - Data clean enough for: [list suitable analyses].
    - Fix before proceeding: [list issues that must be resolved].
    - Consider: [forward fill, winsorize, exclude date range, etc.].

**Related prompts**: Use data_pipeline to fix identified issues,
data_exploration for exploratory analysis.
""",
                },
            }
        ]

    @mcp.prompt()
    def multi_asset_setup(
        tickers: str = "SPY,TLT,GLD,VIX,DXY",
    ) -> list[dict]:
        """Set up a multi-asset workspace with aligned data and returns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Set up a complete multi-asset workspace for [{tickers}]. This prepares
aligned datasets for portfolio analysis, correlation studies, and cross-asset modeling.

---

## Phase 1: Individual Asset Loading

1. **Fetch each asset**: For each ticker in [{tickers}]:
   - Run fetch_yahoo to load OHLCV data.
   - Request the maximum common date range across all assets.
   - Store as prices_<ticker>.

2. **Compute individual returns**: For each asset:
   - Run compute_returns to get daily returns.
   - Run compute_log_returns for log returns (used in portfolio optimization).

---

## Phase 2: Alignment & Merging

3. **Align dates**: Run align_datasets to find the common date range.
   - Report: earliest common date, latest common date, number of common observations.
   - Flag any assets with significantly fewer observations (data starts later).
   - If an asset has < 80% of the maximum observations, consider excluding it.

4. **Create merged dataset**: Run merge_datasets to create:
   - **prices_universe**: All close prices side-by-side (columns = tickers).
   - **returns_universe**: All returns side-by-side.
   - These are the inputs for correlation_analysis, detect_regimes, and portfolio optimization.

5. **Verify alignment**: Use query_data to check the first and last 5 rows
   of the merged dataset. Ensure no NaN values (misaligned dates).

---

## Phase 3: Quick Characterization

6. **Descriptive stats**: Run analyze() on each column of returns_universe.
   Create a comparison table:

   | Asset | Ann Return | Ann Vol | Sharpe | Skew | Kurt |
   |-------|-----------|---------|--------|------|------|
   | SPY | ... | ... | ... | ... | ... |
   | TLT | ... | ... | ... | ... | ... |

7. **Correlation matrix**: Run correlation_analysis on returns_universe.
   Report: key correlations, diversification assessment.
   - SPY-TLT: Negative = good portfolio diversifier.
   - Highly correlated pairs (> 0.7): limited diversification benefit.

8. **Regime check**: Run detect_regimes on the primary asset (usually the equity index).
   Current regime? This sets context for all downstream analysis.

---

## Phase 4: Workspace Catalog

9. **Final inventory**: Run workspace_status.
   List all datasets with metadata:
   - Individual price and return datasets.
   - Merged universe datasets.
   - Date range and observation count.

10. **Ready for analysis**:
    - Portfolio optimization: Use returns_universe with opt/ tools.
    - Cross-asset study: Use returns_universe with correlation_analysis.
    - Risk analysis: Use returns_universe with risk/ tools.
    - Regime analysis: Use returns_universe with detect_regimes.

**Related prompts**: Use data_quality_audit to verify each dataset,
cross_asset_study for correlation regime analysis.
""",
                },
            }
        ]

    @mcp.prompt()
    def data_exploration(dataset: str = "new_data") -> list[dict]:
        """Explore and summarize a new dataset: structure, distributions, patterns."""
        return [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""
First load the wraquant_system_context prompt for full module context.

Explore and summarize {dataset} -- a new dataset that we haven't analyzed before.
This is a structured exploratory data analysis workflow using data/ and stats/ tools.

---

## Phase 1: Structure Discovery

1. **Basic info**: Run describe_dataset on {dataset}.
   Report:
   - Shape (rows x columns).
   - Column names and their types (numeric, datetime, categorical).
   - Index type: DatetimeIndex? RangeIndex?
   - Memory usage.

2. **Sample inspection**: Use query_data to view:
   - First 5 rows: understand the structure.
   - Last 5 rows: verify data extends to expected end date.
   - Random sample of 10 rows: check for formatting issues.

3. **Column classification**: Categorize each column:
   - Price columns: open, high, low, close, adj_close
   - Volume columns: volume, dollar_volume
   - Return columns: already computed returns
   - Factor columns: beta, momentum, value, etc.
   - Metadata columns: ticker, sector, exchange

---

## Phase 2: Univariate Analysis

4. **Numeric summary**: For each numeric column, run analyze() or
   use describe_dataset summary statistics:
   - Central tendency: mean, median. Large difference = skewed distribution.
   - Dispersion: std, min, max, IQR. Large max/min relative to mean = outliers.
   - Shape: skewness, kurtosis. Fat tails? Asymmetry?

5. **Distribution analysis**: For the primary column (likely returns or close):
   Run distribution_fit to identify the best-fitting distribution.
   - Normal: Simplest, usually rejected for financial data.
   - Student-t: Handles fat tails. Common for returns.
   - Skewed-t: Handles both asymmetry and fat tails.

6. **Time series properties**: Run analyze() for:
   - ADF test: Is the series stationary?
   - Ljung-Box: Is there significant autocorrelation?
   - If non-stationary: compute first differences or returns.

---

## Phase 3: Multivariate Analysis (if multi-column)

7. **Correlation structure**: Run correlation_analysis on all numeric columns.
   - Which variables are strongly correlated (|r| > 0.7)?
   - Any unexpected correlations (domain-specific red flags)?
   - Near-perfect correlation (|r| > 0.95) = potential multicollinearity issue.

8. **Lead-lag relationships**: If time series, check if any columns lead or lag
   others. Run correlation_analysis on lagged data.
   - Does volume lead returns? (Liquidity-driven)
   - Does volatility lead returns? (Risk premium)

---

## Phase 4: Pattern Discovery

9. **Regime structure**: Run detect_regimes on the primary variable.
   Are there distinct regimes in the data? How many? What characterizes each?

10. **Seasonality**: If data spans multiple years:
    - Group by month or day-of-week using query_data.
    - Any systematic seasonal patterns?

11. **Anomalies**: Flag any unusual patterns:
    - Sudden level shifts (structural breaks).
    - Periods of zero variance (stale data).
    - Monotonic trends in otherwise random data (drift or data issue).

---

## Phase 5: Summary

12. **Dataset profile**:
    - What kind of data is this? (Prices, returns, factors, panel data)
    - Time range and frequency (daily, hourly, monthly).
    - Quality assessment: clean or needs work?
    - Key statistical properties: stationarity, distribution, autocorrelation.
    - Number of distinct assets/entities (if panel data).

13. **Recommended next steps**: Based on what we found:
    - If OHLCV price data: use data_pipeline for full preparation.
    - If returns data: ready for stats/, vol/, risk/ analysis.
    - If multi-asset: use multi_asset_setup for alignment.
    - If panel data: use panel_data_analysis.

**Related prompts**: Use data_quality_audit for deeper quality checks,
data_pipeline for preparation, equity_deep_dive for single-stock analysis.
""",
                },
            }
        ]
