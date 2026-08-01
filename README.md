# ETF Research Studio

ETF Research Studio is a read-only research application with two deliberately
separate experiences: an ETF Allocation Workbench for fixed, explainable
portfolio policies and the original walk-forward ML sector study. The workbench
is the first tab; the existing ML allocation, backtest, and research results are
unchanged and remain clearly labeled as ML research.

Live app: https://etf-rebalancer.streamlit.app/

## What It Demonstrates

- Time-series ML workflow with monthly walk-forward retraining
- No-lookahead backtesting with transaction-cost assumptions
- One common monthly engine for every workbench strategy and benchmark
- Fixed Volatility Balanced, Volatility Balanced + Trend, and Equal Weight policies
- Explicit analytical cash, current-weight validation, and reconciled ETF tickets
- Portfolio construction with long-only, top-sector, and max-weight constraints
- Benchmarking against `SPY`, equal-weight sectors, and a simple momentum baseline
- Interactive Streamlit UX for allocation review, analytical trade-ticket generation,
  and a separately labeled exploratory ML remix
- Lightweight validation tests and a local benchmark script for reproducibility

## Current Research Snapshot

The committed artifacts were generated with market data through `2026-04-30`.

| Strategy | CAGR | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: |
| ML Signal Blend | 10.43% | 0.74 | -19.33% |
| Equal-Weight Sectors | 11.84% | 0.81 | -23.60% |
| 6M Momentum Top 3 | 10.01% | 0.70 | -15.29% |
| SPY Buy & Hold | 13.55% | 0.90 | -23.93% |

This project does **not** claim that the ML strategy beats SPY. The value is in the end-to-end research system: feature engineering, walk-forward validation, allocation logic, benchmarking, explainability, and an interactive portfolio workflow.

## Product Features

- Curated 1–8 ETF workbench with a mixed `SPY` / `IEF` / `GLD` default
- Historical comparisons against Equal Weight and analytical cash; Buy & Hold is
  available only for a single ETF, and duplicate one-ETF paths are hidden
- Date-range and transaction-cost controls with every selected historical range
  restarted from cash and charged its entry cost
- Full-artifact proposed current targets with signal/execution/artifact provenance,
  allocation/turnover/cost history, drawdowns,
  concise metrics, “Why this weight?” diagnostics, and CSV downloads
- Optional current ETF plus explicit cash weights. Inputs are never normalized;
  invalid weights leave strategy research running but disable Current Mix,
  Portfolio Lab transfer, and the ticket
- An explicit authoritative-target transfer to Portfolio Lab. Its percentage-point
  and optional dollar ticket keeps analytical cash separate from ETF orders
- Current sector ETF allocation with forecast, momentum, and stability signals
- Walk-forward backtest from `2015-01-31` through the latest available month
- Equity curve, drawdown, annual return, turnover, and model-driver views
- Portfolio Lab for exact synced workbench targets and analytical rebalance tickets
- An isolated `ML Sandbox — exploratory, non-authoritative` signal remix that
  cannot read or write the workbench target, transfer, or ticket
- Read-only dashboard backed by committed `artifacts/latest` files, so visitors are not triggering model training or data writes

## Quickstart

```bash
git clone https://github.com/sabdulmajid/ml-etf-rebalancer.git
cd ml-etf-rebalancer
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

The dashboard opens in your browser and works immediately from the two committed
artifact bundles. No public runtime data download occurs.

To refresh the research artifacts with current market data:

```bash
python run_pipeline.py
streamlit run dashboard/app.py
```

Useful local commands:

```bash
make test        # run validation tests
make benchmark   # refresh artifacts and benchmark local health
make refresh     # rebuild artifacts only
make app         # launch the dashboard
```

## Methodology

### ETF Allocation Workbench

The workbench reads only the validated files in `artifacts/workbench`. It uses
adjusted daily closes to form completed monthly signal/execution periods and uses
the common engine in `backtest/engine.py` for drift, one-way turnover,
transaction costs, entry behavior, cash, and performance metrics for every
strategy and comparison. Strategy policy is fixed in `strategies/allocation.py`;
there are no methodology sliders in the UI. No ETF can receive more than 150% of
its equal-weight share.

Latest targets always use the entire artifact. Changing the historical chart
range only changes the displayed backtest, which restarts from 100% cash. A
valid entered Current Mix is a hypothetical constant target reset monthly, not
a reconstruction of actual holdings history.

**Cash — U.S. overnight-rate proxy** uses official EFFR before April 2, 2018 and
official SOFR from that date, with Actual/360 accrual. It is analytical and
non-investable. `BIL` is a separately selectable ETF;
`CASH:USD_OVERNIGHT` is never a ticker or security order.

### Existing ML Sector Study

Universe: `XLB`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLU`, `XLV`, `XLY`

Benchmark: `SPY`

Pipeline:

1. Download adjusted daily ETF prices with `yfinance`.
2. Build monthly technical features: momentum, volatility, moving-average ratios, and relative strength.
3. Train regularized regression models in a monthly walk-forward loop.
4. Blend model forecast, six-month momentum, and inverse-volatility stability signals.
5. Construct a long-only portfolio using top-sector selection and max-weight constraints.
6. Subtract transaction costs from turnover.
7. Export stable artifacts for the dashboard.

## Validation

The repository includes tests for allocator constraints, artifact schema, strategy outputs, and dashboard smoke execution:

```bash
pytest -q
python tools/benchmark.py --pipeline
```

Recent local validation:

```text
pytest -q                              7 passed
python tools/benchmark.py              dashboard bare execution: ~2.0s
python run_pipeline.py                 full artifact refresh: ~22s
```

## Project Structure

```text
artifacts/latest/       Stable dashboard-ready research outputs
artifacts/workbench/    Validated ETF and analytical-cash bundle
dashboard/app.py        Streamlit research terminal
dashboard/workbench.py  Workbench calculations, downloads, and UI
backtest/               Common monthly accounting and metrics
data/features.py        Feature engineering
portfolio/rebalance.py  ML allocation helper plus workbench ticket validation
portfolio/research.py   Walk-forward research engine
strategies/allocation.py Fixed workbench target generators
tests/                  Local validation tests
tools/benchmark.py      Local health and benchmark script
run_pipeline.py         Artifact refresh entrypoint
```

## Scope And Limitations

- This is an educational research project, not investment advice.
- It is not a live trading system and does not place orders.
- Analytical cash is not investable and is not a substitute for a deposit,
  money-market fund, Treasury bill, or executable cash return.
- It does not claim predictive superiority over passive indexing.
- It uses Yahoo Finance data via `yfinance`, which is appropriate for research/demo use but not institutional-grade market data.
- Backtest results depend on the selected universe, assumptions, rebalance timing, and transaction-cost model.

The ML Signal Remix is retained only inside `ML Sandbox — exploratory,
non-authoritative`; it is disconnected from the authoritative workbench target,
Portfolio Lab session state, and ticket. The hindsight scenario stress test was
removed because applying a current target to past regimes implied a holdings
history that did not exist. These UI decisions change no feature engineering,
model fitting, signals, targets, artifacts, or stored ML results.
