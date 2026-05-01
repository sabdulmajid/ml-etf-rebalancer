# ETF Signal Lab

ETF Signal Lab is an interactive research terminal for U.S. sector ETF rotation. It combines walk-forward model forecasts, trailing momentum, and volatility-aware allocation rules to produce a current sector allocation, benchmarked performance analytics, and a hands-on portfolio lab.

Live app: https://etf-rebalancer.streamlit.app/

## What It Demonstrates

- Time-series ML workflow with monthly walk-forward retraining
- No-lookahead backtesting with transaction-cost assumptions
- Portfolio construction with long-only, top-sector, and max-weight constraints
- Benchmarking against `SPY`, equal-weight sectors, and a simple momentum baseline
- Interactive Streamlit UX for allocation review, trade-ticket generation, signal remixing, and historical stress tests
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

- Current sector ETF allocation with forecast, momentum, and stability signals
- Walk-forward backtest from `2015-01-31` through the latest available month
- Equity curve, drawdown, annual return, turnover, and model-driver views
- Portfolio Lab for editable current holdings, target trade tickets, signal remixing, and scenario stress tests
- Read-only dashboard backed by committed `artifacts/latest` files, so visitors are not triggering model training or data writes

## Quickstart

```bash
git clone https://github.com/sabdulmajid/ml-etf-rebalancer.git
cd ml-etf-rebalancer
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

The dashboard opens in your browser and works immediately from the committed artifacts.

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
dashboard/app.py        Streamlit research terminal
data/features.py        Feature engineering
portfolio/rebalance.py  Allocation constraints and rebalance helpers
portfolio/research.py   Walk-forward research engine
tests/                  Local validation tests
tools/benchmark.py      Local health and benchmark script
run_pipeline.py         Artifact refresh entrypoint
```

## Scope And Limitations

- This is an educational research project, not investment advice.
- It is not a live trading system and does not place orders.
- It does not claim predictive superiority over passive indexing.
- It uses Yahoo Finance data via `yfinance`, which is appropriate for research/demo use but not institutional-grade market data.
- Backtest results depend on the selected universe, assumptions, rebalance timing, and transaction-cost model.

