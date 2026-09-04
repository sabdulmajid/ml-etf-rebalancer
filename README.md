# ETF Research Studio

[Open ETF Research Studio](https://etf-rebalancer.streamlit.app/)

ETF Research Studio is a browser-based research application for ETF portfolio
design, strategy comparison, and forecast analysis.

## What You Can Do

- Build a portfolio from one to eight ETFs.
- Compare equal weight, volatility-based allocation, trend, cash, and a
  TimesFM-3 forecast filter on the same dates.
- Review returns, drawdowns, turnover, costs, and monthly target weights.
- See why each ETF receives its latest weight.
- Compare a separate machine-learning sector strategy with SPY and two simple
  sector benchmarks.
- Move a selected target into Portfolio Lab and create a percentage or dollar
  rebalance ticket.

## Research Views

| View | Main question |
| --- | --- |
| ETF Allocation Workbench | How do different allocation rules change this ETF portfolio? |
| TimesFM-3 Forecast Analysis | How does a one-month price forecast change ETF-versus-cash decisions? |
| ML Sector Study | How does a monthly sector model compare with SPY, equal-weight sectors, and momentum? |
| Portfolio Lab | What changes move the current portfolio to the selected target? |

## Current Research Snapshot

- The ETF price library covers **January 29, 1993 through August 31, 2026**.
  Each workbench comparison starts when the selected ETFs have enough shared data.
- The completed TimesFM-3 check covers signals from **June 30, 2009 through
  June 30, 2026**, with results through **August 3, 2026**.
- The ML sector study covers **January 31, 2015 through April 30, 2026**.

| ML sector comparison | Annualized return | Sharpe ratio | Maximum drawdown |
| --- | ---: | ---: | ---: |
| ML Signal Blend | 10.43% | 0.74 | -19.33% |
| Equal-Weight Sectors | 11.84% | 0.81 | -23.60% |
| 6-Month Momentum Top 3 | 10.01% | 0.70 | -15.29% |
| SPY Buy & Hold | 13.55% | 0.90 | -23.93% |

Annualized return is the compound yearly growth rate. The Sharpe ratio compares
return with return variability. Maximum drawdown is the largest decline from a
previous portfolio high.

## How The Research Works

- [Research guide](docs/research/README.md)
- [ETF allocation methods](docs/research/etf-allocation.md)
- [TimesFM-3 forecast method](docs/research/timesfm-3.md)
- [ML sector study](docs/research/ml-sector-study.md)

Developer setup, data refresh, and deployment instructions are in
[DEPLOYMENT.md](DEPLOYMENT.md).
