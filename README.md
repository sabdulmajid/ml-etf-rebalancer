# ETF Research Studio

[Open ETF Research Studio](https://etf-rebalancer.streamlit.app/) · Direct URL:
https://etf-rebalancer.streamlit.app/

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

## Example: Study A Portfolio Through The COVID Drawdown

Suppose you want to know how U.S. stocks, Treasury bonds, and gold worked
together before, during, and after the 2020 market decline.

1. Open **ETF Allocation Workbench**.
2. Select `SPY`, `IEF`, and `GLD`.
3. Select **Forecast filter**, **Trend filter**, **Equal Weight**, and
   **SPY reference** under Comparison series.
4. Set the historical range to **January 2019 through December 2021**.
5. Keep the transaction-cost setting fixed so every portfolio uses the same cost.
6. Use **Growth of $1 after estimated costs** to compare the value of each
   portfolio over the full period.
7. Use **Decline from each prior peak** to compare the size and timing of the
   February–March 2020 drawdown.
8. Use **How the proposed allocation changed** to see when an approach changed
   ETF weights or moved weight to cash.

This example answers four practical questions:

- Which portfolio had the smallest decline?
- Which portfolio recovered sooner?
- Did the trend or forecast rule use cash during the stress period?
- How much trading did each result require?

For example, a drawdown closer to 0% than SPY shows a smaller historical decline.
If the same portfolio also has a higher Sharpe above cash, it produced more
excess return for its return variability during the selected period. Turnover
and cost drag then show how much trading was required to produce that path.

You can then change one input and run the same comparison again. Remove `GLD`
to test a stock-and-bond portfolio. Increase the transaction cost to test a
more expensive rebalance. Enter your current weights to compare your portfolio
with a selected target and create a rebalance ticket.

## What The Workbench Approaches Mean

| Approach | What it does |
| --- | --- |
| Volatility Balanced | Gives more weight to ETFs with lower recent price variability. |
| Volatility Balanced + Trend | Uses the same weights, but assigns 0% to an ETF below its 10-month price average. |
| Forecast filter | Compares each ETF's one-month TimesFM-3 forecast with cash. ETFs forecast above cash can receive weight. |
| Equal Weight | Gives every selected ETF the same weight and resets it monthly. |
| SPY reference | Shows broad U.S. equity performance on the same dates. |
| Cash | Shows the U.S. overnight-rate proxy on the same dates. |

## How To Read The Results

| Measure | What it tells you | How to use it |
| --- | --- | --- |
| Annualized return | The compound yearly growth rate. | Compare long-run portfolio growth. |
| Annualized volatility | How widely monthly returns moved. | Compare how stable or variable each result was. |
| Sharpe above cash | Return above cash relative to the variability of that excess return. | Compare portfolios with different return and risk levels. A higher value means more excess return for the variability taken. |
| Maximum drawdown | The largest decline from a previous portfolio high. | Compare behavior during market stress. A value closer to 0% means a smaller decline. |
| Annualized turnover | The average percentage of the portfolio replaced each year. | Identify strategies that require more trading. |
| Annualized cost drag | The difference between results before and after estimated trading costs. | Test whether trading activity changes the result materially. |

## Current Research Snapshot

- The ETF price library covers **January 29, 1993 through August 31, 2026**.
  Each workbench comparison starts when the selected ETFs have enough shared data.
- The completed TimesFM-3 check covers signals from **June 30, 2009 through
  June 30, 2026**, with results through **August 3, 2026**.
- The ML sector study covers **January 31, 2015 through April 30, 2026**.

| ML sector comparison | What it does | Annualized return | Sharpe ratio | Maximum drawdown |
| --- | --- | ---: | ---: | ---: |
| ML Signal Blend | Combines a return forecast, six-month momentum, and price stability; then holds up to four sectors. | 10.43% | 0.74 | -19.33% |
| Equal-Weight Sectors | Gives the same monthly weight to each of the nine sector ETFs. | 11.84% | 0.81 | -23.60% |
| 6-Month Momentum Top 3 | Holds the three sectors with the highest prior six-month return. | 10.01% | 0.70 | -15.29% |
| SPY Buy & Hold | Holds the broad U.S. equity ETF. | 13.55% | 0.90 | -23.93% |

For this ML snapshot, the Sharpe ratio uses a 0% cash-rate assumption. The ETF
Allocation Workbench uses its U.S. overnight-rate proxy as the cash reference.

## How The Research Works

- [Research guide](docs/research/README.md)
- [ETF allocation methods](docs/research/etf-allocation.md)
- [TimesFM-3 forecast method](docs/research/timesfm-3.md)
- [ML sector study](docs/research/ml-sector-study.md)

Developer setup, data refresh, and deployment instructions are in
[DEPLOYMENT.md](DEPLOYMENT.md).
