# Research Guide

ETF Research Studio connects three tasks:

1. Define a portfolio rule.
2. Test the rule on past data with fixed timing and cost rules.
3. Convert the latest target into a clear rebalance plan.

The application contains two research systems. They answer different questions.

## ETF Allocation Workbench

The workbench starts with ETFs selected by the user. It compares simple portfolio
rules on the same dates. Each rule produces target weights. A common backtest
engine then applies returns, cash interest, portfolio drift, trades, and costs.

Read [ETF allocation methods](etf-allocation.md) for the formulas and timing.

## TimesFM-3 Forecast Analysis

TimesFM-3 reads recent ETF prices and estimates future prices. The application
uses the median estimate to make one decision: hold an ETF or hold cash for that
part of the portfolio. The size of the forecast does not set the ETF weight.

Read [TimesFM-3 forecast method](timesfm-3.md) for the model inputs, decision
rule, portfolio rule, and evaluation measures.

## ML Sector Study

The sector study ranks nine U.S. sector ETFs each month. It combines a statistical
return forecast with momentum and price stability. It then holds up to four
sectors. The application compares the result with three reference portfolios.

Read [ML sector study](ml-sector-study.md) for the features, model, score, and
comparison rules.

## Common Terms

**Signal date** is the date when the strategy reads its inputs.

**Execution date** is the date when the portfolio moves to the new target.

**Holding period** is the interval from one execution date to the next execution
date.

**Target weight** is the required portfolio percentage after a rebalance.

**Pre-trade weight** is the portfolio percentage after market movement and before
the next rebalance.

**Turnover** is the percentage of the portfolio that changes side at a rebalance.

**Drawdown** is the percentage decline from the prior portfolio high.
