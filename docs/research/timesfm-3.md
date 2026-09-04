# TimesFM-3 Forecast Method

## Research Question

The TimesFM-3 analysis answers two questions:

1. Can a general time-series model estimate the next month's ETF price path?
2. Does an ETF-versus-cash filter change portfolio results after trading costs?

[TimesFM-3](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
is a pretrained forecasting model. This project uses the published model with
fixed parameters. The model produces a range of future price estimates for each
ETF.

## What The Model Reads

The model reads the 14 ETF adjusted-price histories together:

`SPY`, `IWM`, `EFA`, `EEM`, `AGG`, `BIL`, `SHY`, `IEF`, `TLT`, `TIP`, `LQD`,
`HYG`, `GLD`, and `DBC`.

At each completed month-end, the model receives the last 512 New York Stock
Exchange sessions. This is approximately two years of daily prices. Each ETF
has the same dates in the input matrix.

The input contains adjusted prices only. The model parameters stay fixed.

## What The Model Produces

The first forecast date is the next exchange session. The last forecast date is
the following monthly execution date. The model estimates nine price levels for
each future date. These levels are the 10th through 90th percentiles.

The 50th-percentile value is the median forecast. The project uses this value to
calculate the forecast holding return:

```math
\widehat R_i=\frac{\widehat P_{i,50,end}}{\widehat P_{i,50,execution}}-1
```

The lower and upper displayed prices are the 10th- and 90th-percentile price
estimates for the end of the holding period.

## How The Forecast Changes A Portfolio

The strategy name is **Volatility Balanced + Forecast**.

The forecast makes one eligibility decision for each selected ETF. First, the
strategy calculates the cash return for the same holding period. Then it
calculates forecast edge:

```math
E_i=\widehat R_i-R_{cash}
```

An ETF is eligible when \(E_i>0\). An ETF with \(E_i\leq0\) receives a zero ETF
weight. The Volatility Balanced rule sets weights for the eligible ETFs. The
forecast value controls eligibility only.

The adaptive position limit is the same limit used by the other workbench
strategies:

```math
c(n)=\min\left(100\%,\frac{150\%}{n}\right)
```

Unused weight goes to analytical cash. If all selected ETFs have a forecast
return at or below cash, the target is 100% cash.

With one selected ETF, the rule becomes a direct ETF-versus-cash decision.

## What The Four Summary Values Mean

**ETF-month forecasts** counts completed ETF forecasts with a known result. One
forecast month for 14 ETFs adds 14 observations.

**Correct up-or-down calls** is the percentage of observations for which the
median forecast and the realized return had the same direction.

**Average forecast difference** is the mean absolute difference between the
forecast one-month return and the realized one-month return. A value of 2.43%
means that the two returns differed by 2.43 percentage points on average.

**Final prices inside the forecast range** is the percentage of realized final
prices between the 10th- and 90th-percentile price estimates.

## Historical Forecast Review

The application also shows three forecast calculations:

| Calculation | Purpose |
| --- | --- |
| Last observed price | Simple reference that assumes the price does not change |
| TimesFM-3, one ETF at a time | Tests each ETF without cross-ETF inputs |
| TimesFM-3, all 14 ETFs together | Tests whether the joint price matrix changes forecast results |

The completed forecast check uses signal dates from June 30, 2009 through June
30, 2026. The final result date is August 3, 2026. It contains 2,870 completed
ETF-month forecasts for each model setup. The stored bundle also contains later
signals whose result dates were still pending when the bundle was built.

The application evaluates only forecasts with a known period-end result. It
keeps the latest pending forecast separate from the completed history.

The model results are built before deployment. The Streamlit application reads
the result files and performs portfolio calculations locally. This keeps the
browser workflow stable and makes repeated inputs deterministic.

## Portfolio Comparison

For one selected ETF, the main reference is Buy & Hold. For two or more selected
ETFs, the main reference is Equal Weight.

The forecast strategy and its reference use the same dates, starting value,
cash return, and transaction-cost setting. The result table shows their
annualized return, volatility, Sharpe above cash, drawdown, and turnover.
