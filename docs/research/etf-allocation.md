# ETF Allocation Methods

## Research Question

The workbench answers this question:

> How do simple allocation rules change the risk, return, drawdown, turnover,
> cash use, and current target of a selected ETF portfolio?

The user selects one to eight ETFs. The application applies each rule to the
same selected ETFs. It uses one accounting engine for all workbench strategies
and comparisons.

## Data And Timing

The workbench uses adjusted daily closing prices. An adjusted price includes the
effect of distributions and splits.

The application uses only completed calendar months.

For each monthly period:

1. The signal date is the last exchange session of a completed month.
2. The strategy uses data available on or before the signal date.
3. The execution date is the first exchange session after the signal date.
4. The portfolio holds the target until the next monthly execution date.

This order prevents a strategy from using future prices.

## Volatility Balanced

This strategy gives more weight to ETFs with lower recent price variability.

For ETF \(i\), define the daily adjusted-price return as:

```math
r_{i,d}=\frac{P_{i,d}}{P_{i,d-1}}-1
```

The strategy uses the last 126 daily returns. It requires at least 120 valid
returns. It calculates annualized volatility as:

```math
\sigma_i=\operatorname{stdev}(r_{i,d-125:d})\sqrt{252}
```

The first weight score is the inverse of volatility:

```math
u_i=\frac{1}{\sigma_i}
```

The raw portfolio weight is:

```math
\tilde w_i=\frac{u_i}{\sum_j u_j}
```

The strategy then applies one position limit. If the user selects \(n\) ETFs,
the limit is:

```math
c(n)=\min\left(100\%,\frac{150\%}{n}\right)
```

No ETF can receive more than 150% of its equal-weight share. If one ETF reaches
the limit, the strategy moves the remaining weight to the other eligible ETFs.

## Volatility Balanced + Trend

This strategy adds one price-trend test before it calculates final weights.

For each ETF, the strategy calculates the mean of the last 10 completed
month-end prices:

```math
M_{i,t}=\frac{1}{10}\sum_{k=0}^{9}P_{i,t-k}
```

An ETF passes when its current completed month-end price is above this mean:

```math
P_{i,t}>M_{i,t}
```

An ETF that passes can receive a Volatility Balanced weight. An ETF that does
not pass receives a zero ETF weight. The position limit still applies.

The strategy assigns unused weight to cash. If all selected ETFs are below
their trends, the target is 100% cash.

When the user selects one ETF, the position limit is 100%. The strategy therefore
holds the ETF when it passes the trend test. It holds cash when it does not pass.

## Equal Weight

Equal Weight assigns the same percentage to each selected ETF:

```math
w_i=\frac{1}{n}
```

The backtest resets these weights each month.

## Cash

The cash label is **Cash — U.S. overnight-rate proxy**. Its internal identifier
is `CASH:USD_OVERNIGHT`.

The cash history uses the official Effective Federal Funds Rate (EFFR) before
April 2, 2018. It uses official Secured Overnight Financing Rate (SOFR) from
April 2, 2018. The [New York Fed started SOFR publication on April 3, 2018](https://www.newyorkfed.org/markets/opolicy/operating_policy_180403)
for the April 2 effective date. This date is the point at which the source
changes.

For an annual rate \(q_d\), held for \(D\) calendar days, cash growth is:

```math
1+\frac{q_d}{100}\frac{D}{360}
```

The series is an analytical cash balance. BIL is a separate, selectable ETF.
The rebalance ticket shows analytical cash as a balance and excludes it from
ETF orders.

## Common Backtest Accounting

Each selected date range starts with 100% cash. The first ETF purchase is a
trade and has a transaction cost.

Let \(w_t^-\) be the weights before the rebalance. Let \(w_t^*\) be the new
target. The trade vector is:

```math
\Delta w_t=w_t^*-w_t^-
```

One-way turnover is:

```math
T_t=\frac{1}{2}\sum_i|\Delta w_{i,t}|
```

For a transaction-cost setting of \(b\) basis points, the cost rate is:

```math
C_t=T_t\frac{b}{10{,}000}
```

Let \(R_{i,t}\) be the holding-period return of asset \(i\). Gross portfolio
growth is:

```math
G_t=\sum_i w_{i,t}^*(1+R_{i,t})
```

Net portfolio return is:

```math
R_t^{net}=(1-C_t)G_t-1
```

The ending weight becomes the next pre-trade weight:

```math
w_{i,t}^{end}=\frac{w_{i,t}^*(1+R_{i,t})}{G_t}
```

This step includes portfolio drift. Turnover therefore compares the new target
with the actual drifted weights, not with the prior target.

## Current Mix

Current Mix is available when the user enters ETF and cash weights that total
100%. Its chart name is **Current Mix — monthly rebalanced**.

For the historical comparison, the application resets the entered weights each
month. The result is a constant-weight comparison. The latest rebalance ticket
uses the entered weights as the current portfolio.

## Measures In The Application

**Annualized return** is the compound yearly growth rate.

**Annualized volatility** is the standard deviation of monthly returns, scaled
to one year.

**Sharpe above cash** is the average monthly portfolio return above cash divided
by the variability of that excess return. The result is scaled to one year.

**Maximum drawdown** is the largest percentage decline from a previous high.

**Return / drawdown** divides annualized return by the absolute maximum drawdown.

**Worst month** is the lowest monthly net return.

**Annualized turnover** is average monthly one-way turnover multiplied by 12.

**Annualized cost drag** is gross annualized return minus net annualized return.
