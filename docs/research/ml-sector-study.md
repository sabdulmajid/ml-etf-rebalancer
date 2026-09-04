# ML Sector Study

## Research Question

The sector study asks this question:

> Can a monthly ranking model combine price patterns, momentum, and stability to
> allocate across U.S. equity sectors?

The study uses nine sector ETFs:

`XLB`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLU`, `XLV`, and `XLY`.

It uses `SPY` as the broad U.S. equity reference.

## Input Features

For each sector ETF, the study calculates:

- price momentum over 20, 60, 125, and 250 trading days;
- daily return volatility over 20 and 60 trading days;
- the ratio of the 50-day moving average to the 200-day moving average; and
- six-month sector strength relative to the average sector price series.

The feature values are shifted before model use. This keeps future prices out of
the input row.

## Walk-Forward Model

The model is ridge regression with standardized inputs. Ridge regression is a
linear model that limits large coefficients.

The process starts after at least 48 monthly training rows are available. For
each new month:

1. Use only rows dated before that month for model training.
2. Fit one model for each sector ETF.
3. Estimate each sector's next monthly return.
4. Build a score for each sector.
5. Convert the scores into portfolio weights.

The model is fit again for every month in the historical study. This is the
walk-forward process.

## ML Signal Blend

The strategy combines three cross-sector scores.

The forecast score is the standardized ridge-regression return estimate.

The momentum score is the standardized compounded return over the prior six
months.

The stability score is the standardized inverse of monthly return volatility.
Lower volatility gives a higher stability score.

The combined score is:

```math
S_i=0.40Z(\widehat R_i)+0.45Z(M_i)+0.15Z(1/\sigma_i)
```

Here, \(Z\) is a cross-sector standard score. \(M_i\) is six-month momentum.
\(\sigma_i\) is monthly return volatility.

The strategy subtracts the lowest score from all scores. It selects the four
largest adjusted scores. It gives higher weights to higher scores. A sector
weight cannot exceed 35%.

The ML Signal Blend uses its original monthly accounting. Its turnover measure
is the sum of the absolute target-weight changes:

```math
T_t=\sum_i|w_{i,t}-w_{i,t-1}|
```

Its monthly cost is \(T_t\) times 8 basis points. The first period compares the
target with zero ETF weights. The 6-Month Momentum Top 3 comparison uses the
same cost setting. Equal-Weight Sectors and SPY Buy & Hold use their recorded
returns without a cost deduction.

## Comparison Portfolios

**Equal-Weight Sectors** gives 1/9 of the portfolio to each sector ETF.

**6-Month Momentum Top 3** selects the three sectors with the highest prior
six-month returns. It gives one-third of the portfolio to each selected sector.

**SPY Buy & Hold** holds the broad U.S. equity ETF.

The sector study Sharpe ratio is annualized compound return divided by annualized
return volatility. It uses a 0% cash-rate assumption. The ETF Allocation
Workbench uses its separate cash-based Sharpe and common accounting definitions.

## Current Snapshot

The current committed study starts on January 31, 2015 and ends on April 30,
2026. It contains 136 monthly return observations.

| Strategy | Annualized return | Annualized volatility | Sharpe ratio | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| ML Signal Blend | 10.43% | 14.12% | 0.74 | -19.33% |
| Equal-Weight Sectors | 11.84% | 14.55% | 0.81 | -23.60% |
| 6-Month Momentum Top 3 | 10.01% | 14.22% | 0.70 | -15.29% |
| SPY Buy & Hold | 13.55% | 15.06% | 0.90 | -23.93% |

These values come from `artifacts/latest/metrics.csv`. The application displays
the same committed research snapshot.
