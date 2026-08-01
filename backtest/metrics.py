"""Performance metrics shared by every workbench strategy and benchmark."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


MONTHS_PER_YEAR = 12


def _validated_series(values, name):
    series = pd.Series(values, copy=True, dtype=float)
    if series.empty:
        raise ValueError(f"{name} must contain at least one observation")
    if not series.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonically increasing")
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values")
    return series


def _validated_returns(values, name):
    returns = _validated_series(values, name)
    if (returns <= -1.0).any():
        raise ValueError(f"{name} cannot contain returns less than or equal to -100%")
    return returns


def calculate_drawdowns(returns):
    """Return drawdowns from a return series, including the initial value as a peak."""
    returns = _validated_returns(returns, "returns")
    wealth = (1.0 + returns).cumprod()
    running_peak = pd.concat(
        [pd.Series([1.0], dtype=float), wealth.reset_index(drop=True)],
        ignore_index=True,
    ).cummax().iloc[1:]
    drawdowns = wealth.to_numpy() / running_peak.to_numpy() - 1.0
    return pd.Series(drawdowns, index=returns.index, name="drawdown")


def annualized_return(returns, periods_per_year=MONTHS_PER_YEAR):
    """Calculate a geometrically annualized return from periodic returns."""
    returns = _validated_returns(returns, "returns")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    growth = float((1.0 + returns).prod())
    return growth ** (periods_per_year / len(returns)) - 1.0


def annualized_volatility(returns, periods_per_year=MONTHS_PER_YEAR):
    """Calculate sample volatility annualized by the square-root-of-time rule."""
    returns = _validated_returns(returns, "returns")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if len(returns) < 2:
        return math.nan
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def excess_return_sharpe(returns, cash_returns, periods_per_year=MONTHS_PER_YEAR):
    """Calculate annualized arithmetic Sharpe from periodic returns over cash."""
    returns = _validated_returns(returns, "returns")
    cash_returns = _validated_returns(cash_returns, "cash_returns")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if not returns.index.equals(cash_returns.index):
        raise ValueError("returns and cash_returns must have identical indexes")
    if len(returns) < 2:
        return math.nan

    excess = returns - cash_returns
    excess_std = float(excess.std(ddof=1))
    if math.isclose(excess_std, 0.0, abs_tol=1e-15):
        return math.nan
    return float(excess.mean() / excess_std * math.sqrt(periods_per_year))


def calculate_metrics(
    net_returns,
    gross_returns,
    cash_returns,
    turnover,
    periods_per_year=MONTHS_PER_YEAR,
):
    """Calculate the concise metric set used by the ETF Allocation Workbench.

    All inputs must describe the same periodic observations. Transaction-cost drag
    is the gross CAGR less the net CAGR, so it includes compounding interactions.
    """
    net = _validated_returns(net_returns, "net_returns")
    gross = _validated_returns(gross_returns, "gross_returns")
    cash = _validated_returns(cash_returns, "cash_returns")
    turnover = _validated_series(turnover, "turnover")

    for name, values in (
        ("gross_returns", gross),
        ("cash_returns", cash),
        ("turnover", turnover),
    ):
        if not net.index.equals(values.index):
            raise ValueError(f"net_returns and {name} must have identical indexes")
    if (turnover < 0.0).any():
        raise ValueError("turnover cannot be negative")

    net_cagr = annualized_return(net, periods_per_year)
    gross_cagr = annualized_return(gross, periods_per_year)
    drawdowns = calculate_drawdowns(net)
    max_drawdown = float(drawdowns.min())
    calmar = math.nan
    if max_drawdown < -1e-15:
        calmar = net_cagr / abs(max_drawdown)

    return {
        "periods": int(len(net)),
        "total_return": float((1.0 + net).prod() - 1.0),
        "annualized_return": net_cagr,
        "annualized_volatility": annualized_volatility(net, periods_per_year),
        "excess_return_sharpe": excess_return_sharpe(net, cash, periods_per_year),
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "worst_month": float(net.min()),
        "annualized_turnover": float(turnover.mean() * periods_per_year),
        "transaction_cost_drag": gross_cagr - net_cagr,
        "gross_annualized_return": gross_cagr,
    }
