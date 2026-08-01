import math

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    annualized_return,
    annualized_volatility,
    calculate_drawdowns,
    calculate_metrics,
    excess_return_sharpe,
)


def _series(values):
    return pd.Series(values, index=pd.date_range("2020-01-31", periods=len(values), freq="ME"))


def test_drawdown_includes_initial_portfolio_value_as_peak():
    returns = _series([-0.1, 0.2, -0.05])

    drawdowns = calculate_drawdowns(returns)

    assert drawdowns.to_numpy() == pytest.approx([-0.1, 0.0, -0.05])
    assert drawdowns.index.equals(returns.index)


def test_annualized_return_is_geometric():
    returns = _series([0.1, -0.1])

    result = annualized_return(returns, periods_per_year=2)

    assert result == pytest.approx(-0.01)


def test_annualized_volatility_uses_sample_standard_deviation():
    returns = _series([0.0, 0.1, -0.1])

    result = annualized_volatility(returns, periods_per_year=12)

    assert result == pytest.approx(float(returns.std(ddof=1) * math.sqrt(12)))


def test_excess_return_sharpe_uses_periodic_arithmetic_excess_returns():
    returns = _series([0.02, 0.01, -0.01, 0.03])
    cash = _series([0.002, 0.002, 0.001, 0.002])
    excess = returns - cash

    result = excess_return_sharpe(returns, cash)

    expected = excess.mean() / excess.std(ddof=1) * math.sqrt(12)
    assert result == pytest.approx(expected)


def test_excess_return_sharpe_is_undefined_for_constant_excess_returns():
    returns = _series([0.01, 0.01, 0.01])
    cash = _series([0.0, 0.0, 0.0])

    assert math.isnan(excess_return_sharpe(returns, cash))


def test_calculate_metrics_reconciles_required_metric_set():
    net = _series([0.02, -0.03, 0.01, 0.04])
    gross = _series([0.021, -0.029, 0.011, 0.041])
    cash = _series([0.001, 0.001, 0.001, 0.001])
    turnover = _series([1.0, 0.2, 0.3, 0.1])

    result = calculate_metrics(net, gross, cash, turnover)

    expected_keys = {
        "periods",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "excess_return_sharpe",
        "max_drawdown",
        "calmar_ratio",
        "worst_month",
        "annualized_turnover",
        "transaction_cost_drag",
        "gross_annualized_return",
    }
    assert set(result) == expected_keys
    assert result["periods"] == 4
    assert result["total_return"] == pytest.approx(float((1.0 + net).prod() - 1.0))
    assert result["max_drawdown"] == pytest.approx(-0.03)
    assert result["worst_month"] == pytest.approx(-0.03)
    assert result["annualized_turnover"] == pytest.approx(turnover.mean() * 12)
    assert result["transaction_cost_drag"] == pytest.approx(
        annualized_return(gross) - annualized_return(net)
    )


def test_calmar_is_nan_when_no_drawdown_occurs():
    net = gross = _series([0.01, 0.02])
    cash = _series([0.0, 0.0])
    turnover = _series([0.0, 0.0])

    metrics = calculate_metrics(net, gross, cash, turnover)

    assert metrics["max_drawdown"] == pytest.approx(0.0)
    assert math.isnan(metrics["calmar_ratio"])


@pytest.mark.parametrize("values", [[-1.0], [np.nan], [np.inf]])
def test_metrics_reject_invalid_returns(values):
    with pytest.raises(ValueError):
        annualized_return(_series(values))


def test_metrics_require_aligned_indexes():
    returns = _series([0.01, 0.02])
    cash = returns.copy()
    cash.index = cash.index + pd.offsets.Day(1)

    with pytest.raises(ValueError, match="identical indexes"):
        excess_return_sharpe(returns, cash)


def test_metrics_reject_unsorted_return_index():
    unsorted = _series([0.01, 0.02, 0.03]).sort_index(ascending=False)

    with pytest.raises(ValueError, match="monotonically increasing"):
        annualized_return(unsorted)


def test_metrics_reject_unsorted_turnover_index():
    net = _series([0.01, 0.02, 0.03])
    gross = net.copy()
    cash = _series([0.001, 0.001, 0.001])
    turnover = _series([0.1, 0.2, 0.3]).sort_index(ascending=False)

    with pytest.raises(ValueError, match="monotonically increasing"):
        calculate_metrics(net, gross, cash, turnover)
