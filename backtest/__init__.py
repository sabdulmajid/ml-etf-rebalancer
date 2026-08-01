"""Shared, strategy-agnostic portfolio backtesting primitives."""

from backtest.engine import (
    CASH_ASSET,
    BacktestResult,
    TargetWeightSchedule,
    build_completed_monthly_periods,
    completed_month_end_dates,
    run_backtest,
)
from backtest.metrics import calculate_drawdowns, calculate_metrics

__all__ = [
    "CASH_ASSET",
    "BacktestResult",
    "TargetWeightSchedule",
    "build_completed_monthly_periods",
    "calculate_drawdowns",
    "calculate_metrics",
    "completed_month_end_dates",
    "run_backtest",
]
