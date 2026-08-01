"""Strategy-agnostic monthly execution and portfolio accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.metrics import calculate_metrics


CASH_ASSET = "CASH:USD_OVERNIGHT"
WEIGHT_TOLERANCE = 1e-9


def _normalized_datetime_index(values, name):
    try:
        index = pd.DatetimeIndex(pd.to_datetime(values))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain valid dates") from exc
    if index.hasnans:
        raise ValueError(f"{name} must contain valid dates")
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    index = index.normalize()
    index.name = name
    return index


def _normalized_timestamp(value, name):
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid date")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def _date_series(values, execution_dates, name):
    if isinstance(values, pd.Series):
        value_index = _normalized_datetime_index(values.index, "execution_date")
        if not value_index.equals(execution_dates):
            raise ValueError(f"{name} index must equal the execution-date index")
        raw_values = values.to_numpy()
    else:
        raw_values = values
    dates = _normalized_datetime_index(raw_values, name)
    if len(dates) != len(execution_dates):
        raise ValueError(f"{name} must contain one value per execution date")
    return pd.Series(dates, index=execution_dates, name=name)


@dataclass(frozen=True)
class TargetWeightSchedule:
    """Target weights and the information/execution timing they represent.

    ``weights`` is indexed by execution date. Each return supplied to the engine
    for that row must cover the close-to-close interval from its execution date
    through ``period_end_dates``. A target may use only information available at
    its strictly earlier ``signal_dates`` value.
    """

    weights: pd.DataFrame
    signal_dates: pd.Series
    period_end_dates: pd.Series

    def __post_init__(self):
        if not isinstance(self.weights, pd.DataFrame):
            raise TypeError("weights must be a pandas DataFrame")
        if self.weights.empty:
            raise ValueError("weights must contain at least one target")
        if not self.weights.columns.is_unique:
            raise ValueError("weight columns must be unique")
        if CASH_ASSET not in self.weights.columns:
            raise ValueError(f"weights must include explicit {CASH_ASSET}")

        execution_dates = _normalized_datetime_index(
            self.weights.index, "execution_date"
        )
        if not execution_dates.is_unique:
            raise ValueError("execution dates must be unique")
        if not execution_dates.is_monotonic_increasing:
            raise ValueError("execution dates must be strictly increasing")

        try:
            weights = self.weights.astype(float).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError("weights must contain only numeric values") from exc
        weights.index = execution_dates
        weights.index.name = "execution_date"
        if weights.isna().any().any() or not np.isfinite(weights.to_numpy()).all():
            raise ValueError("weights must contain only finite values")
        if (weights < 0.0).any().any():
            raise ValueError("weights must be long-only")
        if (weights > 1.0 + WEIGHT_TOLERANCE).any().any():
            raise ValueError("an individual target weight cannot exceed one")
        row_sums = weights.sum(axis=1)
        invalid_sums = ~np.isclose(
            row_sums.to_numpy(), 1.0, rtol=0.0, atol=WEIGHT_TOLERANCE
        )
        if invalid_sums.any():
            bad_date = row_sums.index[np.flatnonzero(invalid_sums)[0]]
            raise ValueError(
                f"target weights must sum to one on {bad_date.date()}"
            )

        signal_dates = _date_series(
            self.signal_dates, execution_dates, "signal_date"
        )
        period_end_dates = _date_series(
            self.period_end_dates, execution_dates, "period_end_date"
        )
        if not signal_dates.is_unique or not signal_dates.is_monotonic_increasing:
            raise ValueError("signal dates must be unique and strictly increasing")
        if (
            not period_end_dates.is_unique
            or not period_end_dates.is_monotonic_increasing
        ):
            raise ValueError(
                "period-end dates must be unique and strictly increasing"
            )
        if not (signal_dates.to_numpy() < execution_dates.to_numpy()).all():
            raise ValueError("every signal date must precede its execution date")
        if not (period_end_dates.to_numpy() > execution_dates.to_numpy()).all():
            raise ValueError("every period end must follow its execution date")
        if len(execution_dates) > 1:
            adjacent_ends = period_end_dates.iloc[:-1].to_numpy()
            adjacent_executions = execution_dates[1:].to_numpy()
            if not (adjacent_ends == adjacent_executions).all():
                raise ValueError(
                    "each period end must equal the following execution date"
                )

        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "signal_dates", signal_dates)
        object.__setattr__(self, "period_end_dates", period_end_dates)

    @property
    def execution_dates(self):
        return self.weights.index

    def select(self, start=None, end=None):
        """Select a date range; the engine restarts it from cash on execution."""
        mask = pd.Series(True, index=self.execution_dates)
        if start is not None:
            start_date = _normalized_timestamp(start, "start")
            mask &= self.execution_dates >= start_date
        if end is not None:
            end_date = _normalized_timestamp(end, "end")
            mask &= self.period_end_dates <= end_date
        selected_dates = self.execution_dates[mask.to_numpy()]
        if selected_dates.empty:
            raise ValueError("the selected date range contains no complete periods")
        return TargetWeightSchedule(
            weights=self.weights.loc[selected_dates],
            signal_dates=self.signal_dates.loc[selected_dates],
            period_end_dates=self.period_end_dates.loc[selected_dates],
        )


@dataclass(frozen=True)
class BacktestResult:
    """Auditable accounting output from ``run_backtest``."""

    periods: pd.DataFrame
    target_weights: pd.DataFrame
    pretrade_weights: pd.DataFrame
    trades: pd.DataFrame
    ending_weights: pd.DataFrame
    metrics: dict


def completed_month_end_dates(observation_dates, as_of=None):
    """Return the last supplied observation for each month completed before as-of.

    This timing primitive does not prove that the supplied observations cover a
    complete month. Upstream market-data validation is responsible for checking
    coverage before treating the returned observations as month-end prices.
    """
    dates = _normalized_datetime_index(observation_dates, "observation_date")
    if not dates.is_unique:
        raise ValueError("observation dates must be unique")
    if not dates.is_monotonic_increasing:
        raise ValueError("observation dates must be strictly increasing")
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC")
    as_of = _normalized_timestamp(as_of, "as_of")
    completed = dates[dates.to_period("M") < as_of.to_period("M")]
    if completed.empty:
        return pd.DatetimeIndex([], name="signal_date")
    month_ends = (
        pd.Series(completed, index=completed.to_period("M"))
        .groupby(level=0)
        .max()
    )
    return pd.DatetimeIndex(month_ends.to_numpy(), name="signal_date")


def build_completed_monthly_periods(observation_dates, as_of=None):
    """Build non-overlapping monthly signal/execution/return-period timing.

    Signals are the last observation of each completed calendar month. Execution
    occurs at the first subsequent observation. A backtest row is emitted only
    once the following execution is known, so every returned holding period is
    fully observed.
    """
    dates = _normalized_datetime_index(observation_dates, "observation_date")
    if not dates.is_unique:
        raise ValueError("observation dates must be unique")
    if not dates.is_monotonic_increasing:
        raise ValueError("observation dates must be strictly increasing")
    signals = completed_month_end_dates(dates, as_of=as_of)

    executed_signals = []
    execution_dates = []
    for signal_date in signals:
        position = dates.searchsorted(signal_date, side="right")
        if position < len(dates):
            executed_signals.append(signal_date)
            execution_dates.append(dates[position])

    columns = ["signal_date", "period_end_date"]
    if len(execution_dates) < 2:
        empty_index = pd.DatetimeIndex([], name="execution_date")
        return pd.DataFrame(index=empty_index, columns=columns)

    timing = pd.DataFrame(
        {
            "signal_date": executed_signals[:-1],
            "period_end_date": execution_dates[1:],
        },
        index=pd.DatetimeIndex(execution_dates[:-1], name="execution_date"),
    )
    return timing


def _validated_return_inputs(schedule, asset_returns, cash_returns):
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame")
    if CASH_ASSET in asset_returns.columns:
        raise ValueError("cash returns must be supplied separately")
    if not asset_returns.columns.is_unique:
        raise ValueError("asset return columns must be unique")

    expected_assets = [column for column in schedule.weights if column != CASH_ASSET]
    if expected_assets and asset_returns.empty:
        raise ValueError("asset_returns must contain holding-period returns")
    if set(asset_returns.columns) != set(expected_assets):
        missing = sorted(set(expected_assets) - set(asset_returns.columns))
        extra = sorted(set(asset_returns.columns) - set(expected_assets))
        raise ValueError(
            f"asset return columns do not match targets; missing={missing}, extra={extra}"
        )

    return_dates = _normalized_datetime_index(
        asset_returns.index, "execution_date"
    )
    if not return_dates.is_unique:
        raise ValueError("asset return dates must be unique")
    if not return_dates.is_monotonic_increasing:
        raise ValueError("asset return dates must be strictly increasing")
    returns = asset_returns.astype(float).copy()
    returns.index = return_dates

    cash = pd.Series(cash_returns, copy=True, dtype=float)
    cash_dates = _normalized_datetime_index(cash.index, "execution_date")
    if not cash_dates.is_unique:
        raise ValueError("cash return dates must be unique")
    if not cash_dates.is_monotonic_increasing:
        raise ValueError("cash return dates must be strictly increasing")
    cash.index = cash_dates
    cash.name = CASH_ASSET

    missing_asset_dates = schedule.execution_dates.difference(returns.index)
    missing_cash_dates = schedule.execution_dates.difference(cash.index)
    if not missing_asset_dates.empty:
        raise ValueError(
            "asset returns are missing execution dates: "
            + ", ".join(date.strftime("%Y-%m-%d") for date in missing_asset_dates)
        )
    if not missing_cash_dates.empty:
        raise ValueError(
            "cash returns are missing execution dates: "
            + ", ".join(date.strftime("%Y-%m-%d") for date in missing_cash_dates)
        )

    returns = returns.loc[schedule.execution_dates, expected_assets]
    cash = cash.loc[schedule.execution_dates]
    if returns.isna().any().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("asset returns must contain only finite values")
    if cash.isna().any() or not np.isfinite(cash.to_numpy()).all():
        raise ValueError("cash returns must contain only finite values")
    if (returns <= -1.0).any().any():
        raise ValueError("asset returns cannot be less than or equal to -100%")
    if (cash <= -1.0).any():
        raise ValueError("cash returns cannot be less than or equal to -100%")
    return returns, cash


def run_backtest(
    schedule,
    asset_returns,
    cash_returns,
    transaction_cost_bps=0.0,
    initial_value=1.0,
    start=None,
    end=None,
):
    """Execute a target schedule with common drift, turnover, cost, and cash rules.

    ``asset_returns`` and ``cash_returns`` are indexed by execution date; each row
    is the realized close-to-close return ending on the corresponding value in
    ``schedule.period_end_dates``. The selected range always starts from 100%
    analytical cash and therefore charges any initial entry trade.
    """
    if not isinstance(schedule, TargetWeightSchedule):
        raise TypeError("schedule must be a TargetWeightSchedule")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be finite and nonnegative")
    if transaction_cost_bps >= 10000.0:
        raise ValueError("transaction_cost_bps must be less than 10000")
    if not np.isfinite(initial_value) or initial_value <= 0.0:
        raise ValueError("initial_value must be finite and positive")

    selected = (
        schedule.select(start=start, end=end)
        if start is not None or end is not None
        else schedule
    )
    returns, cash = _validated_return_inputs(selected, asset_returns, cash_returns)

    columns = list(selected.weights.columns)
    combined_returns = returns.copy()
    combined_returns[CASH_ASSET] = cash
    combined_returns = combined_returns.reindex(columns=columns)

    pretrade_rows = []
    trade_rows = []
    ending_rows = []
    period_rows = []

    target_values = selected.weights.to_numpy(dtype=float)
    return_values = combined_returns.to_numpy(dtype=float)
    cash_values = cash.to_numpy(dtype=float)
    pretrade = np.zeros(len(columns), dtype=float)
    pretrade[columns.index(CASH_ASSET)] = 1.0
    net_equity = float(initial_value)
    gross_equity = float(initial_value)
    running_peak = float(initial_value)

    for row_number, execution_date in enumerate(selected.execution_dates):
        target = target_values[row_number]
        trades = target - pretrade
        if not np.isclose(
            float(np.sum(trades)), 0.0, rtol=0.0, atol=WEIGHT_TOLERANCE
        ):
            raise ArithmeticError("trades do not sum to zero")

        turnover = 0.5 * float(np.sum(np.abs(trades)))
        cost_rate = turnover * float(transaction_cost_bps) / 10000.0
        if cost_rate >= 1.0:
            raise ValueError("transaction costs would exhaust the portfolio")

        starting_equity = net_equity
        cost_amount = starting_equity * cost_rate
        value_after_cost = starting_equity - cost_amount
        period_asset_returns = return_values[row_number]
        gross_growth = float(np.dot(target, 1.0 + period_asset_returns))
        if not np.isfinite(gross_growth) or gross_growth <= 0.0:
            raise ArithmeticError("holding-period gross growth must be positive")

        gross_return = gross_growth - 1.0
        net_return = (1.0 - cost_rate) * gross_growth - 1.0
        gross_equity *= gross_growth
        net_equity = value_after_cost * gross_growth
        ending = target * (1.0 + period_asset_returns) / gross_growth
        if not np.isclose(
            float(np.sum(ending)), 1.0, rtol=0.0, atol=WEIGHT_TOLERANCE
        ):
            raise ArithmeticError("ending weights do not sum to one")

        running_peak = max(running_peak, net_equity)
        drawdown = net_equity / running_peak - 1.0

        pretrade_rows.append(pretrade.copy())
        trade_rows.append(trades.copy())
        ending_rows.append(ending.copy())
        period_rows.append(
            {
                "execution_date": execution_date,
                "signal_date": selected.signal_dates.loc[execution_date],
                "period_end_date": selected.period_end_dates.loc[execution_date],
                "starting_equity": starting_equity,
                "turnover": turnover,
                "cost_rate": cost_rate,
                "transaction_cost": cost_amount,
                "value_after_cost": value_after_cost,
                "cash_return": float(cash_values[row_number]),
                "gross_return": gross_return,
                "net_return": net_return,
                "gross_equity": gross_equity,
                "net_equity": net_equity,
                "drawdown": drawdown,
            }
        )
        pretrade = ending

    result_index = pd.DatetimeIndex(
        selected.execution_dates, name="execution_date"
    )
    target_weights = selected.weights.copy()
    pretrade_weights = pd.DataFrame(
        pretrade_rows, index=result_index, columns=columns
    )
    trades_frame = pd.DataFrame(trade_rows, index=result_index, columns=columns)
    ending_weights = pd.DataFrame(
        ending_rows, index=result_index, columns=columns
    )
    for frame in (
        target_weights,
        pretrade_weights,
        trades_frame,
        ending_weights,
    ):
        frame.index = pd.DatetimeIndex(frame.index, name="execution_date")
        frame.index.freq = None
        frame.columns.name = "asset"

    periods = pd.DataFrame(period_rows).set_index("execution_date")
    periods.index = pd.DatetimeIndex(periods.index, name="execution_date")
    metrics = calculate_metrics(
        periods["net_return"],
        periods["gross_return"],
        periods["cash_return"],
        periods["turnover"],
    )
    return BacktestResult(
        periods=periods,
        target_weights=target_weights,
        pretrade_weights=pretrade_weights,
        trades=trades_frame,
        ending_weights=ending_weights,
        metrics=metrics,
    )
