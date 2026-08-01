"""Pure monthly ETF target generators, separate from return accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from backtest.engine import (
    CASH_ASSET,
    TargetWeightSchedule,
    build_completed_monthly_periods,
    completed_month_end_dates,
)


VOLATILITY_BALANCED = "volatility_balanced"
VOLATILITY_BALANCED_TREND = "volatility_balanced_trend"
EQUAL_WEIGHT = "equal_weight"
SUPPORTED_STRATEGIES = (
    VOLATILITY_BALANCED,
    VOLATILITY_BALANCED_TREND,
    EQUAL_WEIGHT,
)


@dataclass(frozen=True)
class StrategyPolicy:
    """The centrally versioned, non-configurable version-one strategy policy."""

    version: str = "allocation-policy-v1"
    volatility_lookback_days: int = 126
    minimum_volatility_observations: int = 120
    volatility_annualization: int = 252
    trend_months: int = 10
    cap_multiplier: float = 1.5
    minimum_selected_etfs: int = 1
    maximum_selected_etfs: int = 8
    cap_explanation: str = (
        "No ETF can receive more than 150% of its equal-weight share."
    )


DEFAULT_POLICY = StrategyPolicy()


def position_cap(selected_count, policy=DEFAULT_POLICY):
    """Return the adaptive cap ``min(100%, 1.5 / selected ETF count)``."""
    if not isinstance(selected_count, (int, np.integer)):
        raise TypeError("selected_count must be an integer")
    if not policy.minimum_selected_etfs <= selected_count <= policy.maximum_selected_etfs:
        raise ValueError(
            f"selected_count must be between {policy.minimum_selected_etfs} "
            f"and {policy.maximum_selected_etfs}"
        )
    return min(1.0, policy.cap_multiplier / selected_count)


def _validate_prices_and_selection(prices, selected, policy):
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.empty or not prices.columns.is_unique:
        raise ValueError("prices must be nonempty with unique columns")
    selected = tuple(selected)
    if not policy.minimum_selected_etfs <= len(selected) <= policy.maximum_selected_etfs:
        raise ValueError(
            f"select between {policy.minimum_selected_etfs} and "
            f"{policy.maximum_selected_etfs} ETFs"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("selected ETFs must be unique")
    if CASH_ASSET in selected:
        raise ValueError(f"{CASH_ASSET} is analytical cash, not a selectable ETF")
    missing = [ticker for ticker in selected if ticker not in prices.columns]
    if missing:
        raise ValueError(f"selected ETFs are absent from prices: {missing}")

    frame = prices.loc[:, selected].copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("price dates must be unique and increasing")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    valid = frame.stack(future_stack=True).dropna()
    if valid.empty or not np.isfinite(valid.to_numpy()).all() or (valid <= 0.0).any():
        raise ValueError("prices must contain finite positive observations")
    return frame, selected


def _capped_waterfill(scores, cap):
    """Allocate positive scores proportionally subject to a long-only cap.

    When the eligible set has less than 100% aggregate cap capacity, the result
    deliberately sums to less than one and the caller assigns the residual to
    analytical cash.
    """
    scores = pd.Series(scores, dtype=float)
    scores = scores[(scores > 0.0) & np.isfinite(scores)]
    result = pd.Series(0.0, index=scores.index, dtype=float)
    if scores.empty:
        return result
    target_mass = min(1.0, cap * len(scores))
    remaining = list(scores.index)
    remaining_mass = target_mass
    while remaining and remaining_mass > 1e-15:
        denominator = float(scores.loc[remaining].sum())
        proposal = scores.loc[remaining] / denominator * remaining_mass
        over_cap = proposal[proposal > cap + 1e-15]
        if over_cap.empty:
            result.loc[remaining] = proposal
            break
        capped_names = list(over_cap.index)
        result.loc[capped_names] = cap
        remaining_mass -= cap * len(capped_names)
        remaining = [name for name in remaining if name not in capped_names]
    return result


def _normalize_rows(scores):
    totals = scores.sum(axis=1).replace(0.0, np.nan)
    return scores.div(totals, axis=0).fillna(0.0)


def _diagnostic_frame(
    signal_dates,
    selected,
    volatility,
    trend_ready,
    trend_pass,
    eligible,
    unfiltered_raw,
    filtered_raw,
    final_etf,
    strategy,
    cap,
):
    rows = []
    for signal_date in signal_dates:
        for ticker in selected:
            if strategy == EQUAL_WEIGHT:
                reason = "Included; equal weight ignores volatility and trend"
            elif not pd.notna(volatility.loc[signal_date, ticker]):
                reason = "Insufficient volatility history"
            elif volatility.loc[signal_date, ticker] <= 1e-12:
                reason = "Invalid or zero volatility"
            elif not trend_ready.loc[signal_date, ticker]:
                reason = "Insufficient trend history"
            elif (
                strategy == VOLATILITY_BALANCED_TREND
                and eligible.loc[signal_date, ticker]
                and not trend_pass.loc[signal_date, ticker]
            ):
                reason = "Below trend; receives zero ETF weight"
            elif (
                strategy != EQUAL_WEIGHT
                and final_etf.loc[signal_date, ticker] >= cap - 1e-12
            ):
                reason = "Included at the adaptive position cap"
            elif final_etf.loc[signal_date, ticker] > 0.0:
                reason = "Included"
            else:
                reason = "Ineligible"
            if not trend_ready.loc[signal_date, ticker]:
                trend_status = "Insufficient history"
            elif trend_pass.loc[signal_date, ticker]:
                trend_status = "Above trend"
            else:
                trend_status = "Below trend"
            rows.append(
                {
                    "signal_date": signal_date,
                    "ticker": ticker,
                    "trend_status": trend_status,
                    "trailing_volatility": volatility.loc[signal_date, ticker],
                    "unfiltered_inverse_vol_weight": unfiltered_raw.loc[
                        signal_date, ticker
                    ],
                    "filtered_raw_weight": filtered_raw.loc[signal_date, ticker],
                    "cap_adjustment": final_etf.loc[signal_date, ticker]
                    - filtered_raw.loc[signal_date, ticker],
                    "final_target_weight": final_etf.loc[signal_date, ticker],
                    "eligibility_reason": reason,
                }
            )
    return pd.DataFrame(rows).set_index(["signal_date", "ticker"])


@dataclass(frozen=True)
class AllocationResult:
    strategy: str
    policy: StrategyPolicy
    selected_etfs: tuple[str, ...]
    schedule: TargetWeightSchedule
    diagnostics: pd.DataFrame
    latest_target: pd.Series
    latest_diagnostics: pd.DataFrame
    common_start_signal_date: pd.Timestamp
    latest_signal_date: pd.Timestamp
    latest_execution_date: pd.Timestamp | None


def generate_allocation_targets(
    prices,
    selected_etfs,
    strategy,
    as_of=None,
    policy=DEFAULT_POLICY,
):
    """Generate historical and latest targets without calculating any returns.

    All selected ETFs must complete the fixed volatility and trend warm-up before
    the common start.  Signals use completed month-end observations only and are
    mapped to the next common ETF trading close for execution.
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"unsupported strategy: {strategy}")
    prices, selected = _validate_prices_and_selection(prices, selected_etfs, policy)
    common_prices = prices.loc[:, selected].dropna(how="any")
    if common_prices.empty:
        raise ValueError("selected ETFs have no shared price history")
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC")
    as_of = pd.Timestamp(as_of)
    if as_of.tz is not None:
        as_of = as_of.tz_convert("UTC").tz_localize(None)

    monthly_signals = completed_month_end_dates(common_prices.index, as_of=as_of)
    if monthly_signals.empty:
        raise ValueError("selected ETFs have no completed signal month")

    daily_returns = prices.loc[:, selected].pct_change(fill_method=None)
    rolling_volatility = (
        daily_returns.rolling(
            window=policy.volatility_lookback_days,
            min_periods=policy.minimum_volatility_observations,
        ).std(ddof=1)
        * math.sqrt(policy.volatility_annualization)
    )
    monthly_prices = prices.loc[:, selected].reindex(monthly_signals)
    trend_average = monthly_prices.rolling(
        window=policy.trend_months, min_periods=policy.trend_months
    ).mean()

    signal_volatility = rolling_volatility.reindex(monthly_signals)
    trend_ready = monthly_prices.notna() & trend_average.notna()
    trend_pass = trend_ready & (monthly_prices > trend_average)
    history_ready = (
        monthly_prices.notna()
        & signal_volatility.notna()
        & trend_ready
    )
    eligible = history_ready & (signal_volatility > 1e-12)
    common_ready = (
        history_ready.all(axis=1)
        if strategy == EQUAL_WEIGHT
        else eligible.all(axis=1)
    )
    if not common_ready.any():
        raise ValueError("selected ETFs do not have enough shared warm-up history")
    common_start = pd.Timestamp(common_ready.index[common_ready][0])

    usable_signals = monthly_signals[monthly_signals >= common_start]
    latest_signal = pd.Timestamp(usable_signals[-1])
    signal_volatility = signal_volatility.loc[usable_signals]
    trend_ready = trend_ready.loc[usable_signals]
    trend_pass = trend_pass.loc[usable_signals]
    eligible = eligible.loc[usable_signals]

    inverse_scores = (1.0 / signal_volatility).where(eligible, 0.0)
    inverse_scores = inverse_scores.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    unfiltered_raw = _normalize_rows(inverse_scores)
    if strategy == VOLATILITY_BALANCED_TREND:
        allocation_scores = inverse_scores.where(trend_pass, 0.0)
    elif strategy == EQUAL_WEIGHT:
        allocation_scores = pd.DataFrame(
            1.0, index=usable_signals, columns=selected
        )
    else:
        allocation_scores = inverse_scores
    filtered_raw = _normalize_rows(allocation_scores)

    cap = position_cap(len(selected), policy)
    if strategy == EQUAL_WEIGHT:
        final_etf = filtered_raw.copy()
    else:
        final_etf = pd.DataFrame(0.0, index=usable_signals, columns=selected)
        for signal_date in usable_signals:
            allocated = _capped_waterfill(allocation_scores.loc[signal_date], cap)
            final_etf.loc[signal_date, allocated.index] = allocated
    target_by_signal = final_etf.copy()
    target_by_signal[CASH_ASSET] = (1.0 - final_etf.sum(axis=1)).clip(lower=0.0)

    later_dates = common_prices.index[common_prices.index > latest_signal]
    latest_execution = pd.Timestamp(later_dates[0]) if len(later_dates) else None

    timing = build_completed_monthly_periods(common_prices.index, as_of=as_of)
    timing = timing.loc[timing["signal_date"] >= common_start]
    if timing.empty:
        raise ValueError("selected ETFs have no fully observed holding periods")
    weight_frame = target_by_signal.loc[timing["signal_date"].to_numpy()].copy()
    weight_frame.index = timing.index
    schedule = TargetWeightSchedule(
        weights=weight_frame,
        signal_dates=timing["signal_date"],
        period_end_dates=timing["period_end_date"],
    )
    diagnostic_dates = pd.DatetimeIndex(timing["signal_date"].unique())
    diagnostics = _diagnostic_frame(
        diagnostic_dates,
        selected,
        signal_volatility,
        trend_ready,
        trend_pass,
        eligible,
        unfiltered_raw,
        filtered_raw,
        final_etf,
        strategy,
        cap,
    )
    latest_diagnostics = _diagnostic_frame(
        [latest_signal],
        selected,
        signal_volatility,
        trend_ready,
        trend_pass,
        eligible,
        unfiltered_raw,
        filtered_raw,
        final_etf,
        strategy,
        cap,
    )
    return AllocationResult(
        strategy=strategy,
        policy=policy,
        selected_etfs=selected,
        schedule=schedule,
        diagnostics=diagnostics,
        latest_target=target_by_signal.loc[latest_signal].reindex(
            [*selected, CASH_ASSET]
        ),
        latest_diagnostics=latest_diagnostics,
        common_start_signal_date=common_start,
        latest_signal_date=latest_signal,
        latest_execution_date=latest_execution,
    )
