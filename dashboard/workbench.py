"""ETF Allocation Workbench calculations and Streamlit rendering.

All return paths in this module are executed by ``backtest.engine.run_backtest``.
The UI performs no market-data HTTP calls and treats the validated PR2 bundle as
its sole data source.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import exchange_calendars as xcals

from backtest.engine import (
    CASH_ASSET,
    WEIGHT_TOLERANCE,
    BacktestResult,
    TargetWeightSchedule,
    run_backtest,
)
from data.cash import CASH_LABEL
from data.workbench import (
    DEFAULT_BUNDLE_PATH,
    PUBLIC_FILENAMES,
    WorkbenchBundle,
    file_sha256,
    load_workbench_bundle,
)
from data.timesfm import (
    DEFAULT_BUNDLE_PATH as DEFAULT_TIMESFM_BUNDLE_PATH,
    PUBLIC_FILENAMES as TIMESFM_PUBLIC_FILENAMES,
    load_timesfm_bundle,
)
from forecasting.timesfm import MULTIVARIATE_MODE
from portfolio.rebalance import (
    RebalanceTicket,
    build_rebalance_ticket,
    reconcile_current_weights,
    validate_portfolio_weights,
)
from strategies.allocation import (
    AllocationResult,
    EQUAL_WEIGHT,
    VOLATILITY_BALANCED,
    VOLATILITY_BALANCED_TREND,
    position_cap,
    generate_allocation_targets,
)
from strategies.forecast import (
    ForecastAllocationResult,
    generate_forecast_allocation_targets,
)


LOGGER = logging.getLogger(__name__)


VOL_BALANCED_LABEL = "Volatility Balanced"
VOL_TREND_LABEL = "Volatility Balanced + Trend"
VOL_FORECAST_LABEL = "Volatility Balanced + Forecast"
EQUAL_WEIGHT_LABEL = "Equal Weight"
CASH_LABEL_SHORT = CASH_LABEL
BUY_HOLD_LABEL = "Buy & Hold"
CURRENT_MIX_LABEL = "Current Mix — monthly rebalanced"
SPY_REFERENCE_LABEL = "SPY — U.S. equity reference"
DEFAULT_SELECTION = ("SPY", "IEF", "GLD")

CHART_COLORS = {
    VOL_BALANCED_LABEL: "#315f78",
    VOL_TREND_LABEL: "#9a632b",
    VOL_FORECAST_LABEL: "#155b46",
    EQUAL_WEIGHT_LABEL: "#6d7773",
    CASH_LABEL_SHORT: "#b28b52",
    BUY_HOLD_LABEL: "#2b3431",
    CURRENT_MIX_LABEL: "#765b75",
    SPY_REFERENCE_LABEL: "#2b3431",
}
ALLOCATION_COLORS = (
    "#155b46",
    "#315f78",
    "#9a632b",
    "#765b75",
    "#6d7773",
    "#8a6f4a",
    "#54766f",
    "#52667a",
    "#b28b52",
)

ALLOCATION_LABELS = {
    VOL_BALANCED_LABEL: VOLATILITY_BALANCED,
    VOL_TREND_LABEL: VOLATILITY_BALANCED_TREND,
    EQUAL_WEIGHT_LABEL: EQUAL_WEIGHT,
}

COMPARISON_SELECTOR_LABELS = {
    VOL_BALANCED_LABEL: "Balanced",
    VOL_TREND_LABEL: "Trend filter",
    VOL_FORECAST_LABEL: "Forecast filter",
    EQUAL_WEIGHT_LABEL: "Equal Weight",
    CASH_LABEL_SHORT: "Cash",
    SPY_REFERENCE_LABEL: "SPY reference",
    BUY_HOLD_LABEL: "Buy & Hold",
    CURRENT_MIX_LABEL: "Current Mix",
}


def comparison_selector_label(label):
    """Return compact chip text without changing the comparison identifier."""
    try:
        return COMPARISON_SELECTOR_LABELS[label]
    except KeyError as exc:
        raise ValueError(f"unknown comparison label: {label}") from exc


@dataclass(frozen=True)
class WorkbenchStudy:
    selected_etfs: tuple[str, ...]
    allocation_results: dict[str, AllocationResult]
    backtests: dict[str, BacktestResult]
    latest_targets: dict[str, pd.Series]
    forecast_result: ForecastAllocationResult | None
    forecast_manifest: dict | None
    forecast_freshness: dict | None
    current_weights: pd.Series | None
    start: pd.Timestamp | None
    end: pd.Timestamp | None

    @property
    def earliest_execution(self):
        return pd.Timestamp(
            self.backtests[VOL_BALANCED_LABEL].periods.index[0]
        )

    @property
    def latest_period_end(self):
        return pd.Timestamp(
            self.backtests[VOL_BALANCED_LABEL].periods["period_end_date"].iloc[-1]
        )


def bundle_fingerprint(path=DEFAULT_BUNDLE_PATH):
    """Return a content cache key for exactly the four public bundle files."""
    root = Path(path).resolve()
    return tuple(
        (name, file_sha256(root / name)) for name in PUBLIC_FILENAMES
    )


@st.cache_data(show_spinner=False)
def _load_cached_bundle(path_string, fingerprint):
    if tuple(name for name, _digest in fingerprint) != PUBLIC_FILENAMES:
        raise ValueError("bundle cache fingerprint must cover the four public files")
    return load_workbench_bundle(Path(path_string))


def load_cached_bundle(path=DEFAULT_BUNDLE_PATH):
    path = Path(path).resolve()
    return _load_cached_bundle(str(path), bundle_fingerprint(path))


def timesfm_bundle_fingerprint(path=DEFAULT_TIMESFM_BUNDLE_PATH):
    """Return a content cache key for the complete local TimesFM bundle."""
    root = Path(path).resolve()
    return tuple(
        (name, file_sha256(root / name)) for name in TIMESFM_PUBLIC_FILENAMES
    )


@st.cache_data(show_spinner=False)
def _load_cached_timesfm_bundle(
    path_string,
    fingerprint,
    workbench_path_string,
    workbench_fingerprint,
):
    if tuple(name for name, _digest in fingerprint) != TIMESFM_PUBLIC_FILENAMES:
        raise ValueError("TimesFM cache fingerprint must cover the three public files")
    if tuple(name for name, _digest in workbench_fingerprint) != PUBLIC_FILENAMES:
        raise ValueError("workbench cache fingerprint must cover the four public files")
    return load_timesfm_bundle(
        Path(path_string),
        workbench_path=Path(workbench_path_string),
    )


def load_cached_timesfm_bundle(
    path=DEFAULT_TIMESFM_BUNDLE_PATH,
    workbench_path=DEFAULT_BUNDLE_PATH,
):
    path = Path(path).resolve()
    workbench_path = Path(workbench_path).resolve()
    return _load_cached_timesfm_bundle(
        str(path),
        timesfm_bundle_fingerprint(path),
        str(workbench_path),
        bundle_fingerprint(workbench_path),
    )


def generate_workbench_allocations(bundle, selected_etfs):
    """Generate full-artifact targets once for a validated selected ETF set."""
    selected = tuple(selected_etfs)
    prices = bundle.adjusted_close.loc[:, selected]
    return {
        label: generate_allocation_targets(
            prices,
            selected,
            strategy,
            as_of=bundle.signal_as_of,
        )
        for label, strategy in ALLOCATION_LABELS.items()
    }


@st.cache_data(show_spinner=False)
def _cached_allocation_results(path_string, fingerprint, selected_etfs):
    bundle = _load_cached_bundle(path_string, fingerprint)
    return generate_workbench_allocations(bundle, selected_etfs)


@st.cache_data(show_spinner=False)
def _cached_forecast_allocation_result(
    workbench_path_string,
    workbench_fingerprint,
    timesfm_path_string,
    timesfm_fingerprint,
    selected_etfs,
):
    bundle = _load_cached_bundle(workbench_path_string, workbench_fingerprint)
    forecast_bundle = _load_cached_timesfm_bundle(
        timesfm_path_string,
        timesfm_fingerprint,
        workbench_path_string,
        workbench_fingerprint,
    )
    return generate_forecast_allocation_targets(
        bundle.adjusted_close.loc[:, selected_etfs],
        selected_etfs,
        forecast_bundle.signals,
        as_of=bundle.signal_as_of,
    )


def clear_workbench_caches():
    """Clear only workbench bundle/allocation caches for deterministic tests."""
    _load_cached_bundle.clear()
    _cached_allocation_results.clear()
    _load_cached_timesfm_bundle.clear()
    _cached_forecast_allocation_result.clear()


def available_comparisons(
    selected_etfs,
    current_weights_valid=False,
    forecast_available=False,
):
    """Return only comparisons compatible with the selected set and inputs."""
    selected = tuple(selected_etfs)
    if len(selected) == 1:
        # Volatility Balanced and Equal Weight are necessarily 100% in the lone
        # ETF, exactly duplicating Buy & Hold. Represent that path once.
        options = [VOL_TREND_LABEL]
        if forecast_available:
            options.append(VOL_FORECAST_LABEL)
        options.extend([BUY_HOLD_LABEL, CASH_LABEL_SHORT])
    else:
        options = [
            VOL_BALANCED_LABEL,
            VOL_TREND_LABEL,
        ]
        if forecast_available:
            options.append(VOL_FORECAST_LABEL)
        options.extend([EQUAL_WEIGHT_LABEL, CASH_LABEL_SHORT, SPY_REFERENCE_LABEL])
    if len(selected) == 1 and selected[0] != "SPY":
        options.append(SPY_REFERENCE_LABEL)
    if current_weights_valid:
        options.append(CURRENT_MIX_LABEL)
    return options


def default_comparisons(selected_etfs, options):
    """Keep the first chart legible while retaining every compatible option."""
    selected = tuple(selected_etfs)
    preferred = (
        [VOL_FORECAST_LABEL, BUY_HOLD_LABEL, CASH_LABEL_SHORT]
        if len(selected) == 1
        else [VOL_FORECAST_LABEL, EQUAL_WEIGHT_LABEL, SPY_REFERENCE_LABEL]
    )
    defaults = [label for label in preferred if label in options]
    if defaults:
        return defaults
    fallback = (
        [VOL_TREND_LABEL, BUY_HOLD_LABEL, CASH_LABEL_SHORT]
        if len(selected) == 1
        else [VOL_BALANCED_LABEL, EQUAL_WEIGHT_LABEL, SPY_REFERENCE_LABEL]
    )
    return [label for label in fallback if label in options]


def _constant_schedule(base, selected_etfs, weights):
    selected = tuple(selected_etfs)
    values = validate_portfolio_weights(weights, selected, name="target weights")
    frame = pd.DataFrame(
        np.tile(values.to_numpy(), (len(base.execution_dates), 1)),
        index=base.execution_dates,
        columns=[*selected, CASH_ASSET],
    )
    return TargetWeightSchedule(
        weights=frame,
        signal_dates=base.signal_dates,
        period_end_dates=base.period_end_dates,
    )


def holding_period_returns(bundle, schedule, selected_etfs):
    """Build engine-ready monthly ETF/cash returns from the validated bundle."""
    selected = tuple(selected_etfs)
    execution = schedule.execution_dates
    period_end = pd.DatetimeIndex(schedule.period_end_dates.to_numpy())
    start_prices = bundle.adjusted_close.reindex(execution).loc[:, selected]
    end_prices = bundle.adjusted_close.reindex(period_end).loc[:, selected]
    if start_prices.isna().any().any() or end_prices.isna().any().any():
        raise ValueError("selected ETFs lack a price at a required holding-period boundary")
    returns = end_prices.to_numpy(dtype=float) / start_prices.to_numpy(dtype=float) - 1.0
    asset_returns = pd.DataFrame(returns, index=execution, columns=selected)
    cash_returns = bundle.cash_returns(execution, period_end)
    cash_returns.index = execution
    return asset_returns, cash_returns


def build_workbench_study(
    bundle,
    selected_etfs,
    *,
    transaction_cost_bps=5.0,
    current_weights=None,
    start=None,
    end=None,
    allocation_results=None,
    forecast_result=None,
    forecast_manifest=None,
    forecast_freshness=None,
    comparison_labels=None,
):
    """Calculate every compatible series through the common PR1 engine."""
    if not isinstance(bundle, WorkbenchBundle):
        raise TypeError("bundle must be a validated WorkbenchBundle")
    selected = tuple(selected_etfs)
    if not 1 <= len(selected) <= 8:
        raise ValueError("select between one and eight ETFs")
    if len(set(selected)) != len(selected):
        raise ValueError("selected ETFs must be unique")

    if allocation_results is None:
        allocation_results = generate_workbench_allocations(bundle, selected)
    elif set(allocation_results) != set(ALLOCATION_LABELS) or any(
        result.selected_etfs != selected for result in allocation_results.values()
    ):
        raise ValueError("cached allocation results do not match the selected ETFs")
    # The validated analytical cash history begins in 2000. Restrict every
    # comparison to that common observable horizon before constructing returns.
    cash_history_start = pd.Timestamp(bundle.cash_index.index.min())
    schedules = {
        label: result.schedule.select(start=cash_history_start)
        for label, result in allocation_results.items()
    }
    align_to_forecast = forecast_result is not None and (
        comparison_labels is None or VOL_FORECAST_LABEL in set(comparison_labels)
    )
    if forecast_result is not None:
        if (
            not isinstance(forecast_result, ForecastAllocationResult)
            or forecast_result.selected_etfs != selected
        ):
            raise ValueError("forecast allocation does not match the selected ETFs")
        forecast_schedule = forecast_result.schedule.select(start=cash_history_start)
        if align_to_forecast:
            schedules = {
                label: schedule.select(
                    start=forecast_schedule.execution_dates[0],
                    end=forecast_schedule.period_end_dates.iloc[-1],
                )
                for label, schedule in schedules.items()
            }
            expected_dates = forecast_schedule.execution_dates
            if any(
                not schedule.execution_dates.equals(expected_dates)
                for schedule in schedules.values()
            ):
                raise ValueError(
                    "forecast and workbench histories do not share exact periods"
                )
        schedules[VOL_FORECAST_LABEL] = forecast_schedule
    base_schedule = schedules[VOL_BALANCED_LABEL]
    asset_returns, cash_returns = holding_period_returns(
        bundle, base_schedule, selected
    )

    latest_targets = {
        label: result.latest_target.copy()
        for label, result in allocation_results.items()
    }
    if forecast_result is not None:
        latest_targets[VOL_FORECAST_LABEL] = forecast_result.latest_target.copy()

    cash_target = pd.Series(0.0, index=[*selected, CASH_ASSET], dtype=float)
    cash_target.loc[CASH_ASSET] = 1.0
    schedules[CASH_LABEL_SHORT] = _constant_schedule(
        base_schedule, selected, cash_target
    )
    latest_targets[CASH_LABEL_SHORT] = cash_target

    if len(selected) == 1:
        buy_target = pd.Series(
            [1.0, 0.0], index=[selected[0], CASH_ASSET], dtype=float
        )
        schedules[BUY_HOLD_LABEL] = _constant_schedule(
            base_schedule, selected, buy_target
        )
        latest_targets[BUY_HOLD_LABEL] = buy_target

    validated_current = None
    if current_weights is not None:
        validated_current = validate_portfolio_weights(
            current_weights, selected, name="current weights"
        )
        schedules[CURRENT_MIX_LABEL] = _constant_schedule(
            base_schedule, selected, validated_current
        )
        latest_targets[CURRENT_MIX_LABEL] = validated_current.copy()

    if "SPY" in selected and not (len(selected) == 1 and selected[0] == "SPY"):
        spy_target = pd.Series(
            0.0,
            index=[*selected, CASH_ASSET],
            dtype=float,
        )
        spy_target.loc["SPY"] = 1.0
        schedules[SPY_REFERENCE_LABEL] = _constant_schedule(
            base_schedule, selected, spy_target
        )

    backtests = {}
    for label, schedule in schedules.items():
        if schedule.execution_dates.equals(base_schedule.execution_dates):
            label_asset_returns, label_cash_returns = asset_returns, cash_returns
        else:
            label_asset_returns, label_cash_returns = holding_period_returns(
                bundle, schedule, selected
            )
        # A non-selected forecast retains its complete replay for the research
        # detail view; it cannot narrow the displayed standard-strategy range.
        label_start = (
            None
            if label == VOL_FORECAST_LABEL and not align_to_forecast
            else start
        )
        label_end = (
            None
            if label == VOL_FORECAST_LABEL and not align_to_forecast
            else end
        )
        backtests[label] = run_backtest(
            schedule,
            label_asset_returns,
            label_cash_returns,
            transaction_cost_bps=transaction_cost_bps,
            start=label_start,
            end=label_end,
        )
    if "SPY" not in selected:
        external_target = pd.Series(
            {"SPY": 1.0, CASH_ASSET: 0.0}, dtype=float
        )
        external_schedule = _constant_schedule(
            base_schedule,
            ("SPY",),
            external_target,
        )
        spy_returns, spy_cash = holding_period_returns(
            bundle, external_schedule, ("SPY",)
        )
        backtests[SPY_REFERENCE_LABEL] = run_backtest(
            external_schedule,
            spy_returns,
            spy_cash,
            transaction_cost_bps=transaction_cost_bps,
            start=start,
            end=end,
        )
    return WorkbenchStudy(
        selected_etfs=selected,
        allocation_results=allocation_results,
        backtests=backtests,
        latest_targets=latest_targets,
        forecast_result=forecast_result,
        forecast_manifest=forecast_manifest,
        forecast_freshness=forecast_freshness,
        current_weights=validated_current,
        start=None if start is None else pd.Timestamp(start),
        end=None if end is None else pd.Timestamp(end),
    )


def historical_download(study, labels):
    """Return the exact displayed historical results in tidy CSV-ready form."""
    frames = []
    for label in labels:
        periods = study.backtests[label].periods.reset_index().copy()
        periods.insert(0, "series", label)
        frames.append(periods)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def allocation_history_download(study, label):
    """Return exact allocations with all three dates named unambiguously."""
    result = study.backtests[label]
    frame = result.target_weights.reset_index().copy()
    frame = frame.rename(columns={"execution_date": "rebalance_date"})
    frame.insert(0, "strategy", label)
    frame.insert(2, "signal_date", result.periods["signal_date"].to_numpy())
    frame.insert(
        3,
        "holding_period_end",
        result.periods["period_end_date"].to_numpy(),
    )
    frame["turnover"] = result.periods["turnover"].to_numpy(dtype=float)
    frame["estimated_cost_rate"] = result.periods["cost_rate"].to_numpy(dtype=float)
    frame["transaction_cost_value"] = result.periods[
        "transaction_cost"
    ].to_numpy(dtype=float)
    return frame


def allocation_chart_data(study, label):
    """Return plotted allocations indexed by their actual rebalance date."""
    result = study.backtests[label]
    frame = result.target_weights.copy()
    frame.index = pd.DatetimeIndex(frame.index, name="rebalance_date")
    return frame


def standard_target_freshness(bundle, allocation, as_of=None):
    """Validate the intended next-close window for a standard tactical target."""
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC")
    as_of_stamp = pd.Timestamp(as_of)
    if as_of_stamp.tz is not None:
        as_of_stamp = as_of_stamp.tz_convert("UTC").tz_localize(None)
    signal_date = pd.Timestamp(allocation.latest_signal_date).normalize()
    calendar = xcals.get_calendar(
        "XNYS",
        start=signal_date - pd.Timedelta(days=7),
        end=signal_date + pd.Timedelta(days=70),
    )
    execution_date = pd.Timestamp(
        calendar.date_to_session(signal_date + pd.Timedelta(days=1), direction="next")
    ).tz_localize(None)
    period_end_month = signal_date.to_period("M") + 2
    period_end_date = pd.Timestamp(
        calendar.date_to_session(period_end_month.start_time, direction="next")
    ).tz_localize(None)
    execution_close = pd.Timestamp(calendar.session_close(execution_date))
    if execution_close.tz is not None:
        execution_close = execution_close.tz_convert("UTC").tz_localize(None)
    period_end_close = pd.Timestamp(calendar.session_close(period_end_date))
    if period_end_close.tz is not None:
        period_end_close = period_end_close.tz_convert("UTC").tz_localize(None)
    generated_at = pd.Timestamp(bundle.manifest["generated_at_utc"])
    if generated_at.tz is not None:
        generated_at = generated_at.tz_convert("UTC").tz_localize(None)

    if generated_at > as_of_stamp:
        status, reason = "disabled", "target artifact was generated in the future"
    elif generated_at > execution_close:
        status, reason = (
            "disabled",
            "target artifact was generated after its intended execution close",
        )
    elif as_of_stamp > period_end_close:
        status, reason = "disabled", "latest target holding period has expired"
    elif as_of_stamp < execution_close:
        status, reason = (
            "scheduled",
            "latest target awaits its intended execution close",
        )
    else:
        status, reason = "current", None
    return {
        "latest_target_status": status,
        "reason": reason,
        "latest_signal_date": str(signal_date.date()),
        "latest_execution_date": str(execution_date.date()),
        "latest_execution_cutoff_utc": execution_close.isoformat() + "Z",
        "latest_period_end_date": str(period_end_date.date()),
        "latest_period_end_cutoff_utc": period_end_close.isoformat() + "Z",
        "bundle_refreshed_at_utc": generated_at.isoformat() + "Z",
    }


def target_provenance(bundle, study, label):
    """Return authoritative timing and artifact provenance for a latest target."""
    extra = {
        "model_generation_id": None,
        "model_generated_at_utc": None,
        "execution_cutoff_utc": None,
        "bundle_refreshed_at_utc": bundle.manifest["generated_at_utc"],
        "workbench_bundle_refreshed_at_utc": bundle.manifest["generated_at_utc"],
        "forecast_bundle_refreshed_at_utc": None,
    }
    if label == CASH_LABEL_SHORT:
        signal_as_of = "not_applicable_no_tactical_signal"
        execution_status = "constant_target_effective_for_analytical_ticket"
        policy_version = "analytical-cash-comparison-v1"
    elif label == BUY_HOLD_LABEL:
        signal_as_of = "not_applicable_no_tactical_signal"
        execution_status = "constant_target_effective_for_analytical_ticket"
        policy_version = "single-etf-buy-hold-v1"
    elif label == CURRENT_MIX_LABEL:
        signal_as_of = "not_applicable_user_entered_mix"
        execution_status = "constant_target_effective_for_analytical_ticket"
        policy_version = "user-entered-monthly-mix-v1"
    elif label == EQUAL_WEIGHT_LABEL:
        signal_as_of = "not_applicable_no_tactical_signal"
        execution_status = "constant_target_effective_for_analytical_ticket"
        policy_version = "equal-weight-monthly-v1"
    elif label == VOL_FORECAST_LABEL:
        if study.forecast_result is None or study.forecast_manifest is None:
            raise ValueError("forecast target provenance is unavailable")
        allocation = study.forecast_result
        signal_as_of = str(allocation.latest_signal_date.date())
        execution_status = (
            study.forecast_freshness or {}
        ).get("latest_target_status", "disabled")
        policy_version = allocation.forecast_policy_version
        generation_id = allocation.latest_model_generation_id
        generation = next(
            (
                item
                for item in study.forecast_manifest["model_generations"]
                if item["id"] == generation_id
            ),
            None,
        )
        if generation is None:
            raise ValueError("forecast target references unknown model provenance")
        extra = {
            "model_generation_id": generation_id,
            "model_generated_at_utc": generation["generated_at_utc"],
            "execution_cutoff_utc": (study.forecast_freshness or {}).get(
                "latest_execution_cutoff_utc"
            ),
            "bundle_refreshed_at_utc": study.forecast_manifest["bundle_refresh"][
                "generated_at_utc"
            ],
            "workbench_bundle_refreshed_at_utc": bundle.manifest[
                "generated_at_utc"
            ],
            "forecast_bundle_refreshed_at_utc": study.forecast_manifest[
                "bundle_refresh"
            ]["generated_at_utc"],
        }
    else:
        allocation = study.allocation_results[label]
        standard_freshness = standard_target_freshness(
            bundle, allocation, as_of=_test_as_of()
        )
        signal_as_of = standard_freshness["latest_signal_date"]
        execution_status = standard_freshness["latest_target_status"]
        policy_version = allocation.policy.version
        extra["execution_cutoff_utc"] = standard_freshness[
            "latest_execution_cutoff_utc"
        ]
    result = {
        "signal_as_of": signal_as_of,
        "execution_status": execution_status,
        "artifact_generated_at_utc": (
            study.forecast_manifest["bundle_refresh"]["generated_at_utc"]
            if label == VOL_FORECAST_LABEL
            else bundle.manifest["generated_at_utc"]
        ),
        "price_data_as_of": bundle.manifest["price_data_as_of"],
        "policy_version": policy_version,
        "displayed_history_through": str(
            pd.Timestamp(
                study.backtests[label].periods["period_end_date"].iloc[-1]
            ).date()
        ),
        **extra,
    }
    if label == VOL_FORECAST_LABEL:
        result.update(
            {
                "model_checkpoint": study.forecast_manifest["model_policy"][
                    "checkpoint_id"
                ],
                "model_checkpoint_revision": study.forecast_manifest[
                    "model_policy"
                ]["checkpoint_revision"],
                "evaluation_classification": study.forecast_manifest[
                    "evaluation_classification"
                ],
                "pretraining_overlap": study.forecast_manifest["pretraining_overlap"],
            }
        )
    return result


def target_provenance_summary(provenance):
    """Translate machine-readable target provenance into compact UI language."""
    if provenance["signal_as_of"] == "not_applicable_no_tactical_signal":
        signal_text = "No tactical signal"
    elif provenance["signal_as_of"] == "not_applicable_user_entered_mix":
        signal_text = "User-entered mix; no tactical signal"
    else:
        signal_text = f"Signal as of {provenance['signal_as_of']}"

    if provenance["execution_status"] == "current":
        execution_text = "Current target; intended execution close has passed"
    elif provenance["execution_status"] == "scheduled":
        execution_text = "Target scheduled; intended execution close has not arrived"
    elif provenance["execution_status"] == "disabled":
        execution_text = "Historical research only; latest target is not transferable"
    elif provenance["execution_status"] == "pending_next_trading_close":
        execution_text = "Awaiting the next rebalance trading close"
    elif (
        provenance["execution_status"]
        == "constant_target_effective_for_analytical_ticket"
    ):
        execution_text = "Constant target; effective for the analytical ticket"
    else:
        execution_text = f"Execution close {provenance['execution_status']}"
    return f"{signal_text} · {execution_text}"


def latest_target_download(bundle, study, label):
    target = study.latest_targets[label]
    provenance = target_provenance(bundle, study, label)
    # The latest target is independent of the user's historical chart range.
    # Keep range-specific context in the UI/transfer payload, not in this file.
    target_provenance_fields = {
        key: value
        for key, value in provenance.items()
        if key not in {"displayed_history_through", "artifact_generated_at_utc"}
    }
    return pd.DataFrame(
        {
            "strategy": label,
            **target_provenance_fields,
            "asset": target.index,
            "target_weight": target.to_numpy(dtype=float),
            "asset_type": [
                "analytical_cash" if asset == CASH_ASSET else "tradeable_etf"
                for asset in target.index
            ],
        }
    )


def why_this_weight(bundle, study, label):
    """Return the latest strategy explanation, including explicit cash."""
    registry = bundle.instruments.set_index("ticker")
    target = study.latest_targets[label]
    forecast_fields = {
        "median_forecast_return": np.nan,
        "cash_hurdle": np.nan,
        "forecast_status": "Not used",
        "q10_period_end_price": np.nan,
        "q50_period_end_price": np.nan,
        "q90_period_end_price": np.nan,
    }
    if label == VOL_FORECAST_LABEL:
        if study.forecast_result is None:
            raise ValueError("forecast explanation is unavailable")
        diagnostics = (
            study.forecast_result.latest_diagnostics.reset_index().set_index("ticker")
        )
        rows = []
        for ticker in study.selected_etfs:
            row = diagnostics.loc[ticker]
            rows.append(
                {
                    "asset": ticker,
                    "role": registry.loc[ticker, "role"],
                    "trend": "Not used",
                    "trailing_volatility": row["trailing_volatility"],
                    "raw_weight": row["raw_risk_balanced_weight"],
                    "filtered_raw_weight": np.nan,
                    "final_weight": row["final_target_weight"],
                    "change_vs_uncapped_inverse_vol": (
                        row["final_target_weight"]
                        - row["raw_risk_balanced_weight"]
                    ),
                    "median_forecast_return": row["median_forecast_return"],
                    "cash_hurdle": row["cash_hurdle"],
                    "forecast_status": row["forecast_status"],
                    "q10_period_end_price": row["q10_period_end_price"],
                    "q50_period_end_price": row["q50_period_end_price"],
                    "q90_period_end_price": row["q90_period_end_price"],
                    "reason": row["reason"],
                }
            )
        cash_weight = float(target.loc[CASH_ASSET])
        eligible = sum(row["forecast_status"] == "Eligible" for row in rows)
        if eligible == 0:
            reason = (
                "No selected ETF's median forecast cleared cash, so the research "
                "target is 100% analytical cash."
            )
        elif cash_weight > WEIGHT_TOLERANCE:
            reason = (
                "Cash holds the amount displaced by ETFs held in cash and position caps."
            )
        else:
            reason = "All capital is allocated among ETFs whose forecasts cleared cash."
    elif label in study.allocation_results:
        allocation = study.allocation_results[label]
        diagnostics = allocation.latest_diagnostics.reset_index().set_index("ticker")
        rows = []
        for ticker in study.selected_etfs:
            row = diagnostics.loc[ticker]
            equal_weight = label == EQUAL_WEIGHT_LABEL
            trend_used = label == VOL_TREND_LABEL
            raw = (
                np.nan
                if equal_weight
                else float(row["unfiltered_inverse_vol_weight"])
            )
            final = float(row["final_target_weight"])
            if equal_weight:
                reason = "Equal share of the selected ETFs; volatility and trend are not used."
            elif label == VOL_BALANCED_LABEL:
                reason = (
                    "Inverse-volatility weight, limited by the position cap."
                    if "cap" in str(row["eligibility_reason"]).lower()
                    else "Inverse-volatility weight; trend is not used."
                )
            else:
                reason = str(row["eligibility_reason"])
            rows.append(
                {
                    "asset": ticker,
                    "role": registry.loc[ticker, "role"],
                    "trend": row["trend_status"] if trend_used else "Not used",
                    "trailing_volatility": row["trailing_volatility"],
                    "raw_weight": raw,
                    "filtered_raw_weight": (
                        np.nan if equal_weight else row["filtered_raw_weight"]
                    ),
                    "final_weight": final,
                    "change_vs_uncapped_inverse_vol": (
                        np.nan if equal_weight else final - raw
                    ),
                    **forecast_fields,
                    "reason": reason,
                }
            )
        cash_weight = float(target.loc[CASH_ASSET])
        if label == VOL_TREND_LABEL and np.isclose(cash_weight, 1.0):
            reason = (
                "All selected ETFs failed the trend rule, so the proposal is "
                "100% analytical cash."
            )
        elif label == VOL_TREND_LABEL and cash_weight > WEIGHT_TOLERANCE:
            reason = (
                "Analytical cash holds the amount not allocated after trend exclusions "
                "and position caps."
            )
        elif label == VOL_TREND_LABEL:
            reason = "Every dollar is allocated to ETFs that passed the trend rule."
        elif label == VOL_BALANCED_LABEL and cash_weight > WEIGHT_TOLERANCE:
            reason = "Analytical cash holds any amount the eligible capped ETFs cannot take."
        elif label == VOL_BALANCED_LABEL:
            reason = "No residual cash; the eligible ETFs use the full allocation."
        else:
            reason = "Equal Weight is fully allocated across the selected ETFs."
    else:
        descriptions = {
            CASH_LABEL_SHORT: "Analytical cash comparison",
            BUY_HOLD_LABEL: "Single ETF held without a tactical cash signal",
            CURRENT_MIX_LABEL: "User-entered mix, rebalanced monthly without normalization",
        }
        rows = [
            {
                "asset": ticker,
                "role": registry.loc[ticker, "role"],
                "trend": "Not used",
                "trailing_volatility": np.nan,
                "raw_weight": np.nan,
                "filtered_raw_weight": np.nan,
                "final_weight": target.loc[ticker],
                "change_vs_uncapped_inverse_vol": np.nan,
                **forecast_fields,
                "reason": descriptions[label],
            }
            for ticker in study.selected_etfs
        ]
        if label == CASH_LABEL_SHORT:
            reason = "This comparison holds 100% analytical cash."
        elif label == BUY_HOLD_LABEL:
            reason = "Buy & Hold stays fully invested in the selected ETF, so cash is 0%."
        else:
            reason = "Cash is kept at exactly the percentage entered in Current Mix."
    rows.append(
        {
            "asset": CASH_LABEL,
            "role": "Analytical, non-investable balance",
            "trend": "Not applicable",
            "trailing_volatility": np.nan,
            "raw_weight": np.nan,
            "filtered_raw_weight": np.nan,
            "final_weight": target.loc[CASH_ASSET],
            "change_vs_uncapped_inverse_vol": np.nan,
            **forecast_fields,
            "reason": reason,
        }
    )
    return pd.DataFrame(rows)


def _format_metrics(study, labels):
    rows = []
    for label in labels:
        metrics = study.backtests[label].metrics
        rows.append(
            {
                "Series": label,
                "Annualized return ↑": metrics["annualized_return"],
                "Annualized volatility ↓": metrics["annualized_volatility"],
                "Sharpe above cash ↑": metrics["excess_return_sharpe"],
                "Maximum drawdown ↑": metrics["max_drawdown"],
                "Return / drawdown ↑": metrics["calmar_ratio"],
                "Worst month ↑": metrics["worst_month"],
                "Annualized one-way turnover ↓": metrics["annualized_turnover"],
                "Annualized cost drag ↓": metrics["transaction_cost_drag"],
            }
        )
    return pd.DataFrame(rows).replace({None: np.nan}).set_index("Series")


def forecast_check_summary(metrics):
    """Return a compact, display-ready view of validated offline forecast checks."""
    overall = metrics.loc[metrics["scope"] == "overall"].copy()
    labels = {
        "last_value": "Last value (point baseline)",
        "timesfm3_univariate": "TimesFM-3 — each ETF separately",
        "timesfm3_multivariate": "TimesFM-3 — all 14 ETFs together",
    }
    overall["Model"] = overall["model_mode"].map(labels)
    if overall["Model"].isna().any() or set(overall["model_mode"]) != set(labels):
        raise ValueError("forecast checks do not contain the expected model modes")
    return overall[
        [
            "Model",
            "observations",
            "holding_return_mae",
            "period_end_price_mase",
            "directional_accuracy",
            "mean_pinball_loss",
            "interval_80_coverage",
        ]
    ].rename(
        columns={
            "observations": "Completed ETF-months",
            "holding_return_mae": "Average return difference ↓",
            "period_end_price_mase": "Scaled final-price difference ↓",
            "directional_accuracy": "Correct up-or-down calls ↑",
            "mean_pinball_loss": "Probability forecast difference ↓",
            "interval_80_coverage": "Final price inside 10%–90% range",
        }
    )


def forecast_checks_for_display(checks):
    """Format a numeric forecast-check summary with explicit unavailable marks."""
    formats = {
        "Model": None,
        "Completed ETF-months": "{:,.0f}",
        "Average return difference ↓": "{:.2%}",
        "Scaled final-price difference ↓": "{:.2f}",
        "Correct up-or-down calls ↑": "{:.2%}",
        "Probability forecast difference ↓": "{:.4f}",
        "Final price inside 10%–90% range": "{:.2%}",
    }
    if list(checks.columns) != list(formats):
        raise ValueError("forecast-check display received unexpected columns")
    display = checks.copy()
    for column, template in formats.items():
        if template is None:
            display[column] = display[column].astype(str)
        else:
            display[column] = display[column].map(
                lambda value, pattern=template: (
                    "—" if pd.isna(value) else pattern.format(value)
                )
            )
    return display


def forecast_skill_snapshot(metrics):
    """Summarize the three offline forecast experiments in user-facing units."""
    required = {"last_value", "timesfm3_univariate", "timesfm3_multivariate"}
    overall = metrics.loc[metrics["scope"] == "overall"].set_index("model_mode")
    if set(overall.index) != required or not overall.index.is_unique:
        raise ValueError("forecast metrics do not contain one overall row per model mode")
    baseline = overall.loc["last_value"]
    univariate = overall.loc["timesfm3_univariate"]
    multivariate = overall.loc["timesfm3_multivariate"]
    baseline_mae = float(baseline["holding_return_mae"])
    univariate_mae = float(univariate["holding_return_mae"])
    multivariate_mae = float(multivariate["holding_return_mae"])
    if min(baseline_mae, univariate_mae, multivariate_mae) <= 0.0:
        raise ValueError("forecast MAE values must be positive")
    return {
        "observations": int(multivariate["observations"]),
        "directional_accuracy": float(multivariate["directional_accuracy"]),
        "univariate_directional_accuracy": float(
            univariate["directional_accuracy"]
        ),
        "directional_lift_vs_univariate": float(
            multivariate["directional_accuracy"]
            - univariate["directional_accuracy"]
        ),
        "return_mae": multivariate_mae,
        "univariate_return_mae": univariate_mae,
        "last_value_return_mae": baseline_mae,
        "mae_improvement_vs_univariate": 1.0 - multivariate_mae / univariate_mae,
        "mae_improvement_vs_last_value": 1.0 - multivariate_mae / baseline_mae,
        "interval_80_coverage": float(multivariate["interval_80_coverage"]),
    }


def forecast_portfolio_comparison(study):
    """Compare the forecast filter with the simplest investable reference."""
    if VOL_FORECAST_LABEL not in study.backtests:
        raise ValueError("forecast backtest is unavailable")
    benchmark = BUY_HOLD_LABEL if len(study.selected_etfs) == 1 else EQUAL_WEIGHT_LABEL
    if benchmark not in study.backtests:
        raise ValueError(f"required comparison is unavailable: {benchmark}")
    forecast_periods = study.backtests[VOL_FORECAST_LABEL].periods.index
    benchmark_periods = study.backtests[benchmark].periods.index
    if not forecast_periods.equals(benchmark_periods):
        raise ValueError("forecast and benchmark must use identical holding periods")
    forecast = study.backtests[VOL_FORECAST_LABEL].metrics
    reference = study.backtests[benchmark].metrics
    return {
        "benchmark": benchmark,
        "start": pd.Timestamp(forecast_periods.min()),
        "end": pd.Timestamp(
            study.backtests[VOL_FORECAST_LABEL]
            .periods.loc[forecast_periods.max(), "period_end_date"]
        ),
        "forecast_return": float(forecast["annualized_return"]),
        "benchmark_return": float(reference["annualized_return"]),
        "return_delta": float(
            forecast["annualized_return"] - reference["annualized_return"]
        ),
        "forecast_sharpe": float(forecast["excess_return_sharpe"]),
        "benchmark_sharpe": float(reference["excess_return_sharpe"]),
        "sharpe_delta": float(
            forecast["excess_return_sharpe"] - reference["excess_return_sharpe"]
        ),
        "forecast_drawdown": float(forecast["max_drawdown"]),
        "benchmark_drawdown": float(reference["max_drawdown"]),
        "drawdown_improvement": float(
            forecast["max_drawdown"] - reference["max_drawdown"]
        ),
        "forecast_turnover": float(forecast["annualized_turnover"]),
        "benchmark_turnover": float(reference["annualized_turnover"]),
        "turnover_delta": float(
            forecast["annualized_turnover"] - reference["annualized_turnover"]
        ),
    }


def forecast_portfolio_table(comparison):
    """Return a small side-by-side portfolio table for the TimesFM section."""
    return pd.DataFrame(
        [
            {
                "Portfolio": "Forecast",
                "Annualized return": comparison["forecast_return"],
                "Sharpe above cash": comparison["forecast_sharpe"],
                "Maximum drawdown": comparison["forecast_drawdown"],
                "Annualized turnover": comparison["forecast_turnover"],
            },
            {
                "Portfolio": comparison["benchmark"],
                "Annualized return": comparison["benchmark_return"],
                "Sharpe above cash": comparison["benchmark_sharpe"],
                "Maximum drawdown": comparison["benchmark_drawdown"],
                "Annualized turnover": comparison["benchmark_turnover"],
            },
        ]
    )


def forecast_replay_frame(signals, ticker, *, start=None, end=None):
    """Return realized multivariate decisions for an interactive ETF audit."""
    required = {
        "model_mode",
        "ticker",
        "signal_date",
        "period_end_date",
        "forecast_holding_return",
        "cash_hurdle",
        "forecast_edge",
        "actual_holding_return",
        "q10_period_end_price",
        "q90_period_end_price",
        "actual_period_end_price",
        "evaluation_status",
    }
    missing = sorted(required - set(signals.columns))
    if missing:
        raise ValueError(f"forecast replay is missing columns: {missing}")
    rows = signals.loc[
        (signals["model_mode"] == MULTIVARIATE_MODE)
        & (signals["ticker"] == ticker)
        & (signals["evaluation_status"] == "realized"),
        list(required),
    ].copy()
    if rows.empty:
        raise ValueError(f"no realized multivariate forecasts are available for {ticker}")
    rows["signal_date"] = pd.to_datetime(rows["signal_date"], errors="raise")
    rows["period_end_date"] = pd.to_datetime(rows["period_end_date"], errors="raise")
    if start is not None:
        rows = rows.loc[rows["period_end_date"] >= pd.Timestamp(start)]
    if end is not None:
        rows = rows.loc[rows["period_end_date"] <= pd.Timestamp(end)]
    if rows.empty:
        raise ValueError(f"no realized {ticker} forecasts fall in the selected history")
    numeric = [
        "forecast_holding_return",
        "cash_hurdle",
        "forecast_edge",
        "actual_holding_return",
        "q10_period_end_price",
        "q90_period_end_price",
        "actual_period_end_price",
    ]
    rows[numeric] = rows[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(rows[numeric].to_numpy(dtype=float)).all():
        raise ValueError("forecast replay contains non-finite values")
    rows["actual_edge"] = rows["actual_holding_return"] - rows["cash_hurdle"]
    rows["cash_gate_correct"] = (rows["forecast_edge"] > 0.0) == (
        rows["actual_edge"] > 0.0
    )
    rows["interval_hit"] = (
        (rows["actual_period_end_price"] >= rows["q10_period_end_price"])
        & (rows["actual_period_end_price"] <= rows["q90_period_end_price"])
    )
    return rows.sort_values("period_end_date").reset_index(drop=True)


def forecast_replay_summary(replay):
    """Summarize exactly the realized rows displayed by the replay explorer."""
    if replay.empty:
        raise ValueError("forecast replay cannot be empty")
    return {
        "observations": int(len(replay)),
        "cash_gate_accuracy": float(replay["cash_gate_correct"].mean()),
        "return_mae": float(
            (replay["forecast_holding_return"] - replay["actual_holding_return"])
            .abs()
            .mean()
        ),
        "interval_80_coverage": float(replay["interval_hit"].mean()),
    }


def _line_chart(study, labels, column, title, percent=False):
    figure = go.Figure()
    for label in labels:
        periods = study.backtests[label].periods
        y = periods[column]
        figure.add_trace(
            go.Scatter(
                x=periods["period_end_date"],
                y=y,
                mode="lines",
                name=label,
                line=dict(
                    color=CHART_COLORS.get(label, "#6d7773"),
                    width=2.5 if label in (VOL_FORECAST_LABEL, VOL_TREND_LABEL) else 1.8,
                ),
                hovertemplate=(
                    "%{x|%b %Y}<br>%{y:.1%}<extra>%{fullData.name}</extra>"
                    if percent
                    else "%{x|%b %Y}<br>%{y:,.2f}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(
        height=390 if not percent else 280,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#17211f"),
        title=dict(text=title, x=0, xanchor="left"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(
            tickformat=".0%" if percent else ",.2f",
            gridcolor="#e1e5e2",
            zeroline=False,
        ),
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch")


def _forecast_replay_chart(replay, ticker):
    """Plot the forecast, result, and cash return for each completed month."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=replay["period_end_date"],
            y=replay["actual_holding_return"],
            mode="lines",
            name="ETF return",
            line=dict(color="#17211f", width=2),
            hovertemplate="%{x|%b %Y}<br>%{y:+.2%}<extra>Realized</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=replay["period_end_date"],
            y=replay["forecast_holding_return"],
            mode="lines",
            name="TimesFM-3 forecast return",
            line=dict(color="#0c6148", width=2),
            hovertemplate="%{x|%b %Y}<br>%{y:+.2%}<extra>Forecast</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=replay["period_end_date"],
            y=replay["cash_hurdle"],
            mode="lines",
            name="Cash return",
            line=dict(color="#b6782f", width=1.5, dash="dot"),
            hovertemplate="%{x|%b %Y}<br>%{y:.2%}<extra>Cash return</extra>",
        )
    )
    figure.add_hline(y=0.0, line_width=1, line_color="rgba(23,33,31,0.25)")
    figure.update_layout(
        title=f"{ticker}: one-month forecast and result",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#17211f"),
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(tickformat="+.0%", gridcolor="#e1e5e2", zeroline=False),
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch", key=f"wb_forecast_replay_{ticker}")


def _allocation_chart(study, label, *, key):
    allocations = allocation_chart_data(study, label)
    figure = go.Figure()
    for position, asset in enumerate(allocations.columns):
        display = CASH_LABEL if asset == CASH_ASSET else asset
        figure.add_trace(
            go.Scatter(
                x=allocations.index,
                y=allocations[asset],
                mode="lines",
                name=display,
                stackgroup="allocation",
                line=dict(
                    color=ALLOCATION_COLORS[position % len(ALLOCATION_COLORS)],
                    width=0.8,
                ),
                hovertemplate="%{x|%b %Y}<br>%{y:.1%}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#17211f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(
            tickformat=".0%", range=[0, 1], gridcolor="#e1e5e2", zeroline=False
        ),
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch", key=key)


def _test_as_of():
    """Optional deterministic local/AppTest hook; unset in production."""
    value = os.environ.get("ETF_WORKBENCH_TEST_AS_OF")
    return None if not value else pd.Timestamp(value)


def _current_editor(selected):
    """Render and reconcile optional current weights in percentage units."""
    enabled = st.checkbox(
        "Enter current weights",
        key="wb_current_enabled",
        help=(
            "Optional. These exact weights are never normalized and affect only "
            "Current Mix and your Portfolio Lab ticket."
        ),
    )
    state_key = "wb_current_weight_values"
    prior_key = "wb_current_previous_selection"
    if state_key not in st.session_state:
        initial = pd.Series(0.0, index=[*selected, CASH_ASSET])
        initial.loc[CASH_ASSET] = 1.0
        st.session_state[state_key] = initial.to_dict()
        st.session_state[prior_key] = tuple(selected)
    prior = tuple(st.session_state.get(prior_key, selected))
    changed = set(prior).symmetric_difference(selected)
    for ticker in changed:
        # Streamlit widget keys otherwise retain a removed value when the ETF is
        # later re-added, overriding the reconciled zero-value default.
        st.session_state.pop(f"wb_current_pct_{ticker}", None)
    if changed:
        # Cash itself is not a changed selection, but its reconciled value moves
        # whenever an ETF is removed. Clear the old widget value as well.
        st.session_state.pop(f"wb_current_pct_{CASH_ASSET}", None)
    reconciled = reconcile_current_weights(
        prior, selected, st.session_state[state_key]
    )
    st.session_state[state_key] = reconciled.to_dict()
    st.session_state[prior_key] = tuple(selected)
    if not enabled:
        return None, None

    st.caption(
        "Enter the portfolio you hold now, including cash. The values must total "
        "100% and are never normalized."
    )
    columns = st.columns(min(4, len(selected) + 1))
    edited = {}
    for position, asset in enumerate([*selected, CASH_ASSET]):
        label = CASH_LABEL if asset == CASH_ASSET else asset
        edited[asset] = columns[position % len(columns)].number_input(
            label,
            min_value=0.0,
            max_value=100.0,
            value=float(reconciled.loc[asset] * 100.0),
            step=0.5,
            key=f"wb_current_pct_{asset}",
        ) / 100.0
    weights = pd.Series(edited, dtype=float)
    st.session_state[state_key] = weights.to_dict()
    status, message = current_weight_status(weights)
    if status == "ready":
        st.success(message)
    elif status == "under":
        st.info(message)
    else:
        st.warning(message)
    try:
        valid = validate_portfolio_weights(weights, selected)
    except ValueError as exc:
        return weights, str(exc)
    return valid, None


def current_weight_status(weights):
    """Give an exact, actionable status for non-normalized current weights."""
    total = float(pd.Series(weights, dtype=float).sum())
    difference_points = (1.0 - total) * 100.0
    if np.isclose(total, 1.0, rtol=0.0, atol=WEIGHT_TOLERANCE):
        return "ready", "Current total: 100.00%. Ready to compare and build a ticket."
    if total < 1.0:
        return (
            "under",
            f"Current total: {total:.2%}. Add {difference_points:.2f} percentage "
            "points to reach 100%.",
        )
    return (
        "over",
        f"Current total: {total:.2%}. Remove {abs(difference_points):.2f} "
        "percentage points to reach 100%.",
    )


def proposed_target_summary(target):
    """Summarize a target in plain language while preserving exact totals."""
    values = pd.Series(target, dtype=float)
    cash_weight = float(values.get(CASH_ASSET, 0.0))
    etf_weight = float(values.drop(labels=[CASH_ASSET], errors="ignore").sum())
    total = float(values.sum())
    funded = values.drop(labels=[CASH_ASSET], errors="ignore")
    funded = funded[funded > WEIGHT_TOLERANCE].sort_values(ascending=False)
    return (
        f"{len(funded)} ETF{'s' if len(funded) != 1 else ''} funded · "
        f"ETFs {etf_weight:.2%} · Cash {cash_weight:.2%} · {total:.2%} total"
    )


def ticket_action_summary(ticket):
    """Describe the exact current-to-target moves in a rebalance ticket."""
    actions = []
    for row in ticket.security_orders.itertuples(index=False):
        change = float(row.percentage_point_change)
        if abs(change) <= WEIGHT_TOLERANCE * 100.0:
            continue
        verb = "Buy" if change > 0.0 else "Sell"
        actions.append(f"{verb} {row.asset} by {abs(change):.2f} percentage points")
    cash_change = float(ticket.cash_balance["percentage_point_change"].iloc[0])
    if abs(cash_change) > WEIGHT_TOLERANCE * 100.0:
        verb = "increase" if cash_change > 0.0 else "decrease"
        actions.append(f"{verb} cash by {abs(cash_change):.2f} percentage points")
    if not actions:
        return "No allocation changes are needed; current and target weights match."
    return "; ".join(actions) + "."


def ticket_dollar_summary(ticket):
    """Summarize optional ETF notionals without treating cash or costs as orders."""
    if "trade_amount" not in ticket.security_orders:
        return None
    amounts = ticket.security_orders["trade_amount"].astype(float)
    buys = float(amounts.clip(lower=0.0).sum())
    sells = float((-amounts.clip(upper=0.0)).sum())
    cash_change = float(ticket.cash_balance["trade_amount"].iloc[0])
    cash_change_text = (
        f"+${cash_change:,.2f}"
        if cash_change >= 0.0
        else f"-${abs(cash_change):,.2f}"
    )
    return (
        f"ETF notionals: buy ${buys:,.2f}, sell ${sells:,.2f}; analytical cash "
        f"changes by {cash_change_text}. Estimated trading cost is "
        f"${ticket.estimated_cost_amount:,.2f} separately and is not deducted from "
        "these notionals."
    )


def render_workbench(
    bundle_path=DEFAULT_BUNDLE_PATH,
    timesfm_bundle_path=DEFAULT_TIMESFM_BUNDLE_PATH,
):
    """Render the isolated first-tab workbench and populate Portfolio Lab state."""
    st.markdown("## ETF Allocation Workbench")
    st.caption(
        "Choose up to eight ETFs. Compare allocation rules on the same history, "
        "then use one current target to build a rebalance plan."
    )
    try:
        bundle = load_cached_bundle(bundle_path)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        LOGGER.exception("Validated workbench data could not be loaded")
        st.error("Workbench unavailable: validated local data could not be loaded.")
        st.info("The ML research tabs remain available.")
        st.session_state.pop("portfolio_lab_transfer", None)
        return None

    freshness = bundle.freshness(as_of=_test_as_of())
    st.caption(
        f"Data through {bundle.manifest['last_complete_month']} · "
        f"status: {freshness['status']}"
    )
    with st.expander("Data details", expanded=False):
        st.markdown(
            f"""
            | Dataset | Latest observation |
            |:--|:--|
            | Adjusted ETF prices | {bundle.manifest['price_data_as_of']} |
            | U.S. overnight cash rates | {bundle.manifest['cash_rate_as_of']} |
            | Latest complete month | {bundle.manifest['last_complete_month']} |

            **{CASH_LABEL}** uses official EFFR before 2018-04-02 and official
            SOFR from 2018-04-02. It is an analytical balance, not a tradeable
            security. BIL remains a separately selectable ETF.
            """
        )
    if freshness["status"] == "warning":
        st.warning(f"Workbench data warning: {freshness['reason']}.")
    elif freshness["status"] == "disabled":
        st.error(f"Workbench disabled: {freshness['reason']}.")
        st.info("The ML research tabs remain available.")
        st.session_state.pop("portfolio_lab_transfer", None)
        return None

    forecast_bundle = None
    forecast_freshness = None
    forecast_error = None
    try:
        with st.spinner("Loading local forecast research…"):
            forecast_bundle = load_cached_timesfm_bundle(
                timesfm_bundle_path,
                workbench_path=bundle_path,
            )
        forecast_freshness = forecast_bundle.freshness(as_of=_test_as_of())
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        forecast_error = "unavailable"
        LOGGER.warning("Local forecast research could not be loaded: %s", exc)

    if forecast_error:
        st.info(
            "Forecast research is temporarily unavailable, so the standard "
            "workbench is still running without it."
        )

    labels = {
        row.ticker: f"{row.ticker} — {row.role}"
        for row in bundle.instruments.itertuples(index=False)
    }
    selected = st.multiselect(
        "ETFs (select up to 8)",
        options=list(bundle.tickers),
        default=list(DEFAULT_SELECTION),
        max_selections=8,
        format_func=lambda ticker: labels[ticker],
        key="wb_selected_etfs",
    )
    if not selected:
        st.warning("Select at least one ETF to run the workbench.")
        st.session_state.pop("portfolio_lab_transfer", None)
        return None

    current_weights, current_error = _current_editor(selected)
    current_valid = current_weights is not None and current_error is None
    if current_error:
        st.warning(
            f"Current weights are invalid: {current_error}. Strategy research remains "
            "available; Current Mix, Portfolio Lab transfer, and ticket are disabled."
        )

    controls = st.columns(2)
    transaction_cost_bps = controls[0].number_input(
        "Estimated trading cost (bps)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
        key="wb_cost_bps",
        help=(
            "A basis point (bp) is 0.01%, so 5 bps is 0.05%. At 100% one-way "
            "turnover, the modeled cost is $5 per $10,000 of portfolio value."
        ),
    )
    try:
        path = Path(bundle.path).resolve()
        allocation_results = _cached_allocation_results(
            str(path), bundle_fingerprint(path), tuple(selected)
        )
        forecast_result = None
        if forecast_bundle is not None:
            try:
                timesfm_path = Path(forecast_bundle.path).resolve()
                forecast_result = _cached_forecast_allocation_result(
                    str(path),
                    bundle_fingerprint(path),
                    str(timesfm_path),
                    timesfm_bundle_fingerprint(timesfm_path),
                    tuple(selected),
                )
            except (ValueError, ArithmeticError, KeyError) as exc:
                forecast_error = "unavailable"
                forecast_bundle = None
                forecast_freshness = None
                LOGGER.warning("Forecast target calculation failed: %s", exc)
                st.info(
                    "Forecast calculations are unavailable for this ETF set; "
                    "the standard workbench remains available."
                )
        standard_freshness = standard_target_freshness(
            bundle,
            allocation_results[VOL_BALANCED_LABEL],
            as_of=_test_as_of(),
        )
    except (ValueError, ArithmeticError) as exc:
        st.error(f"This ETF set cannot be calculated: {exc}")
        st.session_state.pop("portfolio_lab_transfer", None)
        return None

    unavailable_tactical = []
    if standard_freshness["latest_target_status"] != "current":
        unavailable_tactical.append("volatility/trend")
    if (
        forecast_freshness is not None
        and forecast_freshness["latest_target_status"] != "current"
    ):
        unavailable_tactical.append("TimesFM")
    if unavailable_tactical:
        st.caption(
            "Latest target status: "
            + " and ".join(unavailable_tactical)
            + (" targets are" if len(unavailable_tactical) > 1 else " target is")
            + " outside the current rebalance window. Historical results remain available."
        )

    options = available_comparisons(
        selected,
        current_valid,
        forecast_available=forecast_result is not None,
    )
    selection_key = "|".join(selected)
    saved_by_selection = dict(
        st.session_state.get("wb_comparisons_by_selection", {})
    )
    stored = [
        item
        for item in saved_by_selection.get(
            selection_key, default_comparisons(selected, options)
        )
        if item in options
    ]
    if not stored:
        stored = list(options)
    comparison_widget_key = f"wb_comparisons__{selection_key}"
    existing_widget_values = st.session_state.get(comparison_widget_key)
    if existing_widget_values is not None and any(
        item not in options for item in existing_widget_values
    ):
        st.session_state.pop(comparison_widget_key, None)
    comparison_kwargs = (
        {} if comparison_widget_key in st.session_state else {"default": stored}
    )
    comparisons = st.multiselect(
        "Portfolios to compare",
        options=options,
        format_func=comparison_selector_label,
        key=comparison_widget_key,
        **comparison_kwargs,
    )
    saved_by_selection[selection_key] = list(comparisons)
    st.session_state["wb_comparisons_by_selection"] = saved_by_selection
    if not comparisons:
        st.warning("Select at least one comparison series.")
        return None
    forecast_selected = VOL_FORECAST_LABEL in comparisons
    bounds_schedule = (
        forecast_result.schedule
        if forecast_selected
        else allocation_results[VOL_BALANCED_LABEL].schedule.select(
            start=pd.Timestamp(bundle.cash_index.index.min())
        )
    )
    earliest_execution = pd.Timestamp(bounds_schedule.execution_dates[0])
    latest_period_end = pd.Timestamp(bounds_schedule.period_end_dates.iloc[-1])
    default_start = max(
        earliest_execution,
        latest_period_end - pd.DateOffset(years=10),
    )
    date_key = "wb_date_range"
    saved_range = st.session_state.get(date_key)
    if isinstance(saved_range, (tuple, list)) and len(saved_range) == 2:
        saved_start, saved_end = (pd.Timestamp(value) for value in saved_range)
        if saved_start < earliest_execution or saved_end > latest_period_end:
            st.session_state.pop(date_key, None)
    date_range = controls[1].date_input(
        "Historical range",
        value=(default_start.date(), latest_period_end.date()),
        min_value=earliest_execution.date(),
        max_value=latest_period_end.date(),
        key=date_key,
    )
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Choose both a start and end date.")
        return None
    start, end = date_range
    alignment_text = (
        "All displayed lines use the completed TimesFM forecast dates."
        if forecast_selected
        else "The standard approaches use their full common history."
    )
    st.caption(
        "History restarts at $1 with entry costs. Changing the range does not change "
        "the latest target. "
        + alignment_text
    )
    cap = position_cap(len(selected))
    equal_share = 1.0 / len(selected)
    with st.expander("How the portfolio rules work"):
        st.markdown(
            f"""
            - **{VOL_BALANCED_LABEL}:** Lower-volatility ETFs receive more weight.
            - **{VOL_TREND_LABEL}:** An ETF receives weight only when its price is
              above its 10-month average.
            - **{VOL_FORECAST_LABEL}:** An ETF receives weight only when its TimesFM-3
              forecast return is above the cash return for the same month.
            - **{EQUAL_WEIGHT_LABEL}:** Each selected ETF receives the same weight.
            - **{CURRENT_MIX_LABEL}:** The entered weights reset each month for the
              historical comparison.
            - **Timing:** A completed month-end signal sets the target for the next
              trading close.
            - **Position limit:** With {len(selected)} selected ETF{'s' if len(selected) != 1 else ''},
              equal weight is {equal_share:.1%} and the maximum ETF weight is {cap:.1%}.
            """
        )
        if len(selected) == 1:
            st.caption(
                "With one ETF, the filtered approaches switch between that ETF and "
                "cash. Duplicate fully invested comparisons are hidden."
            )
        if current_valid:
            st.caption(
                "Current Mix resets your entered weights monthly for comparison. It "
                "does not reconstruct your actual transaction history."
            )
        st.markdown(
            "[Read the formulas, timing, cash method, and metric definitions]"
            "(https://github.com/sabdulmajid/ml-etf-rebalancer/blob/master/"
            "docs/research/etf-allocation.md)."
        )

    try:
        study = build_workbench_study(
            bundle,
            selected,
            transaction_cost_bps=transaction_cost_bps,
            current_weights=current_weights if current_valid else None,
            start=start,
            end=end,
            allocation_results=allocation_results,
            forecast_result=forecast_result,
            forecast_manifest=(
                None if forecast_bundle is None else forecast_bundle.manifest
            ),
            forecast_freshness=forecast_freshness,
            comparison_labels=comparisons,
        )
    except (ValueError, ArithmeticError) as exc:
        st.warning(f"Selected range is unavailable: {exc}")
        return None

    show_forecast_detail = False
    if forecast_result is not None:
        show_forecast_detail = st.toggle(
            "Inspect TimesFM-3 forecast research",
            value=False,
            key="wb_show_forecast_detail",
            help=(
                "Open the model evaluation, latest ETF forecasts, and completed "
                "forecast replay."
            ),
        )
    if forecast_result is not None and show_forecast_detail:
        latest_forecast = why_this_weight(bundle, study, VOL_FORECAST_LABEL)
        forecast_etfs = latest_forecast.loc[
            latest_forecast["asset"] != CASH_LABEL
        ].copy()
        skill = forecast_skill_snapshot(forecast_bundle.metrics)
        st.divider()
        st.markdown("## TimesFM-3 Forecast Analysis")
        st.write(
            "TimesFM-3 reads recent ETF prices and estimates each ETF's price one "
            "month ahead. The portfolio holds an ETF when its median forecast return "
            "is above cash. Volatility Balanced then sets the ETF weights."
        )
        st.markdown(
            "**1. Read prices.** All 14 ETF histories enter the model together.  "
            "**2. Estimate next month.** The model produces a median price path and "
            "a range of possible final prices.  "
            "**3. Set weights.** ETFs forecast above cash can receive a weight."
        )
        st.markdown(
            "[Read the TimesFM-3 inputs, formulas, and evaluation measures]"
            "(https://github.com/sabdulmajid/ml-etf-rebalancer/blob/master/"
            "docs/research/timesfm-3.md)."
        )
        scorecard_row_one = st.columns(2)
        scorecard_row_two = st.columns(2)
        scorecards = (*scorecard_row_one, *scorecard_row_two)
        scorecards[0].metric(
            "ETF-month forecasts",
            f"{skill['observations']:,}",
        )
        scorecards[0].caption("Completed forecasts with a known result.")
        scorecards[1].metric(
            "Correct up-or-down calls",
            f"{skill['directional_accuracy']:.1%}",
        )
        scorecards[1].caption(
            "The median forecast called the return direction correctly."
        )
        scorecards[2].metric(
            "Average forecast difference",
            f"{skill['return_mae']:.2%}",
        )
        scorecards[2].caption(
            "Average gap between forecast and realized one-month return."
        )
        scorecards[3].metric(
            "Final prices inside range",
            f"{skill['interval_80_coverage']:.1%}",
        )
        scorecards[3].caption("Share inside the model's 10%–90% price range.")
        st.caption(
            "Joint-input comparison: using all 14 ETF histories produced "
            f"{skill['directional_accuracy']:.1%} correct directions and a "
            f"{skill['return_mae']:.2%} average return difference. Using one ETF "
            f"history at a time produced {skill['univariate_directional_accuracy']:.1%} "
            f"and {skill['univariate_return_mae']:.2%}, respectively."
        )
        if forecast_selected:
            portfolio_comparison = forecast_portfolio_comparison(study)
            st.markdown("### Forecast Portfolio Comparison")
            st.caption(
                f"{portfolio_comparison['start'].date()} through "
                f"{portfolio_comparison['end'].date()} · same dates and transaction "
                "cost for both portfolios. The one-ETF reference is Buy & Hold. The "
                "multi-ETF reference is Equal Weight."
            )
            st.dataframe(
                forecast_portfolio_table(portfolio_comparison).style.format(
                    {
                        "Annualized return": "{:.2%}",
                        "Sharpe above cash": "{:.2f}",
                        "Maximum drawdown": "{:.2%}",
                        "Annualized turnover": "{:.2%}",
                    },
                    na_rep="—",
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "The last-observed-price reference had a "
                f"{skill['last_value_return_mae']:.2%} average return difference."
            )
        else:
            st.info(
                "Select Forecast under Portfolios to compare to add its portfolio results "
                "for the chosen dates."
            )

        st.markdown("### Latest Available TimesFM-3 Signal")
        st.write(
            "The table compares each ETF's one-month forecast with cash for the same "
            "period. A passing ETF can receive a Volatility Balanced weight."
        )
        st.caption(
            f"Prices read through {forecast_result.latest_signal_date.date()} · target "
            f"date {forecast_result.latest_execution_date.date()} · period ends "
            f"{forecast_result.latest_period_end_date.date()}"
        )
        forecast_display = forecast_etfs[
            [
                "asset",
                "role",
                "median_forecast_return",
                "cash_hurdle",
                "forecast_status",
                "final_weight",
            ]
        ].rename(
            columns={
                "asset": "ETF",
                "role": "Role",
                "median_forecast_return": "Forecast one-month return",
                "cash_hurdle": "Cash one-month return",
                "forecast_status": "Portfolio rule",
                "final_weight": "Target weight",
            }
        )
        forecast_display["Portfolio rule"] = forecast_display["Portfolio rule"].replace(
            {"Eligible": "Use ETF", "Held in cash": "Use cash for this share"}
        )
        st.dataframe(
            forecast_display.style.format(
                {
                    "Forecast one-month return": "{:+.2%}",
                    "Cash one-month return": "{:.2%}",
                    "Target weight": "{:.2%}",
                },
                na_rep="—",
            ),
            hide_index=True,
            width="stretch",
        )

        with st.expander("Review completed forecasts", expanded=False):
            replay_key = "wb_forecast_replay_ticker"
            if st.session_state.get(replay_key) not in (None, *selected):
                st.session_state.pop(replay_key, None)
            replay_ticker = st.selectbox(
                "ETF to inspect",
                options=list(selected),
                key=replay_key,
                help="The chart uses only completed forecasts with known outcomes.",
            )
            try:
                replay = forecast_replay_frame(
                    forecast_bundle.signals,
                    replay_ticker,
                    start=start,
                    end=end,
                )
            except ValueError as exc:
                st.info(str(exc))
            else:
                replay_stats = forecast_replay_summary(replay)
                replay_row_one = st.columns(2)
                replay_row_two = st.columns(2)
                replay_cards = (*replay_row_one, replay_row_two[0])
                replay_cards[0].metric(
                    "Correct ETF-or-cash decisions",
                    f"{replay_stats['cash_gate_accuracy']:.1%}",
                )
                replay_cards[0].caption(
                    f"{replay_stats['observations']} completed months in this chart."
                )
                replay_cards[1].metric(
                    "Average forecast difference",
                    f"{replay_stats['return_mae']:.2%}",
                )
                replay_cards[1].caption("Average forecast-to-result gap.")
                replay_cards[2].metric(
                    "Final prices inside range",
                    f"{replay_stats['interval_80_coverage']:.1%}",
                )
                replay_cards[2].caption("Share inside the 10%–90% price range.")
                _forecast_replay_chart(replay, replay_ticker)
                st.caption(
                    "A decision is correct when the model selects the asset with the "
                    "higher realized return: the ETF or cash. Hover over the chart to "
                    "review a month."
                )

        with st.expander(
            "Forecast range, dates, and allocation history",
            expanded=False,
        ):
            st.caption(
                "The model receives all 14 ETF histories together. Your ETF selection "
                "controls the portfolio, not the model input."
            )
            st.caption(
                f"Signal {forecast_result.latest_signal_date.date()} · intended "
                f"execution {forecast_result.latest_execution_date.date()} · holding "
                f"period ends {forecast_result.latest_period_end_date.date()}"
            )
            st.caption(
                f"Model generation `{forecast_result.latest_model_generation_id}` · "
                f"model ran {forecast_freshness['latest_model_generated_at_utc']} · "
                f"execution cutoff {forecast_freshness['latest_execution_cutoff_utc']} · "
                "the bundle refresh timestamp is separate from model generation."
            )
            price_display = forecast_etfs[
                [
                    "asset",
                    "q10_period_end_price",
                    "q50_period_end_price",
                    "q90_period_end_price",
                ]
            ].rename(
                columns={
                    "asset": "ETF",
                    "q10_period_end_price": "10% lower price",
                    "q50_period_end_price": "Median price",
                    "q90_period_end_price": "90% upper price",
                }
            )
            st.caption(
                "The 10% price is the lower estimate, the 50% price is the median, and "
                "the 90% price is the upper estimate for the end of the period."
            )
            st.dataframe(
                price_display.style.format(
                    {
                        name: "{:,.2f}"
                        for name in (
                            "10% lower price",
                            "Median price",
                            "90% upper price",
                        )
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.markdown("#### How the forecast-filtered allocation changed")
            st.caption(
                "These are the research targets on their actual monthly execution dates. "
                + (
                    "They use your selected historical range because Forecast is selected."
                    if forecast_selected
                    else "They show the complete forecast replay because Forecast is not a displayed comparison."
                )
            )
            _allocation_chart(
                study,
                VOL_FORECAST_LABEL,
                key="wb_forecast_history_chart",
            )
            st.download_button(
                "Download forecast allocation history CSV",
                allocation_history_download(study, VOL_FORECAST_LABEL).to_csv(
                    index=False
                ),
                file_name="etf_workbench_forecast_allocation_history.csv",
                mime="text/csv",
                key="wb_forecast_allocation_download",
            )
            st.caption(
                "The displayed history uses completed monthly forecasts and the same "
                "portfolio accounting as the other workbench lines."
            )
        with st.expander("Compare forecast methods", expanded=False):
            st.caption(
                "Average return difference measures the forecast-to-result gap. Scaled "
                "final-price difference compares the price gap with a normal one-session "
                "price move. Probability forecast difference scores all nine forecast "
                "percentiles. Lower values indicate a smaller difference."
            )
            checks = forecast_check_summary(forecast_bundle.metrics)
            st.dataframe(
                forecast_checks_for_display(checks),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "The last-price reference supplies one price value. Measures that need "
                "a direction or a forecast range therefore show a dash."
            )

    target_options = [
        item
        for item in options
        if item not in (CURRENT_MIX_LABEL, SPY_REFERENCE_LABEL)
    ]
    if (
        VOL_FORECAST_LABEL in target_options
        and forecast_freshness["latest_target_status"] == "current"
    ):
        default_target = VOL_FORECAST_LABEL
    elif (
        VOL_TREND_LABEL in target_options
        and standard_freshness["latest_target_status"] == "current"
    ):
        default_target = VOL_TREND_LABEL
    elif EQUAL_WEIGHT_LABEL in target_options:
        default_target = EQUAL_WEIGHT_LABEL
    else:
        default_target = target_options[0]
    stored_target = st.session_state.get("wb_authoritative_target", default_target)
    if stored_target not in target_options:
        stored_target = (
            EQUAL_WEIGHT_LABEL
            if EQUAL_WEIGHT_LABEL in target_options
            else target_options[0]
        )
        if "wb_authoritative_target" in st.session_state:
            st.session_state["wb_authoritative_target"] = stored_target
    target_kwargs = (
        {} if "wb_authoritative_target" in st.session_state else {
            "index": target_options.index(stored_target)
        }
    )
    st.divider()
    st.markdown("## Current proposal")
    authoritative = st.selectbox(
        "Choose a proposed portfolio",
        options=target_options,
        key="wb_authoritative_target",
        help="This choice controls the proposal shown here and sent to Portfolio Lab.",
        **target_kwargs,
    )
    target = study.latest_targets[authoritative]
    provenance = target_provenance(bundle, study, authoritative)
    target_transferable = provenance["execution_status"] in {
        "current",
        "constant_target_effective_for_analytical_ticket",
    }
    st.markdown("### Target weights")
    st.caption(proposed_target_summary(target))
    st.caption(target_provenance_summary(provenance))
    if not target_transferable:
        st.warning(
            "This target is available for research and download, but its timing status "
            "does not permit a Portfolio Lab transfer. Choose a current or constant "
            "target to create a rebalance ticket."
        )
    if authoritative in (VOL_BALANCED_LABEL, VOL_TREND_LABEL):
        st.caption(study.allocation_results[authoritative].policy.cap_explanation)
    with st.expander("Target dates and data details"):
        st.markdown(
            f"""
            - **Latest-target status:** {target_provenance_summary(provenance)}
            - **Displayed historical results through:** {provenance['displayed_history_through']}
            - **Price artifact through:** {provenance['price_data_as_of']}
            - **Workbench data bundle refreshed:** {provenance['workbench_bundle_refreshed_at_utc']}
            - **Rule version:** `{provenance['policy_version']}`

            The latest proposal always uses the full validated artifact. The historical
            chart endpoint above follows your selected date range and may be earlier.
            """
        )
        if provenance["model_generation_id"] is not None:
            st.markdown(
                f"""
                - **Model generation ID:** `{provenance['model_generation_id']}`
                - **Model generated:** {provenance['model_generated_at_utc']}
                - **Intended execution cutoff:** {provenance['execution_cutoff_utc']}
                - **Forecast bundle refreshed:** {provenance['forecast_bundle_refreshed_at_utc']}
                """
            )
        elif provenance["execution_cutoff_utc"] is not None:
            st.markdown(
                f"- **Intended execution cutoff:** {provenance['execution_cutoff_utc']}"
            )
    target_frame = latest_target_download(bundle, study, authoritative)
    target_display = target_frame[["asset", "target_weight", "asset_type"]].copy()
    target_display["asset"] = target_display["asset"].replace({CASH_ASSET: CASH_LABEL})
    target_display = target_display.rename(
        columns={
            "asset": "Asset",
            "target_weight": "Proposed target",
            "asset_type": "Type",
        }
    )
    target_display["Type"] = target_display["Type"].replace(
        {"analytical_cash": "Analytical cash", "tradeable_etf": "Tradeable ETF"}
    )
    st.dataframe(
        target_display.style.format({"Proposed target": "{:.2%}"}),
        hide_index=True,
        width="stretch",
    )

    if current_valid and target_transferable:
        ticket = build_rebalance_ticket(
            current_weights,
            target,
            selected,
            transaction_cost_bps=transaction_cost_bps,
        )
        payload = {
            "selected_etfs": tuple(selected),
            "strategy": authoritative,
            **provenance,
            "transaction_cost_bps": float(transaction_cost_bps),
            "current_weights": current_weights.to_dict(),
            "target_weights": target.to_dict(),
            "target_csv": target_frame.to_csv(index=False),
            "turnover": ticket.turnover,
        }
        existing = st.session_state.get("portfolio_lab_transfer")
        comparable_keys = (
            "selected_etfs",
            "strategy",
            "signal_as_of",
            "execution_status",
            "artifact_generated_at_utc",
            "price_data_as_of",
            "policy_version",
            "model_generation_id",
            "model_generated_at_utc",
            "execution_cutoff_utc",
            "bundle_refreshed_at_utc",
            "workbench_bundle_refreshed_at_utc",
            "forecast_bundle_refreshed_at_utc",
            "transaction_cost_bps",
            "current_weights",
            "target_weights",
        )
        if existing is not None and any(
            existing.get(key) != payload.get(key) for key in comparable_keys
        ):
            st.session_state.pop("portfolio_lab_transfer", None)
        if st.button(
            "Send proposed target to Portfolio Lab",
            type="primary",
            key="wb_send_to_portfolio_lab",
        ):
            st.session_state["portfolio_lab_transfer"] = payload
        if st.session_state.get("portfolio_lab_transfer") is not None:
            st.success(
                "Sent successfully. Open the Portfolio Lab tab to review the exact "
                "current-to-target moves and download a ticket."
            )
    else:
        st.session_state.pop("portfolio_lab_transfer", None)
        st.button(
            "Send proposed target to Portfolio Lab",
            disabled=True,
            key="wb_send_to_portfolio_lab_disabled",
            help=(
                "Choose a current or constant target before transferring."
                if current_valid
                else "Enter current weights that total exactly 100% before transferring."
            ),
        )
        if current_valid:
            st.caption(
                "Next step: choose a current or constant target if you want a ticket. "
                "This historical-only target remains available for analysis and download."
            )
        elif current_error:
            st.caption("Next step: adjust the current-weight total to exactly 100%.")
        else:
            st.caption(
                "To build a ticket, enter current weights—including cash—that total 100%."
            )

    st.markdown("### Why each asset has this weight")
    explanation = why_this_weight(bundle, study, authoritative)
    explanation_display = explanation[
        ["asset", "final_weight", "role", "trend", "reason"]
    ].rename(
        columns={
            "asset": "Asset",
            "final_weight": "Proposed target",
            "role": "Role",
            "trend": "Trend status",
            "reason": "Reason",
        }
    )
    explanation_display["Role and signal"] = (
        explanation_display["Role"]
        + " · "
        + explanation_display["Trend status"]
    )
    explanation_display = explanation_display[
        ["Asset", "Proposed target", "Role and signal", "Reason"]
    ]
    st.table(
        explanation_display.style.format(
            {"Proposed target": "{:.2%}"}, na_rep="—"
        ).hide(axis="index")
    )

    with st.expander("Show calculation details"):
        technical = explanation[
            [
                "asset",
                "trailing_volatility",
                "raw_weight",
                "filtered_raw_weight",
                "final_weight",
                "change_vs_uncapped_inverse_vol",
            ]
        ].rename(
            columns={
                "asset": "Asset",
                "trailing_volatility": "Trailing 126-day volatility",
                "raw_weight": "Uncapped inverse-vol weight",
                "filtered_raw_weight": "Post-filter raw weight",
                "final_weight": "Proposed target",
                "change_vs_uncapped_inverse_vol": "Change vs uncapped inverse-vol",
            }
        )
        st.dataframe(
            technical.style.format(
                {
                    "Trailing 126-day volatility": "{:.2%}",
                    "Uncapped inverse-vol weight": "{:.2%}",
                    "Post-filter raw weight": "{:.2%}",
                    "Proposed target": "{:.2%}",
                    "Change vs uncapped inverse-vol": "{:+.2%}",
                },
                na_rep="—",
            ),
            hide_index=True,
            width="stretch",
        )

    st.divider()
    st.markdown("## Historical comparison")
    st.caption(
        f"The charts below end on {study.latest_period_end.date()}. They restart at "
        "$1 in analytical cash for the selected range; they do not change the latest "
        "proposal shown above."
    )
    _line_chart(study, comparisons, "net_equity", "Growth of $1 after estimated costs")
    _line_chart(study, comparisons, "drawdown", "Decline from each prior peak", percent=True)

    metrics = _format_metrics(study, comparisons)
    st.markdown("### Results at a glance")
    st.caption(
        "Sharpe measures return above cash relative to variability. Drawdown shows the "
        "largest peak-to-trough loss. Turnover and cost drag show how much trading the "
        "approach required."
    )
    performance_columns = [
        "Annualized return ↑",
        "Annualized volatility ↓",
        "Sharpe above cash ↑",
        "Maximum drawdown ↑",
        "Return / drawdown ↑",
        "Worst month ↑",
    ]
    st.dataframe(
        metrics[performance_columns].style.format(
            {
                "Annualized return ↑": "{:.2%}",
                "Annualized volatility ↓": "{:.2%}",
                "Sharpe above cash ↑": "{:.2f}",
                "Maximum drawdown ↑": "{:.2%}",
                "Return / drawdown ↑": "{:.2f}",
                "Worst month ↑": "{:.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )
    st.markdown("#### Trading impact")
    st.dataframe(
        metrics[
            [
                "Annualized one-way turnover ↓",
                "Annualized cost drag ↓",
            ]
        ].style.format(
            {
                "Annualized one-way turnover ↓": "{:.2%}",
                "Annualized cost drag ↓": "{:.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### How the proposed allocation changed")
    st.caption(
        "Monthly weights on each rebalance date. The table shows the last 24 decisions."
    )
    _allocation_chart(
        study,
        authoritative,
        key="wb_authoritative_history_chart",
    )
    allocation_download = allocation_history_download(study, authoritative)
    allocation_history = allocation_download.rename(
        columns={
            "rebalance_date": "Rebalance date",
            "signal_date": "Signal date",
            "holding_period_end": "Holding-period end",
            CASH_ASSET: CASH_LABEL,
            "turnover": "One-way turnover",
            "estimated_cost_rate": "Estimated cost rate at rebalance",
        }
    ).drop(columns=["strategy", "transaction_cost_value"])
    percentage_columns = [
        *selected,
        CASH_LABEL,
        "One-way turnover",
        "Estimated cost rate at rebalance",
    ]
    with st.expander("View monthly allocation data"):
        st.dataframe(
            allocation_history.tail(24).style.format(
                {column: "{:.2%}" for column in percentage_columns}
            ),
            hide_index=True,
            width="stretch",
        )

    downloads = st.columns(3)
    downloads[0].download_button(
        "Download displayed history CSV",
        historical_download(study, comparisons).to_csv(index=False),
        file_name="etf_workbench_history.csv",
        mime="text/csv",
        key="wb_history_download",
    )
    downloads[1].download_button(
        "Download latest target CSV",
        target_frame.to_csv(index=False),
        file_name="etf_workbench_latest_target.csv",
        mime="text/csv",
        key="wb_target_download",
    )
    downloads[2].download_button(
        "Download allocation history CSV",
        allocation_download.to_csv(index=False),
        file_name="etf_workbench_allocation_history.csv",
        mime="text/csv",
        key="wb_allocation_download",
    )

    return study


def render_portfolio_lab():
    """Render the ticket from the exact workbench session target."""
    st.markdown("## Portfolio Lab")
    transfer = st.session_state.get("portfolio_lab_transfer")
    if not transfer:
        st.info(
            "No proposal selected. In ETF Allocation Workbench, enter current weights "
            "that total 100%, then send a proposed portfolio here."
        )
        return

    selected = tuple(transfer["selected_etfs"])
    current = pd.Series(transfer["current_weights"], dtype=float).reindex(
        [*selected, CASH_ASSET]
    )
    target = pd.Series(transfer["target_weights"], dtype=float).reindex(
        [*selected, CASH_ASSET]
    )
    include_dollars = st.checkbox(
        "Add optional dollar estimates",
        key="portfolio_lab_include_dollars",
        help="Percentage-point changes work without a portfolio value.",
    )
    portfolio_value = None
    if include_dollars:
        portfolio_value = st.number_input(
            "Portfolio value",
            min_value=1.0,
            value=100000.0,
            step=1000.0,
            key="portfolio_lab_value",
        )
    try:
        ticket: RebalanceTicket = build_rebalance_ticket(
            current,
            target,
            selected,
            transaction_cost_bps=transfer["transaction_cost_bps"],
            portfolio_value=portfolio_value,
        )
    except ValueError as exc:
        st.warning(f"Ticket disabled: {exc}")
        return

    st.markdown("### Move from your current mix to the proposal")
    st.write(ticket_action_summary(ticket))
    dollar_summary = ticket_dollar_summary(ticket)
    if dollar_summary:
        # Streamlit Markdown otherwise treats pairs of dollar signs as inline
        # math and visually drops the currency symbols from this sentence.
        st.markdown(dollar_summary.replace("$", r"\$"))

    synced_target = pd.DataFrame(
        {
            "Asset": [CASH_LABEL if asset == CASH_ASSET else asset for asset in target.index],
            "Current weight": current.to_numpy(dtype=float),
            "Proposed target": target.to_numpy(dtype=float),
            "Type": [
                "Analytical cash" if asset == CASH_ASSET else "Tradeable ETF"
                for asset in target.index
            ],
        }
    )
    st.dataframe(
        synced_target.style.format(
            {"Current weight": "{:.2%}", "Proposed target": "{:.2%}"}
        ),
        hide_index=True,
        width="stretch",
    )
    if transfer["execution_status"] == "pending_next_trading_close":
        st.warning(
            "This proposed target is awaiting the next rebalance trading close. "
            "The ticket is analytical only and does not represent executable orders."
        )
    with st.expander("Proposal dates and data details"):
        st.markdown(
            f"""
            - **Approach:** {transfer['strategy']}
            - **Status:** {target_provenance_summary(transfer)}
            - **Displayed historical results through when sent:** {transfer['displayed_history_through']}
            - **Workbench data bundle refreshed:** {transfer['workbench_bundle_refreshed_at_utc']}
            - **Rule version:** `{transfer['policy_version']}`
            """
        )
        if transfer.get("model_generation_id") is not None:
            st.markdown(
                f"""
                - **Model generation ID:** `{transfer['model_generation_id']}`
                - **Model generated:** {transfer['model_generated_at_utc']}
                - **Intended execution cutoff:** {transfer['execution_cutoff_utc']}
                - **Forecast bundle refreshed:** {transfer['forecast_bundle_refreshed_at_utc']}
                """
            )

    metrics = st.columns(3)
    metrics[0].metric("One-way turnover", f"{ticket.turnover:.2%}")
    metrics[1].metric("Estimated cost rate", f"{ticket.estimated_cost_rate:.3%}")
    metrics[2].metric(
        "Estimated cost",
        "—" if ticket.estimated_cost_amount is None else f"${ticket.estimated_cost_amount:,.2f}",
    )
    st.caption(
        "One-way turnover is half the sum of absolute weight changes. Estimated cost "
        "equals turnover × the transaction-cost setting. It is shown separately and "
        "is not deducted from ETF or cash notionals."
    )
    st.markdown("### Illustrative ETF changes")
    security_display = ticket.security_orders.rename(
        columns={
            "asset": "ETF",
            "current_weight": "Current weight",
            "target_weight": "Proposed target",
            "percentage_point_change": "Change (p.p.)",
            "action": "Analytical action",
            "trade_amount": "Analytical notional",
        }
    )
    for column in ("Current weight", "Proposed target"):
        security_display[column] = security_display[column].map(lambda value: f"{value:.2%}")
    security_display["Change (p.p.)"] = security_display["Change (p.p.)"].map(
        lambda value: f"{value:+.2f} p.p."
    )
    if "Analytical notional" in security_display:
        security_display["Analytical notional"] = security_display[
            "Analytical notional"
        ].map(lambda value: f"${value:+,.2f}")
    st.dataframe(security_display, hide_index=True, width="stretch")
    st.markdown(f"### {CASH_LABEL} balance (not a security order)")
    cash_display = ticket.cash_balance.rename(
        columns={
            "asset": "Identifier",
            "label": "Balance",
            "current_weight": "Current weight",
            "target_weight": "Proposed target",
            "percentage_point_change": "Change (p.p.)",
            "trade_amount": "Analytical balance change",
        }
    )
    for column in ("Current weight", "Proposed target"):
        cash_display[column] = cash_display[column].map(lambda value: f"{value:.2%}")
    cash_display["Change (p.p.)"] = cash_display["Change (p.p.)"].map(
        lambda value: f"{value:+.2f} p.p."
    )
    if "Analytical balance change" in cash_display:
        cash_display["Analytical balance change"] = cash_display[
            "Analytical balance change"
        ].map(lambda value: f"${value:+,.2f}")
    st.dataframe(cash_display, hide_index=True, width="stretch")
    st.download_button(
        "Download reconciled ticket CSV",
        ticket.download_frame().to_csv(index=False),
        file_name="etf_workbench_rebalance_ticket.csv",
        mime="text/csv",
        key="portfolio_lab_ticket_download",
    )
    st.caption(
        "Percentage-point changes and optional dollar notionals reconcile across ETF "
        "orders and the separate analytical cash balance. The overnight-rate proxy's "
        "yield can differ from the return on your actual cash account. Cash is never a "
        "security order, and no orders are placed."
    )
