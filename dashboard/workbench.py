"""ETF Allocation Workbench calculations and Streamlit rendering.

All return paths in this module are executed by ``backtest.engine.run_backtest``.
The UI performs no market-data HTTP calls and treats the validated PR2 bundle as
its sole data source.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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


VOL_BALANCED_LABEL = "Volatility Balanced"
VOL_TREND_LABEL = "Volatility Balanced + Trend"
EQUAL_WEIGHT_LABEL = "Equal Weight"
CASH_LABEL_SHORT = CASH_LABEL
BUY_HOLD_LABEL = "Buy & Hold"
CURRENT_MIX_LABEL = "Current Mix — monthly rebalanced"
DEFAULT_SELECTION = ("SPY", "IEF", "GLD")

ALLOCATION_LABELS = {
    VOL_BALANCED_LABEL: VOLATILITY_BALANCED,
    VOL_TREND_LABEL: VOLATILITY_BALANCED_TREND,
    EQUAL_WEIGHT_LABEL: EQUAL_WEIGHT,
}


@dataclass(frozen=True)
class WorkbenchStudy:
    selected_etfs: tuple[str, ...]
    allocation_results: dict[str, AllocationResult]
    backtests: dict[str, BacktestResult]
    latest_targets: dict[str, pd.Series]
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


def clear_workbench_caches():
    """Clear only workbench bundle/allocation caches for deterministic tests."""
    _load_cached_bundle.clear()
    _cached_allocation_results.clear()


def available_comparisons(selected_etfs, current_weights_valid=False):
    """Return only comparisons compatible with the selected set and inputs."""
    selected = tuple(selected_etfs)
    if len(selected) == 1:
        # Volatility Balanced and Equal Weight are necessarily 100% in the lone
        # ETF, exactly duplicating Buy & Hold. Represent that path once.
        options = [VOL_TREND_LABEL, BUY_HOLD_LABEL, CASH_LABEL_SHORT]
    else:
        options = [
            VOL_BALANCED_LABEL,
            VOL_TREND_LABEL,
            EQUAL_WEIGHT_LABEL,
            CASH_LABEL_SHORT,
        ]
    if current_weights_valid:
        options.append(CURRENT_MIX_LABEL)
    return options


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
    base_schedule = schedules[VOL_BALANCED_LABEL]
    asset_returns, cash_returns = holding_period_returns(
        bundle, base_schedule, selected
    )

    latest_targets = {
        label: result.latest_target.copy()
        for label, result in allocation_results.items()
    }

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

    backtests = {}
    for label, schedule in schedules.items():
        # All schedules share the selected-set timing contract. Supplying the
        # exact same return matrices enforces identical entry, drift and cost.
        backtests[label] = run_backtest(
            schedule,
            asset_returns,
            cash_returns,
            transaction_cost_bps=transaction_cost_bps,
            start=start,
            end=end,
        )
    return WorkbenchStudy(
        selected_etfs=selected,
        allocation_results=allocation_results,
        backtests=backtests,
        latest_targets=latest_targets,
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


def target_provenance(bundle, study, label):
    """Return authoritative timing and artifact provenance for a latest target."""
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
    else:
        allocation = study.allocation_results[label]
        signal_as_of = str(pd.Timestamp(allocation.latest_signal_date).date())
        execution_status = (
            "pending_next_trading_close"
            if allocation.latest_execution_date is None
            else str(pd.Timestamp(allocation.latest_execution_date).date())
        )
        policy_version = allocation.policy.version
    return {
        "signal_as_of": signal_as_of,
        "execution_status": execution_status,
        "artifact_generated_at_utc": bundle.manifest["generated_at_utc"],
        "price_data_as_of": bundle.manifest["price_data_as_of"],
        "policy_version": policy_version,
        "displayed_history_through": str(
            pd.Timestamp(
                study.backtests[label].periods["period_end_date"].iloc[-1]
            ).date()
        ),
    }


def target_provenance_summary(provenance):
    """Translate machine-readable target provenance into compact UI language."""
    if provenance["signal_as_of"] == "not_applicable_no_tactical_signal":
        signal_text = "No tactical signal"
    elif provenance["signal_as_of"] == "not_applicable_user_entered_mix":
        signal_text = "User-entered mix; no tactical signal"
    else:
        signal_text = f"Signal as of {provenance['signal_as_of']}"

    if provenance["execution_status"] == "pending_next_trading_close":
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
        if key != "displayed_history_through"
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
    if label in study.allocation_results:
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
                "reason": descriptions[label],
            }
            for ticker in study.selected_etfs
        ]
        if label == CASH_LABEL_SHORT:
            reason = "This comparison intentionally holds 100% analytical cash."
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
                hovertemplate=(
                    "%{x|%b %Y}<br>%{y:.1%}<extra>%{fullData.name}</extra>"
                    if percent
                    else "%{x|%b %Y}<br>%{y:,.2f}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(
        title=title,
        height=390 if not percent else 280,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,250,240,0.45)",
        legend=dict(orientation="h"),
        yaxis=dict(tickformat=".0%" if percent else ",.2f"),
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch")


def _allocation_chart(study, label):
    allocations = allocation_chart_data(study, label)
    figure = go.Figure()
    for asset in allocations.columns:
        display = CASH_LABEL if asset == CASH_ASSET else asset
        figure.add_trace(
            go.Scatter(
                x=allocations.index,
                y=allocations[asset],
                mode="lines",
                name=display,
                stackgroup="allocation",
                hovertemplate="%{x|%b %Y}<br>%{y:.1%}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,250,240,0.45)",
        legend=dict(orientation="h"),
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch")


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
        "Enter the portfolio you hold now, including cash. Values are used exactly "
        "as entered—they are not normalized and do not change strategy results. "
        "They only add Current Mix and enable the Portfolio Lab ticket. Removed ETF "
        "weight moves to cash; a newly selected ETF starts at 0%."
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
    if funded.empty:
        allocation_text = "No ETF receives weight."
    else:
        allocation_text = "ETF weights: " + ", ".join(
            f"{asset} {weight:.2%}" for asset, weight in funded.items()
        ) + "."
    return (
        f"This proposal allocates {etf_weight:.2%} to ETFs and {cash_weight:.2%} "
        f"to analytical cash ({total:.2%} total). {allocation_text}"
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


def render_workbench(bundle_path=DEFAULT_BUNDLE_PATH):
    """Render the isolated first-tab workbench and populate Portfolio Lab state."""
    st.markdown("## ETF Allocation Workbench")
    st.write(
        "Build and compare a simple ETF portfolio, see what each approach would "
        "hold now, and turn one proposal into a rebalance checklist."
    )
    guide = st.columns(3)
    guide[0].info("**1 · Choose ETFs**\n\nSelect one to eight investments to study.")
    guide[1].info("**2 · Compare approaches**\n\nReview the proposed mix and historical results.")
    guide[2].info(
        "**3 · Rebalance (optional)**\n\nEnter current weights totaling 100%, then send a proposal to Portfolio Lab."
    )
    try:
        bundle = load_cached_bundle(bundle_path)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        st.error(f"Workbench unavailable: validated local data could not be loaded ({exc}).")
        st.info("The ML research tabs remain available.")
        st.session_state.pop("portfolio_lab_transfer", None)
        return None

    freshness = bundle.freshness(as_of=_test_as_of())
    status_columns = st.columns(4)
    status_columns[0].metric("Bundle status", freshness["status"].title())
    status_columns[1].metric("ETF prices through", bundle.manifest["price_data_as_of"])
    status_columns[2].metric("Cash rates through", bundle.manifest["cash_rate_as_of"])
    status_columns[3].metric("Latest complete month", bundle.manifest["last_complete_month"])
    if freshness["status"] == "warning":
        st.warning(f"Workbench data warning: {freshness['reason']}.")
    elif freshness["status"] == "disabled":
        st.error(f"Workbench disabled: {freshness['reason']}.")
        st.info("The ML research tabs remain available.")
        st.session_state.pop("portfolio_lab_transfer", None)
        return None

    labels = {
        row.ticker: f"{row.ticker} — {row.role}"
        for row in bundle.instruments.itertuples(index=False)
    }
    selected = st.multiselect(
        "Curated ETFs (select 1–8)",
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

    controls = st.columns([1, 1, 2])
    transaction_cost_bps = controls[0].number_input(
        "Transaction cost (bps)",
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
        full_study = build_workbench_study(
            bundle,
            selected,
            transaction_cost_bps=transaction_cost_bps,
            current_weights=current_weights if current_valid else None,
            allocation_results=allocation_results,
        )
    except (ValueError, ArithmeticError) as exc:
        st.error(f"This ETF set cannot be calculated: {exc}")
        st.session_state.pop("portfolio_lab_transfer", None)
        return None

    default_start = max(
        full_study.earliest_execution,
        full_study.latest_period_end - pd.DateOffset(years=10),
    )
    date_range = controls[1].date_input(
        "Historical range",
        value=(default_start.date(), full_study.latest_period_end.date()),
        min_value=full_study.earliest_execution.date(),
        max_value=full_study.latest_period_end.date(),
        key="wb_date_range",
    )
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Choose both a start and end date.")
        return None
    start, end = date_range
    st.caption(
        "Historical-range rule: every selected range restarts at $1 in analytical "
        "cash. The first move into ETFs counts toward turnover and transaction cost. "
        "Changing this range does not change the latest proposed target."
    )

    options = available_comparisons(selected, current_valid)
    selection_key = "|".join(selected)
    saved_by_selection = dict(
        st.session_state.get("wb_comparisons_by_selection", {})
    )
    stored = [
        item for item in saved_by_selection.get(selection_key, options) if item in options
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
    comparisons = controls[2].multiselect(
        "Comparison series",
        options=options,
        key=comparison_widget_key,
        **comparison_kwargs,
    )
    saved_by_selection[selection_key] = list(comparisons)
    st.session_state["wb_comparisons_by_selection"] = saved_by_selection
    if not comparisons:
        st.warning("Select at least one comparison series.")
        return None
    if len(selected) == 1:
        st.info(
            f"With one ETF, compare three distinct choices: {BUY_HOLD_LABEL} keeps "
            f"holding the ETF, {VOL_TREND_LABEL} switches between that ETF and cash, "
            f"and {CASH_LABEL_SHORT} stays in analytical cash. Duplicate fully "
            "invested lines are hidden."
        )
    if current_valid:
        st.caption(
            "Current Mix is a hypothetical constant target reset at every monthly "
            "rebalance; it is not a reconstruction of actual holdings history. To "
            "plot the weights you entered, add Current Mix — monthly rebalanced under "
            "Comparison series."
        )
    cap = position_cap(len(selected))
    equal_share = 1.0 / len(selected)
    st.caption(
        f"{VOL_BALANCED_LABEL} favors steadier ETFs. {VOL_TREND_LABEL} uses the "
        "same method but gives 0% to ETFs below their long-term trend. Passing "
        "ETFs are reweighted within the position limit, and any amount that cannot "
        "be assigned goes to analytical cash. The other lines are reference portfolios."
    )
    with st.expander("How the approaches, timing, and position limit work"):
        st.markdown(
            f"""
            - **{VOL_BALANCED_LABEL}:** gives lower-volatility ETFs more weight using
              the previous 126 trading days. It does not use correlations, trend, or
              a return forecast.
            - **{VOL_TREND_LABEL}:** first requires an ETF's completed month-end price
              to be strictly above its trailing 10-month average, then applies the
              same inverse-volatility weighting. ETFs that fail receive 0%; passing
              ETFs are reweighted subject to the position limit, and any unassigned
              amount goes to analytical cash.
            - **{EQUAL_WEIGHT_LABEL}:** gives every selected ETF the same weight and
              resets it monthly; it uses neither volatility nor trend.
            - **{CURRENT_MIX_LABEL}:** resets the exact weights you entered each month
              for comparison. It does not reconstruct your actual transaction history.
            - **Timing:** completed month-end information sets the target for the next
              trading close, which is held until the following monthly rebalance.
            - **Position limit:** No ETF can receive more than 150% of its equal-weight
              share. With {len(selected)} selected ETF{'s' if len(selected) != 1 else ''},
              equal weight is {equal_share:.1%} and the maximum is {cap:.1%}.

            These rules look backward at recorded prices. They are not forecasts and
            do not promise better future returns.
            """
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
        )
    except (ValueError, ArithmeticError) as exc:
        st.warning(f"Selected range is unavailable: {exc}")
        return None

    target_options = [
        item for item in options if item != CURRENT_MIX_LABEL
    ]
    stored_target = st.session_state.get("wb_authoritative_target", VOL_TREND_LABEL)
    if stored_target not in target_options:
        stored_target = VOL_TREND_LABEL
        if "wb_authoritative_target" in st.session_state:
            st.session_state["wb_authoritative_target"] = stored_target
    target_kwargs = (
        {} if "wb_authoritative_target" in st.session_state else {
            "index": target_options.index(stored_target)
        }
    )
    authoritative = st.selectbox(
        "Choose a proposed portfolio",
        options=target_options,
        key="wb_authoritative_target",
        help="This choice controls the proposal shown here and sent to Portfolio Lab.",
        **target_kwargs,
    )
    target = study.latest_targets[authoritative]
    provenance = target_provenance(bundle, study, authoritative)
    st.markdown("### Latest proposed portfolio")
    st.write(proposed_target_summary(target))
    st.caption(target_provenance_summary(provenance))
    if provenance["execution_status"] == "pending_next_trading_close":
        st.warning(
            "Proposed, not executed: this target uses the latest completed signal "
            "and is waiting for the next monthly rebalance trading close."
        )
    if authoritative in (VOL_BALANCED_LABEL, VOL_TREND_LABEL):
        st.caption(study.allocation_results[authoritative].policy.cap_explanation)
    with st.expander("Target dates and data details"):
        st.markdown(
            f"""
            - **Latest-target status:** {target_provenance_summary(provenance)}
            - **Displayed historical results through:** {provenance['displayed_history_through']}
            - **Price artifact through:** {provenance['price_data_as_of']}
            - **Artifact generated:** {provenance['artifact_generated_at_utc']}
            - **Rule version:** `{provenance['policy_version']}`

            The latest proposal always uses the full validated artifact. The historical
            chart endpoint above follows your selected date range and may be earlier.
            """
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

    if current_valid:
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
            help="Enter current weights that total exactly 100% before transferring.",
        )
        if current_error:
            st.info("Next step: adjust the current-weight total to exactly 100%.")
        else:
            st.info(
                "Next step: turn on ‘Enter current weights,’ include cash, and make "
                "the total exactly 100%."
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
        "Arrows show the generally preferred direction, not a guarantee of quality. "
        "Return, volatility, turnover, and cost drag are annualized. One-way turnover "
        "measures how much of the portfolio is replaced; cost drag is the annualized "
        "difference between gross and after-cost performance. For drawdown and the "
        "worst month, a result closer to 0% is less severe. A dash means the metric "
        "cannot be calculated for the selected history."
    )
    st.dataframe(
        metrics.style.format(
            {
                "Annualized return ↑": "{:.2%}",
                "Annualized volatility ↓": "{:.2%}",
                "Sharpe above cash ↑": "{:.2f}",
                "Maximum drawdown ↑": "{:.2%}",
                "Return / drawdown ↑": "{:.2f}",
                "Worst month ↑": "{:.2%}",
                "Annualized one-way turnover ↓": "{:.2%}",
                "Annualized cost drag ↓": "{:.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

    st.markdown("### How the proposed allocation changed")
    st.caption(
        "Weights are plotted on rebalance (execution) dates. The table separates "
        "the earlier signal date, rebalance date, and end of the resulting holding period."
    )
    _allocation_chart(study, authoritative)
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

    st.markdown(
        f"**{CASH_LABEL}** uses official EFFR before 2018-04-02 and official "
        "SOFR from 2018-04-02. It is an analytical, non-investable series. BIL "
        "remains a separately selectable ETF; analytical cash is never a ticker "
        "or security order."
    )
    return study


def render_portfolio_lab():
    """Render the ticket from the exact workbench session target."""
    st.markdown("## Portfolio Lab")
    transfer = st.session_state.get("portfolio_lab_transfer")
    if not transfer:
        st.info(
            "No proposal has been sent yet. In ETF Allocation Workbench: (1) choose "
            "ETFs, (2) turn on Enter current weights and make the total exactly 100%, "
            "and (3) choose a proposed portfolio and select Send to Portfolio Lab."
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
            - **Artifact generated:** {transfer['artifact_generated_at_utc']}
            - **Rule version:** `{transfer['policy_version']}`
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
