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
    """Return the exact displayed authoritative allocation and accounting rows."""
    result = study.backtests[label]
    frame = result.target_weights.reset_index().copy()
    frame.insert(0, "strategy", label)
    frame["signal_date"] = result.periods["signal_date"].to_numpy()
    frame["period_end_date"] = result.periods["period_end_date"].to_numpy()
    frame["turnover"] = result.periods["turnover"].to_numpy(dtype=float)
    frame["transaction_cost"] = result.periods["transaction_cost"].to_numpy(dtype=float)
    return frame


def target_provenance(bundle, study, label):
    """Return authoritative timing and artifact provenance for a latest target."""
    accounting_schedule = study.allocation_results[VOL_BALANCED_LABEL]
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
        "historical_accounting_schedule_as_of": str(
            pd.Timestamp(accounting_schedule.latest_signal_date).date()
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
    return pd.DataFrame(
        {
            "strategy": label,
            **{key: value for key, value in provenance.items()},
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
            raw = (
                np.nan
                if equal_weight
                else float(row["unfiltered_inverse_vol_weight"])
            )
            final = float(row["final_target_weight"])
            rows.append(
                {
                    "asset": ticker,
                    "role": registry.loc[ticker, "role"],
                    "trend": "Not used" if equal_weight else row["trend_status"],
                    "trailing_volatility": row["trailing_volatility"],
                    "raw_weight": raw,
                    "filtered_raw_weight": (
                        np.nan if equal_weight else row["filtered_raw_weight"]
                    ),
                    "final_weight": final,
                    "change_vs_uncapped_inverse_vol": (
                        np.nan if equal_weight else final - raw
                    ),
                    "reason": row["eligibility_reason"],
                }
            )
        reason = (
            "Residual created by trend exclusions and the adaptive ETF cap"
            if target.loc[CASH_ASSET] > 0.0
            else "No residual cash target"
        )
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
        reason = descriptions[label]
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
                "Annualized return": metrics["annualized_return"],
                "Volatility": metrics["annualized_volatility"],
                "Excess-return Sharpe": metrics["excess_return_sharpe"],
                "Max drawdown": metrics["max_drawdown"],
                "Calmar": metrics["calmar_ratio"],
                "Worst month": metrics["worst_month"],
                "Annual turnover": metrics["annualized_turnover"],
                "Cost drag": metrics["transaction_cost_drag"],
            }
        )
    return pd.DataFrame(rows).set_index("Series")


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
    result = study.backtests[label]
    figure = go.Figure()
    for asset in result.target_weights.columns:
        display = CASH_LABEL if asset == CASH_ASSET else asset
        figure.add_trace(
            go.Scatter(
                x=result.periods["signal_date"],
                y=result.target_weights[asset],
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
        help="Weights are validated as entered and are never normalized.",
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
        "Removed ETF weight moves to cash; a newly selected ETF starts at 0%. "
        "The total must equal 100%."
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
    try:
        valid = validate_portfolio_weights(weights, selected)
    except ValueError as exc:
        return weights, str(exc)
    return valid, None


def render_workbench(bundle_path=DEFAULT_BUNDLE_PATH):
    """Render the isolated first-tab workbench and populate Portfolio Lab state."""
    st.markdown("## ETF Allocation Workbench")
    st.caption(
        "Research fixed allocation policies across a curated ETF set. No strategy "
        "parameters are exposed or fitted in the browser."
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
            "With one ETF, Volatility Balanced, Equal Weight, and Buy & Hold can "
            "produce the same fully invested path. The duplicate tactical curves "
            "are hidden and represented once as Buy & Hold."
        )
    if current_valid:
        st.caption(
            "Current Mix is a hypothetical constant target reset at every monthly "
            "rebalance; it is not a reconstruction of actual holdings history."
        )
    st.caption(
        "Policy guide — Volatility Balanced: inverse volatility with the fixed "
        "position cap. + Trend: also excludes ETFs below their 10-month trend. "
        "Equal Weight: 1/N, reset monthly. Current Mix: entered weights, reset "
        f"monthly. {CASH_LABEL}: analytical overnight-rate comparison."
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

    _line_chart(study, comparisons, "net_equity", "Equity curves (start = 1.00)")
    _line_chart(study, comparisons, "drawdown", "Drawdowns", percent=True)

    metrics = _format_metrics(study, comparisons)
    st.markdown("### Concise metrics")
    st.dataframe(
        metrics.style.format(
            {
                "Annualized return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Excess-return Sharpe": "{:.2f}",
                "Max drawdown": "{:.2%}",
                "Calmar": "{:.2f}",
                "Worst month": "{:.2%}",
                "Annual turnover": "{:.2%}",
                "Cost drag": "{:.2%}",
            },
            na_rep="—",
        ),
        width="stretch",
    )

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
        "Authoritative rebalance target",
        options=target_options,
        key="wb_authoritative_target",
        help="This single choice controls the latest target, transfer, and Portfolio Lab ticket.",
        **target_kwargs,
    )
    target = study.latest_targets[authoritative]
    provenance = target_provenance(bundle, study, authoritative)
    scope_text = (
        "The proposed target uses the full artifact and does not change with the "
        "historical chart range."
        if authoritative in study.allocation_results
        else "The target is constant; historical results use the committed accounting "
        f"schedule through {provenance['historical_accounting_schedule_as_of']}."
    )
    st.markdown("### Proposed current target")
    st.caption(
        target_provenance_summary(provenance)
        + f" · Artifact generated {provenance['artifact_generated_at_utc']} · "
        f"Prices through {provenance['price_data_as_of']} · "
        f"Policy {provenance['policy_version']}. {scope_text}"
    )
    if authoritative in study.allocation_results:
        st.caption(study.allocation_results[authoritative].policy.cap_explanation)
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
            "historical_accounting_schedule_as_of",
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
            st.success("Exact proposed target and current weights are synced to Portfolio Lab.")
    else:
        st.session_state.pop("portfolio_lab_transfer", None)
        st.button(
            "Send proposed target to Portfolio Lab",
            disabled=True,
            key="wb_send_to_portfolio_lab_disabled",
            help="Enter valid current weights before transferring a target.",
        )

    st.markdown("### Why this weight?")
    explanation = why_this_weight(bundle, study, authoritative)
    explanation_display = explanation.rename(
        columns={
            "asset": "Asset",
            "role": "Role",
            "trend": "Trend status",
            "trailing_volatility": "Trailing volatility",
            "raw_weight": "Uncapped inverse-vol weight",
            "filtered_raw_weight": "Post-filter raw weight",
            "final_weight": "Proposed target",
            "change_vs_uncapped_inverse_vol": "Change vs uncapped inverse-vol",
            "reason": "Reason",
        }
    )
    st.dataframe(
        explanation_display.style.format(
            {
                "Trailing volatility": "{:.2%}",
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

    history = study.backtests[authoritative]
    st.markdown("### Allocation, turnover, and cost history")
    _allocation_chart(study, authoritative)
    allocation_history = history.target_weights.copy()
    allocation_history[CASH_LABEL] = allocation_history.pop(CASH_ASSET)
    allocation_history.insert(0, "Signal date", history.periods["signal_date"])
    allocation_history["One-way turnover"] = history.periods["turnover"]
    allocation_history[
        "Transaction cost (% of portfolio at rebalance)"
    ] = history.periods["cost_rate"]
    percentage_columns = [
        *selected,
        CASH_LABEL,
        "One-way turnover",
        "Transaction cost (% of portfolio at rebalance)",
    ]
    st.dataframe(
        allocation_history.tail(24).style.format(
            {column: "{:.2%}" for column in percentage_columns}
        ),
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
    allocation_download = allocation_history_download(study, authoritative)
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
    """Render the ticket from the exact authoritative workbench session target."""
    st.markdown("## Portfolio Lab")
    transfer = st.session_state.get("portfolio_lab_transfer")
    if not transfer:
        st.info(
            "Enter valid current weights in ETF Allocation Workbench to sync its "
            "authoritative latest target and create a ticket."
        )
        return

    selected = tuple(transfer["selected_etfs"])
    current = pd.Series(transfer["current_weights"], dtype=float).reindex(
        [*selected, CASH_ASSET]
    )
    target = pd.Series(transfer["target_weights"], dtype=float).reindex(
        [*selected, CASH_ASSET]
    )
    st.markdown("### Exact synced proposed target")
    st.caption(
        f"{transfer['strategy']} · "
        + target_provenance_summary(transfer)
        + f" · artifact generated {transfer['artifact_generated_at_utc']} · "
        f"prices through {transfer['price_data_as_of']} · "
        f"policy {transfer['policy_version']}. These are the exact synced values "
        "used to build the analytical ticket below."
    )
    synced_target = pd.DataFrame(
        {
            "Asset": [CASH_LABEL if asset == CASH_ASSET else asset for asset in target.index],
            "Synced proposed target": target.to_numpy(dtype=float),
            "Type": [
                "Analytical cash" if asset == CASH_ASSET else "Tradeable ETF"
                for asset in target.index
            ],
        }
    )
    st.dataframe(
        synced_target.style.format({"Synced proposed target": "{:.2%}"}),
        hide_index=True,
        width="stretch",
    )
    if transfer["execution_status"] == "pending_next_trading_close":
        st.warning(
            "This proposed target is awaiting the next rebalance trading close. "
            "The ticket is analytical only and does not represent executable orders."
        )
    portfolio_value = st.number_input(
        "Portfolio value for optional dollar tickets",
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

    metrics = st.columns(3)
    metrics[0].metric("One-way turnover", f"{ticket.turnover:.2%}")
    metrics[1].metric("Estimated cost rate", f"{ticket.estimated_cost_rate:.3%}")
    metrics[2].metric("Estimated cost", f"${ticket.estimated_cost_amount:,.2f}")
    st.markdown("### ETF security orders")
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
        "orders and the separate analytical cash balance. No orders are placed."
    )
