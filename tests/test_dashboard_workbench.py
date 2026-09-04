import os
import shutil
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from backtest.engine import CASH_ASSET, run_backtest
from dashboard.workbench import (
    BUY_HOLD_LABEL,
    CASH_LABEL_SHORT,
    CURRENT_MIX_LABEL,
    DEFAULT_SELECTION,
    EQUAL_WEIGHT_LABEL,
    VOL_BALANCED_LABEL,
    VOL_FORECAST_LABEL,
    VOL_TREND_LABEL,
    SPY_REFERENCE_LABEL,
    _format_metrics,
    allocation_chart_data,
    allocation_history_download,
    available_comparisons,
    build_workbench_study,
    bundle_fingerprint,
    clear_workbench_caches,
    current_weight_status,
    forecast_check_summary,
    historical_download,
    holding_period_returns,
    latest_target_download,
    load_cached_bundle,
    load_cached_timesfm_bundle,
    proposed_target_summary,
    standard_target_freshness,
    target_provenance,
    target_provenance_summary,
    ticket_action_summary,
    ticket_dollar_summary,
    timesfm_bundle_fingerprint,
    why_this_weight,
)
from data.cash import CASH_LABEL
from data.timesfm import load_timesfm_bundle
from data.workbench import DEFAULT_BUNDLE_PATH, load_workbench_bundle
from data.timesfm import DEFAULT_BUNDLE_PATH as DEFAULT_TIMESFM_BUNDLE_PATH
from portfolio.rebalance import build_rebalance_ticket
from strategies.forecast import generate_forecast_allocation_targets


@pytest.fixture(scope="module")
def bundle():
    return load_workbench_bundle()


def test_content_cache_detects_same_size_same_mtime_bundle_corruption(tmp_path):
    copied = tmp_path / "workbench"
    shutil.copytree(DEFAULT_BUNDLE_PATH, copied)
    prices_path = copied / "adjusted_close.csv"
    clear_workbench_caches()
    try:
        load_cached_bundle(copied)
        initial_fingerprint = bundle_fingerprint(copied)
        initial_stat = prices_path.stat()
        contents = bytearray(prices_path.read_bytes())
        position = contents.find(b"1993")
        assert position >= 0
        contents[position + 3] = ord("4")
        prices_path.write_bytes(contents)
        os.utime(
            prices_path,
            ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
        )

        mutated_stat = prices_path.stat()
        assert mutated_stat.st_size == initial_stat.st_size
        assert mutated_stat.st_mtime_ns == initial_stat.st_mtime_ns
        assert bundle_fingerprint(copied) != initial_fingerprint
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_cached_bundle(copied)
    finally:
        clear_workbench_caches()


def test_timesfm_content_cache_invalidates_same_size_same_mtime_corruption(tmp_path):
    copied = tmp_path / "timesfm"
    shutil.copytree(DEFAULT_TIMESFM_BUNDLE_PATH, copied)
    signals_path = copied / "forecast_signals.csv"
    clear_workbench_caches()
    try:
        load_cached_timesfm_bundle(copied, DEFAULT_BUNDLE_PATH)
        initial_fingerprint = timesfm_bundle_fingerprint(copied)
        initial_stat = signals_path.stat()
        contents = bytearray(signals_path.read_bytes())
        position = contents.find(b"timesfm3_multivariate")
        assert position >= 0
        contents[position + 9] = ord("x")
        signals_path.write_bytes(contents)
        os.utime(
            signals_path,
            ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
        )

        assert signals_path.stat().st_size == initial_stat.st_size
        assert signals_path.stat().st_mtime_ns == initial_stat.st_mtime_ns
        assert timesfm_bundle_fingerprint(copied) != initial_fingerprint
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_cached_timesfm_bundle(copied, DEFAULT_BUNDLE_PATH)
    finally:
        clear_workbench_caches()


def test_committed_timesfm_bundle_is_historical_only_after_late_generation():
    bundle = load_cached_timesfm_bundle()
    freshness = bundle.freshness(as_of="2026-09-05")
    assert freshness["historical_available"] is True
    assert freshness["latest_target_status"] == "disabled"
    assert "after its execution cutoff" in freshness["reason"]


def test_standard_tactical_target_uses_intended_close_without_future_price_row(bundle):
    study = build_workbench_study(bundle, DEFAULT_SELECTION)
    allocation = study.allocation_results[VOL_BALANCED_LABEL]

    late = standard_target_freshness(bundle, allocation, as_of="2026-09-05")
    assert late["latest_target_status"] == "disabled"
    assert late["latest_signal_date"] == "2026-08-31"
    assert late["latest_execution_date"] == "2026-09-01"
    assert late["latest_execution_cutoff_utc"] == "2026-09-01T20:00:00Z"
    assert late["latest_period_end_date"] == "2026-10-01"
    assert late["latest_period_end_cutoff_utc"] == "2026-10-01T20:00:00Z"
    assert "after its intended execution close" in late["reason"]

    timely_manifest = deepcopy(bundle.manifest)
    timely_manifest["generated_at_utc"] = "2026-09-01T19:00:00Z"
    timely_bundle = replace(bundle, manifest=timely_manifest)
    before_close = standard_target_freshness(
        timely_bundle, allocation, as_of="2026-09-01T19:30:00Z"
    )
    after_close = standard_target_freshness(
        timely_bundle, allocation, as_of="2026-09-01T20:01:00Z"
    )
    assert before_close["latest_target_status"] == "scheduled"
    assert "execution close" in before_close["reason"]
    assert after_close["latest_target_status"] == "current"
    before_period_end_close = standard_target_freshness(
        timely_bundle, allocation, as_of="2026-10-01T19:59:00Z"
    )
    after_period_end_close = standard_target_freshness(
        timely_bundle, allocation, as_of="2026-10-01T20:01:00Z"
    )
    assert before_period_end_close["latest_target_status"] == "current"
    assert after_period_end_close["latest_target_status"] == "disabled"


def test_selected_comparisons_control_forecast_history_alignment(bundle):
    forecasts = load_timesfm_bundle()
    forecast_result = generate_forecast_allocation_targets(
        bundle.adjusted_close.loc[:, DEFAULT_SELECTION],
        DEFAULT_SELECTION,
        forecasts.signals,
        as_of=bundle.signal_as_of,
    )
    native = build_workbench_study(
        bundle,
        DEFAULT_SELECTION,
        forecast_result=forecast_result,
        forecast_manifest=forecasts.manifest,
        forecast_freshness=forecasts.freshness(as_of="2026-09-05"),
        comparison_labels=[VOL_BALANCED_LABEL, CASH_LABEL_SHORT],
    )
    aligned = build_workbench_study(
        bundle,
        DEFAULT_SELECTION,
        forecast_result=forecast_result,
        forecast_manifest=forecasts.manifest,
        forecast_freshness=forecasts.freshness(as_of="2026-09-05"),
        comparison_labels=[VOL_BALANCED_LABEL, VOL_FORECAST_LABEL],
    )

    assert (
        native.backtests[VOL_BALANCED_LABEL].periods.index[0]
        < native.backtests[VOL_FORECAST_LABEL].periods.index[0]
    )
    assert native.backtests[CASH_LABEL_SHORT].periods.index.equals(
        native.backtests[VOL_BALANCED_LABEL].periods.index
    )
    assert aligned.backtests[VOL_BALANCED_LABEL].periods.index.equals(
        aligned.backtests[VOL_FORECAST_LABEL].periods.index
    )


def test_forecast_provenance_separates_model_generation_and_bundle_refresh(bundle):
    forecasts = load_timesfm_bundle()
    result = generate_forecast_allocation_targets(
        bundle.adjusted_close.loc[:, DEFAULT_SELECTION],
        DEFAULT_SELECTION,
        forecasts.signals,
        as_of=bundle.signal_as_of,
    )
    freshness = forecasts.freshness(as_of="2026-09-05")
    study = build_workbench_study(
        bundle,
        DEFAULT_SELECTION,
        forecast_result=result,
        forecast_manifest=forecasts.manifest,
        forecast_freshness=freshness,
        comparison_labels=[VOL_FORECAST_LABEL, CASH_LABEL_SHORT],
    )
    provenance = target_provenance(bundle, study, VOL_FORECAST_LABEL)
    generation = next(
        item
        for item in forecasts.manifest["model_generations"]
        if item["id"] == result.latest_model_generation_id
    )

    assert provenance["model_generation_id"] == generation["id"]
    assert provenance["model_generated_at_utc"] == generation["generated_at_utc"]
    assert provenance["execution_cutoff_utc"] == freshness[
        "latest_execution_cutoff_utc"
    ]
    assert provenance["workbench_bundle_refreshed_at_utc"] == bundle.manifest[
        "generated_at_utc"
    ]
    assert provenance["forecast_bundle_refreshed_at_utc"] == forecasts.manifest[
        "bundle_refresh"
    ]["generated_at_utc"]
    download = latest_target_download(bundle, study, VOL_FORECAST_LABEL)
    assert "artifact_generated_at_utc" not in download.columns
    assert download["model_generation_id"].unique().tolist() == [generation["id"]]


def test_forecast_check_summary_is_compact_and_includes_price_mase():
    checks = forecast_check_summary(load_timesfm_bundle().metrics)
    assert checks.columns.tolist() == [
        "Model",
        "Forecasts",
        "Return MAE ↓",
        "Period-end price MASE ↓",
        "Direction accuracy ↑",
        "Pinball loss ↓",
        "q10–q90 price coverage",
    ]
    assert "TimesFM-3 — all 14 ETFs together" in checks["Model"].tolist()
    baseline = checks.loc[checks["Model"] == "Last value (point baseline)"].iloc[0]
    assert pd.isna(baseline["Pinball loss ↓"])


def test_freshness_policy_allows_one_month_warning_and_disables_two_or_future(bundle):
    assert bundle.freshness(as_of="2026-09-15")["status"] == "current"
    assert bundle.freshness(as_of="2026-10-15")["status"] == "warning"
    assert bundle.freshness(as_of="2026-11-15")["status"] == "disabled"
    future = bundle.freshness(as_of="2026-08-01T12:00:00Z")
    assert future["status"] == "disabled"
    assert "future" in future["reason"]


@pytest.mark.parametrize(
    "selected",
    [
        ("SPY",),
        ("SPY", "IEF"),
        DEFAULT_SELECTION,
        ("AGG", "SHY", "IEF", "TLT"),
        ("SPY", "IWM", "EFA", "EEM", "AGG", "BIL", "IEF", "GLD"),
    ],
    ids=["one", "two", "default", "fixed-income", "eight"],
)
def test_supported_selected_sets_build_common_engine_results(bundle, selected):
    study = build_workbench_study(bundle, selected, transaction_cost_bps=7)

    required = {
        VOL_BALANCED_LABEL,
        VOL_TREND_LABEL,
        EQUAL_WEIGHT_LABEL,
        CASH_LABEL_SHORT,
    }
    assert required.issubset(study.backtests)
    assert (BUY_HOLD_LABEL in study.backtests) == (len(selected) == 1)
    for label, result in study.backtests.items():
        assert not result.periods.empty, label
        if label == SPY_REFERENCE_LABEL and "SPY" not in selected:
            assert result.target_weights.columns.tolist() == ["SPY", CASH_ASSET]
        else:
            assert result.target_weights.columns.tolist() == [*selected, CASH_ASSET]
        assert result.target_weights.sum(axis=1).to_numpy() == pytest.approx(1.0)
        assert result.periods["net_equity"].iloc[-1] > 0


def test_workbench_calculation_exactly_agrees_with_common_engine(bundle):
    study = build_workbench_study(bundle, DEFAULT_SELECTION, transaction_cost_bps=11)
    allocation = study.allocation_results[VOL_TREND_LABEL]
    assets, cash = holding_period_returns(bundle, allocation.schedule, DEFAULT_SELECTION)
    direct = run_backtest(
        allocation.schedule, assets, cash, transaction_cost_bps=11
    )

    pd.testing.assert_frame_equal(
        study.backtests[VOL_TREND_LABEL].periods, direct.periods
    )
    pd.testing.assert_frame_equal(
        study.backtests[VOL_TREND_LABEL].trades, direct.trades
    )


def test_selected_range_restarts_from_cash_and_latest_target_stays_full_artifact(bundle):
    full = build_workbench_study(bundle, DEFAULT_SELECTION, transaction_cost_bps=20)
    start = full.backtests[VOL_TREND_LABEL].periods.index[-24]
    end = full.latest_period_end
    selected = build_workbench_study(
        bundle,
        DEFAULT_SELECTION,
        transaction_cost_bps=20,
        start=start,
        end=end,
    )

    first = selected.backtests[VOL_TREND_LABEL].periods.iloc[0]
    first_target = selected.backtests[VOL_TREND_LABEL].target_weights.iloc[0]
    expected_turnover = 0.5 * (
        first_target.drop(CASH_ASSET).abs().sum()
        + abs(first_target[CASH_ASSET] - 1.0)
    )
    assert first["turnover"] == pytest.approx(expected_turnover)
    assert first["cost_rate"] == pytest.approx(expected_turnover * 20 / 10000)
    pd.testing.assert_series_equal(
        selected.latest_targets[VOL_TREND_LABEL],
        full.latest_targets[VOL_TREND_LABEL],
    )


def test_current_mix_requires_valid_unnormalized_current_weights(bundle):
    assert CURRENT_MIX_LABEL not in build_workbench_study(
        bundle, DEFAULT_SELECTION
    ).backtests
    valid = pd.Series({"SPY": 0.2, "IEF": 0.3, "GLD": 0.1, CASH_ASSET: 0.4})
    study = build_workbench_study(
        bundle, DEFAULT_SELECTION, current_weights=valid
    )
    assert CURRENT_MIX_LABEL in study.backtests
    assert study.latest_targets[CURRENT_MIX_LABEL].to_dict() == pytest.approx(
        valid.to_dict()
    )
    with pytest.raises(ValueError, match="sum to 100%"):
        build_workbench_study(
            bundle,
            DEFAULT_SELECTION,
            current_weights={"SPY": 0.2, "IEF": 0.3, "GLD": 0.1, CASH_ASSET: 0.1},
        )


@pytest.mark.parametrize(
    ("weights", "status", "message"),
    [
        ({"SPY": 0.9}, "under", "Add 10.00 percentage points"),
        ({"SPY": 1.0}, "ready", "Ready to compare"),
        ({"SPY": 1.1}, "over", "Remove 10.00 percentage points"),
    ],
)
def test_current_weight_status_is_actionable(weights, status, message):
    actual_status, actual_message = current_weight_status(weights)
    assert actual_status == status
    assert message in actual_message


def test_comparison_options_prevent_incompatible_and_duplicate_series():
    one = available_comparisons(["SPY"], current_weights_valid=False)
    assert one == [VOL_TREND_LABEL, BUY_HOLD_LABEL, CASH_LABEL_SHORT]
    assert VOL_BALANCED_LABEL not in one
    assert EQUAL_WEIGHT_LABEL not in one
    assert CURRENT_MIX_LABEL not in one
    assert BUY_HOLD_LABEL not in available_comparisons(["SPY", "IEF"])
    assert CURRENT_MIX_LABEL in available_comparisons(
        ["SPY", "IEF"], current_weights_valid=True
    )


def test_explanation_has_required_semantics_and_cash_label(bundle):
    study = build_workbench_study(bundle, DEFAULT_SELECTION)
    explanation = why_this_weight(bundle, study, EQUAL_WEIGHT_LABEL)

    assert explanation.columns.tolist() == [
        "asset",
        "role",
        "trend",
        "trailing_volatility",
        "raw_weight",
        "filtered_raw_weight",
        "final_weight",
        "change_vs_uncapped_inverse_vol",
        "median_forecast_return",
        "cash_hurdle",
        "forecast_status",
        "q10_period_end_price",
        "q50_period_end_price",
        "q90_period_end_price",
        "reason",
    ]
    etfs = explanation[explanation["asset"] != CASH_LABEL]
    assert (etfs["trend"] == "Not used").all()
    assert etfs["raw_weight"].isna().all()
    assert etfs["filtered_raw_weight"].isna().all()
    assert etfs["change_vs_uncapped_inverse_vol"].isna().all()
    assert explanation.iloc[-1]["asset"] == CASH_LABEL
    assert pd.isna(explanation.iloc[-1]["change_vs_uncapped_inverse_vol"])
    assert explanation["final_weight"].sum() == pytest.approx(1.0)
    assert not any("cash_displacement" in column for column in explanation.columns)

    trend = why_this_weight(bundle, study, VOL_TREND_LABEL)
    trend_etfs = trend[trend["asset"] != CASH_LABEL]
    expected_change = trend_etfs["final_weight"] - trend_etfs["raw_weight"]
    assert trend_etfs["change_vs_uncapped_inverse_vol"].to_numpy() == pytest.approx(
        expected_change.to_numpy()
    )
    assert trend["final_weight"].sum() == pytest.approx(1.0)

    balanced = why_this_weight(bundle, study, VOL_BALANCED_LABEL)
    balanced_etfs = balanced[balanced["asset"] != CASH_LABEL]
    assert (balanced_etfs["trend"] == "Not used").all()
    assert all("trend is not used" in reason.lower() or "position cap" in reason.lower()
               for reason in balanced_etfs["reason"])

    all_fail = build_workbench_study(bundle, ["AGG"])
    all_fail_explanation = why_this_weight(bundle, all_fail, VOL_TREND_LABEL)
    cash_reason = all_fail_explanation.loc[
        all_fail_explanation["asset"] == CASH_LABEL, "reason"
    ].item()
    assert "All selected ETFs failed" in cash_reason
    assert "100% analytical cash" in cash_reason


def test_forecast_explanation_status_and_all_cash_reason_reconcile(bundle):
    forecasts = load_timesfm_bundle()
    selected = DEFAULT_SELECTION
    signals = forecasts.signals.copy()
    mask = (signals["model_mode"] == "timesfm3_multivariate") & signals[
        "ticker"
    ].isin(selected)
    signals.loc[mask, "forecast_edge"] = -0.01
    signals.loc[mask, "forecast_holding_return"] = (
        signals.loc[mask, "cash_hurdle"] - 0.01
    )
    result = generate_forecast_allocation_targets(
        bundle.adjusted_close.loc[:, selected],
        selected,
        signals,
        as_of=bundle.signal_as_of,
    )
    study = build_workbench_study(
        bundle,
        selected,
        forecast_result=result,
        forecast_manifest=forecasts.manifest,
        forecast_freshness=forecasts.freshness(as_of="2026-09-05"),
        comparison_labels=[VOL_FORECAST_LABEL],
    )
    explanation = why_this_weight(bundle, study, VOL_FORECAST_LABEL)
    etfs = explanation.loc[explanation["asset"] != CASH_LABEL]
    cash = explanation.loc[explanation["asset"] == CASH_LABEL].iloc[0]

    assert set(etfs["forecast_status"]) == {"Held in cash"}
    assert (etfs["final_weight"] == 0.0).all()
    assert cash["final_weight"] == pytest.approx(1.0)
    assert "median forecast cleared cash" in cash["reason"]
    assert "100% analytical cash" in cash["reason"]


def test_downloads_exactly_reconcile_displayed_results_and_targets(bundle, monkeypatch):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-09-05")
    study = build_workbench_study(bundle, DEFAULT_SELECTION, transaction_cost_bps=9)
    labels = [VOL_TREND_LABEL, CASH_LABEL_SHORT]
    history = historical_download(study, labels)
    for label in labels:
        expected = study.backtests[label].periods.reset_index()
        actual = history.loc[history["series"] == label].drop(columns="series")
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)

    target = latest_target_download(bundle, study, VOL_TREND_LABEL)
    assert "displayed_history_through" not in target.columns
    assert target["target_weight"].sum() == pytest.approx(1.0)
    assert target.loc[target["asset"] == CASH_ASSET, "asset_type"].item() == "analytical_cash"
    assert target["signal_as_of"].nunique() == 1
    assert target["execution_status"].unique().tolist() == [
        "disabled"
    ]
    assert "artifact_generated_at_utc" not in target.columns
    assert target["workbench_bundle_refreshed_at_utc"].unique().tolist() == [
        bundle.manifest["generated_at_utc"]
    ]
    assert target["price_data_as_of"].unique().tolist() == [
        bundle.manifest["price_data_as_of"]
    ]
    assert target["policy_version"].unique().tolist() == ["allocation-policy-v1"]
    pd.testing.assert_series_equal(
        target.set_index("asset")["target_weight"],
        study.latest_targets[VOL_TREND_LABEL],
        check_names=False,
    )

    allocation = allocation_history_download(study, VOL_TREND_LABEL)
    assert allocation.columns[:4].tolist() == [
        "strategy",
        "rebalance_date",
        "signal_date",
        "holding_period_end",
    ]
    weight_columns = [*DEFAULT_SELECTION, CASH_ASSET]
    assert allocation[weight_columns].sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert allocation["turnover"].to_numpy() == pytest.approx(
        study.backtests[VOL_TREND_LABEL].periods["turnover"].to_numpy()
    )
    assert allocation["estimated_cost_rate"].to_numpy() == pytest.approx(
        study.backtests[VOL_TREND_LABEL].periods["cost_rate"].to_numpy()
    )
    assert allocation["transaction_cost_value"].to_numpy() == pytest.approx(
        study.backtests[VOL_TREND_LABEL].periods["transaction_cost"].to_numpy()
    )
    assert not np.allclose(
        allocation["estimated_cost_rate"].to_numpy(),
        allocation["transaction_cost_value"].to_numpy(),
    )
    plotted = allocation_chart_data(study, VOL_TREND_LABEL)
    assert plotted.index.name == "rebalance_date"
    assert plotted.index[0] == study.backtests[VOL_TREND_LABEL].target_weights.index[0]
    assert plotted.index[0] != study.backtests[VOL_TREND_LABEL].periods["signal_date"].iloc[0]


def test_target_provenance_keeps_cash_distinct_from_tactical_strategy(
    bundle, monkeypatch
):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-09-05")
    study = build_workbench_study(bundle, DEFAULT_SELECTION)
    strategy = target_provenance(bundle, study, VOL_TREND_LABEL)
    cash = target_provenance(bundle, study, CASH_LABEL_SHORT)

    assert strategy["execution_status"] == "disabled"
    assert strategy["execution_cutoff_utc"] == "2026-09-01T20:00:00Z"
    assert strategy["signal_as_of"] == str(
        study.allocation_results[VOL_TREND_LABEL].latest_signal_date.date()
    )
    assert cash["signal_as_of"] == "not_applicable_no_tactical_signal"
    assert cash["execution_status"] == "constant_target_effective_for_analytical_ticket"
    assert cash["policy_version"] == "analytical-cash-comparison-v1"
    assert cash["displayed_history_through"] == str(
        study.backtests[CASH_LABEL_SHORT].periods["period_end_date"].iloc[-1].date()
    )
    assert "historical_accounting_schedule_as_of" not in cash
    assert cash != target_provenance(bundle, study, VOL_BALANCED_LABEL)
    summary = target_provenance_summary(cash)
    assert "No tactical signal" in summary
    assert "Constant target" in summary
    assert "not_applicable" not in summary
    assert "None" not in summary
    download = latest_target_download(bundle, study, CASH_LABEL_SHORT)
    assert download["policy_version"].unique().tolist() == [
        "analytical-cash-comparison-v1"
    ]


def test_target_provenance_keeps_buy_hold_distinct_from_tactical_strategy(bundle):
    study = build_workbench_study(bundle, ["SPY"])
    buy_hold = target_provenance(bundle, study, BUY_HOLD_LABEL)

    assert buy_hold["signal_as_of"] == "not_applicable_no_tactical_signal"
    assert (
        buy_hold["execution_status"]
        == "constant_target_effective_for_analytical_ticket"
    )
    assert buy_hold["policy_version"] == "single-etf-buy-hold-v1"
    assert buy_hold["displayed_history_through"] == str(
        study.backtests[BUY_HOLD_LABEL].periods["period_end_date"].iloc[-1].date()
    )
    summary = target_provenance_summary(buy_hold)
    assert "No tactical signal" in summary
    assert "effective for the analytical ticket" in summary
    assert "not_applicable" not in summary
    assert "None" not in summary
    download = latest_target_download(bundle, study, BUY_HOLD_LABEL)
    assert download["policy_version"].unique().tolist() == [
        "single-etf-buy-hold-v1"
    ]


def test_equal_weight_is_fixed_not_a_pending_tactical_signal(bundle):
    full = build_workbench_study(bundle, DEFAULT_SELECTION)
    end = full.backtests[EQUAL_WEIGHT_LABEL].periods["period_end_date"].iloc[-12]
    selected = build_workbench_study(bundle, DEFAULT_SELECTION, end=end)
    provenance = target_provenance(bundle, selected, EQUAL_WEIGHT_LABEL)

    assert provenance["signal_as_of"] == "not_applicable_no_tactical_signal"
    assert provenance["execution_status"] == "constant_target_effective_for_analytical_ticket"
    assert provenance["policy_version"] == "equal-weight-monthly-v1"
    assert provenance["displayed_history_through"] == str(end.date())


def test_plain_summaries_and_metrics_never_render_none(bundle):
    study = build_workbench_study(bundle, DEFAULT_SELECTION)
    summary = proposed_target_summary(study.latest_targets[VOL_TREND_LABEL])
    assert "100.00% total" in summary
    assert "None" not in summary
    metrics = _format_metrics(study, [VOL_TREND_LABEL])
    assert not metrics.map(lambda value: value is None).any().any()

    ticket = build_rebalance_ticket(
        {"SPY": 0.2, "IEF": 0.3, "GLD": 0.1, CASH_ASSET: 0.4},
        study.latest_targets[VOL_TREND_LABEL],
        DEFAULT_SELECTION,
        transaction_cost_bps=5,
        portfolio_value=100_000,
    )
    action = ticket_action_summary(ticket)
    dollars = ticket_dollar_summary(ticket)
    assert "percentage points" in action
    assert "cash" in action
    assert "separately" in dollars
    assert "not deducted" in dollars
    assert "+$" in dollars or "-$" in dollars
