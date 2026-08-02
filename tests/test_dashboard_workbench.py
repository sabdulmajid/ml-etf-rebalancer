import os
import shutil

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
    VOL_TREND_LABEL,
    allocation_history_download,
    available_comparisons,
    build_workbench_study,
    bundle_fingerprint,
    clear_workbench_caches,
    historical_download,
    holding_period_returns,
    latest_target_download,
    load_cached_bundle,
    target_provenance,
    target_provenance_summary,
    why_this_weight,
)
from data.cash import CASH_LABEL
from data.workbench import DEFAULT_BUNDLE_PATH, load_workbench_bundle


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


def test_freshness_policy_allows_one_month_warning_and_disables_two_or_future(bundle):
    assert bundle.freshness(as_of="2026-09-15")["status"] == "warning"
    assert bundle.freshness(as_of="2026-10-15")["status"] == "disabled"
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


def test_downloads_exactly_reconcile_displayed_results_and_targets(bundle):
    study = build_workbench_study(bundle, DEFAULT_SELECTION, transaction_cost_bps=9)
    labels = [VOL_TREND_LABEL, CASH_LABEL_SHORT]
    history = historical_download(study, labels)
    for label in labels:
        expected = study.backtests[label].periods.reset_index()
        actual = history.loc[history["series"] == label].drop(columns="series")
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)

    target = latest_target_download(bundle, study, VOL_TREND_LABEL)
    assert target["target_weight"].sum() == pytest.approx(1.0)
    assert target.loc[target["asset"] == CASH_ASSET, "asset_type"].item() == "analytical_cash"
    assert target["signal_as_of"].nunique() == 1
    assert target["execution_status"].unique().tolist() == [
        "pending_next_trading_close"
    ]
    assert target["artifact_generated_at_utc"].unique().tolist() == [
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
    weight_columns = [*DEFAULT_SELECTION, CASH_ASSET]
    assert allocation[weight_columns].sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert allocation["turnover"].to_numpy() == pytest.approx(
        study.backtests[VOL_TREND_LABEL].periods["turnover"].to_numpy()
    )


def test_target_provenance_keeps_cash_distinct_from_tactical_strategy(bundle):
    study = build_workbench_study(bundle, DEFAULT_SELECTION)
    strategy = target_provenance(bundle, study, VOL_TREND_LABEL)
    cash = target_provenance(bundle, study, CASH_LABEL_SHORT)

    assert strategy["execution_status"] == "pending_next_trading_close"
    assert strategy["signal_as_of"] == str(
        study.allocation_results[VOL_TREND_LABEL].latest_signal_date.date()
    )
    assert cash["signal_as_of"] == "not_applicable_no_tactical_signal"
    assert cash["execution_status"] == "constant_target_effective_for_analytical_ticket"
    assert cash["policy_version"] == "analytical-cash-comparison-v1"
    assert cash["historical_accounting_schedule_as_of"] == str(
        study.allocation_results[VOL_BALANCED_LABEL].latest_signal_date.date()
    )
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
    assert buy_hold["historical_accounting_schedule_as_of"] == str(
        study.allocation_results[VOL_BALANCED_LABEL].latest_signal_date.date()
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
