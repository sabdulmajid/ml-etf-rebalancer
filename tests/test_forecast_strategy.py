import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.engine import CASH_ASSET, run_backtest
from dashboard.workbench import (
    CASH_LABEL_SHORT,
    DEFAULT_SELECTION,
    SPY_REFERENCE_LABEL,
    VOL_FORECAST_LABEL,
    available_comparisons,
    build_workbench_study,
    holding_period_returns,
)
from data.timesfm import load_timesfm_bundle
from data.workbench import load_workbench_bundle
from strategies.allocation import position_cap
from strategies.forecast import generate_forecast_allocation_targets


@pytest.fixture(scope="module")
def bundles():
    return load_workbench_bundle(), load_timesfm_bundle()


def _force_edges(signals, tickers, edge):
    forced = signals.copy()
    mask = (forced["model_mode"] == "timesfm3_multivariate") & forced[
        "ticker"
    ].isin(tickers)
    forced.loc[mask, "forecast_edge"] = edge
    forced.loc[mask, "forecast_holding_return"] = (
        forced.loc[mask, "cash_hurdle"] + edge
    )
    return forced


@pytest.mark.parametrize("count", range(1, 9))
def test_forecast_targets_support_one_through_eight_etfs_and_cap(bundles, count):
    workbench, forecasts = bundles
    selected = tuple(workbench.tickers[:count])
    result = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        forecasts.signals,
        as_of=workbench.signal_as_of,
    )

    assert result.schedule.weights.columns.tolist() == [*selected, CASH_ASSET]
    assert result.schedule.weights.sum(axis=1).to_numpy() == pytest.approx(1.0)
    assert result.latest_target.sum() == pytest.approx(1.0)
    assert result.schedule.weights[list(selected)].max().max() <= position_cap(count) + 1e-12
    assert result.latest_target.loc[list(selected)].max() <= position_cap(count) + 1e-12
    latest_generation_ids = forecasts.latest()["model_generation_id"].unique().tolist()
    assert result.latest_model_generation_id == latest_generation_ids[0]


def test_one_etf_forecast_filter_naturally_switches_between_etf_and_cash(bundles):
    workbench, forecasts = bundles
    selected = ("SPY",)
    positive = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        _force_edges(forecasts.signals, selected, 0.01),
        as_of=workbench.signal_as_of,
    )
    negative = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        _force_edges(forecasts.signals, selected, -0.01),
        as_of=workbench.signal_as_of,
    )

    assert (positive.schedule.weights["SPY"] == 1.0).all()
    assert (positive.schedule.weights[CASH_ASSET] == 0.0).all()
    assert (negative.schedule.weights["SPY"] == 0.0).all()
    assert (negative.schedule.weights[CASH_ASSET] == 1.0).all()
    assert negative.latest_target.to_dict() == pytest.approx(
        {"SPY": 0.0, CASH_ASSET: 1.0}
    )
    assert negative.latest_diagnostics["forecast_status"].unique().tolist() == [
        "Held in cash"
    ]
    assert positive.latest_diagnostics["forecast_status"].unique().tolist() == [
        "Eligible"
    ]


def test_forecast_filter_removes_some_or_all_assets_without_magnitude_weighting(bundles):
    workbench, forecasts = bundles
    selected = DEFAULT_SELECTION
    some = forecasts.signals.copy()
    some = _force_edges(some, ("SPY", "GLD"), 0.50)
    some = _force_edges(some, ("IEF",), -0.01)
    result = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        some,
        as_of=workbench.signal_as_of,
    )
    assert (result.schedule.weights["IEF"] == 0.0).all()
    assert result.latest_target["IEF"] == 0.0

    smaller_scores = _force_edges(some, ("SPY", "GLD"), 0.001)
    same_weights = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        smaller_scores,
        as_of=workbench.signal_as_of,
    )
    pd.testing.assert_frame_equal(result.schedule.weights, same_weights.schedule.weights)
    pd.testing.assert_series_equal(result.latest_target, same_weights.latest_target)

    all_fail = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        _force_edges(forecasts.signals, selected, -0.01),
        as_of=workbench.signal_as_of,
    )
    assert (all_fail.schedule.weights[list(selected)] == 0.0).all().all()
    assert (all_fail.schedule.weights[CASH_ASSET] == 1.0).all()
    assert all_fail.latest_target[CASH_ASSET] == 1.0


def test_actual_outcomes_cannot_change_forecast_targets(bundles):
    workbench, forecasts = bundles
    selected = DEFAULT_SELECTION
    original = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        forecasts.signals,
        as_of=workbench.signal_as_of,
    )
    mutated = forecasts.signals.copy()
    outcome_columns = [column for column in mutated if column.startswith("actual_")]
    mutated.loc[:, outcome_columns] = -999_999.0
    after = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        mutated,
        as_of=workbench.signal_as_of,
    )

    pd.testing.assert_frame_equal(original.schedule.weights, after.schedule.weights)
    pd.testing.assert_series_equal(original.latest_target, after.latest_target)
    pd.testing.assert_frame_equal(original.diagnostics, after.diagnostics)


def test_future_or_incomplete_month_forecast_origin_cannot_be_selected(bundles):
    workbench, forecasts = bundles
    selected = ("SPY",)
    result = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        forecasts.signals,
        as_of="2026-08-15",
    )

    assert result.latest_signal_date == pd.Timestamp("2026-07-31")
    assert result.latest_signal_date.to_period("M") < pd.Period("2026-08", freq="M")


def test_historical_as_of_excludes_outcomes_realized_only_later(bundles):
    workbench, forecasts = bundles
    selected = ("SPY", "IEF")
    as_of = pd.Timestamp("2025-06-15")
    result = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        forecasts.signals,
        as_of=as_of,
    )

    assert result.latest_signal_date == pd.Timestamp("2025-05-30")
    assert (result.schedule.period_end_dates <= as_of).all()
    assert result.schedule.period_end_dates.iloc[-1] == pd.Timestamp("2025-06-02")


def test_forecast_history_uses_common_engine_costs_and_common_alignment(bundles):
    workbench, forecasts = bundles
    selected = DEFAULT_SELECTION
    allocation = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        forecasts.signals,
        as_of=workbench.signal_as_of,
    )
    study = build_workbench_study(
        workbench,
        selected,
        transaction_cost_bps=17.0,
        forecast_result=allocation,
        forecast_manifest=forecasts.manifest,
        forecast_freshness=forecasts.freshness(),
    )
    assets, cash = holding_period_returns(workbench, allocation.schedule, selected)
    direct = run_backtest(
        allocation.schedule,
        assets,
        cash,
        transaction_cost_bps=17.0,
    )
    pd.testing.assert_frame_equal(study.backtests[VOL_FORECAST_LABEL].periods, direct.periods)
    pd.testing.assert_frame_equal(study.backtests[VOL_FORECAST_LABEL].trades, direct.trades)
    timing = {
        tuple(result.periods[["signal_date", "period_end_date"]].to_numpy().ravel())
        for result in study.backtests.values()
    }
    assert len(timing) == 1


def test_spy_reference_is_available_once_and_is_never_a_proposed_target(bundles):
    workbench, forecasts = bundles
    assert SPY_REFERENCE_LABEL not in available_comparisons(
        ["SPY"], forecast_available=True
    )
    assert available_comparisons(
        ["IEF"], forecast_available=True
    ).count(SPY_REFERENCE_LABEL) == 1
    assert available_comparisons(
        ["SPY", "IEF"], forecast_available=True
    ).count(SPY_REFERENCE_LABEL) == 1

    selected = ("IEF",)
    allocation = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        forecasts.signals,
        as_of=workbench.signal_as_of,
    )
    study = build_workbench_study(
        workbench,
        selected,
        forecast_result=allocation,
        forecast_manifest=forecasts.manifest,
        forecast_freshness=forecasts.freshness(),
    )
    assert SPY_REFERENCE_LABEL in study.backtests
    assert SPY_REFERENCE_LABEL not in study.latest_targets
    assert CASH_LABEL_SHORT in study.latest_targets


def test_streamlit_runtime_loaders_do_not_import_model_dependencies():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dashboard.workbench; "
                "assert 'timesfm3' not in sys.modules; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
