import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.engine import (
    CASH_ASSET,
    TargetWeightSchedule,
    build_completed_monthly_periods,
    completed_month_end_dates,
    run_backtest,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden_backtest.json"
ASSETS = ["A", "B", "C", CASH_ASSET]


def _golden_inputs():
    fixture = json.loads(FIXTURE_PATH.read_text())
    periods = fixture["periods"]
    execution_dates = pd.to_datetime(
        [period["execution_date"] for period in periods]
    )
    weights = pd.DataFrame(
        [period["target"] for period in periods],
        index=execution_dates,
        columns=fixture["assets"],
    )
    schedule = TargetWeightSchedule(
        weights=weights,
        signal_dates=pd.Series(
            pd.to_datetime([period["signal_date"] for period in periods]),
            index=execution_dates,
        ),
        period_end_dates=pd.Series(
            pd.to_datetime([period["period_end_date"] for period in periods]),
            index=execution_dates,
        ),
    )
    realized = pd.DataFrame(
        [period["returns"] for period in periods],
        index=execution_dates,
        columns=fixture["assets"],
    )
    return fixture, schedule, realized.drop(columns=CASH_ASSET), realized[CASH_ASSET]


def _two_period_schedule(weights=None):
    execution_dates = pd.to_datetime(["2020-02-03", "2020-03-02"])
    if weights is None:
        weights = pd.DataFrame(
            [[0.6, 0.4], [0.5, 0.5]],
            index=execution_dates,
            columns=["A", CASH_ASSET],
        )
    return TargetWeightSchedule(
        weights=weights,
        signal_dates=pd.Series(
            pd.to_datetime(["2020-01-31", "2020-02-28"]),
            index=execution_dates,
        ),
        period_end_dates=pd.Series(
            pd.to_datetime(["2020-03-02", "2020-04-01"]),
            index=execution_dates,
        ),
    )


def test_completed_months_exclude_partial_current_month():
    dates = pd.bdate_range("2020-01-01", "2020-04-15")

    signals = completed_month_end_dates(dates, as_of="2020-04-15")

    assert list(signals) == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
        pd.Timestamp("2020-03-31"),
    ]
    assert pd.Timestamp("2020-04-15") not in signals


def test_monthly_period_builder_separates_signal_execution_and_return_period():
    dates = pd.bdate_range("2020-01-01", "2020-05-15")

    timing = build_completed_monthly_periods(dates, as_of="2020-05-15")

    expected = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2020-01-31", "2020-02-28", "2020-03-31"]
            ),
            "period_end_date": pd.to_datetime(
                ["2020-03-02", "2020-04-01", "2020-05-01"]
            ),
        },
        index=pd.DatetimeIndex(
            pd.to_datetime(["2020-02-03", "2020-03-02", "2020-04-01"]),
            name="execution_date",
        ),
    )
    pd.testing.assert_frame_equal(timing, expected, check_freq=False)


def test_monthly_period_builder_emits_only_fully_observed_periods():
    dates = pd.bdate_range("2020-01-01", "2020-03-15")

    timing = build_completed_monthly_periods(dates, as_of="2020-03-15")

    assert list(timing.index) == [pd.Timestamp("2020-02-03")]
    assert timing.iloc[0]["signal_date"] == pd.Timestamp("2020-01-31")
    assert timing.iloc[0]["period_end_date"] == pd.Timestamp("2020-03-02")


def test_target_schedule_normalizes_dates_and_preserves_explicit_cash():
    schedule = _two_period_schedule()

    assert schedule.weights.index.name == "execution_date"
    assert list(schedule.weights.columns) == ["A", CASH_ASSET]
    assert schedule.weights.sum(axis=1).to_numpy() == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([[0.6, 0.3], [0.5, 0.5]], "sum to one"),
        ([[1.1, -0.1], [0.5, 0.5]], "long-only"),
        ([[np.nan, np.nan], [0.5, 0.5]], "finite"),
    ],
)
def test_target_schedule_rejects_invalid_targets(weights, message):
    dates = pd.to_datetime(["2020-02-03", "2020-03-02"])
    frame = pd.DataFrame(weights, index=dates, columns=["A", CASH_ASSET])

    with pytest.raises(ValueError, match=message):
        _two_period_schedule(frame)


def test_target_schedule_requires_explicit_cash():
    dates = pd.to_datetime(["2020-02-03", "2020-03-02"])
    weights = pd.DataFrame([[1.0], [1.0]], index=dates, columns=["A"])

    with pytest.raises(ValueError, match="explicit"):
        _two_period_schedule(weights)


@pytest.mark.parametrize(
    ("signal_dates", "period_end_dates", "message"),
    [
        (
            ["2020-02-03", "2020-02-28"],
            ["2020-03-02", "2020-04-01"],
            "signal date must precede",
        ),
        (
            ["2020-01-31", "2020-02-28"],
            ["2020-03-03", "2020-04-01"],
            "following execution date",
        ),
        (
            ["2020-01-31", "2020-01-31"],
            ["2020-03-02", "2020-04-01"],
            "unique and strictly increasing",
        ),
    ],
)
def test_target_schedule_rejects_invalid_timing(
    signal_dates, period_end_dates, message
):
    dates = pd.to_datetime(["2020-02-03", "2020-03-02"])
    weights = pd.DataFrame(
        [[0.6, 0.4], [0.5, 0.5]],
        index=dates,
        columns=["A", CASH_ASSET],
    )

    with pytest.raises(ValueError, match=message):
        TargetWeightSchedule(
            weights=weights,
            signal_dates=pd.Series(pd.to_datetime(signal_dates), index=dates),
            period_end_dates=pd.Series(pd.to_datetime(period_end_dates), index=dates),
        )


def test_golden_backtest_reconciles_every_period():
    fixture, schedule, asset_returns, cash_returns = _golden_inputs()

    result = run_backtest(
        schedule,
        asset_returns,
        cash_returns,
        transaction_cost_bps=fixture["transaction_cost_bps"],
        initial_value=fixture["initial_value"],
    )

    expected_target = []
    expected_pretrade = []
    expected_trades = []
    expected_ending = []
    scalar_columns = [
        "starting_equity",
        "turnover",
        "cost_rate",
        "transaction_cost",
        "value_after_cost",
        "cash_return",
        "gross_return",
        "net_return",
        "gross_equity",
        "net_equity",
        "drawdown",
    ]
    expected_scalars = {column: [] for column in scalar_columns}
    for period in fixture["periods"]:
        expected_target.append(period["target"])
        expected_pretrade.append(period["pretrade"])
        expected_trades.append(period["trades"])
        expected_ending.append(period["ending"])
        for column in scalar_columns:
            if column == "cash_return":
                expected_scalars[column].append(period["returns"][-1])
            else:
                expected_scalars[column].append(period[column])

    np.testing.assert_allclose(
        result.target_weights, expected_target, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        result.pretrade_weights, expected_pretrade, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        result.trades, expected_trades, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        result.ending_weights, expected_ending, rtol=0.0, atol=1e-14
    )
    for column, expected in expected_scalars.items():
        np.testing.assert_allclose(
            result.periods[column], expected, rtol=0.0, atol=1e-14
        )

    np.testing.assert_allclose(
        result.trades.sum(axis=1), 0.0, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        result.target_weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        result.pretrade_weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        result.ending_weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-14
    )

    compounded_net_equity = fixture["initial_value"] * float(
        (1.0 + result.periods["net_return"]).prod()
    )
    assert result.periods.iloc[-1]["net_equity"] == pytest.approx(
        compounded_net_equity, rel=0.0, abs=1e-14
    )

    expected_metrics = fixture["expected_metrics"]
    assert set(expected_metrics).issubset(result.metrics)
    for metric, expected in expected_metrics.items():
        assert result.metrics[metric] == pytest.approx(
            expected, rel=0.0, abs=1e-14
        )
    assert np.isnan(result.metrics["calmar_ratio"])


def test_selected_range_restarts_from_cash_and_charges_new_entry_cost():
    _, schedule, asset_returns, cash_returns = _golden_inputs()

    result = run_backtest(
        schedule,
        asset_returns,
        cash_returns,
        transaction_cost_bps=10,
        start="2020-03-02",
        end="2020-05-01",
    )

    first_date = pd.Timestamp("2020-03-02")
    assert list(result.periods.index) == [
        pd.Timestamp("2020-03-02"),
        pd.Timestamp("2020-04-01"),
    ]
    assert result.pretrade_weights.loc[first_date].to_dict() == {
        "A": 0.0,
        "B": 0.0,
        "C": 0.0,
        CASH_ASSET: 1.0,
    }
    assert result.periods.loc[first_date, "turnover"] == pytest.approx(0.8)
    assert result.periods.loc[first_date, "cost_rate"] == pytest.approx(0.0008)


def test_genuine_cash_only_schedule_accepts_empty_asset_return_matrix():
    dates = pd.to_datetime(["2020-02-03", "2020-03-02"])
    weights = pd.DataFrame(
        [[1.0], [1.0]],
        index=dates,
        columns=[CASH_ASSET],
    )
    schedule = TargetWeightSchedule(
        weights=weights,
        signal_dates=pd.Series(
            pd.to_datetime(["2020-01-31", "2020-02-28"]), index=dates
        ),
        period_end_dates=pd.Series(
            pd.to_datetime(["2020-03-02", "2020-04-01"]), index=dates
        ),
    )
    asset_returns = pd.DataFrame(index=dates)
    cash_returns = pd.Series([0.001, 0.002], index=dates)

    result = run_backtest(
        schedule, asset_returns, cash_returns, transaction_cost_bps=50
    )

    assert result.periods["turnover"].to_numpy() == pytest.approx([0.0, 0.0])
    assert result.periods["transaction_cost"].to_numpy() == pytest.approx([0.0, 0.0])
    assert result.periods["net_return"].to_numpy() == pytest.approx([0.001, 0.002])
    assert result.periods.iloc[-1]["net_equity"] == pytest.approx(1.001 * 1.002)


def test_empty_asset_return_matrix_fails_when_schedule_requires_an_asset():
    schedule = _two_period_schedule()
    empty_returns = pd.DataFrame(index=schedule.execution_dates)
    cash = pd.Series([0.001, 0.001], index=schedule.execution_dates)

    with pytest.raises(ValueError, match="must contain holding-period returns"):
        run_backtest(schedule, empty_returns, cash)


def test_zero_cost_gross_and_net_paths_are_identical():
    _, schedule, asset_returns, cash_returns = _golden_inputs()

    result = run_backtest(schedule, asset_returns, cash_returns)

    np.testing.assert_allclose(
        result.periods["gross_return"],
        result.periods["net_return"],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        result.periods["gross_equity"],
        result.periods["net_equity"],
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("asset_returns", "cash_returns", "message"),
    [
        ([0.01], [0.001, 0.001], "missing execution dates"),
        ([0.01, np.nan], [0.001, 0.001], "finite"),
        ([0.01, -1.0], [0.001, 0.001], "-100%"),
    ],
)
def test_engine_rejects_missing_or_invalid_asset_returns(
    asset_returns, cash_returns, message
):
    schedule = _two_period_schedule()
    asset_dates = schedule.execution_dates[: len(asset_returns)]
    returns = pd.DataFrame({"A": asset_returns}, index=asset_dates)
    cash = pd.Series(cash_returns, index=schedule.execution_dates)

    with pytest.raises(ValueError, match=message):
        run_backtest(schedule, returns, cash)


def test_engine_rejects_missing_cash_return():
    schedule = _two_period_schedule()
    returns = pd.DataFrame({"A": [0.01, 0.02]}, index=schedule.execution_dates)
    cash = pd.Series([0.001], index=schedule.execution_dates[:1])

    with pytest.raises(ValueError, match="cash returns are missing"):
        run_backtest(schedule, returns, cash)


@pytest.mark.parametrize(
    ("cost_bps", "message"),
    [(-1, "nonnegative"), (np.nan, "finite"), (10000, "less than 10000")],
)
def test_engine_rejects_invalid_transaction_cost(cost_bps, message):
    schedule = _two_period_schedule()
    returns = pd.DataFrame({"A": [0.01, 0.02]}, index=schedule.execution_dates)
    cash = pd.Series([0.001, 0.001], index=schedule.execution_dates)

    with pytest.raises(ValueError, match=message):
        run_backtest(schedule, returns, cash, transaction_cost_bps=cost_bps)
