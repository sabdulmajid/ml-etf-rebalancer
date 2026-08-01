import numpy as np
import pandas as pd
import pytest

from backtest.engine import CASH_ASSET
from strategies.allocation import (
    EQUAL_WEIGHT,
    VOLATILITY_BALANCED,
    VOLATILITY_BALANCED_TREND,
    StrategyPolicy,
    generate_allocation_targets,
    position_cap,
)


def _synthetic_prices(tickers=("A", "B", "C"), periods=900):
    dates = pd.bdate_range("2018-01-01", periods=periods)
    steps = np.arange(periods, dtype=float)
    data = {}
    for number, ticker in enumerate(tickers, start=1):
        trend = 100 * np.exp((0.0002 * number) * steps)
        cycle = 1 + (0.0025 * number) * np.sin(steps / 7.0)
        data[ticker] = trend * cycle
    return pd.DataFrame(data, index=dates)


@pytest.mark.parametrize("count", range(1, 9))
def test_adaptive_cap_one_through_eight(count):
    expected = min(1.0, 1.5 / count)
    assert position_cap(count) == pytest.approx(expected)


@pytest.mark.parametrize("count", range(1, 9))
def test_one_through_eight_selected_etfs_generate_deterministic_feasible_targets(count):
    tickers = tuple("ABCDEFGH")[:count]
    prices = _synthetic_prices(tickers, periods=700)
    first = generate_allocation_targets(
        prices, tickers, VOLATILITY_BALANCED, as_of="2020-08-31"
    )
    second = generate_allocation_targets(
        prices, tickers, VOLATILITY_BALANCED, as_of="2020-08-31"
    )
    pd.testing.assert_series_equal(first.latest_target, second.latest_target)
    assert first.latest_target.sum() == pytest.approx(1.0)
    assert (
        first.latest_target[list(tickers)] <= position_cap(count) + 1e-12
    ).all()


def test_policy_contains_approved_fixed_methodology():
    policy = StrategyPolicy()
    assert policy.volatility_lookback_days == 126
    assert policy.minimum_volatility_observations == 120
    assert policy.trend_months == 10
    assert policy.cap_explanation == (
        "No ETF can receive more than 150% of its equal-weight share."
    )


def test_one_etf_trend_naturally_becomes_etf_or_cash():
    rising = _synthetic_prices(("A",), periods=900)
    result = generate_allocation_targets(
        rising, ["A"], VOLATILITY_BALANCED_TREND, as_of="2021-06-30"
    )
    assert result.latest_target["A"] == pytest.approx(1.0)
    assert result.latest_target[CASH_ASSET] == pytest.approx(0.0)

    falling = rising.copy()
    falling.loc[falling.index[-180] :, "A"] = np.linspace(
        falling.iloc[-181, 0], falling.iloc[-181, 0] * 0.5, 180
    )
    result = generate_allocation_targets(
        falling, ["A"], VOLATILITY_BALANCED_TREND, as_of="2021-06-30"
    )
    assert result.latest_target["A"] == pytest.approx(0.0)
    assert result.latest_target[CASH_ASSET] == pytest.approx(1.0)


def test_volatility_balanced_weights_and_cash_sum_to_one_and_respect_cap():
    prices = _synthetic_prices()
    result = generate_allocation_targets(
        prices, ["A", "B", "C"], VOLATILITY_BALANCED, as_of="2021-06-30"
    )
    assert result.latest_target.sum() == pytest.approx(1.0)
    assert (result.latest_target[["A", "B", "C"]] <= 0.5 + 1e-12).all()
    assert result.schedule.weights.sum(axis=1).to_numpy() == pytest.approx(1.0)


def test_trend_filter_removes_several_or_all_assets_and_leaves_residual_cash():
    prices = _synthetic_prices(("A", "B", "C", "D"), periods=900)
    for ticker in ["B", "C", "D"]:
        start = prices.index[-180]
        prices.loc[start:, ticker] = np.linspace(
            prices.loc[start, ticker], prices.loc[start, ticker] * 0.4, 180
        )
    one_pass = generate_allocation_targets(
        prices,
        ["A", "B", "C", "D"],
        VOLATILITY_BALANCED_TREND,
        as_of="2021-06-30",
    )
    assert one_pass.latest_target["A"] == pytest.approx(0.375)
    assert one_pass.latest_target[CASH_ASSET] == pytest.approx(0.625)

    start = prices.index[-180]
    prices.loc[start:, "A"] = np.linspace(
        prices.loc[start, "A"], prices.loc[start, "A"] * 0.4, 180
    )
    all_fail = generate_allocation_targets(
        prices,
        ["A", "B", "C", "D"],
        VOLATILITY_BALANCED_TREND,
        as_of="2021-06-30",
    )
    assert all_fail.latest_target.drop(CASH_ASSET).sum() == pytest.approx(0.0)
    assert all_fail.latest_target[CASH_ASSET] == pytest.approx(1.0)


def test_equal_weight_uses_same_timing_contract():
    prices = _synthetic_prices(("A", "B"))
    result = generate_allocation_targets(
        prices, ["A", "B"], EQUAL_WEIGHT, as_of="2021-06-30"
    )
    assert result.schedule.weights[["A", "B"]].to_numpy() == pytest.approx(0.5)
    assert (result.schedule.weights[CASH_ASSET] == 0.0).all()
    assert (
        result.schedule.signal_dates.to_numpy()
        < result.schedule.execution_dates.to_numpy()
    ).all()


def test_equal_weight_includes_flat_price_etf_without_cash_residual():
    dates = pd.bdate_range("2018-01-01", periods=700)
    prices = pd.DataFrame(
        {
            "FLAT": np.full(len(dates), 100.0),
            "MOVING": 100.0 + np.arange(len(dates)) * 0.05,
        },
        index=dates,
    )
    result = generate_allocation_targets(
        prices, ["FLAT", "MOVING"], EQUAL_WEIGHT, as_of="2020-08-31"
    )
    assert result.latest_target["FLAT"] == pytest.approx(0.5)
    assert result.latest_target["MOVING"] == pytest.approx(0.5)
    assert result.latest_target[CASH_ASSET] == pytest.approx(0.0)
    assert (
        result.latest_diagnostics.loc[(result.latest_signal_date, "FLAT"), "eligibility_reason"]
        == "Included; equal weight ignores volatility and trend"
    )


def test_common_start_waits_for_later_etf_warmup_and_latest_uses_completed_month():
    prices = _synthetic_prices(("A", "B"), periods=900)
    prices.loc[prices.index[:300], "B"] = np.nan
    result = generate_allocation_targets(
        prices, ["A", "B"], VOLATILITY_BALANCED, as_of="2021-06-15"
    )
    assert result.common_start_signal_date >= prices.index[300]
    assert result.latest_signal_date.month == 5
    assert result.latest_signal_date.year == 2021


def test_future_prices_cannot_change_an_earlier_signal_target():
    prices = _synthetic_prices(("A", "B", "C"), periods=900)
    base = generate_allocation_targets(
        prices, ["A", "B", "C"], VOLATILITY_BALANCED_TREND, as_of="2021-06-30"
    )
    comparison_date = base.schedule.execution_dates[-8]
    comparison_signal = base.schedule.signal_dates.loc[comparison_date]

    changed = prices.copy()
    changed.loc[changed.index > comparison_signal, "A"] *= 4.0
    revised = generate_allocation_targets(
        changed, ["A", "B", "C"], VOLATILITY_BALANCED_TREND, as_of="2021-06-30"
    )
    revised_date = revised.schedule.signal_dates[
        revised.schedule.signal_dates == comparison_signal
    ].index[0]
    pd.testing.assert_series_equal(
        base.schedule.weights.loc[comparison_date],
        revised.schedule.weights.loc[revised_date],
        check_names=False,
    )


def test_target_generation_rejects_cash_and_more_than_eight_etfs():
    prices = _synthetic_prices(tuple("ABCDEFGHI"))
    with pytest.raises(ValueError, match="analytical cash"):
        generate_allocation_targets(prices, [CASH_ASSET], VOLATILITY_BALANCED)
    with pytest.raises(ValueError, match="between"):
        generate_allocation_targets(prices, list("ABCDEFGHI"), VOLATILITY_BALANCED)
