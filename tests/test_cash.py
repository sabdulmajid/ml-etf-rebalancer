import numpy as np
import pandas as pd
import pytest

from data.cash import (
    CASH_ASSET,
    SOFR_SWITCH_DATE,
    cash_returns_between,
    cash_values_on_dates,
    construct_cash_index,
    validate_against_sofr_index,
)


def _rates():
    effr = pd.Series(
        [1.0, 1.2], index=pd.to_datetime(["2018-03-29", "2018-03-30"])
    )
    sofr = pd.Series(
        [1.8, 1.9, 2.0],
        index=pd.to_datetime(["2018-04-02", "2018-04-03", "2018-04-04"]),
    )
    return effr, sofr


def test_cash_switch_and_weekend_actual_360_compounding():
    effr, sofr = _rates()
    cash = construct_cash_index(effr, sofr, valuation_through="2018-04-05")

    assert set(cash.loc[cash.index < SOFR_SWITCH_DATE, "source_series"]) == {"EFFR"}
    assert set(cash.loc[cash.index >= SOFR_SWITCH_DATE, "source_series"]) == {"SOFR"}
    assert cash.loc[pd.Timestamp("2018-03-30"), "accrual_days"] == 3
    expected = (1 + 0.01 / 360) * (1 + 0.012 * 3 / 360)
    assert cash.loc[SOFR_SWITCH_DATE, "cash_index"] == pytest.approx(expected)


def test_cash_values_use_simple_accrual_within_one_rate_interval():
    effr, sofr = _rates()
    cash = construct_cash_index(effr, sofr, valuation_through="2018-04-05")

    values = cash_values_on_dates(
        cash, ["2018-03-30", "2018-03-31", "2018-04-01", "2018-04-02"], "2018-04-05"
    )
    base = values.iloc[0]
    assert values.iloc[1] / base == pytest.approx(1 + 0.012 / 360)
    assert values.iloc[2] / base == pytest.approx(1 + 0.012 * 2 / 360)
    assert values.iloc[3] / base == pytest.approx(1 + 0.012 * 3 / 360)


def test_cash_return_pairs_keep_analytical_identifier():
    effr, sofr = _rates()
    cash = construct_cash_index(effr, sofr, valuation_through="2018-04-05")
    returns = cash_returns_between(
        cash, ["2018-03-30", "2018-04-02"], ["2018-04-02", "2018-04-05"], "2018-04-05"
    )
    assert returns.name == CASH_ASSET
    assert (returns > 0).all()


def test_sofr_index_validation_normalizes_at_switch():
    effr, sofr = _rates()
    cash = construct_cash_index(effr, sofr, valuation_through="2018-04-05")
    official_dates = pd.to_datetime(["2018-04-02", "2018-04-03", "2018-04-04"])
    reconstructed = cash_values_on_dates(cash, official_dates, "2018-04-05")
    official = reconstructed / reconstructed.iloc[0]

    validation = validate_against_sofr_index(
        cash, official, tolerance=1e-12, valuation_through="2018-04-05"
    )

    assert validation.passed
    assert validation.maximum_absolute_error == pytest.approx(0.0)


def test_cash_rejects_missing_switch_date_and_future_valuation():
    effr, sofr = _rates()
    with pytest.raises(ValueError, match="must begin"):
        construct_cash_index(effr, sofr.iloc[1:])
    cash = construct_cash_index(effr, sofr, valuation_through="2018-04-05")
    with pytest.raises(ValueError, match="exceeds"):
        cash_values_on_dates(cash, ["2018-04-06"], "2018-04-05")
