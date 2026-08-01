"""Construction and valuation of the analytical U.S. overnight cash series.

The combined history is deliberately *not* called SOFR.  It uses the official
New York Fed EFFR before 2018-04-02 and official SOFR from that effective date.
Rates are annual percentages on an Actual/360 basis.  A published rate accrues
from its effective date until the next effective date without intra-period
compounding, matching the SOFR Index convention across weekends and holidays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CASH_ASSET = "CASH:USD_OVERNIGHT"
CASH_LABEL = "Cash — U.S. overnight-rate proxy"
SOFR_SWITCH_DATE = pd.Timestamp("2018-04-02")
DAY_COUNT_DENOMINATOR = 360.0
INDEX_TOLERANCE = 5e-8


def _rate_series(values, name):
    series = pd.Series(values, copy=True, dtype=float)
    try:
        series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).tz_localize(None)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have a valid date index") from exc
    series.index = series.index.normalize()
    series.index.name = "effective_date"
    series = series.sort_index()
    if series.empty:
        raise ValueError(f"{name} must not be empty")
    if not series.index.is_unique:
        raise ValueError(f"{name} effective dates must be unique")
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{name} must contain finite rates")
    if (series <= -100.0).any():
        raise ValueError(f"{name} contains an invalid rate at or below -100%")
    series.name = "annual_rate"
    return series


def construct_cash_index(effr_rates, sofr_rates, valuation_through=None):
    """Combine official rate observations into an auditable cash-index table.

    ``cash_index`` is the beginning-of-effective-date index. ``annual_rate``
    applies for ``accrual_days`` until the next effective date (or the supplied
    final valuation date for the last row).  The table therefore retains the
    exact source rate used for every compounding interval.
    """
    effr = _rate_series(effr_rates, "EFFR")
    sofr = _rate_series(sofr_rates, "SOFR")
    effr = effr.loc[effr.index < SOFR_SWITCH_DATE]
    sofr = sofr.loc[sofr.index >= SOFR_SWITCH_DATE]
    if effr.empty:
        raise ValueError("EFFR must cover dates before the SOFR switch")
    if sofr.empty or sofr.index[0] != SOFR_SWITCH_DATE:
        raise ValueError("SOFR must begin on the 2018-04-02 switch date")
    effr_gaps = effr.index.to_series().diff().dt.days.dropna()
    if (effr_gaps > 7).any():
        raise ValueError("EFFR history contains an impossible gap longer than seven days")
    if (SOFR_SWITCH_DATE - effr.index[-1]).days > 7:
        raise ValueError("EFFR does not provide an active observation before the switch")

    rates = pd.concat(
        [
            pd.DataFrame({"annual_rate": effr, "source_series": "EFFR"}),
            pd.DataFrame({"annual_rate": sofr, "source_series": "SOFR"}),
        ]
    ).sort_index()
    if not rates.index.is_unique:
        raise ValueError("combined overnight-rate dates must be unique")

    if valuation_through is None:
        valuation_through = rates.index[-1]
    valuation_through = pd.Timestamp(valuation_through).tz_localize(None).normalize()
    if valuation_through < rates.index[-1]:
        rates = rates.loc[rates.index <= valuation_through]
    if rates.empty:
        raise ValueError("valuation_through precedes the overnight-rate history")

    next_dates = rates.index.to_series().shift(-1)
    next_dates.iloc[-1] = max(valuation_through, rates.index[-1])
    accrual_days = (next_dates - rates.index.to_series()).dt.days.astype(int)
    if (accrual_days < 0).any():
        raise ValueError("cash accrual intervals cannot be negative")

    index_values = np.empty(len(rates), dtype=float)
    index_values[0] = 1.0
    for position in range(1, len(rates)):
        prior_rate = float(rates.iloc[position - 1]["annual_rate"])
        days = int(accrual_days.iloc[position - 1])
        growth = 1.0 + prior_rate / 100.0 * days / DAY_COUNT_DENOMINATOR
        if growth <= 0.0:
            raise ValueError("an overnight rate produced nonpositive cash growth")
        index_values[position] = index_values[position - 1] * growth

    result = rates.copy()
    result.insert(0, "cash_index", index_values)
    result["effective_date"] = result.index
    result["accrual_days"] = accrual_days.to_numpy()
    result.index.name = "date"
    return result[
        [
            "cash_index",
            "annual_rate",
            "source_series",
            "effective_date",
            "accrual_days",
        ]
    ]


def cash_values_on_dates(cash_index, valuation_dates, valuation_through=None):
    """Value the index on arbitrary dates without forward-filling rate data."""
    if not isinstance(cash_index, pd.DataFrame):
        raise TypeError("cash_index must be a pandas DataFrame")
    required = {
        "cash_index",
        "annual_rate",
        "source_series",
        "effective_date",
        "accrual_days",
    }
    if not required.issubset(cash_index.columns):
        raise ValueError("cash_index has missing required columns")
    frame = cash_index.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    if frame.empty or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("cash_index dates must be nonempty, unique, and increasing")

    dates = pd.DatetimeIndex(pd.to_datetime(valuation_dates)).tz_localize(None).normalize()
    if dates.hasnans:
        raise ValueError("valuation_dates must be valid dates")
    if len(dates) == 0:
        return pd.Series(dtype=float, index=dates, name="cash_index")
    if dates.min() < frame.index[0]:
        raise ValueError("valuation date precedes the cash history")

    if valuation_through is None:
        last_row = frame.iloc[-1]
        valuation_through = frame.index[-1] + pd.Timedelta(
            days=int(last_row["accrual_days"])
        )
    valuation_through = pd.Timestamp(valuation_through).tz_localize(None).normalize()
    if dates.max() > valuation_through:
        raise ValueError("valuation date exceeds the published cash horizon")

    positions = frame.index.searchsorted(dates, side="right") - 1
    base = frame.iloc[positions]
    elapsed = (dates - frame.index[positions]).days.astype(float)
    values = base["cash_index"].to_numpy(dtype=float) * (
        1.0
        + base["annual_rate"].to_numpy(dtype=float)
        / 100.0
        * elapsed
        / DAY_COUNT_DENOMINATOR
    )
    return pd.Series(values, index=dates, name="cash_index")


def cash_returns_between(cash_index, start_dates, end_dates, valuation_through=None):
    """Calculate cash returns over paired valuation intervals."""
    starts = pd.DatetimeIndex(pd.to_datetime(start_dates)).tz_localize(None).normalize()
    ends = pd.DatetimeIndex(pd.to_datetime(end_dates)).tz_localize(None).normalize()
    if len(starts) != len(ends):
        raise ValueError("start_dates and end_dates must have equal length")
    if not (ends.to_numpy() > starts.to_numpy()).all():
        raise ValueError("every cash return end date must follow its start date")
    start_values = cash_values_on_dates(cash_index, starts, valuation_through)
    end_values = cash_values_on_dates(cash_index, ends, valuation_through)
    returns = end_values.to_numpy() / start_values.to_numpy() - 1.0
    return pd.Series(returns, index=starts, name=CASH_ASSET)


@dataclass(frozen=True)
class IndexValidation:
    observations: int
    maximum_absolute_error: float
    tolerance: float

    @property
    def passed(self):
        return self.observations > 0 and self.maximum_absolute_error <= self.tolerance


def validate_against_sofr_index(
    cash_index,
    official_sofr_index,
    tolerance=INDEX_TOLERANCE,
    valuation_through=None,
):
    """Reconstruct the published SOFR Index portion and compare point-by-point."""
    official = pd.Series(official_sofr_index, copy=True, dtype=float).dropna()
    official.index = pd.DatetimeIndex(pd.to_datetime(official.index)).tz_localize(None).normalize()
    official = official.sort_index()
    if official.empty or not official.index.is_unique:
        raise ValueError("official_sofr_index must be nonempty with unique dates")
    official = official.loc[official.index >= SOFR_SWITCH_DATE]
    available_end = min(
        official.index[-1],
        pd.Timestamp(valuation_through).normalize()
        if valuation_through is not None
        else official.index[-1],
    )
    official = official.loc[:available_end]
    values = cash_values_on_dates(
        cash_index,
        official.index,
        valuation_through=valuation_through,
    )
    switch_value = float(
        cash_values_on_dates(
            cash_index,
            [SOFR_SWITCH_DATE],
            valuation_through=valuation_through,
        ).iloc[0]
    )
    reconstructed = values / switch_value
    maximum_error = float(np.max(np.abs(reconstructed.to_numpy() - official.to_numpy())))
    validation = IndexValidation(len(official), maximum_error, float(tolerance))
    if not validation.passed:
        raise ValueError(
            "reconstructed SOFR Index exceeds tolerance: "
            f"max_error={maximum_error:.12g}, tolerance={tolerance:.12g}"
        )
    return validation
