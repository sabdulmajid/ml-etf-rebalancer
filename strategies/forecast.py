"""TimesFM-filtered workbench targets built from validated offline forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import CASH_ASSET, TargetWeightSchedule
from forecasting.timesfm import MULTIVARIATE_MODE
from strategies.allocation import (
    DEFAULT_POLICY,
    StrategyPolicy,
    allocate_inverse_volatility,
    position_cap,
    trailing_volatility_on_dates,
    validate_price_selection,
)


VOLATILITY_BALANCED_FORECAST = "volatility_balanced_forecast"
FORECAST_POLICY_VERSION = "timesfm-filter-policy-v1"

_REQUIRED_FORECAST_COLUMNS = {
    "model_mode",
    "model_generation_id",
    "signal_date",
    "execution_date",
    "period_end_date",
    "ticker",
    "forecast_holding_return",
    "cash_hurdle",
    "forecast_edge",
    "q10_period_end_price",
    "q50_period_end_price",
    "q90_period_end_price",
    "evaluation_status",
}


@dataclass(frozen=True)
class ForecastAllocationResult:
    """Forecast eligibility decisions and targets, separate from accounting."""

    strategy: str
    policy: StrategyPolicy
    forecast_policy_version: str
    model_mode: str
    selected_etfs: tuple[str, ...]
    schedule: TargetWeightSchedule
    diagnostics: pd.DataFrame
    latest_target: pd.Series
    latest_diagnostics: pd.DataFrame
    latest_signal_date: pd.Timestamp
    latest_execution_date: pd.Timestamp
    latest_period_end_date: pd.Timestamp
    latest_model_generation_id: str


def _normalize_forecasts(forecasts, selected):
    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError("forecasts must be a pandas DataFrame")
    missing = sorted(_REQUIRED_FORECAST_COLUMNS - set(forecasts.columns))
    if missing:
        raise ValueError(f"forecasts are missing required columns: {missing}")
    rows = forecasts.loc[
        (forecasts["model_mode"] == MULTIVARIATE_MODE)
        & forecasts["ticker"].isin(selected),
        list(_REQUIRED_FORECAST_COLUMNS),
    ].copy()
    if rows.empty:
        raise ValueError("no multivariate TimesFM forecasts match the selected ETFs")
    for column in ("signal_date", "execution_date", "period_end_date"):
        rows[column] = pd.to_datetime(rows[column], errors="raise").dt.normalize()
    numeric = [
        "forecast_holding_return",
        "cash_hurdle",
        "forecast_edge",
        "q10_period_end_price",
        "q50_period_end_price",
        "q90_period_end_price",
    ]
    rows[numeric] = rows[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(rows[numeric].to_numpy(dtype=float)).all():
        raise ValueError("forecast decision fields must be finite")
    if rows.duplicated(["signal_date", "ticker"]).any():
        raise ValueError("forecasts must be unique by signal date and ticker")
    if set(rows["evaluation_status"]) - {"realized", "pending"}:
        raise ValueError("forecast evaluation status is invalid")
    if not np.allclose(
        rows["forecast_holding_return"] - rows["cash_hurdle"],
        rows["forecast_edge"],
        rtol=0.0,
        atol=5e-11,
    ):
        raise ValueError("forecast edge does not reconcile with the cash hurdle")
    if not (
        (rows["signal_date"] < rows["execution_date"])
        & (rows["execution_date"] < rows["period_end_date"])
    ).all():
        raise ValueError("forecast signal, execution, and holding dates are invalid")
    expected = set(selected)
    for signal_date, group in rows.groupby("signal_date"):
        if set(group["ticker"]) != expected:
            raise ValueError(
                f"forecast origin {signal_date.date()} does not contain every selected ETF"
            )
        if group[
            [
                "execution_date",
                "period_end_date",
                "evaluation_status",
                "model_generation_id",
            ]
        ].nunique().max() != 1:
            raise ValueError(f"forecast origin {signal_date.date()} has inconsistent timing")
    return rows.sort_values(["signal_date", "ticker"]).reset_index(drop=True)


def _targets_and_diagnostics(rows, selected, volatility_by_signal, policy):
    target_rows = []
    diagnostic_rows = []
    cap = position_cap(len(selected), policy)
    for signal_date, origin in rows.groupby("signal_date", sort=True):
        origin = origin.set_index("ticker").reindex(selected)
        try:
            volatility = volatility_by_signal.loc[signal_date].reindex(selected)
        except KeyError as exc:
            raise ValueError(
                f"price history has no volatility signal for {signal_date.date()}"
            ) from exc
        volatility_ready = volatility.notna() & (volatility > 1e-12)
        inverse_scores = (1.0 / volatility).where(volatility_ready, 0.0)
        score_total = float(inverse_scores.sum())
        raw_weight = (
            inverse_scores / score_total
            if score_total > 0.0
            else pd.Series(0.0, index=selected, dtype=float)
        )
        forecast_pass = origin["forecast_edge"] > 0.0
        eligible = volatility_ready & forecast_pass
        etf_weights = allocate_inverse_volatility(volatility, eligible, policy=policy)
        cash_weight = max(0.0, 1.0 - float(etf_weights.sum()))
        target_rows.append(
            {
                "signal_date": signal_date,
                **etf_weights.to_dict(),
                CASH_ASSET: cash_weight,
            }
        )
        for ticker in selected:
            if not volatility_ready.loc[ticker]:
                status = "Insufficient volatility history"
                reason = "Insufficient or invalid trailing volatility."
            elif not forecast_pass.loc[ticker]:
                status = "Held in cash"
                reason = "Median forecast did not clear the cash return hurdle."
            elif etf_weights.loc[ticker] >= cap - 1e-12:
                status = "Eligible"
                reason = "Forecast cleared cash; weight is limited by the position cap."
            else:
                status = "Eligible"
                reason = "Forecast cleared cash; weighted by trailing volatility."
            diagnostic_rows.append(
                {
                    "signal_date": signal_date,
                    "ticker": ticker,
                    "median_forecast_return": origin.at[
                        ticker, "forecast_holding_return"
                    ],
                    "cash_hurdle": origin.at[ticker, "cash_hurdle"],
                    "forecast_edge": origin.at[ticker, "forecast_edge"],
                    "forecast_status": status,
                    "q10_period_end_price": origin.at[
                        ticker, "q10_period_end_price"
                    ],
                    "q50_period_end_price": origin.at[
                        ticker, "q50_period_end_price"
                    ],
                    "q90_period_end_price": origin.at[
                        ticker, "q90_period_end_price"
                    ],
                    "trailing_volatility": volatility.loc[ticker],
                    "raw_risk_balanced_weight": raw_weight.loc[ticker],
                    "final_target_weight": etf_weights.loc[ticker],
                    "reason": reason,
                }
            )
    targets = pd.DataFrame(target_rows).set_index("signal_date")
    diagnostics = pd.DataFrame(diagnostic_rows).set_index(["signal_date", "ticker"])
    return targets, diagnostics


def generate_forecast_allocation_targets(
    prices,
    selected_etfs,
    forecasts,
    *,
    as_of=None,
    policy=DEFAULT_POLICY,
):
    """Generate forecast-filtered targets from multivariate TimesFM outputs only.

    ``forecast_edge > 0`` is a binary eligibility decision. Passing assets use
    the same inverse-volatility and adaptive-cap implementation as Volatility
    Balanced; forecast magnitude never changes an asset's weight. Only realized
    origins enter the historical schedule. The newest origin, realized or
    pending, supplies the latest research target.
    """
    selected = tuple(selected_etfs)
    prices, selected = validate_price_selection(prices, selected, policy)
    rows = _normalize_forecasts(forecasts, selected)
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC")
    as_of_stamp = pd.Timestamp(as_of)
    if as_of_stamp.tz is not None:
        as_of_stamp = as_of_stamp.tz_convert("UTC").tz_localize(None)
    completed = rows["signal_date"].dt.to_period("M") < as_of_stamp.to_period("M")
    rows = rows.loc[completed].copy()
    if rows.empty:
        raise ValueError("no forecast origin is from a month completed before as_of")
    signal_dates = pd.DatetimeIndex(rows["signal_date"].unique()).sort_values()
    volatility = trailing_volatility_on_dates(
        prices, signal_dates, policy=policy
    )
    targets, diagnostics = _targets_and_diagnostics(
        rows, selected, volatility, policy
    )

    realized_dates = pd.DatetimeIndex(
        rows.loc[
            (rows["evaluation_status"] == "realized")
            & (rows["period_end_date"] <= as_of_stamp),
            "signal_date",
        ].unique()
    ).sort_values()
    if realized_dates.empty:
        raise ValueError("forecast bundle has no realized historical periods")
    realized = rows.loc[rows["signal_date"].isin(realized_dates)]
    timing = (
        realized[["signal_date", "execution_date", "period_end_date"]]
        .drop_duplicates()
        .sort_values("execution_date")
        .set_index("execution_date")
    )
    weights = targets.loc[timing["signal_date"].to_numpy(), [*selected, CASH_ASSET]].copy()
    weights.index = pd.DatetimeIndex(timing.index, name="execution_date")
    schedule = TargetWeightSchedule(
        weights=weights,
        signal_dates=pd.Series(
            timing["signal_date"].to_numpy(), index=weights.index
        ),
        period_end_dates=pd.Series(
            timing["period_end_date"].to_numpy(), index=weights.index
        ),
    )

    latest_signal = pd.Timestamp(rows["signal_date"].max())
    latest_rows = rows.loc[rows["signal_date"] == latest_signal]
    latest_timing = latest_rows[
        ["execution_date", "period_end_date"]
    ].drop_duplicates()
    if len(latest_timing) != 1:
        raise ValueError("latest forecast origin has ambiguous timing")
    generation_ids = latest_rows["model_generation_id"].unique().tolist()
    if len(generation_ids) != 1:
        raise ValueError("latest forecast origin has ambiguous model provenance")
    return ForecastAllocationResult(
        strategy=VOLATILITY_BALANCED_FORECAST,
        policy=policy,
        forecast_policy_version=FORECAST_POLICY_VERSION,
        model_mode=MULTIVARIATE_MODE,
        selected_etfs=selected,
        schedule=schedule,
        diagnostics=diagnostics.loc[
            diagnostics.index.get_level_values("signal_date").isin(realized_dates)
        ],
        latest_target=targets.loc[latest_signal, [*selected, CASH_ASSET]],
        latest_diagnostics=diagnostics.loc[[latest_signal]],
        latest_signal_date=latest_signal,
        latest_execution_date=pd.Timestamp(latest_timing["execution_date"].iloc[0]),
        latest_period_end_date=pd.Timestamp(latest_timing["period_end_date"].iloc[0]),
        latest_model_generation_id=generation_ids[0],
    )
