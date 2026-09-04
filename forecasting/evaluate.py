"""Forecast-quality metrics kept separate from portfolio return accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.timesfm import LAST_VALUE_MODE, QUANTILE_LEVELS


METRIC_COLUMNS = (
    "model_mode",
    "scope",
    "ticker",
    "observations",
    "holding_return_mae",
    "holding_return_rmse",
    "directional_accuracy",
    "period_end_price_mase",
    "mean_pinball_loss",
    "interval_80_coverage",
    "mean_interval_80_width",
    "mean_cross_sectional_rank_correlation",
)


def _mean_rank_correlation(frame):
    correlations = []
    for _, month in frame.groupby("signal_date", sort=True):
        if len(month) < 2:
            continue
        predicted = month["forecast_holding_return"].rank(method="average")
        actual = month["actual_holding_return"].rank(method="average")
        if predicted.nunique() < 2 or actual.nunique() < 2:
            continue
        value = predicted.corr(actual)
        if pd.notna(value):
            correlations.append(float(value))
    return float(np.mean(correlations)) if correlations else np.nan


def _metric_row(frame, model_mode, scope, ticker):
    actual_return = frame["actual_holding_return"].to_numpy(dtype=float)
    predicted_return = frame["forecast_holding_return"].to_numpy(dtype=float)
    return_error = predicted_return - actual_return

    actual_end = frame["actual_period_end_price"].to_numpy(dtype=float)
    predicted_end = frame["q50_period_end_price"].to_numpy(dtype=float)
    daily_scale = frame["context_mean_absolute_change"].to_numpy(dtype=float)
    valid_scale = daily_scale > 0.0
    mase = (
        float(np.mean(np.abs(predicted_end[valid_scale] - actual_end[valid_scale]) / daily_scale[valid_scale]))
        if valid_scale.any()
        else np.nan
    )

    origin = frame["signal_price"].to_numpy(dtype=float)
    point_only = model_mode == LAST_VALUE_MODE
    if point_only:
        # A flat forecast is a tie, neither an up nor down prediction.  It also
        # has no probabilistic forecast, even though repeated point values occupy
        # the common artifact columns.
        directional_accuracy = np.nan
        pinball = np.nan
        coverage = np.nan
        interval_width = np.nan
    else:
        directional_accuracy = float(
            np.mean((predicted_return > 0.0) == (actual_return > 0.0))
        )
        pinball_values = []
        for level in QUANTILE_LEVELS:
            label = int(round(level * 100))
            prediction = frame[f"q{label}_period_end_price"].to_numpy(dtype=float)
            error = actual_end - prediction
            loss = np.maximum(level * error, (level - 1.0) * error) / origin
            pinball_values.append(loss)
        pinball = float(np.mean(np.column_stack(pinball_values)))
        low = frame["q10_period_end_price"].to_numpy(dtype=float)
        high = frame["q90_period_end_price"].to_numpy(dtype=float)
        coverage = float(np.mean((actual_end >= low) & (actual_end <= high)))
        interval_width = float(np.mean((high - low) / origin))
    return {
        "model_mode": model_mode,
        "scope": scope,
        "ticker": ticker,
        "observations": int(len(frame)),
        "holding_return_mae": float(np.mean(np.abs(return_error))),
        "holding_return_rmse": float(np.sqrt(np.mean(return_error**2))),
        "directional_accuracy": directional_accuracy,
        "period_end_price_mase": mase,
        "mean_pinball_loss": pinball,
        "interval_80_coverage": coverage,
        "mean_interval_80_width": interval_width,
        "mean_cross_sectional_rank_correlation": (
            _mean_rank_correlation(frame)
            if scope == "overall" and not point_only
            else np.nan
        ),
    }


def evaluate_forecasts(signals):
    """Aggregate only fully realized forecast rows.

    Pending latest forecasts stay in ``forecast_signals.csv`` but are excluded
    here, so future outcomes can never be inferred or silently backfilled.
    """
    if not isinstance(signals, pd.DataFrame):
        raise TypeError("signals must be a pandas DataFrame")
    required = {
        "model_mode",
        "signal_date",
        "ticker",
        "signal_price",
        "context_mean_absolute_change",
        "forecast_holding_return",
        "actual_holding_return",
        "actual_period_end_price",
        "evaluation_status",
        *{
            f"q{int(round(level * 100))}_period_end_price"
            for level in QUANTILE_LEVELS
        },
    }
    missing = sorted(required - set(signals.columns))
    if missing:
        raise ValueError(f"forecast signals lack evaluation columns: {missing}")
    realized = signals.loc[signals["evaluation_status"] == "realized"].copy()
    if realized.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    numeric = [
        "signal_price",
        "context_mean_absolute_change",
        "forecast_holding_return",
        "actual_holding_return",
        "actual_period_end_price",
        *[
            f"q{int(round(level * 100))}_period_end_price"
            for level in QUANTILE_LEVELS
        ],
    ]
    for column in numeric:
        realized[column] = pd.to_numeric(realized[column], errors="raise")
    if not np.isfinite(realized[numeric].to_numpy()).all():
        raise ValueError("realized forecast rows must contain finite evaluation values")

    rows = []
    for mode, mode_frame in realized.groupby("model_mode", sort=True):
        rows.append(_metric_row(mode_frame, mode, "overall", "ALL"))
        for ticker, ticker_frame in mode_frame.groupby("ticker", sort=True):
            rows.append(_metric_row(ticker_frame, mode, "ticker", ticker))
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)
