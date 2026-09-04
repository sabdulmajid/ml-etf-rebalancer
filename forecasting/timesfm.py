"""Exact offline TimesFM-3 inputs, timing, and inference adapter."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from time import perf_counter

import numpy as np
import pandas as pd
import exchange_calendars as xcals

from backtest.engine import completed_month_end_dates
from data.workbench import xnys_sessions


CHECKPOINT_ID = "google/timesfm-3.0-pytorch"
CHECKPOINT_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
TIMESFM_SOURCE_REVISION = "aa480150652811e732d87a3c5344b235234104e3"
CONTEXT_SESSIONS = 512
PER_CORE_BATCH_SIZE = 16
QUANTILE_LEVELS = tuple(value / 10.0 for value in range(1, 10))
LAST_VALUE_MODE = "last_value"
UNIVARIATE_MODE = "timesfm3_univariate"
MULTIVARIATE_MODE = "timesfm3_multivariate"
MODEL_MODES = (LAST_VALUE_MODE, UNIVARIATE_MODE, MULTIVARIATE_MODE)

EVALUATOR_POLICY = {
    "return_quantiles": True,
    "use_symmetric_averaging": True,
    "make_positive": True,
    "sort_quantiles": True,
    "use_znorm": False,
    "padding_mode": "none",
}


def model_policy():
    """Return the complete, versioned inference policy used for reuse checks."""
    return {
        "policy_version": "timesfm3-zero-shot-policy-v2",
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "timesfm_source_revision": TIMESFM_SOURCE_REVISION,
        "context_sessions": CONTEXT_SESSIONS,
        "universe_mode": "all-14-etfs-jointly",
        "covariates": "none",
        "fine_tuning": False,
        "quantile_levels": list(QUANTILE_LEVELS),
        "holding_score": "q50(period_end) / q50(execution) - 1",
        "cash_hurdle": (
            "last overnight-rate effective date strictly before signal; "
            "Actual/360 over execution-to-period-end calendar days"
        ),
        "evaluator": EVALUATOR_POLICY,
        "primary_mode": MULTIVARIATE_MODE,
        "per_core_batch_size": PER_CORE_BATCH_SIZE,
    }


def model_policy_sha256():
    payload = json.dumps(model_policy(), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _date(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("dates must be valid")
    if stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize()


def _first_xnys_session(calendar, month):
    period = pd.Period(month, freq="M")
    session = calendar.date_to_session(period.start_time, direction="next")
    return pd.Timestamp(session).tz_localize(None).normalize()


def _context_digest(context):
    digest = sha256()
    digest.update("|".join(context.columns).encode("utf-8"))
    digest.update(context.index.to_numpy(dtype="datetime64[ns]").tobytes())
    digest.update(np.ascontiguousarray(context.to_numpy(dtype="float64")).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ForecastWindow:
    """One completed-month forecast origin and its exact trading-day horizon."""

    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    period_end_date: pd.Timestamp
    output_sessions: pd.DatetimeIndex
    context: pd.DataFrame
    context_sha256: str

    @property
    def horizon(self):
        return len(self.output_sessions)


@dataclass(frozen=True)
class ForecastValues:
    """Forecast arrays in ticker-by-horizon order."""

    quantiles: np.ndarray
    inference_seconds: float


def _validate_price_matrix(prices):
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.empty or not prices.columns.is_unique:
        raise ValueError("prices must be nonempty with unique ticker columns")
    frame = prices.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    if (
        frame.index.hasnans
        or not frame.index.is_unique
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError("price dates must be valid, unique, and increasing")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    common = frame.dropna(how="any")
    if common.empty:
        raise ValueError("the forecast universe has no shared complete history")
    if frame.loc[common.index[0] :].isna().any().any():
        raise ValueError("forecast inputs contain an internal missing price")
    values = common.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("forecast prices must be finite and strictly positive")
    expected_sessions = xnys_sessions(common.index[0], common.index[-1])
    if not common.index.equals(expected_sessions):
        raise ValueError("forecast inputs must contain every XNYS session")
    return common


def build_forecast_windows(prices, as_of=None, context_sessions=CONTEXT_SESSIONS):
    """Build all eligible monthly windows, including the latest pending origin.

    The model sees exactly ``context_sessions`` complete XNYS observations through
    a completed month-end signal.  Forecast step one is the next XNYS session
    (execution), and the final step is the following monthly execution date.
    """
    if not isinstance(context_sessions, (int, np.integer)) or context_sessions <= 1:
        raise ValueError("context_sessions must be an integer greater than one")
    common = _validate_price_matrix(prices)
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC")
    as_of = _date(as_of)
    signals = completed_month_end_dates(common.index, as_of=as_of)
    calendar = xcals.get_calendar(
        "XNYS",
        start=common.index[0] - pd.Timedelta(days=7),
        end=max(as_of, common.index[-1]) + pd.Timedelta(days=100),
    )
    windows = []
    for signal_date in signals:
        signal_position = common.index.get_loc(signal_date)
        if signal_position + 1 < context_sessions:
            continue
        signal_month = signal_date.to_period("M")
        execution_date = _first_xnys_session(calendar, signal_month + 1)
        period_end_date = _first_xnys_session(calendar, signal_month + 2)
        output_sessions = pd.DatetimeIndex(
            calendar.sessions_in_range(execution_date, period_end_date)
        ).tz_localize(None)
        if output_sessions[0] != execution_date or output_sessions[-1] != period_end_date:
            raise ValueError("forecast horizon does not reconcile with execution timing")
        context = common.iloc[
            signal_position + 1 - context_sessions : signal_position + 1
        ].copy()
        if context.index[-1] != signal_date or len(context) != context_sessions:
            raise ValueError("forecast context construction failed")
        windows.append(
            ForecastWindow(
                signal_date=pd.Timestamp(signal_date),
                execution_date=execution_date,
                period_end_date=period_end_date,
                output_sessions=output_sessions,
                context=context,
                context_sha256=_context_digest(context),
            )
        )
    if not windows:
        raise ValueError("price history does not support one complete forecast window")
    return tuple(windows)


def last_value_forecast(window):
    """Return the auditable flat-price benchmark for one window."""
    last_values = window.context.iloc[-1].to_numpy(dtype=float)
    values = np.broadcast_to(
        last_values[:, None, None],
        (len(last_values), window.horizon, len(QUANTILE_LEVELS)),
    ).copy()
    return ForecastValues(quantiles=values, inference_seconds=0.0)


class TimesFM3Runner:
    """Small optional adapter around the official TimesFM-3 evaluator.

    Importing this module does not import Torch or TimesFM.  Those optional
    dependencies are loaded only when an offline builder instantiates this class.
    """

    def __init__(
        self,
        checkpoint_path=CHECKPOINT_ID,
        checkpoint_revision=CHECKPOINT_REVISION,
        device=None,
        cache_dir=None,
        local_files_only=False,
        per_core_batch_size=PER_CORE_BATCH_SIZE,
    ):
        try:
            from timesfm3 import ModelConfig, TimesFM3Evaluator
        except ImportError as exc:
            raise RuntimeError(
                "TimesFM-3 is an offline optional dependency; install "
                "requirements-timesfm.txt before building model artifacts"
            ) from exc

        config = ModelConfig(
            checkpoint_path=checkpoint_path,
            revision=checkpoint_revision,
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            per_core_batch_size=per_core_batch_size,
        )
        self._evaluator = TimesFM3Evaluator(config)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_revision = checkpoint_revision
        self.device = str(self._evaluator.device)
        self.per_core_batch_size = int(per_core_batch_size)

    def predict(self, window, mode):
        if mode not in (UNIVARIATE_MODE, MULTIVARIATE_MODE):
            raise ValueError(f"TimesFM runner does not support mode: {mode}")
        context = window.context.to_numpy(dtype=np.float32).T
        if context.shape != (len(window.context.columns), CONTEXT_SESSIONS):
            raise ValueError("TimesFM context shape does not match the fixed policy")
        if not np.isfinite(context).all() or (context <= 0.0).any():
            raise ValueError(
                "TimesFM context must be complete, finite, and positive; "
                "internal interpolation is not permitted"
            )

        start = perf_counter()
        outputs = list(
            self._evaluator.predict_batch(
                contexts=[context],
                horizon=window.horizon,
                past_only_covariates=None,
                past_future_covariates=None,
                return_quantiles=EVALUATOR_POLICY["return_quantiles"],
                use_symmetric_averaging=EVALUATOR_POLICY[
                    "use_symmetric_averaging"
                ],
                make_positive=EVALUATOR_POLICY["make_positive"],
                sort_quantiles=EVALUATOR_POLICY["sort_quantiles"],
                use_znorm=EVALUATOR_POLICY["use_znorm"],
                padding_mode=EVALUATOR_POLICY["padding_mode"],
                univariate=mode == UNIVARIATE_MODE,
            )
        )
        elapsed = perf_counter() - start
        if len(outputs) != 1 or outputs[0].quantiles is None:
            raise RuntimeError("TimesFM returned an unexpected result count")
        quantiles = np.asarray(outputs[0].quantiles, dtype=float)
        expected_shape = (len(window.context.columns), window.horizon, 9)
        if quantiles.shape != expected_shape:
            raise RuntimeError(
                f"TimesFM quantile shape {quantiles.shape} != {expected_shape}"
            )
        if not np.isfinite(quantiles).all() or (quantiles <= 0.0).any():
            raise RuntimeError("TimesFM produced a nonpositive or nonfinite price")
        if (np.diff(quantiles, axis=2) < -1e-10).any():
            raise RuntimeError("TimesFM quantiles are not ordered")
        return ForecastValues(quantiles=quantiles, inference_seconds=elapsed)
