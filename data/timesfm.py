"""Validated local-only bundle of offline TimesFM research forecasts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from data.workbench import DEFAULT_BUNDLE_PATH as DEFAULT_WORKBENCH_PATH
from data.workbench import canonical_instrument_registry, file_sha256
from forecasting.evaluate import METRIC_COLUMNS
from forecasting.timesfm import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    CONTEXT_SESSIONS,
    EVALUATOR_POLICY,
    LAST_VALUE_MODE,
    MODEL_MODES,
    MULTIVARIATE_MODE,
    QUANTILE_LEVELS,
)


BUNDLE_SCHEMA_VERSION = "timesfm-research-bundle-v1"
PUBLIC_FILENAMES = ("forecast_signals.csv", "forecast_metrics.csv", "manifest.json")
DEFAULT_BUNDLE_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "timesfm"
DATE_COLUMNS = (
    "signal_date",
    "execution_date",
    "period_end_date",
    "context_start",
    "context_end",
    "known_rate_effective_date",
)
QUANTILE_PRICE_COLUMNS = tuple(
    f"q{int(round(level * 100))}_{point}_price"
    for point in ("execution", "period_end")
    for level in QUANTILE_LEVELS
)
SIGNAL_COLUMNS = (
    "model_mode",
    "signal_date",
    "execution_date",
    "period_end_date",
    "ticker",
    "context_start",
    "context_end",
    "context_observations",
    "context_sha256",
    "horizon_sessions",
    "signal_price",
    "context_mean_absolute_change",
    *QUANTILE_PRICE_COLUMNS,
    "forecast_holding_return",
    "known_rate_effective_date",
    "known_rate_source",
    "known_annual_rate",
    "cash_hurdle",
    "forecast_edge",
    "actual_execution_price",
    "actual_period_end_price",
    "actual_holding_return",
    "evaluation_status",
    "inference_seconds",
)


@dataclass(frozen=True)
class TimesFMResearchBundle:
    signals: pd.DataFrame
    metrics: pd.DataFrame
    manifest: dict
    path: Path

    def latest(self, mode=MULTIVARIATE_MODE):
        rows = self.signals.loc[self.signals["model_mode"] == mode]
        if rows.empty:
            raise ValueError(f"forecast bundle contains no rows for {mode}")
        latest_date = rows["signal_date"].max()
        return rows.loc[rows["signal_date"] == latest_date].copy()


def _validate_manifest(manifest, require_clean):
    required = {
        "schema_version",
        "generated_at_utc",
        "artifact_kind",
        "publication_allowed",
        "model_execution_status",
        "checkpoint",
        "configuration",
        "model_modes",
        "ticker_order",
        "workbench_input",
        "forecast_signal_start",
        "forecast_signal_end",
        "row_counts",
        "file_sha256",
        "dependency_versions",
        "git_sha",
        "git_dirty_at_build",
        "pipeline_version",
        "refresh",
        "runtime",
        "performance",
        "validation_status",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"TimesFM manifest is missing required keys: {missing}")
    if manifest["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported TimesFM research artifact schema")
    if manifest["artifact_kind"] != "timesfm3-zero-shot-research":
        raise ValueError("TimesFM artifact kind is invalid")
    if manifest["publication_allowed"] is not True:
        raise ValueError("TimesFM bundle is not approved for publication")
    if manifest["model_execution_status"] != "complete":
        raise ValueError("TimesFM model execution is not complete")
    if manifest["validation_status"] != "passed":
        raise ValueError("TimesFM artifact validation did not pass")
    if manifest["checkpoint"] != {
        "id": CHECKPOINT_ID,
        "revision": CHECKPOINT_REVISION,
    }:
        raise ValueError("TimesFM checkpoint identity or revision changed")
    configuration = manifest["configuration"]
    if configuration.get("context_sessions") != CONTEXT_SESSIONS:
        raise ValueError("TimesFM context policy changed")
    if configuration.get("quantile_levels") != list(QUANTILE_LEVELS):
        raise ValueError("TimesFM quantile policy changed")
    if configuration.get("evaluator") != EVALUATOR_POLICY:
        raise ValueError("TimesFM evaluator policy changed")
    if configuration.get("per_core_batch_size") != 16:
        raise ValueError("TimesFM offline batch policy changed")
    if manifest["model_modes"] != list(MODEL_MODES):
        raise ValueError("TimesFM model-mode set changed")
    expected_tickers = list(canonical_instrument_registry()["ticker"])
    if manifest["ticker_order"] != expected_tickers:
        raise ValueError("TimesFM ticker universe is not the approved 14 ETFs")
    if not isinstance(manifest["git_dirty_at_build"], bool):
        raise ValueError("TimesFM git_dirty_at_build must be boolean")
    if require_clean and manifest["git_dirty_at_build"]:
        raise ValueError("release TimesFM artifacts require a clean Git build")
    if re.fullmatch(r"[0-9a-fA-F]{40}", str(manifest["git_sha"])) is None:
        raise ValueError("TimesFM git_sha must be a 40-character hexadecimal hash")
    workbench = manifest["workbench_input"]
    if not isinstance(workbench, dict) or not {
        "schema_version",
        "price_data_as_of",
        "last_complete_month",
        "file_sha256",
    }.issubset(workbench):
        raise ValueError("TimesFM workbench-input provenance is incomplete")
    if not isinstance(manifest["dependency_versions"], dict):
        raise ValueError("TimesFM dependency provenance is incomplete")


def _validate_signals(signals, manifest):
    if list(signals.columns) != list(SIGNAL_COLUMNS):
        raise ValueError("forecast_signals.csv columns do not match the v1 schema")
    if signals.empty:
        raise ValueError("forecast_signals.csv must not be empty")
    signals = signals.copy()
    for column in DATE_COLUMNS:
        signals[column] = pd.to_datetime(signals[column], errors="raise")
    if signals[list(DATE_COLUMNS)].isna().any().any():
        raise ValueError("forecast timing columns must be complete")
    if not (signals["signal_date"] < signals["execution_date"]).all():
        raise ValueError("every forecast signal must precede execution")
    if not (signals["execution_date"] < signals["period_end_date"]).all():
        raise ValueError("every forecast period end must follow execution")
    if not (signals["context_end"] == signals["signal_date"]).all():
        raise ValueError("every forecast context must end on its signal date")
    if not (signals["known_rate_effective_date"] < signals["signal_date"]).all():
        raise ValueError("cash hurdle uses a rate not safely known at the signal")

    key = ["model_mode", "signal_date", "ticker"]
    if signals.duplicated(key).any():
        raise ValueError("forecast rows must be unique by mode, signal date, and ticker")
    if set(signals["model_mode"]) != set(MODEL_MODES):
        raise ValueError("forecast rows do not contain every required model mode")
    tickers = list(canonical_instrument_registry()["ticker"])
    if set(signals["ticker"]) != set(tickers):
        raise ValueError("forecast rows do not contain the approved ETF universe")
    expected_per_signal = len(MODEL_MODES) * len(tickers)
    counts = signals.groupby("signal_date").size()
    if not (counts == expected_per_signal).all():
        raise ValueError("every signal date must contain every mode and ETF")

    numeric = [
        "context_observations",
        "horizon_sessions",
        "signal_price",
        "context_mean_absolute_change",
        *QUANTILE_PRICE_COLUMNS,
        "forecast_holding_return",
        "known_annual_rate",
        "cash_hurdle",
        "forecast_edge",
        "inference_seconds",
    ]
    for column in numeric:
        signals[column] = pd.to_numeric(signals[column], errors="raise")
    if not np.isfinite(signals[numeric].to_numpy()).all():
        raise ValueError("forecast rows contain nonfinite required values")
    if not (signals["context_observations"] == CONTEXT_SESSIONS).all():
        raise ValueError("forecast rows do not use the fixed context")
    if (signals["horizon_sessions"] <= 1).any():
        raise ValueError("forecast horizons must include execution and period end")
    if (signals[["signal_price", *QUANTILE_PRICE_COLUMNS]] <= 0.0).any().any():
        raise ValueError("forecast price levels must be positive")
    if (signals["context_mean_absolute_change"] <= 0.0).any():
        raise ValueError("forecast context scales must be positive")
    if (signals["inference_seconds"] < 0.0).any():
        raise ValueError("forecast inference time cannot be negative")
    if not signals["context_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("forecast context checksums are invalid")

    for point in ("execution", "period_end"):
        columns = [
            f"q{int(round(level * 100))}_{point}_price"
            for level in QUANTILE_LEVELS
        ]
        if (np.diff(signals[columns].to_numpy(dtype=float), axis=1) < -1e-10).any():
            raise ValueError(f"forecast quantiles cross at {point}")
    expected_return = (
        signals["q50_period_end_price"] / signals["q50_execution_price"] - 1.0
    )
    if not np.allclose(
        expected_return,
        signals["forecast_holding_return"],
        rtol=0.0,
        atol=5e-11,
    ):
        raise ValueError("forecast holding returns do not reconcile with q50 prices")
    if not np.allclose(
        signals["forecast_holding_return"] - signals["cash_hurdle"],
        signals["forecast_edge"],
        rtol=0.0,
        atol=5e-11,
    ):
        raise ValueError("forecast edges do not reconcile with the cash hurdle")
    expected_hurdle = (
        signals["known_annual_rate"]
        / 100.0
        * (signals["period_end_date"] - signals["execution_date"]).dt.days
        / 360.0
    )
    if not np.allclose(
        expected_hurdle, signals["cash_hurdle"], rtol=0.0, atol=5e-11
    ):
        raise ValueError("cash hurdles do not use the recorded Actual/360 rate")

    actual_columns = [
        "actual_execution_price",
        "actual_period_end_price",
        "actual_holding_return",
    ]
    statuses = set(signals["evaluation_status"])
    if not statuses.issubset({"realized", "pending"}):
        raise ValueError("forecast evaluation_status is invalid")
    realized = signals["evaluation_status"] == "realized"
    if signals.loc[realized, actual_columns].isna().any().any():
        raise ValueError("realized forecast rows must contain actual outcomes")
    if signals.loc[~realized, actual_columns].notna().any().any():
        raise ValueError("pending forecast rows cannot contain actual outcomes")
    if realized.any():
        actual = signals.loc[realized, actual_columns].apply(
            pd.to_numeric, errors="raise"
        )
        if not np.isfinite(actual.to_numpy()).all() or (
            actual[["actual_execution_price", "actual_period_end_price"]] <= 0.0
        ).any().any():
            raise ValueError("realized forecast outcomes are invalid")
        expected_actual = (
            actual["actual_period_end_price"] / actual["actual_execution_price"] - 1.0
        )
        if not np.allclose(
            expected_actual,
            actual["actual_holding_return"],
            rtol=0.0,
            atol=5e-11,
        ):
            raise ValueError("actual holding returns do not reconcile")

    baseline = signals["model_mode"] == LAST_VALUE_MODE
    baseline_prices = signals.loc[baseline, QUANTILE_PRICE_COLUMNS]
    expected = signals.loc[baseline, "signal_price"].to_numpy()[:, None]
    if not np.allclose(baseline_prices.to_numpy(), expected, rtol=0.0, atol=5e-11):
        raise ValueError("last-value forecasts are not flat at the signal price")

    start = str(signals["signal_date"].min().date())
    end = str(signals["signal_date"].max().date())
    if manifest["forecast_signal_start"] != start or manifest["forecast_signal_end"] != end:
        raise ValueError("forecast signal range does not reconcile with the manifest")
    return signals.sort_values(key).reset_index(drop=True)


def validate_timesfm_bundle(path=DEFAULT_BUNDLE_PATH, verify_checksums=True, require_clean=True):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"TimesFM artifact directory does not exist: {path}")
    actual = tuple(sorted(item.name for item in path.iterdir()))
    if actual != tuple(sorted(PUBLIC_FILENAMES)):
        raise ValueError("TimesFM artifact directory must contain exactly three files")
    manifest = json.loads((path / "manifest.json").read_text())
    _validate_manifest(manifest, require_clean=require_clean)
    if verify_checksums:
        for filename in PUBLIC_FILENAMES[:-1]:
            if manifest["file_sha256"].get(filename) != file_sha256(path / filename):
                raise ValueError(f"checksum mismatch for {filename}")
    signals = pd.read_csv(
        path / "forecast_signals.csv", float_precision="round_trip"
    )
    metrics = pd.read_csv(
        path / "forecast_metrics.csv", float_precision="round_trip"
    )
    if manifest["row_counts"] != {
        "forecast_signals.csv": len(signals),
        "forecast_metrics.csv": len(metrics),
    }:
        raise ValueError("TimesFM artifact row counts do not reconcile")
    signals = _validate_signals(signals, manifest)
    if list(metrics.columns) != list(METRIC_COLUMNS):
        raise ValueError("forecast_metrics.csv columns do not match the v1 schema")
    if metrics.empty or set(metrics["model_mode"]) != set(MODEL_MODES):
        raise ValueError("forecast metrics do not cover every model mode")
    return TimesFMResearchBundle(signals=signals, metrics=metrics, manifest=manifest, path=path)


def load_timesfm_bundle(path=DEFAULT_BUNDLE_PATH, workbench_path=DEFAULT_WORKBENCH_PATH):
    """Load the local artifact bundle without importing TimesFM or using HTTP."""
    bundle = validate_timesfm_bundle(path)
    expected = bundle.manifest["workbench_input"]["file_sha256"]
    for filename, digest in expected.items():
        source = Path(workbench_path) / filename
        if not source.is_file() or file_sha256(source) != digest:
            raise ValueError(
                f"TimesFM forecasts do not match the current workbench {filename}"
            )
    return bundle
