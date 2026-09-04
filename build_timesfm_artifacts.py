"""Build the small offline TimesFM-3 research artifact bundle.

This script is the only project entrypoint that imports the optional TimesFM
stack.  The public Streamlit app reads committed CSV/JSON outputs and never
loads Torch, downloads a checkpoint, or performs model inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as dependency_version
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

from data.timesfm import (
    BUNDLE_SCHEMA_VERSION,
    DEFAULT_BUNDLE_PATH,
    PUBLIC_FILENAMES,
    SIGNAL_COLUMNS,
    validate_timesfm_bundle,
)
from data.workbench import DEFAULT_BUNDLE_PATH as DEFAULT_WORKBENCH_PATH
from data.workbench import file_sha256, load_workbench_bundle
from forecasting.evaluate import evaluate_forecasts
from forecasting.timesfm import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    CONTEXT_SESSIONS,
    EVALUATOR_POLICY,
    LAST_VALUE_MODE,
    MODEL_MODES,
    MULTIVARIATE_MODE,
    QUANTILE_LEVELS,
    TIMESFM_SOURCE_REVISION,
    UNIVARIATE_MODE,
    TimesFM3Runner,
    build_forecast_windows,
    last_value_forecast,
)


PIPELINE_VERSION = "timesfm-artifact-builder-v1"


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_dirty():
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _dependency_versions():
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "exchange-calendars",
        "timesfm",
        "torch",
        "huggingface-hub",
        "safetensors",
    ):
        try:
            versions[package] = dependency_version(package)
        except PackageNotFoundError:
            if package in {"timesfm", "torch", "huggingface-hub", "safetensors"}:
                versions[package] = None
            else:
                raise
    return versions


def _rate_hurdle(cash_index, signal_date, execution_date, period_end_date):
    history = cash_index.loc[cash_index.index < signal_date]
    if history.empty:
        raise ValueError(f"no safely known overnight rate precedes {signal_date.date()}")
    row = history.iloc[-1]
    effective_date = pd.Timestamp(row["effective_date"]).normalize()
    if effective_date >= signal_date:
        raise ValueError("cash hurdle rate must be effective strictly before the signal")
    annual_rate = float(row["annual_rate"])
    days = int((period_end_date - execution_date).days)
    hurdle = annual_rate / 100.0 * days / 360.0
    return effective_date, str(row["source_series"]), annual_rate, hurdle


def _actual_outcome(prices, execution_date, period_end_date, ticker):
    if execution_date not in prices.index or period_end_date not in prices.index:
        return np.nan, np.nan, np.nan, "pending"
    execution = prices.at[execution_date, ticker]
    period_end = prices.at[period_end_date, ticker]
    if pd.isna(execution) or pd.isna(period_end):
        return np.nan, np.nan, np.nan, "pending"
    execution = float(execution)
    period_end = float(period_end)
    return execution, period_end, period_end / execution - 1.0, "realized"


def _forecast_rows(window, mode, values, cash_index, prices):
    expected_shape = (
        len(window.context.columns),
        window.horizon,
        len(QUANTILE_LEVELS),
    )
    quantiles = np.asarray(values.quantiles, dtype=float)
    if quantiles.shape != expected_shape:
        raise ValueError(f"forecast shape {quantiles.shape} != {expected_shape}")
    rate_date, rate_source, annual_rate, hurdle = _rate_hurdle(
        cash_index,
        window.signal_date,
        window.execution_date,
        window.period_end_date,
    )
    rows = []
    for ticker_position, ticker in enumerate(window.context.columns):
        context = window.context[ticker].to_numpy(dtype=float)
        row = {
            "model_mode": mode,
            "signal_date": window.signal_date,
            "execution_date": window.execution_date,
            "period_end_date": window.period_end_date,
            "ticker": ticker,
            "context_start": window.context.index[0],
            "context_end": window.context.index[-1],
            "context_observations": len(window.context),
            "context_sha256": window.context_sha256,
            "horizon_sessions": window.horizon,
            "signal_price": context[-1],
            "context_mean_absolute_change": float(np.mean(np.abs(np.diff(context)))),
        }
        for point, position in (("execution", 0), ("period_end", -1)):
            for quantile_position, level in enumerate(QUANTILE_LEVELS):
                label = int(round(level * 100))
                row[f"q{label}_{point}_price"] = quantiles[
                    ticker_position, position, quantile_position
                ]
        row["forecast_holding_return"] = (
            row["q50_period_end_price"] / row["q50_execution_price"] - 1.0
        )
        row["known_rate_effective_date"] = rate_date
        row["known_rate_source"] = rate_source
        row["known_annual_rate"] = annual_rate
        row["cash_hurdle"] = hurdle
        row["forecast_edge"] = row["forecast_holding_return"] - hurdle
        (
            row["actual_execution_price"],
            row["actual_period_end_price"],
            row["actual_holding_return"],
            row["evaluation_status"],
        ) = _actual_outcome(
            prices, window.execution_date, window.period_end_date, ticker
        )
        row["inference_seconds"] = float(values.inference_seconds)
        rows.append(row)
    return rows


def _reusable_rows(existing, window, mode, rate_details):
    if existing is None:
        return None
    mask = (existing["model_mode"] == mode) & (
        existing["signal_date"] == window.signal_date
    )
    rows = existing.loc[mask]
    if len(rows) != len(window.context.columns):
        return None
    rate_date, rate_source, annual_rate, hurdle = rate_details
    checks = (
        (rows["context_sha256"] == window.context_sha256).all()
        and (rows["execution_date"] == window.execution_date).all()
        and (rows["period_end_date"] == window.period_end_date).all()
        and (rows["horizon_sessions"] == window.horizon).all()
        and (rows["known_rate_effective_date"] == rate_date).all()
        and (rows["known_rate_source"] == rate_source).all()
        and np.allclose(rows["known_annual_rate"], annual_rate, rtol=0.0, atol=1e-12)
        and np.allclose(rows["cash_hurdle"], hurdle, rtol=0.0, atol=1e-12)
        and set(rows["ticker"]) == set(window.context.columns)
    )
    return rows.copy() if checks else None


def _load_incremental_source(output_dir, full):
    output_dir = Path(output_dir)
    if full or not output_dir.exists():
        return None
    return validate_timesfm_bundle(output_dir, require_clean=False)


def _promote_directory(staging, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    moved_destination = False
    if destination.exists():
        os.replace(destination, backup)
        moved_destination = True
    try:
        os.replace(staging, destination)
    except Exception:
        if moved_destination:
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _write_bundle(staging, signals, metrics, manifest):
    staging.mkdir(parents=True)
    signals.to_csv(
        staging / "forecast_signals.csv",
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
    )
    metrics.to_csv(
        staging / "forecast_metrics.csv", index=False, float_format="%.17g"
    )
    manifest = dict(manifest)
    manifest["row_counts"] = {
        "forecast_signals.csv": len(signals),
        "forecast_metrics.csv": len(metrics),
    }
    manifest["file_sha256"] = {
        filename: file_sha256(staging / filename)
        for filename in PUBLIC_FILENAMES[:-1]
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build_bundle(
    output_dir=DEFAULT_BUNDLE_PATH,
    workbench_dir=DEFAULT_WORKBENCH_PATH,
    *,
    full=False,
    allow_dirty=False,
    runner=None,
    device=None,
    cache_dir=None,
    local_files_only=False,
    generated_at=None,
):
    """Run missing monthly forecasts and atomically promote a validated bundle."""
    git_sha = _git_sha()
    git_dirty = _git_dirty()
    if not isinstance(git_sha, str) or len(git_sha) != 40:
        raise RuntimeError("refusing to build without a valid Git commit SHA")
    if git_dirty is None:
        raise RuntimeError("unable to determine Git working-tree provenance")
    if git_dirty and not allow_dirty:
        raise RuntimeError("refusing to build release forecasts from a dirty Git tree")

    workbench = load_workbench_bundle(workbench_dir)
    as_of = pd.Timestamp(workbench.manifest["generated_at_utc"])
    windows = build_forecast_windows(workbench.adjusted_close, as_of=as_of)
    existing_bundle = _load_incremental_source(output_dir, full=full)
    existing = None if existing_bundle is None else existing_bundle.signals

    rows = []
    reused_groups = 0
    generated_groups = 0
    for window in windows:
        rate_details = _rate_hurdle(
            workbench.cash_index,
            window.signal_date,
            window.execution_date,
            window.period_end_date,
        )
        for mode in MODEL_MODES:
            reusable = _reusable_rows(existing, window, mode, rate_details)
            if reusable is not None:
                # Outcomes may have become observable in a refreshed price bundle.
                reused = []
                for record in reusable.to_dict("records"):
                    (
                        record["actual_execution_price"],
                        record["actual_period_end_price"],
                        record["actual_holding_return"],
                        record["evaluation_status"],
                    ) = _actual_outcome(
                        workbench.adjusted_close,
                        window.execution_date,
                        window.period_end_date,
                        record["ticker"],
                    )
                    reused.append(record)
                rows.extend(reused)
                reused_groups += 1
                continue
            if mode == LAST_VALUE_MODE:
                values = last_value_forecast(window)
            else:
                if runner is None:
                    runner = TimesFM3Runner(
                        checkpoint_path=CHECKPOINT_ID,
                        checkpoint_revision=CHECKPOINT_REVISION,
                        device=device,
                        cache_dir=cache_dir,
                        local_files_only=local_files_only,
                    )
                values = runner.predict(window, mode)
            rows.extend(
                _forecast_rows(
                    window,
                    mode,
                    values,
                    workbench.cash_index,
                    workbench.adjusted_close,
                )
            )
            generated_groups += 1

    signals = pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
    signals = signals.sort_values(["model_mode", "signal_date", "ticker"]).reset_index(
        drop=True
    )
    metrics = evaluate_forecasts(signals)
    performance = {}
    for mode, mode_rows in signals.groupby("model_mode", sort=True):
        timings = mode_rows.drop_duplicates(["model_mode", "signal_date"])[
            "inference_seconds"
        ].to_numpy(dtype=float)
        performance[mode] = {
            "forecast_origins": int(len(timings)),
            "total_inference_seconds": float(timings.sum()),
            "median_inference_seconds": float(np.median(timings)),
            "maximum_inference_seconds": float(timings.max()),
        }
    if generated_at is None:
        generated_at = datetime.now(timezone.utc)
    else:
        generated_at = pd.Timestamp(generated_at)
        if generated_at.tz is None:
            generated_at = generated_at.tz_localize("UTC")
        else:
            generated_at = generated_at.tz_convert("UTC")
        generated_at = generated_at.to_pydatetime()

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z"),
        "artifact_kind": "timesfm3-zero-shot-research",
        "publication_allowed": True,
        "model_execution_status": "complete",
        "checkpoint": {
            "id": CHECKPOINT_ID,
            "revision": CHECKPOINT_REVISION,
            "timesfm_source_revision": TIMESFM_SOURCE_REVISION,
        },
        "configuration": {
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
            "per_core_batch_size": 16,
        },
        "model_modes": list(MODEL_MODES),
        "ticker_order": list(workbench.tickers),
        "workbench_input": {
            "schema_version": workbench.manifest["schema_version"],
            "price_data_as_of": workbench.manifest["price_data_as_of"],
            "last_complete_month": workbench.manifest["last_complete_month"],
            "file_sha256": {
                filename: file_sha256(Path(workbench_dir) / filename)
                for filename in (
                    "adjusted_close.csv",
                    "cash_index.csv",
                    "instruments.csv",
                    "manifest.json",
                )
            },
        },
        "forecast_signal_start": str(signals["signal_date"].min().date()),
        "forecast_signal_end": str(signals["signal_date"].max().date()),
        "dependency_versions": _dependency_versions(),
        "git_sha": git_sha,
        "git_dirty_at_build": git_dirty,
        "pipeline_version": PIPELINE_VERSION,
        "runtime": (
            existing_bundle.manifest["runtime"]
            if runner is None and existing_bundle is not None
            else {
                "device": getattr(runner, "device", "injected-test-runner"),
                "builder_peak_rss_kb": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
            }
        ),
        "performance": performance,
        "refresh": {
            "mode": "full" if full or existing is None else "incremental",
            "reused_mode_signal_groups": reused_groups,
            "generated_mode_signal_groups": generated_groups,
        },
        "validation_status": "passed",
    }

    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-build-", dir=destination.parent)
    )
    shutil.rmtree(staging)
    try:
        _write_bundle(staging, signals, metrics, manifest)
        validate_timesfm_bundle(staging, require_clean=not allow_dirty)
        _promote_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return validate_timesfm_bundle(destination, require_clean=not allow_dirty)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--workbench-dir", type=Path, default=DEFAULT_WORKBENCH_PATH)
    parser.add_argument("--full", action="store_true", help="ignore reusable forecasts")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--device", default=None, help="TimesFM device, e.g. cpu or cuda")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    bundle = build_bundle(
        args.output_dir,
        args.workbench_dir,
        full=args.full,
        allow_dirty=args.allow_dirty,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    realized = int((bundle.signals["evaluation_status"] == "realized").sum())
    print(
        f"Validated {len(bundle.signals):,} forecast rows through "
        f"{bundle.manifest['forecast_signal_end']} ({realized:,} realized rows) "
        f"at {bundle.path}"
    )


if __name__ == "__main__":
    main()
