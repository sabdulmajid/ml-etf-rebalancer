"""Validated local-only bundle of offline TimesFM research forecasts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from data.workbench import DEFAULT_BUNDLE_PATH as DEFAULT_WORKBENCH_PATH
from data.workbench import canonical_instrument_registry, file_sha256
from forecasting.evaluate import METRIC_COLUMNS, evaluate_forecasts
from forecasting.timesfm import (
    CONTEXT_SESSIONS,
    LAST_VALUE_MODE,
    MODEL_MODES,
    MULTIVARIATE_MODE,
    QUANTILE_LEVELS,
    build_forecast_windows,
    model_policy,
    model_policy_sha256,
)


BUNDLE_SCHEMA_VERSION = "timesfm-research-bundle-v2"
PIPELINE_VERSION = "timesfm-artifact-builder-v2"
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
    "model_generation_id",
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

    def freshness(self, as_of=None):
        """Separate durable historical availability from latest-target validity."""
        if as_of is None:
            as_of = pd.Timestamp.now(tz="UTC")
        as_of = pd.Timestamp(as_of)
        if as_of.tz is not None:
            as_of = as_of.tz_convert("UTC").tz_localize(None)
        generated = pd.Timestamp(self.manifest["bundle_refresh"]["generated_at_utc"])
        if generated.tz is not None:
            generated = generated.tz_convert("UTC").tz_localize(None)
        latest = self.latest()
        signal = pd.Timestamp(latest["signal_date"].iloc[0])
        execution = pd.Timestamp(latest["execution_date"].iloc[0])
        period_end = pd.Timestamp(latest["period_end_date"].iloc[0])
        generation_ids = latest["model_generation_id"].unique().tolist()
        if len(generation_ids) != 1:
            raise ValueError("latest forecast has ambiguous model-generation provenance")
        generation = next(
            (
                item
                for item in self.manifest["model_generations"]
                if item["id"] == generation_ids[0]
            ),
            None,
        )
        if generation is None:
            raise ValueError("latest forecast has unknown model-generation provenance")
        generation_time = pd.Timestamp(generation["generated_at_utc"])
        if generation_time.tz is not None:
            generation_time = generation_time.tz_convert("UTC").tz_localize(None)
        calendar = xcals.get_calendar(
            "XNYS",
            start=execution - pd.Timedelta(days=7),
            end=execution + pd.Timedelta(days=7),
        )
        execution_close = pd.Timestamp(calendar.session_close(execution))
        if execution_close.tz is not None:
            execution_close = execution_close.tz_convert("UTC").tz_localize(None)
        price_as_of = pd.Timestamp(
            self.manifest["workbench_input"]["price_data_as_of"]
        )
        if generated > as_of:
            status, reason = "disabled", "forecast artifact was generated in the future"
        elif generation_time > execution_close:
            status, reason = (
                "disabled",
                "latest forecast was generated after its execution cutoff",
            )
        elif as_of > period_end:
            status, reason = "disabled", "latest forecast holding period has expired"
        elif as_of < execution:
            status, reason = "scheduled", "latest target awaits its execution date"
        else:
            status, reason = "current", None
        return {
            "historical_available": True,
            "latest_target_status": status,
            "reason": reason,
            "price_data_as_of": str(price_as_of.date()),
            "latest_signal_date": str(signal.date()),
            "latest_execution_date": str(execution.date()),
            "latest_execution_cutoff_utc": execution_close.isoformat() + "Z",
            "latest_period_end_date": str(period_end.date()),
            "latest_model_generated_at_utc": generation_time.isoformat() + "Z",
        }


def _validate_manifest(manifest, require_clean):
    required = {
        "schema_version",
        "artifact_kind",
        "publication_allowed",
        "model_execution_status",
        "model_policy",
        "model_policy_sha256",
        "model_modes",
        "ticker_order",
        "workbench_input",
        "forecast_signal_start",
        "forecast_signal_end",
        "row_counts",
        "file_sha256",
        "pipeline_version",
        "model_generations",
        "bundle_refresh",
        "evaluation_classification",
        "pretraining_overlap",
        "validation_status",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"TimesFM manifest is missing required keys: {missing}")
    if manifest["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported TimesFM research artifact schema")
    if manifest["pipeline_version"] != PIPELINE_VERSION:
        raise ValueError("TimesFM artifact pipeline policy changed")
    if manifest["artifact_kind"] != "timesfm3-zero-shot-research":
        raise ValueError("TimesFM artifact kind is invalid")
    if manifest["publication_allowed"] is not True:
        raise ValueError("TimesFM bundle is not approved for publication")
    if manifest["model_execution_status"] != "complete":
        raise ValueError("TimesFM model execution is not complete")
    if manifest["validation_status"] != "passed":
        raise ValueError("TimesFM artifact validation did not pass")
    if manifest["model_policy"] != model_policy():
        raise ValueError("TimesFM model policy changed")
    if manifest["model_policy_sha256"] != model_policy_sha256():
        raise ValueError("TimesFM model policy digest changed")
    if manifest["evaluation_classification"] != "historical_replay":
        raise ValueError("TimesFM evaluation must be classified as historical replay")
    if manifest["pretraining_overlap"] != "unknown":
        raise ValueError("TimesFM pretraining-overlap disclosure is invalid")
    if manifest["model_modes"] != list(MODEL_MODES):
        raise ValueError("TimesFM model-mode set changed")
    expected_tickers = list(canonical_instrument_registry()["ticker"])
    if manifest["ticker_order"] != expected_tickers:
        raise ValueError("TimesFM ticker universe is not the approved 14 ETFs")
    refresh = manifest["bundle_refresh"]
    refresh_required = {
        "generated_at_utc",
        "git_sha",
        "git_dirty_at_build",
        "mode",
        "reused_mode_signal_groups",
        "generated_baseline_groups",
        "generated_model_groups",
    }
    if not isinstance(refresh, dict) or not refresh_required.issubset(refresh):
        raise ValueError("TimesFM bundle-refresh provenance is incomplete")
    if re.fullmatch(r"[0-9a-fA-F]{40}", str(refresh["git_sha"])) is None:
        raise ValueError("TimesFM refresh Git SHA is invalid")
    if not isinstance(refresh["git_dirty_at_build"], bool):
        raise ValueError("TimesFM refresh dirty flag must be boolean")
    if require_clean and refresh["git_dirty_at_build"]:
        raise ValueError("release TimesFM artifacts require a clean Git refresh")
    pd.Timestamp(refresh["generated_at_utc"])

    generations = manifest["model_generations"]
    if not isinstance(generations, list) or not generations:
        raise ValueError("TimesFM model-generation provenance is missing")
    generation_ids = set()
    for generation in generations:
        required_generation = {
            "id",
            "generated_at_utc",
            "git_sha",
            "git_dirty_at_build",
            "dependency_versions",
            "runtime",
            "model_policy_sha256",
            "performance",
        }
        if not isinstance(generation, dict) or not required_generation.issubset(generation):
            raise ValueError("TimesFM model-generation provenance is incomplete")
        if generation["id"] in generation_ids:
            raise ValueError("TimesFM model-generation IDs must be unique")
        generation_ids.add(generation["id"])
        if generation["model_policy_sha256"] != model_policy_sha256():
            raise ValueError("TimesFM generation uses a different model policy")
        if re.fullmatch(r"[0-9a-fA-F]{40}", str(generation["git_sha"])) is None:
            raise ValueError("TimesFM generation Git SHA is invalid")
        if not isinstance(generation["git_dirty_at_build"], bool):
            raise ValueError("TimesFM generation dirty flag must be boolean")
        if require_clean and generation["git_dirty_at_build"]:
            raise ValueError("release TimesFM artifacts cannot use dirty model outputs")
        pd.Timestamp(generation["generated_at_utc"])
        dependencies = generation["dependency_versions"]
        if not isinstance(dependencies, dict) or not all(
            isinstance(dependencies.get(name), str) and dependencies[name]
            for name in ("timesfm", "torch", "huggingface-hub", "safetensors")
        ):
            raise ValueError("TimesFM generation dependency provenance is incomplete")
        runtime = generation["runtime"]
        if not isinstance(runtime, dict) or not {
            "device",
            "builder_peak_rss_kb",
        }.issubset(runtime):
            raise ValueError("TimesFM generation runtime provenance is incomplete")
    workbench = manifest["workbench_input"]
    if not isinstance(workbench, dict) or not {
        "schema_version",
        "price_data_as_of",
        "last_complete_month",
        "file_sha256",
    }.issubset(workbench):
        raise ValueError("TimesFM workbench-input provenance is incomplete")


def _validate_signals(signals, manifest):
    if list(signals.columns) != list(SIGNAL_COLUMNS):
        raise ValueError("forecast_signals.csv columns do not match the v2 schema")
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
    model_rows = signals["model_mode"] != LAST_VALUE_MODE
    generation_ids = {item["id"] for item in manifest["model_generations"]}
    if set(signals.loc[model_rows, "model_generation_id"]) - generation_ids:
        raise ValueError("forecast rows reference unknown model-generation provenance")
    if not (signals.loc[~model_rows, "model_generation_id"] == "point-baseline-v1").all():
        raise ValueError("last-value rows must use point-baseline-v1 provenance")

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


def _validate_metrics(metrics, signals):
    if list(metrics.columns) != list(METRIC_COLUMNS):
        raise ValueError("forecast_metrics.csv columns do not match the v2 schema")
    tickers = list(canonical_instrument_registry()["ticker"])
    expected_keys = {
        (mode, "overall", "ALL") for mode in MODEL_MODES
    } | {
        (mode, "ticker", ticker) for mode in MODEL_MODES for ticker in tickers
    }
    keys = set(metrics[["model_mode", "scope", "ticker"]].itertuples(index=False, name=None))
    if keys != expected_keys or metrics.duplicated(
        ["model_mode", "scope", "ticker"]
    ).any():
        raise ValueError("forecast metrics have incomplete or duplicate keys")
    point_columns = [
        "holding_return_mae",
        "holding_return_rmse",
        "period_end_price_mase",
    ]
    probabilistic = [
        "mean_pinball_loss",
        "interval_80_coverage",
        "mean_interval_80_width",
    ]
    metrics = metrics.copy()
    for column in ["observations", *point_columns, "directional_accuracy", *probabilistic, "mean_cross_sectional_rank_correlation"]:
        metrics[column] = pd.to_numeric(metrics[column], errors="raise")
    if metrics[["observations", *point_columns]].isna().any().any():
        raise ValueError("forecast point metrics cannot be missing")
    if (metrics["observations"] <= 0).any() or not np.equal(
        metrics["observations"] % 1, 0
    ).all():
        raise ValueError("forecast metric observation counts are invalid")
    if (metrics[point_columns] < 0.0).any().any() or not np.isfinite(
        metrics[point_columns].to_numpy()
    ).all():
        raise ValueError("forecast point metrics are invalid")
    baseline = metrics["model_mode"] == LAST_VALUE_MODE
    if metrics.loc[baseline, ["directional_accuracy", *probabilistic]].notna().any().any():
        raise ValueError("point-only last-value metrics must be N/A")
    model = ~baseline
    if metrics.loc[model, ["directional_accuracy", *probabilistic]].isna().any().any():
        raise ValueError("TimesFM directional/probabilistic metrics cannot be missing")
    if not metrics.loc[model, "directional_accuracy"].between(0.0, 1.0).all():
        raise ValueError("directional accuracy is outside [0, 1]")
    if not metrics.loc[model, "interval_80_coverage"].between(0.0, 1.0).all():
        raise ValueError("interval coverage is outside [0, 1]")
    if (metrics.loc[model, ["mean_pinball_loss", "mean_interval_80_width"]] < 0.0).any().any():
        raise ValueError("probabilistic metrics cannot be negative")
    if not np.isfinite(
        metrics.loc[
            model,
            ["directional_accuracy", *probabilistic],
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("TimesFM directional/probabilistic metrics must be finite")
    overall_model = model & (metrics["scope"] == "overall")
    ticker_or_baseline = ~overall_model
    rank = metrics["mean_cross_sectional_rank_correlation"]
    if rank.loc[overall_model].isna().any() or not rank.loc[overall_model].between(-1.0, 1.0).all():
        raise ValueError("overall TimesFM rank correlation is invalid")
    if rank.loc[ticker_or_baseline].notna().any():
        raise ValueError("rank correlation is defined only for overall TimesFM rows")

    expected_counts = signals.loc[signals["evaluation_status"] == "realized"].groupby(
        ["model_mode", "ticker"]
    ).size()
    for row in metrics.itertuples(index=False):
        expected = (
            int(expected_counts.loc[row.model_mode].sum())
            if row.scope == "overall"
            else int(expected_counts.loc[(row.model_mode, row.ticker)])
        )
        if int(row.observations) != expected:
            raise ValueError("forecast metric observations do not match realized rows")

    recomputed = evaluate_forecasts(signals).sort_values(
        ["model_mode", "scope", "ticker"]
    ).reset_index(drop=True)
    supplied = metrics.sort_values(["model_mode", "scope", "ticker"]).reset_index(drop=True)
    for column in METRIC_COLUMNS:
        if column in {"model_mode", "scope", "ticker"}:
            if not supplied[column].equals(recomputed[column]):
                raise ValueError("forecast metric keys do not reconcile")
        elif not np.allclose(
            supplied[column].to_numpy(dtype=float),
            recomputed[column].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"forecast metric {column} does not reconcile")
    return metrics


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
    metrics = _validate_metrics(metrics, signals)
    return TimesFMResearchBundle(signals=signals, metrics=metrics, manifest=manifest, path=path)


def load_timesfm_bundle(
    path=DEFAULT_BUNDLE_PATH,
    workbench_path=DEFAULT_WORKBENCH_PATH,
    require_clean=True,
):
    """Load the local artifact bundle without importing TimesFM or using HTTP."""
    bundle = validate_timesfm_bundle(path, require_clean=require_clean)
    expected = bundle.manifest["workbench_input"]["file_sha256"]
    for filename, digest in expected.items():
        source = Path(workbench_path) / filename
        if not source.is_file() or file_sha256(source) != digest:
            raise ValueError(
                f"TimesFM forecasts do not match the current workbench {filename}"
            )
    from data.workbench import load_workbench_bundle

    workbench = load_workbench_bundle(workbench_path)
    windows = build_forecast_windows(
        workbench.adjusted_close, workbench.manifest["generated_at_utc"]
    )
    expected = {
        window.signal_date: window for window in windows
    }
    actual_dates = set(bundle.signals["signal_date"])
    if actual_dates != set(expected):
        raise ValueError("forecast signal dates do not match the exact expected windows")
    for signal_date, rows in bundle.signals.groupby("signal_date", sort=True):
        window = expected[signal_date]
        if not (
            (rows["execution_date"] == window.execution_date).all()
            and (rows["period_end_date"] == window.period_end_date).all()
            and (rows["context_start"] == window.context.index[0]).all()
            and (rows["context_end"] == window.context.index[-1]).all()
            and (rows["context_sha256"] == window.context_sha256).all()
            and (rows["horizon_sessions"] == window.horizon).all()
        ):
            raise ValueError(
                f"forecast timing or context does not match {signal_date.date()}"
            )
        known_rates = workbench.cash_index.loc[
            workbench.cash_index.index < signal_date
        ]
        if known_rates.empty:
            raise ValueError("forecast cash hurdle lacks a safely known rate")
        known = known_rates.iloc[-1]
        rate_date = pd.Timestamp(known["effective_date"])
        days = (window.period_end_date - window.execution_date).days
        hurdle = float(known["annual_rate"]) / 100.0 * days / 360.0
        for ticker, ticker_rows in rows.groupby("ticker", sort=False):
            context = window.context[ticker].to_numpy(dtype=float)
            if not (
                np.allclose(ticker_rows["signal_price"], context[-1])
                and np.allclose(
                    ticker_rows["context_mean_absolute_change"],
                    np.mean(np.abs(np.diff(context))),
                )
                and (ticker_rows["known_rate_effective_date"] == rate_date).all()
                and (ticker_rows["known_rate_source"] == known["source_series"]).all()
                and np.allclose(ticker_rows["known_annual_rate"], known["annual_rate"])
                and np.allclose(ticker_rows["cash_hurdle"], hurdle)
            ):
                raise ValueError(
                    f"forecast inputs do not match {ticker} on {signal_date.date()}"
                )
            if (
                window.execution_date in workbench.adjusted_close.index
                and window.period_end_date in workbench.adjusted_close.index
            ):
                execution_price = float(
                    workbench.adjusted_close.at[window.execution_date, ticker]
                )
                period_end_price = float(
                    workbench.adjusted_close.at[window.period_end_date, ticker]
                )
                holding_return = period_end_price / execution_price - 1.0
                outcomes_match = (
                    (ticker_rows["evaluation_status"] == "realized").all()
                    and np.allclose(
                        ticker_rows["actual_execution_price"], execution_price
                    )
                    and np.allclose(
                        ticker_rows["actual_period_end_price"], period_end_price
                    )
                    and np.allclose(
                        ticker_rows["actual_holding_return"], holding_return
                    )
                )
            else:
                outcomes_match = (
                    (ticker_rows["evaluation_status"] == "pending").all()
                    and ticker_rows[
                        [
                            "actual_execution_price",
                            "actual_period_end_price",
                            "actual_holding_return",
                        ]
                    ]
                    .isna()
                    .all()
                    .all()
                )
            if not outcomes_match:
                raise ValueError(
                    f"forecast outcomes do not match {ticker} on {signal_date.date()}"
                )
    return bundle
