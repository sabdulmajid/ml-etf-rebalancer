import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import build_timesfm_artifacts as builder
from data.timesfm import load_timesfm_bundle, validate_timesfm_bundle
from data.workbench import file_sha256, load_workbench_bundle
from forecasting.evaluate import evaluate_forecasts
from forecasting.timesfm import (
    CONTEXT_SESSIONS,
    EVALUATOR_POLICY,
    LAST_VALUE_MODE,
    MODEL_MODES,
    MULTIVARIATE_MODE,
    QUANTILE_LEVELS,
    TimesFM3Runner,
    build_forecast_windows,
)


FAKE_DEPENDENCY_VERSIONS = {
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "exchange-calendars": "4.13.2",
    "timesfm": "3.0.1",
    "torch": "2.14.0+cpu",
    "huggingface-hub": "1.3.0",
    "safetensors": "0.7.0",
}


class FakeRunner:
    def __init__(self):
        self.calls = []

    def predict(self, window, mode):
        self.calls.append((window.signal_date, mode))
        base = window.context.iloc[-1].to_numpy(dtype=float)
        progress = np.linspace(1.0 / window.horizon, 1.0, window.horizon)
        drift = 0.01 if mode == MULTIVARIATE_MODE else 0.005
        offsets = np.linspace(-0.04, 0.04, len(QUANTILE_LEVELS))
        values = base[:, None, None] * (
            1.0 + drift * progress[None, :, None] + offsets[None, None, :]
        )
        return SimpleNamespace(quantiles=values, inference_seconds=0.001)


@pytest.fixture(scope="module")
def workbench():
    return load_workbench_bundle()


@pytest.fixture(autouse=True)
def deterministic_dependency_versions(monkeypatch):
    monkeypatch.setattr(
        builder, "_dependency_versions", lambda: FAKE_DEPENDENCY_VERSIONS.copy()
    )


def _rewrite_manifest_checksums(output, **row_counts):
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["file_sha256"] = {
        filename: file_sha256(output / filename)
        for filename in ("forecast_signals.csv", "forecast_metrics.csv")
    }
    manifest["row_counts"].update(row_counts)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def test_windows_use_512_sessions_and_keep_two_pending_origins(workbench):
    windows = build_forecast_windows(
        workbench.adjusted_close, workbench.manifest["generated_at_utc"]
    )
    assert len(windows) == 207
    first = windows[0]
    assert first.signal_date == pd.Timestamp("2009-06-30")
    assert first.execution_date == pd.Timestamp("2009-07-01")
    assert first.period_end_date == pd.Timestamp("2009-08-03")
    assert first.horizon == 23
    assert len(first.context) == CONTEXT_SESSIONS
    assert first.context.index[-1] == first.signal_date

    latest = windows[-1]
    assert latest.signal_date == pd.Timestamp("2026-08-31")
    assert latest.execution_date == pd.Timestamp("2026-09-01")
    assert latest.period_end_date == pd.Timestamp("2026-10-01")
    assert latest.horizon == 22
    realized = [
        window
        for window in windows
        if window.period_end_date in workbench.adjusted_close.index
    ]
    assert len(realized) == 205
    assert realized[-1].signal_date == pd.Timestamp("2026-06-30")


def test_window_builder_rejects_internal_missing_price(workbench):
    prices = workbench.adjusted_close.copy()
    prices.loc[pd.Timestamp("2015-01-05"), "SPY"] = np.nan
    with pytest.raises(ValueError, match="internal missing price"):
        build_forecast_windows(prices, workbench.manifest["generated_at_utc"])


def test_timesfm_runner_freezes_evaluator_flags(monkeypatch, workbench):
    calls = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            calls["config"] = kwargs

    class FakeEvaluator:
        def __init__(self, config):
            self.device = "cpu"

        def predict_batch(self, **kwargs):
            calls["predict"] = kwargs
            context = kwargs["contexts"][0]
            shape = (context.shape[0], kwargs["horizon"], len(QUANTILE_LEVELS))
            base = context[:, -1][:, None, None]
            offsets = np.linspace(0.96, 1.04, len(QUANTILE_LEVELS))
            yield SimpleNamespace(
                quantiles=np.broadcast_to(base * offsets, shape).copy()
            )

    monkeypatch.setitem(
        __import__("sys").modules,
        "timesfm3",
        SimpleNamespace(ModelConfig=FakeConfig, TimesFM3Evaluator=FakeEvaluator),
    )
    window = build_forecast_windows(
        workbench.adjusted_close, workbench.manifest["generated_at_utc"]
    )[0]
    runner = TimesFM3Runner(device="cpu", local_files_only=True)
    result = runner.predict(window, MULTIVARIATE_MODE)
    assert result.quantiles.shape == (14, window.horizon, 9)
    assert calls["predict"]["univariate"] is False
    for name, value in EVALUATOR_POLICY.items():
        assert calls["predict"][name] == value
    assert calls["predict"]["past_only_covariates"] is None
    assert calls["predict"]["past_future_covariates"] is None


def test_builder_writes_realized_and_pending_rows_and_validates(
    tmp_path, monkeypatch, workbench
):
    monkeypatch.setattr(builder, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(builder, "_git_dirty", lambda: False)
    runner = FakeRunner()
    output = tmp_path / "timesfm"
    bundle = builder.build_bundle(
        output,
        workbench.path,
        full=True,
        runner=runner,
        generated_at="2026-08-02T12:00:00Z",
    )
    assert len(runner.calls) == 207 * 2
    assert len(bundle.signals) == 207 * 3 * 14
    assert set(bundle.signals["model_mode"]) == set(MODEL_MODES)
    status_counts = bundle.signals.groupby("evaluation_status").size().to_dict()
    assert status_counts == {"pending": 2 * 3 * 14, "realized": 205 * 3 * 14}
    assert bundle.manifest["bundle_refresh"] == {
        "generated_at_utc": "2026-08-02T12:00:00Z",
        "git_sha": "a" * 40,
        "git_dirty_at_build": False,
        "mode": "full",
        "reused_mode_signal_groups": 0,
        "generated_baseline_groups": 207,
        "generated_model_groups": 207 * 2,
    }
    assert bundle.manifest["evaluation_classification"] == "historical_replay"
    assert bundle.manifest["pretraining_overlap"] == "unknown"
    assert len(bundle.manifest["model_generations"]) == 1
    generation = bundle.manifest["model_generations"][0]
    assert generation["generated_at_utc"] == "2026-08-02T12:00:00Z"
    model_rows = bundle.signals["model_mode"] != LAST_VALUE_MODE
    assert set(bundle.signals.loc[model_rows, "model_generation_id"]) == {
        generation["id"]
    }
    assert set(bundle.signals.loc[~model_rows, "model_generation_id"]) == {
        "point-baseline-v1"
    }
    latest = bundle.latest()
    assert len(latest) == 14
    assert latest["actual_holding_return"].isna().all()
    assert (latest["known_rate_effective_date"] < latest["signal_date"]).all()
    expected = (
        latest["q50_period_end_price"] / latest["q50_execution_price"] - 1.0
    )
    assert np.allclose(expected, latest["forecast_holding_return"])
    assert set(bundle.metrics["model_mode"]) == set(MODEL_MODES)

    # The runtime loader is file-only and does not touch HTTP.
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime forecast loader attempted network access")
        ),
    )
    loaded = load_timesfm_bundle(output)
    assert len(loaded.signals) == len(bundle.signals)


def test_incremental_build_reuses_matching_mode_signal_groups(
    tmp_path, monkeypatch, workbench
):
    monkeypatch.setattr(builder, "_git_sha", lambda: "b" * 40)
    monkeypatch.setattr(builder, "_git_dirty", lambda: True)
    output = tmp_path / "timesfm"
    initial_runner = FakeRunner()
    initial = builder.build_bundle(
        output,
        workbench.path,
        full=True,
        allow_dirty=True,
        runner=initial_runner,
    )
    incremental_runner = FakeRunner()
    refreshed = builder.build_bundle(
        output,
        workbench.path,
        allow_dirty=True,
        runner=incremental_runner,
    )
    assert incremental_runner.calls == []
    refresh = refreshed.manifest["bundle_refresh"]
    assert refresh["mode"] == "incremental"
    assert refresh["reused_mode_signal_groups"] == 207 * 3
    assert refresh["generated_baseline_groups"] == 0
    assert refresh["generated_model_groups"] == 0
    assert refreshed.manifest["model_generations"] == initial.manifest[
        "model_generations"
    ]


@pytest.mark.parametrize(
    ("manifest_field", "bad_value"),
    [
        ("model_policy_sha256", "0" * 64),
        ("pipeline_version", "obsolete-builder-policy"),
    ],
)
def test_policy_mismatch_forces_regeneration(
    tmp_path, monkeypatch, workbench, manifest_field, bad_value
):
    monkeypatch.setattr(builder, "_git_sha", lambda: "e" * 40)
    monkeypatch.setattr(builder, "_git_dirty", lambda: True)
    output = tmp_path / "timesfm"
    builder.build_bundle(
        output,
        workbench.path,
        full=True,
        allow_dirty=True,
        runner=FakeRunner(),
        generated_at="2026-08-02T12:00:00Z",
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[manifest_field] = bad_value
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    replacement_runner = FakeRunner()
    regenerated = builder.build_bundle(
        output,
        workbench.path,
        allow_dirty=True,
        runner=replacement_runner,
        generated_at="2026-08-03T12:00:00Z",
    )
    assert len(replacement_runner.calls) == 207 * 2
    assert regenerated.manifest["bundle_refresh"]["mode"] == "full"
    assert regenerated.manifest["bundle_refresh"]["reused_mode_signal_groups"] == 0


def test_bundle_checksum_is_enforced(tmp_path, monkeypatch, workbench):
    monkeypatch.setattr(builder, "_git_sha", lambda: "c" * 40)
    monkeypatch.setattr(builder, "_git_dirty", lambda: True)
    output = tmp_path / "timesfm"
    builder.build_bundle(
        output,
        workbench.path,
        full=True,
        allow_dirty=True,
        runner=FakeRunner(),
    )
    with (output / "forecast_signals.csv").open("a") as handle:
        handle.write("corrupt\n")
    with pytest.raises(ValueError, match="checksum"):
        validate_timesfm_bundle(output, require_clean=False)


def test_timesfm_atomic_promotion_replaces_complete_directory(tmp_path):
    destination = tmp_path / "timesfm"
    destination.mkdir()
    (destination / "old.txt").write_text("old")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new")
    builder._promote_directory(staging, destination)
    assert sorted(path.name for path in destination.iterdir()) == ["new.txt"]
    assert not (tmp_path / "timesfm.previous").exists()


def test_timesfm_atomic_promotion_restores_previous_bundle(tmp_path, monkeypatch):
    destination = tmp_path / "timesfm"
    destination.mkdir()
    (destination / "old.txt").write_text("old")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new")
    real_replace = os.replace
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("promotion failed")
        return real_replace(source, target)

    monkeypatch.setattr(builder.os, "replace", fail_second)
    with pytest.raises(OSError, match="promotion failed"):
        builder._promote_directory(staging, destination)
    assert calls == 3
    assert (destination / "old.txt").read_text() == "old"
    assert (staging / "new.txt").read_text() == "new"
    assert not (tmp_path / "timesfm.previous").exists()


def test_evaluation_excludes_pending_rows(workbench):
    window = build_forecast_windows(
        workbench.adjusted_close, workbench.manifest["generated_at_utc"]
    )[0]
    rows = []
    for mode in MODEL_MODES:
        values = (
            builder.last_value_forecast(window)
            if mode == "last_value"
            else FakeRunner().predict(window, mode)
        )
        rows.extend(
            builder._forecast_rows(
                window,
                mode,
                (
                    "point-baseline-v1"
                    if mode == LAST_VALUE_MODE
                    else "timesfm3-test-generation"
                ),
                values,
                workbench.cash_index,
                workbench.adjusted_close,
            )
        )
    frame = pd.DataFrame(rows)
    frame.loc[frame["ticker"] == "SPY", "evaluation_status"] = "pending"
    frame.loc[
        frame["ticker"] == "SPY",
        ["actual_execution_price", "actual_period_end_price", "actual_holding_return"],
    ] = np.nan
    metrics = evaluate_forecasts(frame)
    overall = metrics.loc[metrics["scope"] == "overall"]
    assert set(overall["observations"]) == {13}
    baseline = metrics["model_mode"] == LAST_VALUE_MODE
    assert metrics.loc[
        baseline,
        [
            "directional_accuracy",
            "mean_pinball_loss",
            "interval_80_coverage",
            "mean_interval_80_width",
            "mean_cross_sectional_rank_correlation",
        ],
    ].isna().all().all()
    assert metrics.loc[
        baseline,
        ["holding_return_mae", "holding_return_rmse", "period_end_price_mase"],
    ].notna().all().all()


def test_root_streamlit_requirements_exclude_optional_model_stack():
    requirements = Path("requirements.txt").read_text().lower()
    assert "timesfm" not in requirements
    assert "torch" not in requirements
    optional = Path("requirements-timesfm.txt").read_text()
    assert "aa480150652811e732d87a3c5344b235234104e3" in optional


def test_manifest_does_not_accept_incomplete_model_execution(
    tmp_path, monkeypatch, workbench
):
    monkeypatch.setattr(builder, "_git_sha", lambda: "d" * 40)
    monkeypatch.setattr(builder, "_git_dirty", lambda: True)
    output = tmp_path / "timesfm"
    builder.build_bundle(
        output,
        workbench.path,
        full=True,
        allow_dirty=True,
        runner=FakeRunner(),
    )
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["model_execution_status"] = "not_run"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="execution is not complete"):
        validate_timesfm_bundle(output, verify_checksums=False, require_clean=False)


def test_loader_rejects_deleted_interior_signal(tmp_path, monkeypatch, workbench):
    monkeypatch.setattr(builder, "_git_sha", lambda: "f" * 40)
    monkeypatch.setattr(builder, "_git_dirty", lambda: False)
    output = tmp_path / "timesfm"
    builder.build_bundle(
        output,
        workbench.path,
        full=True,
        runner=FakeRunner(),
        generated_at="2026-08-02T12:00:00Z",
    )
    signals = pd.read_csv(output / "forecast_signals.csv", float_precision="round_trip")
    interior = sorted(signals["signal_date"].unique())[100]
    signals = signals.loc[signals["signal_date"] != interior]
    signals.to_csv(output / "forecast_signals.csv", index=False, float_format="%.17g")
    metrics = evaluate_forecasts(signals)
    metrics.to_csv(output / "forecast_metrics.csv", index=False, float_format="%.17g")
    _rewrite_manifest_checksums(
        output,
        **{
            "forecast_signals.csv": len(signals),
            "forecast_metrics.csv": len(metrics),
        },
    )
    with pytest.raises(ValueError, match="exact expected windows"):
        load_timesfm_bundle(output, workbench.path)


def test_validator_rejects_tampered_metric(tmp_path, monkeypatch, workbench):
    monkeypatch.setattr(builder, "_git_sha", lambda: "1" * 40)
    monkeypatch.setattr(builder, "_git_dirty", lambda: True)
    output = tmp_path / "timesfm"
    builder.build_bundle(
        output,
        workbench.path,
        full=True,
        allow_dirty=True,
        runner=FakeRunner(),
    )
    metrics = pd.read_csv(output / "forecast_metrics.csv", float_precision="round_trip")
    metrics.loc[0, "holding_return_mae"] += 0.01
    metrics.to_csv(output / "forecast_metrics.csv", index=False, float_format="%.17g")
    _rewrite_manifest_checksums(output)
    with pytest.raises(ValueError, match="holding_return_mae does not reconcile"):
        validate_timesfm_bundle(output, require_clean=False)

    metrics = evaluate_forecasts(
        pd.read_csv(output / "forecast_signals.csv", float_precision="round_trip")
    )
    model_row = metrics["model_mode"] == MULTIVARIATE_MODE
    metrics.loc[
        model_row & (metrics["scope"] == "overall"), "mean_pinball_loss"
    ] = np.nan
    metrics.to_csv(output / "forecast_metrics.csv", index=False, float_format="%.17g")
    _rewrite_manifest_checksums(output)
    with pytest.raises(
        ValueError, match="TimesFM directional/probabilistic metrics cannot be missing"
    ):
        validate_timesfm_bundle(output, require_clean=False)


def test_loader_rejects_outcomes_that_do_not_match_prices(
    tmp_path, monkeypatch, workbench
):
    monkeypatch.setattr(builder, "_git_sha", lambda: "3" * 40)
    monkeypatch.setattr(builder, "_git_dirty", lambda: False)
    output = tmp_path / "timesfm"
    builder.build_bundle(
        output,
        workbench.path,
        full=True,
        runner=FakeRunner(),
        generated_at="2026-08-02T12:00:00Z",
    )
    signals = pd.read_csv(output / "forecast_signals.csv", float_precision="round_trip")
    target = (
        (signals["signal_date"] == "2015-01-30")
        & (signals["ticker"] == "SPY")
    )
    signals.loc[target, "actual_period_end_price"] += 1.0
    signals.loc[target, "actual_holding_return"] = (
        signals.loc[target, "actual_period_end_price"]
        / signals.loc[target, "actual_execution_price"]
        - 1.0
    )
    signals.to_csv(output / "forecast_signals.csv", index=False, float_format="%.17g")
    metrics = evaluate_forecasts(signals)
    metrics.to_csv(output / "forecast_metrics.csv", index=False, float_format="%.17g")
    _rewrite_manifest_checksums(output)
    with pytest.raises(ValueError, match="forecast outcomes do not match SPY"):
        load_timesfm_bundle(output, workbench.path)


def test_clean_refresh_cannot_launder_dirty_model_generation(
    tmp_path, monkeypatch, workbench
):
    monkeypatch.setattr(builder, "_git_sha", lambda: "4" * 40)
    monkeypatch.setattr(builder, "_git_dirty", lambda: True)
    output = tmp_path / "timesfm"
    builder.build_bundle(
        output,
        workbench.path,
        full=True,
        allow_dirty=True,
        runner=FakeRunner(),
    )
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["bundle_refresh"]["git_dirty_at_build"] = False
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="dirty model outputs"):
        validate_timesfm_bundle(output, require_clean=True)


def test_freshness_disables_expired_latest_target(tmp_path, monkeypatch, workbench):
    monkeypatch.setattr(builder, "_git_sha", lambda: "2" * 40)
    monkeypatch.setattr(builder, "_git_dirty", lambda: False)
    bundle = builder.build_bundle(
        tmp_path / "timesfm",
        workbench.path,
        full=True,
        runner=FakeRunner(),
        generated_at="2026-08-31T23:00:00Z",
    )
    scheduled = bundle.freshness("2026-08-31T23:30:00Z")
    assert scheduled["latest_target_status"] == "scheduled"
    assert bundle.freshness("2026-09-15")["latest_target_status"] == "current"
    expired = bundle.freshness("2026-10-02")
    assert expired["historical_available"] is True
    assert expired["latest_target_status"] == "disabled"
    assert expired["reason"] == "latest forecast holding period has expired"
