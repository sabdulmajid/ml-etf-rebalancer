import os

import pandas as pd
import pytest

import build_workbench_artifacts as builder


def test_nyfed_parser_sorts_effective_dates(monkeypatch):
    # Official records always identify their requested reference-rate series.
    monkeypatch.setattr(
        builder,
        "_fetch_json",
        lambda url: {
            "refRates": [
                {"effectiveDate": "2020-01-03", "type": "EFFR", "percentRate": 1.55},
                {"effectiveDate": "2020-01-02", "type": "EFFR", "percentRate": 1.50},
            ]
        },
    )
    series, stats = builder.fetch_nyfed_series(
        "unsecured", "effr", "2020-01-01", "2020-01-04", "percentRate"
    )
    assert list(series.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert list(series) == [1.50, 1.55]
    assert stats["returned_count"] == 2
    assert stats["requested_start"] == "2020-01-01"


def test_nyfed_parser_rejects_wrong_series_type(monkeypatch):
    monkeypatch.setattr(
        builder,
        "_fetch_json",
        lambda url: {
            "refRates": [
                {"effectiveDate": "2020-01-02", "type": "SOFR", "percentRate": 1.5}
            ]
        },
    )
    with pytest.raises(RuntimeError, match="contained types"):
        builder.fetch_nyfed_series(
            "unsecured", "effr", "2020-01-01", "2020-01-04", "percentRate"
        )


def test_atomic_directory_promotion_replaces_the_complete_bundle(tmp_path):
    destination = tmp_path / "workbench"
    destination.mkdir()
    (destination / "old.txt").write_text("old")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new")

    builder._promote_directory(staging, destination)

    assert sorted(item.name for item in destination.iterdir()) == ["new.txt"]
    assert (destination / "new.txt").read_text() == "new"
    assert not staging.exists()
    assert not (tmp_path / "workbench.previous").exists()


def test_first_atomic_rename_failure_preserves_valid_destination(tmp_path, monkeypatch):
    destination = tmp_path / "workbench"
    destination.mkdir()
    (destination / "old.txt").write_text("old")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new")

    def fail_first_replace(source, target):
        raise OSError("first rename")

    monkeypatch.setattr(builder.os, "replace", fail_first_replace)
    with pytest.raises(OSError, match="first rename"):
        builder._promote_directory(staging, destination)

    assert (destination / "old.txt").read_text() == "old"
    assert (staging / "new.txt").read_text() == "new"


def test_second_atomic_rename_failure_restores_backup(tmp_path, monkeypatch):
    destination = tmp_path / "workbench"
    destination.mkdir()
    (destination / "old.txt").write_text("old")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new")
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("second rename")
        return real_replace(source, target)

    monkeypatch.setattr(builder.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="second rename"):
        builder._promote_directory(staging, destination)

    assert calls == 3
    assert (destination / "old.txt").read_text() == "old"
    assert (staging / "new.txt").read_text() == "new"
    assert not (tmp_path / "workbench.previous").exists()


def test_release_builder_fails_dirty_before_network(monkeypatch, tmp_path):
    monkeypatch.setattr(builder, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(builder, "_git_dirty", lambda: True)
    monkeypatch.setattr(
        builder,
        "download_adjusted_close",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network path should not run")
        ),
    )
    with pytest.raises(RuntimeError, match="dirty Git"):
        builder.build_bundle(tmp_path / "bundle", as_of="2026-08-01")


def test_release_builder_rejects_invalid_git_sha_before_network(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(builder, "_git_sha", lambda: "unknown")
    monkeypatch.setattr(builder, "_git_dirty", lambda: False)
    monkeypatch.setattr(
        builder,
        "download_adjusted_close",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network path should not run")
        ),
    )
    with pytest.raises(RuntimeError, match="40-character Git commit SHA"):
        builder.build_bundle(tmp_path / "bundle", as_of="2026-08-01")


def test_completed_session_truncates_partial_month_rows():
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2026-07-30", "2026-07-31", "2026-08-03"]),
    )
    truncated, completed = builder.truncate_to_completed_session(
        prices, "2026-08-15"
    )
    assert completed == pd.Timestamp("2026-07-31")
    assert truncated.index.max() == completed
    assert pd.Timestamp("2026-08-03") not in truncated.index


def test_completed_session_rejects_truncated_prior_month():
    prices = pd.DataFrame(
        {"SPY": [100.0]}, index=pd.to_datetime(["2026-07-30"])
    )
    with pytest.raises(RuntimeError, match="do not cover"):
        builder.truncate_to_completed_session(prices, "2026-08-15")


def test_completed_session_requires_every_selected_etf_price():
    prices = pd.DataFrame(
        {"SPY": [100.0], "IEF": [float("nan")]},
        index=pd.to_datetime(["2026-07-31"]),
    )
    with pytest.raises(RuntimeError, match="missing approved ETFs"):
        builder.truncate_to_completed_session(prices, "2026-08-15")
