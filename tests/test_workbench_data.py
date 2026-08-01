import json
from pathlib import Path

import pandas as pd
import pytest

from data.cash import construct_cash_index
from data.workbench import (
    BUNDLE_SCHEMA_VERSION,
    INSTRUMENT_COLUMNS,
    canonical_instrument_registry,
    file_sha256,
    load_workbench_bundle,
    prior_completed_xnys_session,
)


def _write_bundle(path):
    path.mkdir()
    registry = canonical_instrument_registry()
    dates = pd.to_datetime(["2018-03-26", "2018-03-27", "2018-03-28", "2018-03-29"])
    prices = pd.DataFrame(
        {ticker: [100.0, 100.1, 100.2, 100.3] for ticker in registry["ticker"]},
        index=dates,
    )
    prices.index.name = "date"
    prices.to_csv(path / "adjusted_close.csv")
    registry["price_start"] = dates[0]
    registry["price_end"] = dates[-1]
    registry.to_csv(path / "instruments.csv", index=False)

    effr = pd.Series([1.0], index=[pd.Timestamp("2018-03-30")])
    sofr = pd.Series(
        [1.8, 1.81], index=pd.to_datetime(["2018-04-02", "2018-04-03"])
    )
    cash = construct_cash_index(effr, sofr, valuation_through="2018-04-04")
    cash.to_csv(path / "cash_index.csv")
    checksums = {
        filename: file_sha256(path / filename)
        for filename in ["adjusted_close.csv", "cash_index.csv", "instruments.csv"]
    }
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at_utc": "2018-04-05T12:00:00Z",
        "price_data_as_of": str(dates[-1].date()),
        "last_complete_month": "2018-03-29",
        "cash_rate_as_of": "2018-04-03",
        "cash_valuation_through": "2018-04-04",
        "cash_identifier": "CASH:USD_OVERNIGHT",
        "cash_label": "Cash — U.S. overnight-rate proxy",
        "cash_policy": "test",
        "cash_switch_date": "2018-04-02",
        "sofr_index_validation": {
            "observations": 2,
            "maximum_absolute_error": 1e-10,
            "tolerance": 5e-8,
            "passed": True,
        },
        "instrument_count": 14,
        "dependency_versions": {
            "pandas": "2.3.3",
            "yfinance": "0.2.66",
            "exchange-calendars": "4.13.2",
        },
        "source_queries": {
            "prices": {
                "requested_start": "2018-03-26",
                "requested_end": "2018-03-29",
                "returned_start": "2018-03-26",
                "returned_end": "2018-03-29",
                "returned_count": 4,
            },
            "effr": {
                "requested_start": "2018-03-30",
                "requested_end": "2018-04-01",
                "returned_start": "2018-03-30",
                "returned_end": "2018-03-30",
                "returned_count": 1,
            },
            "sofr": {
                "requested_start": "2018-04-02",
                "requested_end": "2018-04-03",
                "returned_start": "2018-04-02",
                "returned_end": "2018-04-03",
                "returned_count": 2,
            },
            "sofr_index": {
                "requested_start": "2018-04-02",
                "requested_end": "2018-04-03",
                "returned_start": "2018-04-02",
                "returned_end": "2018-04-03",
                "returned_count": 2,
            },
        },
        "row_counts": {
            "adjusted_close.csv": 4,
            "cash_index.csv": 3,
            "instruments.csv": 14,
        },
        "file_sha256": checksums,
        "validation_status": "passed",
        "git_sha": "a" * 40,
        "git_dirty_at_build": False,
        "pipeline_version": "test-builder-v1",
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_local_bundle_loads_without_network(tmp_path, monkeypatch):
    bundle_path = tmp_path / "bundle"
    _write_bundle(bundle_path)

    def fail_network(*args, **kwargs):
        raise AssertionError("runtime loader attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", fail_network)
    bundle = load_workbench_bundle(bundle_path)
    assert len(bundle.tickers) == 14
    assert bundle.manifest["cash_identifier"] == "CASH:USD_OVERNIGHT"
    assert bundle.freshness("2018-04-06")["status"] == "current"
    assert bundle.freshness("2018-05-01")["status"] == "warning"
    assert bundle.freshness("2018-06-01")["status"] == "disabled"


def test_bundle_checksum_and_exact_file_set_are_enforced(tmp_path):
    bundle_path = tmp_path / "bundle"
    _write_bundle(bundle_path)
    (bundle_path / "adjusted_close.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_workbench_bundle(bundle_path)

    _write_bundle(tmp_path / "other")
    (tmp_path / "other" / "extra.csv").write_text("x")
    with pytest.raises(ValueError, match="exactly"):
        load_workbench_bundle(tmp_path / "other")


def test_loader_rejects_manifest_completed_session_inconsistent_with_build_time(
    tmp_path,
):
    bundle_path = tmp_path / "bundle"
    _write_bundle(bundle_path)
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["last_complete_month"] = "2018-03-28"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="final XNYS session"):
        load_workbench_bundle(bundle_path)


def _update_checksum(bundle_path, filename):
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_sha256"][filename] = file_sha256(bundle_path / filename)
    manifest_path.write_text(json.dumps(manifest))


def test_loader_rejects_omitted_common_xnys_session(tmp_path):
    bundle_path = tmp_path / "bundle"
    _write_bundle(bundle_path)
    price_path = bundle_path / "adjusted_close.csv"
    prices = pd.read_csv(price_path)
    prices = prices.loc[prices["date"] != "2018-03-27"]
    prices.to_csv(price_path, index=False)
    _update_checksum(bundle_path, "adjusted_close.csv")

    with pytest.raises(ValueError, match="exactly equal XNYS sessions"):
        load_workbench_bundle(bundle_path)


def test_loader_rejects_cash_index_that_does_not_reconcile(tmp_path):
    bundle_path = tmp_path / "bundle"
    _write_bundle(bundle_path)
    cash_path = bundle_path / "cash_index.csv"
    cash = pd.read_csv(cash_path)
    cash.loc[1, "cash_index"] += 0.01
    cash.to_csv(cash_path, index=False)
    _update_checksum(bundle_path, "cash_index.csv")

    with pytest.raises(ValueError, match="do not reconcile"):
        load_workbench_bundle(bundle_path)


def test_loader_rejects_instrument_range_not_reconciled_to_prices(tmp_path):
    bundle_path = tmp_path / "bundle"
    _write_bundle(bundle_path)
    instrument_path = bundle_path / "instruments.csv"
    instruments = pd.read_csv(instrument_path)
    instruments.loc[0, "price_start"] = "2018-03-27"
    instruments.to_csv(instrument_path, index=False)
    _update_checksum(bundle_path, "instruments.csv")

    with pytest.raises(ValueError, match="price range does not match"):
        load_workbench_bundle(bundle_path)


def test_loader_rejects_incomplete_source_provenance(tmp_path):
    bundle_path = tmp_path / "bundle"
    _write_bundle(bundle_path)
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["dependency_versions"]["yfinance"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="exact yfinance"):
        load_workbench_bundle(bundle_path)


def test_loader_rejects_invalid_git_sha(tmp_path):
    bundle_path = tmp_path / "bundle"
    _write_bundle(bundle_path)
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["git_sha"] = "unknown"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="40-character hexadecimal"):
        load_workbench_bundle(bundle_path)


def test_freshness_disables_future_generated_artifact(tmp_path):
    bundle_path = tmp_path / "bundle"
    _write_bundle(bundle_path)
    bundle = load_workbench_bundle(bundle_path)
    freshness = bundle.freshness("2018-04-01")
    assert freshness["status"] == "disabled"
    assert freshness["reason"] == "artifact generated_at_utc is in the future"


def test_prior_completed_session_observes_xnys_holidays():
    assert prior_completed_xnys_session("2024-04-15") == pd.Timestamp("2024-03-28")


def test_registry_is_exactly_the_approved_tradeable_universe():
    registry = canonical_instrument_registry()
    assert list(registry["ticker"]) == [
        "SPY", "IWM", "EFA", "EEM", "AGG", "BIL", "SHY", "IEF", "TLT",
        "TIP", "LQD", "HYG", "GLD", "DBC",
    ]
    assert registry["tradeable"].all()
    dbc_note = registry.loc[registry["ticker"] == "DBC", "methodology_note"].iloc[0]
    assert "November 10, 2025" in dbc_note
    assert "CASH:USD_OVERNIGHT" not in set(registry["ticker"])
