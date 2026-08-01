"""Offline builder for the committed ETF Allocation Workbench data bundle.

This is the only workbench module that performs network access.  The public
Streamlit app reads the validated files through ``data.workbench`` and never
imports this script.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version as dependency_version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from data.cash import (
    CASH_ASSET,
    CASH_LABEL,
    INDEX_TOLERANCE,
    SOFR_SWITCH_DATE,
    construct_cash_index,
    validate_against_sofr_index,
)
from data.workbench import (
    BUNDLE_SCHEMA_VERSION,
    DEFAULT_BUNDLE_PATH,
    PUBLIC_FILENAMES,
    canonical_instrument_registry,
    file_sha256,
    prior_completed_xnys_session,
    validate_workbench_bundle,
)


NYFED_BASE = "https://markets.newyorkfed.org/api/rates"
PRICE_START = "1993-01-01"
EFFR_START = "2000-07-03"
USER_AGENT = "ml-etf-rebalancer-offline-builder/1.0"


def _fetch_json(url):
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_nyfed_series(section, rate_type, start_date, end_date, value_field):
    query = urlencode(
        {"startDate": str(start_date), "endDate": str(end_date), "type": "rate"}
    )
    payload = _fetch_json(
        f"{NYFED_BASE}/{section}/{rate_type}/search.json?{query}"
    )
    observations = payload.get("refRates", [])
    if not observations:
        raise RuntimeError(f"New York Fed returned no {rate_type.upper()} observations")
    frame = pd.DataFrame(observations)
    if not {"effectiveDate", "type", value_field}.issubset(frame.columns):
        raise RuntimeError(f"New York Fed {rate_type.upper()} response schema changed")
    expected_type = rate_type.upper()
    returned_types = set(frame["type"])
    if returned_types != {expected_type}:
        raise RuntimeError(
            f"New York Fed {expected_type} response contained types {returned_types}"
        )
    series = pd.Series(
        pd.to_numeric(frame[value_field], errors="raise").to_numpy(),
        index=pd.to_datetime(frame["effectiveDate"], errors="raise"),
        name=rate_type.upper(),
    ).sort_index()
    if not series.index.is_unique:
        raise RuntimeError(f"New York Fed returned duplicate {rate_type.upper()} dates")
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    if series.index[0] < requested_start or series.index[-1] > requested_end:
        raise RuntimeError(
            f"New York Fed {expected_type} response escaped the requested range"
        )
    stats = {
        "requested_start": str(requested_start.date()),
        "requested_end": str(requested_end.date()),
        "returned_start": str(series.index[0].date()),
        "returned_end": str(series.index[-1].date()),
        "returned_count": len(series),
    }
    return series, stats


def download_adjusted_close(tickers, start_date, end_date):
    raw = yf.download(
        list(tickers),
        start=str(start_date),
        end=str(end_date),
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no ETF prices")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("Yahoo Finance response has no adjusted-close field")
        adjusted = raw["Adj Close"].copy()
    else:
        if len(tickers) != 1 or "Adj Close" not in raw:
            raise RuntimeError("Yahoo Finance response has an unexpected schema")
        adjusted = raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    adjusted.index = pd.DatetimeIndex(adjusted.index).tz_localize(None).normalize()
    adjusted.index.name = "date"
    adjusted = adjusted.reindex(columns=list(tickers)).sort_index()
    return adjusted


def truncate_to_completed_session(prices, as_of):
    """Require and retain prices through the exact prior-month XNYS session."""
    expected_session = prior_completed_xnys_session(as_of)
    if expected_session not in prices.index:
        raise RuntimeError(
            "downloaded prices do not cover the final completed XNYS session "
            f"{expected_session.date()}"
        )
    missing = prices.loc[expected_session].index[
        prices.loc[expected_session].isna()
    ].tolist()
    if missing:
        raise RuntimeError(
            "downloaded prices are missing approved ETFs on the final completed "
            f"XNYS session {expected_session.date()}: {missing}"
        )
    truncated = prices.loc[:expected_session].copy()
    if truncated.index.max() != expected_session:
        raise RuntimeError("completed-session price truncation failed")
    return truncated, expected_session


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty():
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_bundle(
    staging,
    prices,
    cash,
    instruments,
    generated_at,
    validation,
    source_git_sha,
    source_git_dirty,
    expected_completed_session,
    dependency_versions,
    source_queries,
):
    staging.mkdir(parents=True, exist_ok=False)
    price_path = staging / "adjusted_close.csv"
    cash_path = staging / "cash_index.csv"
    instrument_path = staging / "instruments.csv"
    prices.to_csv(price_path, float_format="%.10f")
    cash.to_csv(cash_path, float_format="%.12f", date_format="%Y-%m-%d")
    instruments.to_csv(instrument_path, index=False, date_format="%Y-%m-%d")

    checksums = {
        filename: file_sha256(staging / filename)
        for filename in PUBLIC_FILENAMES[:-1]
    }
    latest_cash = cash.iloc[-1]
    cash_valuation_through = cash.index[-1] + pd.Timedelta(
        days=int(latest_cash["accrual_days"])
    )
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z"),
        "price_data_as_of": str(prices.index.max().date()),
        "last_complete_month": str(expected_completed_session.date()),
        "cash_rate_as_of": str(cash.index.max().date()),
        "cash_valuation_through": str(cash_valuation_through.date()),
        "cash_identifier": CASH_ASSET,
        "cash_label": CASH_LABEL,
        "cash_policy": (
            "Official New York Fed EFFR before 2018-04-02; official SOFR "
            "from 2018-04-02; analytical and non-investable."
        ),
        "cash_switch_date": str(SOFR_SWITCH_DATE.date()),
        "sofr_index_validation": {
            "observations": validation.observations,
            "maximum_absolute_error": validation.maximum_absolute_error,
            "tolerance": validation.tolerance,
            "passed": validation.passed,
        },
        "instrument_count": len(instruments),
        "source_versions": {
            "prices": "Yahoo Finance adjusted close via yfinance",
            "cash": "Federal Reserve Bank of New York Markets Data API",
            "strategy_data_contract": "allocation-policy-v1",
        },
        "dependency_versions": dependency_versions,
        "source_queries": source_queries,
        "row_counts": {
            "adjusted_close.csv": len(prices),
            "cash_index.csv": len(cash),
            "instruments.csv": len(instruments),
        },
        "file_sha256": checksums,
        "validation_status": "passed",
        "git_sha": source_git_sha,
        "git_dirty_at_build": source_git_dirty,
        "pipeline_version": "workbench-artifact-builder-v1",
        "refresh_error": None,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _promote_directory(staging, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    moved_destination = False
    if destination.exists():
        try:
            os.replace(destination, backup)
            moved_destination = True
        except Exception:
            # The valid destination was never moved; it must remain untouched.
            raise
    try:
        os.replace(staging, destination)
    except Exception:
        if moved_destination:
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def build_bundle(output_dir=DEFAULT_BUNDLE_PATH, as_of=None, allow_dirty=False):
    source_git_sha = _git_sha()
    source_git_dirty = _git_dirty()
    if re.fullmatch(r"[0-9a-fA-F]{40}", source_git_sha or "") is None:
        raise RuntimeError(
            "refusing to build artifacts without a valid 40-character Git commit SHA"
        )
    if source_git_dirty is None:
        raise RuntimeError("unable to determine Git working-tree provenance")
    if source_git_dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to build release artifacts from a dirty Git working tree"
        )
    if as_of is None:
        generated_at = datetime.now(timezone.utc)
    else:
        generated_stamp = pd.Timestamp(as_of)
        if generated_stamp.tz is None:
            generated_stamp = generated_stamp.tz_localize("UTC")
        else:
            generated_stamp = generated_stamp.tz_convert("UTC")
        generated_at = generated_stamp.to_pydatetime()
    today = generated_at.date()
    exclusive_end = today + timedelta(days=1)
    registry = canonical_instrument_registry()
    tickers = tuple(registry["ticker"])
    prices = download_adjusted_close(tickers, PRICE_START, exclusive_end)
    price_query = {
        "requested_start": PRICE_START,
        "requested_end": str(exclusive_end),
        "requested_end_inclusive": False,
        "returned_start": str(prices.index.min().date()),
        "returned_end": str(prices.index.max().date()),
        "returned_count": len(prices),
    }
    prices, expected_completed_session = truncate_to_completed_session(
        prices, generated_at
    )

    effr_end = SOFR_SWITCH_DATE.date() - timedelta(days=1)
    effr, effr_query = fetch_nyfed_series(
        "unsecured", "effr", EFFR_START, effr_end, "percentRate"
    )
    sofr, sofr_query = fetch_nyfed_series(
        "secured", "sofr", SOFR_SWITCH_DATE.date(), today, "percentRate"
    )
    official_index, sofr_index_query = fetch_nyfed_series(
        "secured", "sofrai", date(2020, 3, 2), today, "index"
    )
    # The published SOFR Index for a valuation date proves that every rate
    # needed to value cash through that date has been published.  If prices are
    # newer, fail validation rather than substitute a stale or unpublished rate.
    if official_index.index.max().date() < expected_completed_session.date():
        raise RuntimeError(
            "published SOFR Index does not cover the final completed XNYS session"
        )
    cash_through = expected_completed_session.date()
    cash = construct_cash_index(effr, sofr, valuation_through=cash_through)
    validation = validate_against_sofr_index(
        cash,
        official_index,
        tolerance=INDEX_TOLERANCE,
        valuation_through=cash_through,
    )

    registry["price_start"] = [prices[ticker].first_valid_index() for ticker in tickers]
    registry["price_end"] = [prices[ticker].last_valid_index() for ticker in tickers]
    instrument_columns = [
        "ticker",
        "name",
        "asset_class",
        "role",
        "inception_date",
        "overlap_note",
        "methodology_note",
        "tradeable",
        "price_start",
        "price_end",
    ]
    registry = registry[instrument_columns]

    dependency_versions = {
        name: dependency_version(name)
        for name in ("pandas", "yfinance", "exchange-calendars")
    }
    source_queries = {
        "prices": price_query,
        "effr": effr_query,
        "sofr": sofr_query,
        "sofr_index": sofr_index_query,
    }
    destination = Path(output_dir)
    staging_parent = destination.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-build-", dir=staging_parent)
    )
    shutil.rmtree(staging)
    try:
        _write_bundle(
            staging,
            prices,
            cash,
            registry,
            generated_at,
            validation,
            source_git_sha,
            source_git_dirty,
            expected_completed_session,
            dependency_versions,
            source_queries,
        )
        validate_workbench_bundle(staging, require_clean=not allow_dirty)
        _promote_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return validate_workbench_bundle(destination, require_clean=not allow_dirty)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="build a non-release bundle for local testing",
    )
    args = parser.parse_args()
    bundle = build_bundle(args.output_dir, allow_dirty=args.allow_dirty)
    print(
        f"Validated {len(bundle.tickers)} ETFs through "
        f"{bundle.manifest['price_data_as_of']} at {bundle.path}"
    )


if __name__ == "__main__":
    main()
