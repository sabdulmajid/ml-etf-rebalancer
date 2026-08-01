"""Validated, local-only data bundle for the ETF Allocation Workbench."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import exchange_calendars as xcals

from data.cash import CASH_ASSET, cash_returns_between, cash_values_on_dates


BUNDLE_SCHEMA_VERSION = "workbench-bundle-v1"
REQUIRED_DEPENDENCY_VERSIONS = ("pandas", "yfinance", "exchange-calendars")
REQUIRED_SOURCE_QUERIES = ("prices", "effr", "sofr", "sofr_index")
SOURCE_QUERY_FIELDS = (
    "requested_start",
    "requested_end",
    "returned_start",
    "returned_end",
    "returned_count",
)
PUBLIC_FILENAMES = (
    "adjusted_close.csv",
    "cash_index.csv",
    "instruments.csv",
    "manifest.json",
)
DEFAULT_BUNDLE_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "workbench"


INSTRUMENT_RECORDS = (
    (
        "SPY",
        "State Street SPDR S&P 500 ETF Trust",
        "Equity",
        "U.S. large-cap equity",
        "1993-01-22",
        "Core U.S. equity exposure.",
        "Tracks the S&P 500 Index.",
    ),
    (
        "IWM",
        "iShares Russell 2000 ETF",
        "Equity",
        "U.S. small-cap equity",
        "2000-05-22",
        "Correlated with SPY, but represents smaller U.S. companies.",
        "Tracks U.S. small-cap equities.",
    ),
    (
        "EFA",
        "iShares MSCI EAFE ETF",
        "Equity",
        "Developed ex-U.S. equity",
        "2001-08-14",
        "Adds foreign-currency and regional exposure beyond U.S. equities.",
        "Tracks developed markets outside the U.S. and Canada.",
    ),
    (
        "EEM",
        "iShares MSCI Emerging Markets ETF",
        "Equity",
        "Emerging-market equity",
        "2003-04-07",
        "Overlaps the global cycle but adds emerging-market country/currency risk.",
        "Tracks large- and mid-cap emerging-market equities.",
    ),
    (
        "AGG",
        "iShares Core U.S. Aggregate Bond ETF",
        "Fixed income",
        "Broad U.S. core bonds",
        "2003-09-22",
        "Overlaps Treasury and credit sleeves; useful as a core-bond building block.",
        "Tracks a broad investment-grade U.S. bond index.",
    ),
    (
        "BIL",
        "State Street SPDR Bloomberg 1-3 Month T-Bill ETF",
        "Fixed income",
        "Investable Treasury-bill sleeve",
        "2007-05-25",
        "Unlike analytical cash, it is a tradeable fund with expenses/price moves.",
        "Tracks U.S. Treasury bills with one to three months remaining.",
    ),
    (
        "SHY",
        "iShares 1-3 Year Treasury Bond ETF",
        "Fixed income",
        "Short Treasuries",
        "2002-07-22",
        "More interest-rate duration than BIL and less than IEF.",
        "Tracks U.S. Treasury bonds with one to three years remaining.",
    ),
    (
        "IEF",
        "iShares 7-10 Year Treasury Bond ETF",
        "Fixed income",
        "Intermediate Treasuries",
        "2002-07-22",
        "Overlaps AGG while isolating intermediate Treasury duration.",
        "Tracks U.S. Treasury bonds with seven to ten years remaining.",
    ),
    (
        "TLT",
        "iShares 20+ Year Treasury Bond ETF",
        "Fixed income",
        "Long Treasuries",
        "2002-07-22",
        "High duration can dominate rate risk even after volatility weighting.",
        "Tracks U.S. Treasury bonds with more than twenty years remaining.",
    ),
    (
        "TIP",
        "iShares TIPS Bond ETF",
        "Fixed income",
        "Inflation-linked Treasuries",
        "2003-12-04",
        "Shares real-rate exposure with nominal Treasuries but adds indexation.",
        "Tracks U.S. Treasury inflation-protected securities.",
    ),
    (
        "LQD",
        "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "Fixed income",
        "Investment-grade credit",
        "2002-07-22",
        "Combines Treasury-duration and corporate-credit-spread exposure.",
        "Tracks U.S. dollar investment-grade corporate bonds.",
    ),
    (
        "HYG",
        "iShares iBoxx $ High Yield Corporate Bond ETF",
        "Fixed income",
        "High-yield credit",
        "2007-04-04",
        "Credit-cycle behavior can overlap materially with equities.",
        "Tracks U.S. dollar high-yield corporate bonds.",
    ),
    (
        "GLD",
        "SPDR Gold Shares",
        "Real asset",
        "Gold",
        "2004-11-18",
        "A non-income real asset with unstable relationships to stocks and rates.",
        "Designed to reflect the price of gold bullion, less fund expenses.",
    ),
    (
        "DBC",
        "Invesco DB Commodity Index Tracking Fund",
        "Real asset",
        "Broad commodities",
        "2006-02-03",
        "Futures exposure can overlap inflation assets and includes roll effects.",
        "Futures-based methodology has changed; Invesco announced a benchmark "
        "transition effective November 10, 2025. History spans both methodologies.",
    ),
)


INSTRUMENT_COLUMNS = (
    "ticker",
    "name",
    "asset_class",
    "role",
    "inception_date",
    "overlap_note",
    "methodology_note",
)


def canonical_instrument_registry():
    registry = pd.DataFrame(INSTRUMENT_RECORDS, columns=INSTRUMENT_COLUMNS)
    registry["inception_date"] = pd.to_datetime(registry["inception_date"])
    registry["tradeable"] = True
    return registry


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prior_completed_xnys_session(as_of):
    """Return the final XNYS session in the month immediately before ``as_of``."""
    stamp = pd.Timestamp(as_of)
    if pd.isna(stamp):
        raise ValueError("as_of must be a valid timestamp")
    if stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    prior_month = stamp.to_period("M") - 1
    month_start = prior_month.start_time.normalize()
    month_end = prior_month.end_time.normalize()
    calendar = xcals.get_calendar(
        "XNYS",
        start=month_start - pd.Timedelta(days=7),
        end=month_end + pd.Timedelta(days=2),
    )
    sessions = calendar.sessions_in_range(month_start, month_end)
    if sessions.empty:
        raise ValueError(f"XNYS has no sessions in {prior_month}")
    return pd.Timestamp(sessions[-1]).tz_localize(None).normalize()


def xnys_sessions(start, end):
    """Return the exact XNYS session index within an inclusive date range."""
    start = pd.Timestamp(start).tz_localize(None).normalize()
    end = pd.Timestamp(end).tz_localize(None).normalize()
    if end < start:
        raise ValueError("XNYS session range end must not precede start")
    calendar = xcals.get_calendar(
        "XNYS",
        start=start - pd.Timedelta(days=7),
        end=end + pd.Timedelta(days=2),
    )
    return pd.DatetimeIndex(calendar.sessions_in_range(start, end)).tz_localize(None)


@dataclass(frozen=True)
class WorkbenchBundle:
    adjusted_close: pd.DataFrame
    cash_index: pd.DataFrame
    instruments: pd.DataFrame
    manifest: dict
    path: Path

    @property
    def tickers(self):
        return tuple(self.instruments["ticker"])

    @property
    def signal_as_of(self):
        return pd.Timestamp(self.manifest["generated_at_utc"])

    @property
    def cash_valuation_through(self):
        return pd.Timestamp(self.manifest["cash_valuation_through"])

    def cash_values(self, dates):
        return cash_values_on_dates(
            self.cash_index,
            dates,
            valuation_through=self.cash_valuation_through,
        )

    def cash_returns(self, start_dates, end_dates):
        return cash_returns_between(
            self.cash_index,
            start_dates,
            end_dates,
            valuation_through=self.cash_valuation_through,
        )

    def freshness(self, as_of=None):
        """Return a small deployment status derived from completed months."""
        if as_of is None:
            as_of = pd.Timestamp.now(tz="UTC")
        as_of = pd.Timestamp(as_of)
        if as_of.tz is not None:
            as_of = as_of.tz_convert("UTC").tz_localize(None)
        generated_at = pd.Timestamp(self.manifest["generated_at_utc"])
        if generated_at.tz is not None:
            generated_at = generated_at.tz_convert("UTC").tz_localize(None)
        expected_period = as_of.to_period("M") - 1
        artifact_period = pd.Timestamp(
            self.manifest["last_complete_month"]
        ).to_period("M")
        if generated_at > as_of:
            return {
                "status": "disabled",
                "reason": "artifact generated_at_utc is in the future",
                "months_behind": 0,
                "last_complete_month": str(artifact_period),
                "expected_complete_month": str(expected_period),
            }
        if artifact_period > expected_period:
            return {
                "status": "disabled",
                "reason": "artifact completed period is in the future",
                "months_behind": 0,
                "last_complete_month": str(artifact_period),
                "expected_complete_month": str(expected_period),
            }
        months_behind = max(0, expected_period.ordinal - artifact_period.ordinal)
        status = (
            "current"
            if months_behind == 0
            else "warning"
            if months_behind == 1
            else "disabled"
        )
        return {
            "status": status,
            "reason": (
                None
                if status == "current"
                else "artifact is one completed month behind"
                if status == "warning"
                else "artifact is two or more completed months behind"
            ),
            "months_behind": months_behind,
            "last_complete_month": str(artifact_period),
            "expected_complete_month": str(expected_period),
        }


def _read_dated_csv(path, index_name):
    frame = pd.read_csv(path)
    if index_name not in frame.columns:
        raise ValueError(f"{path.name} must contain a {index_name} column")
    frame[index_name] = pd.to_datetime(frame[index_name], errors="raise")
    return frame.set_index(index_name)


def _validate_prices(prices, instruments):
    if prices.empty:
        raise ValueError("adjusted_close.csv must not be empty")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("adjusted price dates must be unique and increasing")
    expected = list(instruments["ticker"])
    if list(prices.columns) != expected:
        raise ValueError("adjusted price columns must match the instrument registry order")
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    for ticker in expected:
        series = numeric[ticker]
        valid = series.dropna()
        if valid.empty:
            raise ValueError(f"{ticker} has no adjusted price history")
        if not np.isfinite(valid.to_numpy()).all() or (valid <= 0.0).any():
            raise ValueError(f"{ticker} contains invalid adjusted prices")
        internal = series.loc[valid.index[0] : valid.index[-1]]
        if internal.isna().any():
            first_gap = internal.index[internal.isna()][0]
            raise ValueError(
                f"{ticker} has an internal adjusted-price gap on {first_gap.date()}"
            )
    return numeric


def _validate_cash(frame, manifest):
    required = [
        "cash_index",
        "annual_rate",
        "source_series",
        "effective_date",
        "accrual_days",
    ]
    if list(frame.columns) != required:
        raise ValueError("cash_index.csv columns do not match the v1 schema")
    if frame.empty or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("cash index dates must be nonempty, unique, and increasing")
    frame = frame.copy()
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="raise")
    if not frame["effective_date"].equals(pd.Series(frame.index, index=frame.index)):
        raise ValueError("cash effective dates must equal cash row dates")
    for column in ("cash_index", "annual_rate", "accrual_days"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"cash {column} must be finite")
    if (frame["cash_index"] <= 0.0).any() or (frame["accrual_days"] < 0).any():
        raise ValueError("cash index levels and accrual intervals are invalid")
    if not np.equal(frame["accrual_days"] % 1, 0).all():
        raise ValueError("cash accrual_days must be whole calendar days")
    if not set(frame["source_series"]).issubset({"EFFR", "SOFR"}):
        raise ValueError("cash source_series must contain only EFFR or SOFR")
    switch = pd.Timestamp(manifest["cash_switch_date"])
    if (frame.loc[frame.index < switch, "source_series"] != "EFFR").any():
        raise ValueError("cash observations before the switch must use EFFR")
    if (frame.loc[frame.index >= switch, "source_series"] != "SOFR").any():
        raise ValueError("cash observations at/after the switch must use SOFR")
    sofr_dates = frame.index[frame["source_series"] == "SOFR"]
    if sofr_dates.empty or sofr_dates[0] != switch:
        raise ValueError("cash SOFR observations must begin exactly on the switch date")
    effr_dates = frame.index[frame["source_series"] == "EFFR"]
    if effr_dates.empty or (switch - effr_dates[-1]).days > 7:
        raise ValueError("cash EFFR history does not remain active through the switch")

    interval_days = np.diff(frame.index.to_numpy()).astype("timedelta64[D]").astype(int)
    if not np.array_equal(
        frame["accrual_days"].iloc[:-1].to_numpy(dtype=int), interval_days
    ):
        raise ValueError("cash accrual_days do not match consecutive effective dates")
    expected_next = frame["cash_index"].iloc[:-1].to_numpy(dtype=float) * (
        1.0
        + frame["annual_rate"].iloc[:-1].to_numpy(dtype=float)
        / 100.0
        * interval_days
        / 360.0
    )
    if not np.allclose(
        expected_next,
        frame["cash_index"].iloc[1:].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-10,
    ):
        raise ValueError("cash index levels do not reconcile with rates and accruals")
    valuation_through = pd.Timestamp(manifest["cash_valuation_through"])
    expected_last_accrual = (valuation_through - frame.index[-1]).days
    if int(frame["accrual_days"].iloc[-1]) != expected_last_accrual:
        raise ValueError("last cash accrual does not reach cash_valuation_through")
    return frame


def _validate_manifest_provenance(manifest, require_clean):
    required_keys = {
        "schema_version",
        "generated_at_utc",
        "price_data_as_of",
        "last_complete_month",
        "cash_rate_as_of",
        "cash_valuation_through",
        "cash_identifier",
        "cash_label",
        "cash_policy",
        "cash_switch_date",
        "sofr_index_validation",
        "instrument_count",
        "dependency_versions",
        "source_queries",
        "row_counts",
        "file_sha256",
        "validation_status",
        "git_sha",
        "git_dirty_at_build",
        "pipeline_version",
    }
    missing = sorted(required_keys - set(manifest))
    if missing:
        raise ValueError(f"manifest is missing required keys: {missing}")
    if not isinstance(manifest["git_dirty_at_build"], bool):
        raise ValueError("manifest git_dirty_at_build must be boolean")
    if require_clean and manifest["git_dirty_at_build"] is not False:
        raise ValueError("release workbench artifacts require a clean Git build")
    git_sha = manifest["git_sha"]
    if not isinstance(git_sha, str) or re.fullmatch(r"[0-9a-fA-F]{40}", git_sha) is None:
        raise ValueError("manifest git_sha must be a 40-character hexadecimal commit hash")

    dependencies = manifest["dependency_versions"]
    for dependency in REQUIRED_DEPENDENCY_VERSIONS:
        version = dependencies.get(dependency) if isinstance(dependencies, dict) else None
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"manifest lacks exact {dependency} dependency version")

    queries = manifest["source_queries"]
    for source_name in REQUIRED_SOURCE_QUERIES:
        stats = queries.get(source_name) if isinstance(queries, dict) else None
        if not isinstance(stats, dict) or not set(SOURCE_QUERY_FIELDS).issubset(stats):
            raise ValueError(f"manifest source query metadata is incomplete for {source_name}")
        requested_start = pd.Timestamp(stats["requested_start"])
        requested_end = pd.Timestamp(stats["requested_end"])
        returned_start = pd.Timestamp(stats["returned_start"])
        returned_end = pd.Timestamp(stats["returned_end"])
        if not requested_start <= returned_start <= returned_end <= requested_end:
            raise ValueError(f"manifest source range is invalid for {source_name}")
        if not isinstance(stats["returned_count"], int) or stats["returned_count"] <= 0:
            raise ValueError(f"manifest source count is invalid for {source_name}")

    sofr_validation = manifest["sofr_index_validation"]
    required_validation = {"observations", "maximum_absolute_error", "tolerance", "passed"}
    if not isinstance(sofr_validation, dict) or not required_validation.issubset(
        sofr_validation
    ):
        raise ValueError("manifest SOFR Index validation metadata is incomplete")
    if sofr_validation["passed"] is not True:
        raise ValueError("manifest SOFR Index validation did not pass")
    if int(sofr_validation["observations"]) <= 0 or float(
        sofr_validation["maximum_absolute_error"]
    ) > float(sofr_validation["tolerance"]):
        raise ValueError("manifest SOFR Index validation values are invalid")


def validate_workbench_bundle(
    path=DEFAULT_BUNDLE_PATH,
    verify_checksums=True,
    require_clean=True,
):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"workbench artifact directory does not exist: {path}")
    actual_entries = tuple(sorted(item.name for item in path.iterdir()))
    if actual_entries != tuple(sorted(PUBLIC_FILENAMES)):
        raise ValueError(
            "workbench artifact directory must contain exactly the four public files"
        )

    manifest = json.loads((path / "manifest.json").read_text())
    _validate_manifest_provenance(manifest, require_clean=require_clean)
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported workbench artifact schema")
    if manifest.get("cash_identifier") != CASH_ASSET:
        raise ValueError("manifest cash identifier is invalid")
    if manifest.get("validation_status") != "passed":
        raise ValueError("workbench artifact validation status is not passed")
    if verify_checksums:
        expected_checksums = manifest.get("file_sha256", {})
        for filename in PUBLIC_FILENAMES[:-1]:
            if expected_checksums.get(filename) != file_sha256(path / filename):
                raise ValueError(f"checksum mismatch for {filename}")

    instruments = pd.read_csv(path / "instruments.csv")
    required_registry_columns = list(INSTRUMENT_COLUMNS) + [
        "tradeable",
        "price_start",
        "price_end",
    ]
    if list(instruments.columns) != required_registry_columns:
        raise ValueError("instruments.csv columns do not match the v1 schema")
    if len(instruments) != len(INSTRUMENT_RECORDS) or not instruments["ticker"].is_unique:
        raise ValueError("instruments.csv must contain the 14 unique approved ETFs")
    if list(instruments["ticker"]) != [record[0] for record in INSTRUMENT_RECORDS]:
        raise ValueError("instruments.csv does not contain the approved ETF universe")
    instruments["inception_date"] = pd.to_datetime(instruments["inception_date"])
    instruments["price_start"] = pd.to_datetime(instruments["price_start"])
    instruments["price_end"] = pd.to_datetime(instruments["price_end"])
    if not instruments["tradeable"].astype(bool).all():
        raise ValueError("all registered instruments must be tradeable ETFs")

    prices = _read_dated_csv(path / "adjusted_close.csv", "date")
    prices = _validate_prices(prices, instruments)
    cash = _read_dated_csv(path / "cash_index.csv", "date")
    cash = _validate_cash(cash, manifest)

    generated_at = pd.Timestamp(manifest["generated_at_utc"])
    expected_completed_session = prior_completed_xnys_session(generated_at)
    manifest_completed_session = pd.Timestamp(manifest["last_complete_month"])
    if manifest_completed_session != expected_completed_session:
        raise ValueError(
            "manifest last_complete_month is not the final XNYS session of the "
            "calendar month before generated_at_utc"
        )
    if prices.index.max() != expected_completed_session:
        raise ValueError(
            "adjusted prices must end on the final completed XNYS monthly session"
        )
    if prices.loc[expected_completed_session].isna().any():
        raise ValueError("every approved ETF must have a price on last_complete_month")
    expected_sessions = xnys_sessions(prices.index[0], expected_completed_session)
    if not prices.index.equals(expected_sessions):
        missing_sessions = expected_sessions.difference(prices.index)
        extra_dates = prices.index.difference(expected_sessions)
        raise ValueError(
            "adjusted price index must exactly equal XNYS sessions; "
            f"missing={list(missing_sessions[:3])}, extra={list(extra_dates[:3])}"
        )
    if str(prices.index.max().date()) != manifest.get("price_data_as_of"):
        raise ValueError("manifest price_data_as_of does not match adjusted prices")
    if str(cash.index.max().date()) != manifest.get("cash_rate_as_of"):
        raise ValueError("manifest cash_rate_as_of does not match cash index")
    cash_valuation_through = pd.Timestamp(manifest["cash_valuation_through"])
    if cash_valuation_through < prices.index.max():
        raise ValueError("cash valuation horizon does not cover the latest ETF price")
    if int(manifest.get("instrument_count", -1)) != len(instruments):
        raise ValueError("manifest instrument_count is invalid")
    expected_rows = manifest.get("row_counts", {})
    if expected_rows.get("adjusted_close.csv") != len(prices):
        raise ValueError("manifest adjusted-price row count is invalid")
    if expected_rows.get("cash_index.csv") != len(cash):
        raise ValueError("manifest cash-index row count is invalid")
    if expected_rows.get("instruments.csv") != len(instruments):
        raise ValueError("manifest instrument row count is invalid")

    for row in instruments.itertuples(index=False):
        valid = prices[row.ticker].dropna()
        if row.price_start != valid.index[0] or row.price_end != valid.index[-1]:
            raise ValueError(
                f"instrument price range does not match adjusted prices for {row.ticker}"
            )

    return WorkbenchBundle(prices, cash, instruments, manifest, path)


def load_workbench_bundle(path=DEFAULT_BUNDLE_PATH):
    """Load only validated committed files; this function performs no HTTP calls."""
    return validate_workbench_bundle(
        path,
        verify_checksums=True,
        require_clean=True,
    )
