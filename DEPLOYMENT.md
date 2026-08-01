# Deployment

Keep deployment simple: this project is a read-only Streamlit app backed by committed artifacts.

## Local

```bash
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Streamlit Community Cloud

1. Connect the GitHub repository.
2. Set the app entrypoint to `dashboard/app.py`.
3. Use the root `requirements.txt`.
4. Deploy.

The deployed app does not run model training or download market data from the
UI. Both the existing ML experience and the ETF Allocation Workbench read only
committed, reviewable artifacts.

## Refreshing Existing ML Artifacts

```bash
python run_pipeline.py
git add artifacts/latest
git commit -m "Refresh research artifacts"
```

This avoids fragile scheduled jobs and keeps every public result traceable to a reviewed commit.

## Refreshing ETF Allocation Workbench Artifacts

The workbench has one offline builder and exactly four public data files:

```text
artifacts/workbench/
    adjusted_close.csv
    cash_index.csv
    instruments.csv
    manifest.json
```

Run the builder only in a trusted development environment with network access:

```bash
python build_workbench_artifacts.py
python -c "from data.workbench import load_workbench_bundle; load_workbench_bundle()"
git diff --stat -- artifacts/workbench
git diff -- artifacts/workbench/manifest.json artifacts/workbench/instruments.csv
```

The builder downloads adjusted ETF prices through `yfinance` and official EFFR,
SOFR, and SOFR Index observations from the Federal Reserve Bank of New York. It
writes into a staging directory, validates prices, rate provenance, the SOFR
Index reconstruction, schemas, row counts, and checksums, and only then replaces
the previous bundle. A failed build leaves the previous valid directory in
place.

Release builds fail before any download when the Git worktree is dirty. The
manifest records the exact pandas, yfinance, and exchange-calendars versions and
the requested/returned date range and observation count for every source query.
`--allow-dirty` exists only for explicitly non-release local fixtures; public
loading rejects a bundle whose manifest says it was built dirty.

Promotion uses a validated staging directory and two recoverable directory
renames. There is a brief interval between those renames in which the workbench
path is absent to a concurrent reader. A first-rename failure leaves the prior
bundle in place, and a second-rename failure restores it. Deployment should
refresh outside active app startup; this release does not add a versioned
artifact store or pointer framework.

Price coverage is checked against the packaged XNYS exchange calendar. The
bundle must end on the final XNYS session of the calendar month immediately
before its build timestamp, with a valid price for all 14 ETFs. Current partial-
month rows are removed; a truncated completed month fails the build.

The analytical series is always labeled **Cash — U.S. overnight-rate proxy**:

- Official EFFR applies before April 2, 2018.
- Official SOFR applies beginning April 2, 2018.
- Weekend and holiday accrual uses Actual/360 calendar days.
- The combined history is analytical and non-investable; it is not labeled SOFR.
- BIL remains a separately selectable, investable ETF.
- `CASH:USD_OVERNIGHT` is represented as a cash balance, never a security order.

The New York Fed API is the sole source of record for the reference rates. EFFR
is requested only through April 1, 2018, and SOFR must begin exactly on April 2,
2018. Validation rejects wrong response types, out-of-range observations,
impossible long pre-switch EFFR gaps, and a missing active EFFR endpoint. SOFR
continuity is independently checked by reconstructing the official SOFR Index;
the builder does not invent a second holiday calendar or substitute a provider.

Review before committing:

- `validation_status` is `passed`.
- `price_data_as_of`, `cash_rate_as_of`, and `cash_valuation_through` are plausible.
- The reconstructed SOFR Index error is within its recorded tolerance.
- `git_dirty_at_build` is `false`, dependency versions are exact, and every
  source query has requested/returned ranges and counts.
- All three CSV checksums match the manifest.
- Only the approved 14 ETFs appear in `instruments.csv`.
- The current partial month is not reported as `last_complete_month`.

The public application must import `data.workbench`, not the offline builder.
`load_workbench_bundle()` performs no HTTP calls and rejects missing, extra,
corrupt, stale-schema, or checksum-mismatched bundle files. If a workbench bundle
cannot be loaded, deployment should isolate that failure to the workbench while
leaving the existing ML experience available.
