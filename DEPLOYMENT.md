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
3. Keep the root `requirements.txt` as the only Python dependency manifest.
4. Deploy.

Streamlit Community Cloud searches the entrypoint directory before the
repository root when selecting a dependency file. Therefore this repository
intentionally has no `dashboard/requirements.txt`; adding one would shadow the
root manifest. This behavior is covered by `tests/test_deployment.py` and is
documented in Streamlit's
[app-dependencies guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

The deployed app does not run model training or download market data from the
UI. Both the existing ML experience and the ETF Allocation Workbench read only
committed, reviewable artifacts.

The workbench cache key includes the resolved bundle path plus a SHA-256 digest
of each of the four public files. Replacing any file invalidates the cached
validated bundle and full-artifact allocation schedules even when file size and
timestamps are preserved. Historical range and cost reruns reuse those
schedules and rerun only common accounting.

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

At runtime, the workbench compares the bundle's final completed month with the
current completed month. Current data is enabled, one month behind is enabled
with a warning, and data two or more completed months behind is disabled. A
future build timestamp or future completed period is also disabled. These states
affect only the workbench; the committed ML tabs remain available.

The public UI exposes one authoritative workbench target at a time. Transfer to
Portfolio Lab is explicit and requires valid, non-normalized current ETF and cash
weights summing to 100%. Portfolio Lab rebuilds its ticket from the exact target
stored in Streamlit session state. `CASH:USD_OVERNIGHT` appears only in the
separate cash-balance row and never in ETF security orders.

Signal Remix is rendered only under `ML Sandbox — exploratory,
non-authoritative` in ML Research Notes. It does not read or write the
authoritative workbench target, `portfolio_lab_transfer`, or any ticket. The
hindsight scenario stress test remains removed because applying a current target
to past regimes implied a holdings history that did not exist. Existing ML
artifacts and historical results are not modified.

## Deterministic Local UI Capture

For AppTest or a local screenshot review immediately after an artifact build,
an unset-by-default test hook can supply a deterministic clock:

```bash
ETF_WORKBENCH_TEST_AS_OF=2026-08-02 streamlit run dashboard/app.py
```

Choose a timestamp at or after `generated_at_utc` whose prior completed month
matches the bundle. This hook does not alter prices, targets, returns, files, or
capture screenshots; it only controls freshness evaluation. It is not
configured in production. Review screenshots under `docs/screenshots/pr3` were
captured separately from the running local app.
