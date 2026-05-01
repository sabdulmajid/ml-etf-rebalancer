import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "latest"


def load_summary():
    started = time.perf_counter()
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text())
    metrics = pd.read_csv(ARTIFACT_DIR / "metrics.csv", index_col=0)
    allocation = pd.read_csv(ARTIFACT_DIR / "current_allocation.csv")
    weights = pd.read_csv(ARTIFACT_DIR / "weights.csv", index_col=0)
    elapsed = time.perf_counter() - started

    return {
        "artifact_load_seconds": round(elapsed, 4),
        "data_end": manifest["data_end"],
        "backtest_months": int(metrics.loc["ML Signal Blend", "Months"]),
        "ml_signal_cagr": round(float(metrics.loc["ML Signal Blend", "CAGR"]), 4),
        "ml_signal_sharpe": round(float(metrics.loc["ML Signal Blend", "Sharpe Ratio"]), 4),
        "ml_signal_max_drawdown": round(float(metrics.loc["ML Signal Blend", "Max Drawdown"]), 4),
        "allocation_sum": round(float(allocation["weight"].sum()), 8),
        "max_current_weight": round(float(allocation["weight"].max()), 4),
        "historical_weight_sum_min": round(float(weights.sum(axis=1).min()), 8),
        "historical_weight_sum_max": round(float(weights.sum(axis=1).max()), 8),
    }


def time_command(command):
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    elapsed = time.perf_counter() - started
    return elapsed, result


def main():
    parser = argparse.ArgumentParser(description="Benchmark ETF Signal Lab local health.")
    parser.add_argument("--pipeline", action="store_true", help="Also time a full artifact refresh.")
    args = parser.parse_args()

    summary = load_summary()

    app_seconds, app_result = time_command([sys.executable, "dashboard/app.py"])
    summary["dashboard_bare_exec_seconds"] = round(app_seconds, 4)
    summary["dashboard_bare_exec_returncode"] = app_result.returncode

    if args.pipeline:
        pipeline_seconds, pipeline_result = time_command([sys.executable, "run_pipeline.py"])
        summary["pipeline_seconds"] = round(pipeline_seconds, 4)
        summary["pipeline_returncode"] = pipeline_result.returncode
        if pipeline_result.returncode != 0:
            summary["pipeline_stderr_tail"] = pipeline_result.stderr[-1200:]

    print(json.dumps(summary, indent=2))

    if app_result.returncode != 0:
        print(app_result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(app_result.returncode)


if __name__ == "__main__":
    main()
