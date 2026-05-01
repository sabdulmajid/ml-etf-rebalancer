import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "latest"


def test_artifact_manifest_and_schema():
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text())
    required_files = [
        "annual_returns.csv",
        "current_allocation.csv",
        "drawdowns.csv",
        "equity_curves.csv",
        "feature_importance.csv",
        "metrics.csv",
        "monthly_returns.csv",
        "predictions.csv",
        "signal_scores.csv",
        "strategy_detail.csv",
        "strategy_returns.csv",
        "weights.csv",
    ]

    for file_name in required_files:
        assert (ARTIFACT_DIR / file_name).exists(), file_name

    assert manifest["data_end"] >= manifest["backtest_end"]
    assert manifest["latest_signal_date"] == manifest["data_end"]
    assert set(manifest["universe"]) == {"XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"}


def test_allocation_and_weights_are_valid():
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text())
    allocation = pd.read_csv(ARTIFACT_DIR / "current_allocation.csv")
    weights = pd.read_csv(ARTIFACT_DIR / "weights.csv", index_col=0)

    assert allocation["weight"].sum() == pytest.approx(1.0)
    assert allocation["weight"].max() <= manifest["max_weight"] + 1e-12
    assert weights.sum(axis=1).min() == pytest.approx(1.0)
    assert weights.sum(axis=1).max() == pytest.approx(1.0)
    assert weights.max(axis=1).max() <= manifest["max_weight"] + 1e-12


def test_strategy_outputs_are_usable():
    metrics = pd.read_csv(ARTIFACT_DIR / "metrics.csv", index_col=0)
    returns = pd.read_csv(ARTIFACT_DIR / "strategy_returns.csv", index_col=0, parse_dates=True)
    equity = pd.read_csv(ARTIFACT_DIR / "equity_curves.csv", index_col=0, parse_dates=True)

    expected = {"ML Signal Blend", "Equal-Weight Sectors", "6M Momentum Top 3", "SPY Buy & Hold"}
    assert set(metrics.index) == expected
    assert set(returns.columns) == expected
    assert set(equity.columns) == expected
    assert len(returns) >= 100
    assert returns.index.equals(equity.index)
    assert pd.to_numeric(metrics["CAGR"]).notna().all()
