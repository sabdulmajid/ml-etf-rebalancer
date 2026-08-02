import subprocess
import sys
from pathlib import Path
import datetime as dt

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from backtest.engine import CASH_ASSET
from dashboard.workbench import (
    ALLOCATION_LABELS,
    CURRENT_MIX_LABEL,
)
from data.workbench import load_workbench_bundle
from portfolio.rebalance import build_rebalance_ticket
from strategies.allocation import generate_allocation_targets


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_executes_in_bare_mode():
    result = subprocess.run(
        [sys.executable, "dashboard/app.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr[-2000:]


def test_streamlit_workbench_transfer_survives_rerun_and_uses_exact_target(
    monkeypatch,
):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-08-02")

    def no_http(*args, **kwargs):
        raise AssertionError("Streamlit runtime attempted public HTTP")

    monkeypatch.setattr("requests.sessions.Session.request", no_http)
    monkeypatch.setattr("yfinance.download", no_http)
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=20).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs][:2] == [
        "ETF Allocation Workbench",
        "ML Current Allocation",
    ]
    app.checkbox(key="wb_current_enabled").set_value(True).run()
    app.button(key="wb_send_to_portfolio_lab").click().run()
    assert not app.exception

    transfer = app.session_state["portfolio_lab_transfer"]
    selected = tuple(transfer["selected_etfs"])
    target = pd.Series(transfer["target_weights"], dtype=float).reindex(
        [*selected, CASH_ASSET]
    )
    current = pd.Series(transfer["current_weights"], dtype=float).reindex(
        [*selected, CASH_ASSET]
    )
    assert target.sum() == pytest.approx(1.0)
    ticket = build_rebalance_ticket(
        current,
        target,
        selected,
        transaction_cost_bps=transfer["transaction_cost_bps"],
    )
    assert CASH_ASSET not in set(ticket.security_orders["asset"])
    assert transfer["execution_status"] == "pending_next_trading_close"
    assert transfer["policy_version"] == "allocation-policy-v1"
    bundle = load_workbench_bundle()
    library_target = generate_allocation_targets(
        bundle.adjusted_close.loc[:, selected],
        selected,
        ALLOCATION_LABELS[transfer["strategy"]],
        as_of=bundle.signal_as_of,
    ).latest_target
    pd.testing.assert_series_equal(target, library_target, check_names=False)

    historical = app.date_input(key="wb_date_range")
    _, old_end = historical.value
    historical.set_value((dt.date(2020, 1, 1), old_end)).run()
    rerun = app.session_state["portfolio_lab_transfer"]
    assert rerun["target_weights"] == transfer["target_weights"]
    assert rerun["target_csv"] == transfer["target_csv"]
    app.slider(key="ml_sandbox_forecast").set_value(0.2).run()
    sandbox_rerun = app.session_state["portfolio_lab_transfer"]
    assert sandbox_rerun["target_weights"] == transfer["target_weights"]
    assert sandbox_rerun["target_csv"] == transfer["target_csv"]
    assert not app.exception


def test_streamlit_selection_reconciliation_and_invalid_current_controls(monkeypatch):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-08-02")
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=20).run()
    app.checkbox(key="wb_current_enabled").set_value(True).run()
    app.number_input(key="wb_current_pct_GLD").set_value(20.0)
    app.number_input(key=f"wb_current_pct_{CASH_ASSET}").set_value(80.0)
    app.run()

    app.multiselect(key="wb_selected_etfs").set_value(["SPY", "IEF"]).run()
    assert app.number_input(key=f"wb_current_pct_{CASH_ASSET}").value == 100.0
    app.multiselect(key="wb_selected_etfs").set_value(["SPY", "IEF", "GLD"]).run()
    assert app.number_input(key="wb_current_pct_GLD").value == 0.0
    assert app.number_input(key=f"wb_current_pct_{CASH_ASSET}").value == 100.0

    app.number_input(key=f"wb_current_pct_{CASH_ASSET}").set_value(90.0).run()
    comparison = next(
        widget for widget in app.multiselect if widget.label == "Comparison series"
    )
    assert CURRENT_MIX_LABEL not in comparison.options
    assert app.button(key="wb_send_to_portfolio_lab_disabled").disabled
    assert "portfolio_lab_transfer" not in app.session_state
    assert any("Current weights are invalid" in warning.value for warning in app.warning)
    assert not app.exception


def test_comparison_defaults_follow_selected_etf_tuple(monkeypatch):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-08-02")
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=20).run()

    initial = next(widget for widget in app.multiselect if widget.label == "Comparison series")
    initial_default = list(initial.value)
    app.multiselect(key="wb_selected_etfs").set_value(["SPY"]).run()
    one = next(widget for widget in app.multiselect if widget.label == "Comparison series")
    assert one.value == [
        "Volatility Balanced + Trend",
        "Buy & Hold",
        "Cash — U.S. overnight-rate proxy",
    ]
    app.multiselect(key="wb_selected_etfs").set_value(["SPY", "IEF", "GLD"]).run()
    restored = next(widget for widget in app.multiselect if widget.label == "Comparison series")
    assert restored.value == initial_default
    assert not app.exception


@pytest.mark.parametrize(
    ("as_of", "element", "text"),
    [
        ("2026-09-15", "warning", "one completed month behind"),
        ("2026-10-15", "error", "two or more completed months behind"),
    ],
)
def test_streamlit_freshness_warning_and_disabled_state(monkeypatch, as_of, element, text):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", as_of)
    source = """
import streamlit as st
from dashboard.workbench import render_workbench
render_workbench()
st.write('ML sentinel remains available')
"""
    app = AppTest.from_string(source, default_timeout=20).run()

    messages = getattr(app, element)
    assert any(text in message.value for message in messages)
    assert any("ML sentinel remains available" in item.value for item in app.markdown)
    if element == "error":
        assert not any(widget.label == "Comparison series" for widget in app.multiselect)
    assert not app.exception


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_workbench_data_failure_is_isolated_from_other_app_content(
    tmp_path, monkeypatch, kind
):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-08-02")
    bundle_path = tmp_path / "workbench"
    if kind == "corrupt":
        import shutil

        shutil.copytree(ROOT / "artifacts" / "workbench", bundle_path)
        prices = bundle_path / "adjusted_close.csv"
        prices.write_text(prices.read_text() + "\n")

    source = f"""
import streamlit as st
from dashboard.workbench import render_workbench
render_workbench(r{str(bundle_path)!r})
st.write('ML sentinel remains available')
"""
    app = AppTest.from_string(source, default_timeout=10).run()

    assert not app.exception
    assert any("Workbench unavailable" in error.value for error in app.error)
    assert any("ML sentinel remains available" in text.value for text in app.markdown)
