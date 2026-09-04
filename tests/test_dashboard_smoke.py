import subprocess
import sys
from pathlib import Path
import datetime as dt
from copy import deepcopy
from io import StringIO
import json
import shutil

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from backtest.engine import CASH_ASSET
from dashboard.workbench import (
    ALLOCATION_LABELS,
    CURRENT_MIX_LABEL,
    EQUAL_WEIGHT_LABEL,
    VOL_BALANCED_LABEL,
    VOL_FORECAST_LABEL,
    VOL_TREND_LABEL,
    comparison_selector_label,
)
from data.timesfm import load_timesfm_bundle
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
        timeout=45,
    )

    assert result.returncode == 0, result.stderr[-2000:]


def test_streamlit_workbench_transfer_survives_rerun_and_uses_exact_target(
    monkeypatch,
):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-09-05")

    def no_http(*args, **kwargs):
        raise AssertionError("Streamlit runtime attempted public HTTP")

    monkeypatch.setattr("requests.sessions.Session.request", no_http)
    monkeypatch.setattr("yfinance.download", no_http)
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=45).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs][:2] == [
        "ETF Allocation Workbench",
        "ML Current Allocation",
    ]
    info_text = " ".join(str(item.value) for item in app.info)
    assert "1 · Choose ETFs" in info_text
    assert "2 · Compare approaches" in info_text
    assert "3 · Rebalance (optional)" in info_text
    explanation = app.table[0].value
    assert explanation.columns.tolist() == [
        "Asset",
        "Proposed target",
        "Role and signal",
        "Reason",
    ]
    assert explanation["Proposed target"].sum() == pytest.approx(1.0)
    assert explanation["Reason"].str.len().gt(0).all()
    app.checkbox(key="wb_current_enabled").set_value(True).run()
    assert any(
        "add Current Mix — monthly rebalanced" in item.value
        for item in app.caption
    )
    app.button(key="wb_send_to_portfolio_lab").click().run()
    assert not app.exception
    assert any("Open the Portfolio Lab tab" in item.value for item in app.success)
    assert any(
        "Move from your current mix to the proposal" in item.value
        for item in app.markdown
    )

    transfer = deepcopy(app.session_state["portfolio_lab_transfer"])
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
    assert transfer["strategy"] == EQUAL_WEIGHT_LABEL
    assert transfer["execution_status"] == "constant_target_effective_for_analytical_ticket"
    assert transfer["policy_version"] == "equal-weight-monthly-v1"
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
    assert rerun["displayed_history_through"] == transfer["displayed_history_through"]
    app.slider(key="ml_sandbox_forecast").set_value(0.2).run()
    sandbox_rerun = app.session_state["portfolio_lab_transfer"]
    assert sandbox_rerun["target_weights"] == transfer["target_weights"]
    assert sandbox_rerun["target_csv"] == transfer["target_csv"]
    assert not app.exception


def test_portfolio_lab_currency_summary_escapes_streamlit_markdown(monkeypatch):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-09-05")
    app = AppTest.from_file(
        str(ROOT / "dashboard" / "app.py"), default_timeout=45
    ).run()
    app.checkbox(key="wb_current_enabled").set_value(True).run()
    app.button(key="wb_send_to_portfolio_lab").click().run()
    app.checkbox(key="portfolio_lab_include_dollars").set_value(True).run()

    rendered = [item.value for item in app.markdown]
    summary = next(item for item in rendered if item.startswith("ETF notionals:"))
    assert r"\$" in summary
    assert "not deducted from these notionals" in summary
    assert not app.exception


def test_streamlit_selection_reconciliation_and_invalid_current_controls(monkeypatch):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-09-05")
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=45).run()
    app.checkbox(key="wb_current_enabled").set_value(True).run()
    assert any("Current total: 100.00%" in item.value for item in app.success)
    app.number_input(key=f"wb_current_pct_{CASH_ASSET}").set_value(90.0).run()
    assert any("Add 10.00 percentage points" in item.value for item in app.info)
    app.number_input(key="wb_current_pct_SPY").set_value(20.0).run()
    assert any("Remove 10.00 percentage points" in item.value for item in app.warning)
    app.number_input(key=f"wb_current_pct_{CASH_ASSET}").set_value(80.0).run()
    assert any("Ready to compare" in item.value for item in app.success)
    app.number_input(key="wb_current_pct_SPY").set_value(0.0).run()
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
    assert comparison_selector_label(CURRENT_MIX_LABEL) not in comparison.options
    assert app.button(key="wb_send_to_portfolio_lab_disabled").disabled
    assert "portfolio_lab_transfer" not in app.session_state
    assert any("Current weights are invalid" in warning.value for warning in app.warning)
    assert any("adjust the current-weight total" in item.value for item in app.info)
    assert not app.exception


def test_late_forecast_history_is_visible_but_target_is_not_transferable(monkeypatch):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-09-05")
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=45).run()

    comparison = next(
        widget for widget in app.multiselect if widget.label == "Comparison series"
    )
    target = app.selectbox(key="wb_authoritative_target")
    assert comparison_selector_label(VOL_FORECAST_LABEL) in comparison.options
    assert VOL_FORECAST_LABEL in comparison.value
    assert VOL_FORECAST_LABEL not in target.options
    assert VOL_BALANCED_LABEL not in target.options
    assert VOL_TREND_LABEL not in target.options
    assert any(
        "historical research is available" in warning.value
        and "target transfer stays disabled" in warning.value
        for warning in app.warning
    )
    assert any(
        "historical research only" in expander.label.lower()
        for expander in app.expander
    )
    forecast_checks = next(
        item.value
        for item in app.dataframe
        if "Pinball loss ↓" in item.value.columns
    )
    assert "—" in forecast_checks["Pinball loss ↓"].tolist()
    assert not forecast_checks.map(lambda value: value is None).any().any()
    assert not forecast_checks.astype(str).eq("None").any().any()
    assert not app.exception


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_forecast_bundle_failure_leaves_base_workbench_available(
    tmp_path, monkeypatch, kind
):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-09-05")
    forecast_path = tmp_path / "timesfm"
    if kind == "corrupt":
        shutil.copytree(ROOT / "artifacts" / "timesfm", forecast_path)
        signals = forecast_path / "forecast_signals.csv"
        signals.write_text(signals.read_text() + "\n")
    source = f"""
import streamlit as st
from dashboard.workbench import render_workbench
render_workbench(timesfm_bundle_path=r{str(forecast_path)!r})
st.write('base-and-ml-sentinel')
"""
    app = AppTest.from_string(source, default_timeout=45).run()

    assert not app.exception
    assert any("forecast research is temporarily unavailable" in item.value.lower()
               for item in app.info)
    comparison = next(
        widget for widget in app.multiselect if widget.label == "Comparison series"
    )
    assert comparison_selector_label(VOL_BALANCED_LABEL) in comparison.options
    assert comparison_selector_label(VOL_FORECAST_LABEL) not in comparison.options
    assert any("base-and-ml-sentinel" in item.value for item in app.markdown)


def test_timely_forecast_target_transfers_exactly_to_portfolio_lab(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-09-05")
    timely_path = tmp_path / "timesfm"
    shutil.copytree(ROOT / "artifacts" / "timesfm", timely_path)
    manifest_path = timely_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    latest_generation_id = manifest["model_generations"][-1]["id"]
    generation = next(
        item
        for item in manifest["model_generations"]
        if item["id"] == latest_generation_id
    )
    generation["generated_at_utc"] = "2026-09-01T19:00:00Z"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    source = f"""
from dashboard.workbench import render_portfolio_lab, render_workbench
render_workbench(timesfm_bundle_path=r{str(timely_path)!r})
render_portfolio_lab()
"""
    app = AppTest.from_string(source, default_timeout=45).run()
    assert not app.exception
    app.checkbox(key="wb_current_enabled").set_value(True).run()
    target_selector = app.selectbox(key="wb_authoritative_target")
    assert VOL_FORECAST_LABEL in target_selector.options
    target_selector.set_value(VOL_FORECAST_LABEL).run()
    app.button(key="wb_send_to_portfolio_lab").click().run()

    transfer = deepcopy(app.session_state["portfolio_lab_transfer"])
    assert transfer["strategy"] == VOL_FORECAST_LABEL
    transferred_download = pd.read_csv(StringIO(transfer["target_csv"]))
    assert transferred_download["strategy"].unique().tolist() == [
        VOL_FORECAST_LABEL
    ]
    assert transfer["execution_status"] == "current"
    assert transfer["model_generation_id"] == latest_generation_id
    assert transfer["model_generated_at_utc"] == "2026-09-01T19:00:00Z"
    assert transfer["execution_cutoff_utc"] == "2026-09-01T20:00:00Z"
    assert transfer["forecast_bundle_refreshed_at_utc"] == manifest[
        "bundle_refresh"
    ]["generated_at_utc"]
    selected = tuple(transfer["selected_etfs"])
    target = pd.Series(transfer["target_weights"], dtype=float).reindex(
        [*selected, CASH_ASSET]
    )
    workbench = load_workbench_bundle()
    forecasts = load_timesfm_bundle(timely_path)
    from strategies.forecast import generate_forecast_allocation_targets

    expected = generate_forecast_allocation_targets(
        workbench.adjusted_close.loc[:, selected],
        selected,
        forecasts.signals,
        as_of=workbench.signal_as_of,
    ).latest_target
    pd.testing.assert_series_equal(target, expected, check_names=False)
    ticket = build_rebalance_ticket(
        transfer["current_weights"],
        transfer["target_weights"],
        selected,
        transaction_cost_bps=transfer["transaction_cost_bps"],
    )
    assert ticket.download_frame()["target_weight"].sum() == pytest.approx(1.0)
    assert CASH_ASSET not in set(ticket.security_orders["asset"])
    assert any(
        "Move from your current mix to the proposal" in item.value
        for item in app.markdown
    )
    assert not app.exception


def test_timely_standard_tactical_target_transfers_after_intended_close(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-09-05")
    timely_path = tmp_path / "workbench"
    shutil.copytree(ROOT / "artifacts" / "workbench", timely_path)
    manifest_path = timely_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generated_at_utc"] = "2026-09-01T19:00:00Z"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    missing_forecast = tmp_path / "no-forecast-bundle"
    source = f"""
from dashboard.workbench import render_portfolio_lab, render_workbench
render_workbench(
    bundle_path=r{str(timely_path)!r},
    timesfm_bundle_path=r{str(missing_forecast)!r},
)
render_portfolio_lab()
"""
    app = AppTest.from_string(source, default_timeout=45).run()
    assert not app.exception
    app.checkbox(key="wb_current_enabled").set_value(True).run()
    selector = app.selectbox(key="wb_authoritative_target")
    assert VOL_BALANCED_LABEL in selector.options
    assert VOL_TREND_LABEL in selector.options
    selector.set_value(VOL_TREND_LABEL).run()
    app.button(key="wb_send_to_portfolio_lab").click().run()

    transfer = deepcopy(app.session_state["portfolio_lab_transfer"])
    assert transfer["strategy"] == VOL_TREND_LABEL
    assert transfer["execution_status"] == "current"
    assert transfer["execution_cutoff_utc"] == "2026-09-01T20:00:00Z"
    assert transfer["workbench_bundle_refreshed_at_utc"] == (
        "2026-09-01T19:00:00Z"
    )
    selected = tuple(transfer["selected_etfs"])
    expected = generate_allocation_targets(
        load_workbench_bundle(timely_path).adjusted_close.loc[:, selected],
        selected,
        ALLOCATION_LABELS[VOL_TREND_LABEL],
        as_of=load_workbench_bundle(timely_path).signal_as_of,
    ).latest_target
    pd.testing.assert_series_equal(
        pd.Series(transfer["target_weights"], dtype=float).reindex(
            [*selected, CASH_ASSET]
        ),
        expected,
        check_names=False,
    )
    assert any(
        "Move from your current mix to the proposal" in item.value
        for item in app.markdown
    )
    assert not app.exception


def test_comparison_defaults_follow_selected_etf_tuple(monkeypatch):
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-09-05")
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=45).run()

    initial = next(widget for widget in app.multiselect if widget.label == "Comparison series")
    initial_default = list(initial.value)
    app.multiselect(key="wb_selected_etfs").set_value(["SPY"]).run()
    one = next(widget for widget in app.multiselect if widget.label == "Comparison series")
    assert one.value == [
        "Volatility Balanced + Trend",
        "Volatility Balanced + Forecast",
        "Buy & Hold",
        "Cash — U.S. overnight-rate proxy",
    ]
    assert app.session_state["wb_comparisons_by_selection"]["SPY"] == one.value
    assert any(
        "keeps holding the ETF" in item.value
        and "filtered approaches switch between that ETF" in item.value
        for item in app.info
    )
    app.multiselect(key="wb_selected_etfs").set_value(["SPY", "IEF", "GLD"]).run()
    restored = next(widget for widget in app.multiselect if widget.label == "Comparison series")
    assert restored.value == initial_default
    assert not app.exception


@pytest.mark.parametrize(
    ("as_of", "element", "text"),
    [
        ("2026-10-15", "warning", "one completed month behind"),
        ("2026-11-15", "error", "two or more completed months behind"),
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
    app = AppTest.from_string(source, default_timeout=45).run()

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
    monkeypatch.setenv("ETF_WORKBENCH_TEST_AS_OF", "2026-09-05")
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
