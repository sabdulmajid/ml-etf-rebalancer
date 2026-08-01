from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "latest"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio.rebalance import compute_weights
from dashboard.workbench import render_portfolio_lab, render_workbench


st.set_page_config(
    page_title="ETF Research Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #17211f;
        --muted: #63706b;
        --paper: #f4efe4;
        --panel: #fffaf0;
        --panel-strong: #fbf2dd;
        --line: rgba(23, 33, 31, 0.16);
        --green: #0c6148;
        --green-soft: #dcebe2;
        --gold: #b6782f;
        --red: #9d3d2f;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 4%, rgba(182, 120, 47, 0.18), transparent 26rem),
            radial-gradient(circle at 88% 8%, rgba(12, 97, 72, 0.13), transparent 28rem),
            linear-gradient(180deg, #f8f2e7 0%, var(--paper) 48%, #efe7d8 100%);
        color: var(--ink);
    }

    h1, h2, h3 {
        font-family: 'Newsreader', serif !important;
        color: var(--ink);
        letter-spacing: -0.03em;
    }

    p, label, div, span {
        font-family: 'IBM Plex Mono', monospace;
    }

    section[data-testid="stSidebar"] {
        background: rgba(255, 250, 240, 0.84);
        border-right: 1px solid var(--line);
    }

    .hero {
        border: 1px solid var(--line);
        background: rgba(255, 250, 240, 0.82);
        border-radius: 28px;
        padding: 2.2rem 2.4rem;
        box-shadow: 0 24px 80px rgba(46, 35, 20, 0.10);
        margin-bottom: 1.4rem;
    }

    .eyebrow {
        color: var(--green);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-family: 'Newsreader', serif !important;
        font-size: clamp(3rem, 7vw, 6rem);
        line-height: 0.88;
        margin: 0;
        max-width: 850px;
    }

    .hero-copy {
        color: var(--muted);
        max-width: 850px;
        line-height: 1.65;
        font-size: 0.96rem;
        margin-top: 1.1rem;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1.4rem;
    }

    .pill {
        background: var(--panel-strong);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.55rem 0.8rem;
        color: var(--ink);
        font-size: 0.78rem;
    }

    .metric-card {
        border: 1px solid var(--line);
        background: rgba(255, 250, 240, 0.86);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        min-height: 118px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-family: 'Newsreader', serif !important;
        font-size: 2rem;
        color: var(--ink);
        line-height: 1;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.74rem;
        margin-top: 0.45rem;
    }

    .callout {
        border: 1px solid rgba(12, 97, 72, 0.28);
        background: rgba(220, 235, 226, 0.72);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        color: var(--ink);
        font-size: 0.85rem;
        line-height: 1.55;
    }

    .disclaimer {
        border: 1px solid rgba(157, 61, 47, 0.30);
        background: rgba(157, 61, 47, 0.07);
        border-radius: 16px;
        padding: 0.85rem 1rem;
        color: #59322d;
        font-size: 0.78rem;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 250, 240, 0.72);
        border: 1px solid var(--line);
        padding: 0.9rem 1rem;
        border-radius: 18px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: rgba(255, 250, 240, 0.56);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.75rem 1.1rem;
        color: var(--muted);
    }

    .stTabs [aria-selected="true"] {
        background: var(--green-soft);
        color: var(--green);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def require_artifacts():
    manifest_path = ARTIFACT_DIR / "manifest.json"
    if not manifest_path.exists():
        st.error("Research artifacts were not found. Run `python run_pipeline.py` from the repo root.")
        st.stop()


@st.cache_data(show_spinner=False)
def load_artifacts():
    require_artifacts()
    with open(ARTIFACT_DIR / "manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    def read_csv(name, **kwargs):
        return pd.read_csv(ARTIFACT_DIR / name, **kwargs)

    metrics = read_csv("metrics.csv", index_col=0)
    for col in metrics.columns:
        converted = pd.to_numeric(metrics[col], errors="coerce")
        if converted.notna().any():
            metrics[col] = converted.where(converted.notna(), metrics[col])

    return {
        "manifest": manifest,
        "metrics": metrics,
        "allocation": read_csv("current_allocation.csv"),
        "equity": read_csv("equity_curves.csv", index_col=0, parse_dates=True),
        "returns": read_csv("strategy_returns.csv", index_col=0, parse_dates=True),
        "drawdowns": read_csv("drawdowns.csv", index_col=0, parse_dates=True),
        "annual": read_csv("annual_returns.csv", index_col=0),
        "features": read_csv("feature_importance.csv"),
        "signals": read_csv("signal_scores.csv", index_col=0, parse_dates=True),
        "weights": read_csv("weights.csv", index_col=0, parse_dates=True),
    }


def pct(value):
    return f"{float(value):.1%}"


def money(value):
    return f"${float(value):,.0f}"


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_equity(equity, selected):
    fig = go.Figure()
    palette = {
        "ML Signal Blend": "#0c6148",
        "Equal-Weight Sectors": "#b6782f",
        "6M Momentum Top 3": "#4b6f8f",
        "SPY Buy & Hold": "#7d4f36",
    }
    for col in selected:
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity[col],
                mode="lines",
                name=col,
                line=dict(width=3 if col == "ML Signal Blend" else 2, color=palette.get(col)),
                hovertemplate="%{x|%b %Y}<br>%{y:$,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,250,240,0.45)",
        font=dict(family="IBM Plex Mono", color="#17211f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(23,33,31,0.12)", tickprefix="$"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width='stretch')


def plot_drawdown(drawdowns, selected):
    fig = go.Figure()
    for col in selected:
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns[col],
                mode="lines",
                name=col,
                fill="tozeroy",
                hovertemplate="%{x|%b %Y}<br>%{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,250,240,0.45)",
        font=dict(family="IBM Plex Mono", color="#17211f"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(23,33,31,0.12)", tickformat=".0%"),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')


def plot_allocation(allocation):
    alloc = allocation.sort_values("weight", ascending=True)
    colors = ["#d5c2a1" if w == 0 else "#0c6148" for w in alloc["weight"]]
    fig = go.Figure(
        go.Bar(
            x=alloc["weight"],
            y=alloc["sector"] + " (" + alloc["ticker"] + ")",
            orientation="h",
            marker=dict(color=colors),
            text=[pct(x) for x in alloc["weight"]],
            textposition="outside",
            hovertemplate="%{y}<br>Weight %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=50, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,250,240,0.45)",
        font=dict(family="IBM Plex Mono", color="#17211f"),
        xaxis=dict(tickformat=".0%", gridcolor="rgba(23,33,31,0.12)", range=[0, max(0.4, alloc["weight"].max() * 1.18)]),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')


def plot_feature_importance(features):
    data = features.sort_values("importance", ascending=True).tail(14)
    fig = go.Figure(
        go.Bar(
            x=data["importance"],
            y=data["feature"],
            orientation="h",
            marker=dict(color="#b6782f"),
            hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,250,240,0.45)",
        font=dict(family="IBM Plex Mono", color="#17211f"),
        xaxis=dict(gridcolor="rgba(23,33,31,0.12)"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, width='stretch')


def normalize_percentages(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)
    total = values.sum()
    if total <= 0:
        return pd.Series(0.0, index=frame.index)
    return values / total


def build_trade_ticket(current_weights, target_weights, portfolio_value, min_trade_dollars=100):
    ticket = pd.DataFrame(
        {
            "ticker": target_weights.index,
            "current_weight": current_weights.reindex(target_weights.index).fillna(0.0),
            "target_weight": target_weights,
        }
    )
    ticket["trade_weight"] = ticket["target_weight"] - ticket["current_weight"]
    ticket["trade_dollars"] = ticket["trade_weight"] * portfolio_value
    ticket["action"] = np.where(ticket["trade_dollars"] > 0, "BUY", "SELL")
    ticket.loc[ticket["trade_dollars"].abs() < min_trade_dollars, "action"] = "HOLD"
    return ticket.sort_values("trade_dollars", key=lambda s: s.abs(), ascending=False)


def static_portfolio_return(monthly_returns, weights):
    aligned = weights.reindex(monthly_returns.columns).fillna(0.0)
    returns = monthly_returns.mul(aligned, axis=1).sum(axis=1)
    return (1 + returns).prod() - 1


def scenario_table(monthly_returns, strategy_returns, target_weights):
    latest_end = monthly_returns.index.max()
    scenarios = {
        "COVID Shock": ("2020-02-29", "2020-03-31"),
        "Inflation Bear": ("2022-01-31", "2022-10-31"),
        "AI Rally": ("2023-01-31", "2023-12-31"),
        "Last 12 Months": ((latest_end - pd.DateOffset(months=11)).strftime("%Y-%m-%d"), latest_end.strftime("%Y-%m-%d")),
    }
    rows = []
    for name, (start, end) in scenarios.items():
        period_returns = monthly_returns.loc[start:end]
        period_strategy = strategy_returns.loc[start:end]
        if period_returns.empty or period_strategy.empty:
            continue
        rows.append(
            {
                "Scenario": name,
                "Start": period_returns.index.min().strftime("%Y-%m-%d"),
                "End": period_returns.index.max().strftime("%Y-%m-%d"),
                "Current Target": static_portfolio_return(period_returns, target_weights),
                "ML Signal Blend": (1 + period_strategy["ML Signal Blend"]).prod() - 1,
                "SPY Buy & Hold": (1 + period_strategy["SPY Buy & Hold"]).prod() - 1,
                "Equal-Weight Sectors": (1 + period_strategy["Equal-Weight Sectors"]).prod() - 1,
            }
        )
    return pd.DataFrame(rows)


def remix_allocation(allocation, forecast_weight, momentum_weight, stability_weight, max_weight, top_n):
    total = forecast_weight + momentum_weight + stability_weight
    if total <= 0:
        forecast_weight, momentum_weight, stability_weight = 0.4, 0.45, 0.15
        total = 1.0
    score = (
        forecast_weight / total * allocation["forecast_score"]
        + momentum_weight / total * allocation["momentum_score"]
        + stability_weight / total * allocation["stability_score"]
    )
    score = score - score.min()
    weights = compute_weights(
        pd.Series(score.values, index=allocation["ticker"]),
        method="simple",
        min_weight=0.0,
        max_weight=max_weight,
        top_n=top_n,
    )
    return weights


data = load_artifacts()
manifest = data["manifest"]
metrics = data["metrics"]
allocation = data["allocation"]
equity = data["equity"]
drawdowns = data["drawdowns"]
annual = data["annual"]
features = data["features"]

sector_columns = list(manifest["universe"].keys())


with st.sidebar:
    st.markdown("### ML Research Controls")
    st.caption("These controls apply only to the existing ML research tabs.")
    initial_capital = st.number_input(
        "Portfolio value",
        min_value=1000,
        max_value=10000000,
        value=int(manifest.get("initial_value", 10000)),
        step=1000,
    )
    strategies = list(equity.columns)
    selected = st.multiselect(
        "Compare strategies",
        options=strategies,
        default=strategies,
    )
    if not selected:
        selected = ["ML Signal Blend"]

    st.markdown("### Signal Recipe")
    signal_weights = manifest.get("signal_weights", {})
    st.write(f"Forecast: {signal_weights.get('forecast', 0):.0%}")
    st.write(f"Momentum: {signal_weights.get('momentum', 0):.0%}")
    st.write(f"Stability: {signal_weights.get('stability', 0):.0%}")
    st.write(f"Max sector: {manifest.get('max_weight', 0):.0%}")
    st.write(f"Transaction cost: {manifest.get('transaction_cost_bps', 0):.0f} bps/trade")

    st.markdown(
        """
        <div class="disclaimer">
        Educational research tool only. This is not financial advice, not a live trading system,
        and not a recommendation to buy or sell securities.
        </div>
        """,
        unsafe_allow_html=True,
    )


scaled_equity = equity / float(manifest.get("initial_value", 10000)) * initial_capital
primary = metrics.loc["ML Signal Blend"]
primary_final = scaled_equity["ML Signal Blend"].iloc[-1]

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">Transparent ETF portfolio research</div>
        <h1 class="hero-title">ETF Research Studio</h1>
        <div class="hero-copy">
            Explore fixed, explainable ETF allocation policies and inspect the existing walk-forward
            machine-learning sector study. Both experiences use committed, reviewable artifacts and
            are educational analytics only; this application never places orders.
        </div>
        <div class="pill-row">
            <div class="pill">Allocation Workbench: 14 curated ETFs</div>
            <div class="pill">ML study through {manifest["data_end"]}</div>
            <div class="pill">Local committed artifacts</div>
            <div class="pill">No brokerage connection</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tab_workbench, tab_allocation, tab_backtest, tab_lab, tab_research = st.tabs(
    [
        "ETF Allocation Workbench",
        "ML Current Allocation",
        "ML Backtest",
        "Portfolio Lab",
        "ML Research Notes",
    ]
)


with tab_workbench:
    render_workbench()


with tab_allocation:
    st.markdown("### ML sector model snapshot")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Current value", money(primary_final), f"on {money(initial_capital)} initial capital")
    with col2:
        metric_card("CAGR", pct(primary["CAGR"]), "walk-forward signal blend")
    with col3:
        metric_card("Sharpe", f"{float(primary['Sharpe Ratio']):.2f}", "0% risk-free assumption")
    with col4:
        metric_card("Max drawdown", pct(primary["Max Drawdown"]), "largest peak-to-trough loss")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("## Current Allocation")
        plot_allocation(allocation)
    with right:
        st.markdown("## Signal Tape")
        active = allocation[allocation["weight"] > 0].sort_values("weight", ascending=False)
        for _, row in active.iterrows():
            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom: 0.8rem;">
                    <div class="metric-label">{row['ticker']} / {row['sector']}</div>
                    <div class="metric-value">{pct(row['weight'])}</div>
                    <div class="metric-note">
                    Forecast {pct(row['predicted_return'])} monthly |
                    Momentum z {row['momentum_score']:.2f} |
                    Stability z {row['stability_score']:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.dataframe(
        allocation.assign(
            weight=allocation["weight"].map(lambda x: f"{x:.2%}"),
            predicted_return=allocation["predicted_return"].map(lambda x: f"{x:.2%}"),
            composite_score=allocation["composite_score"].map(lambda x: f"{x:.2f}"),
        ),
        hide_index=True,
        width='stretch',
    )

    st.download_button(
        "Download allocation CSV",
        allocation.to_csv(index=False),
        file_name="current_etf_signal_allocation.csv",
        mime="text/csv",
    )


with tab_backtest:
    st.markdown("## Walk-Forward Performance")
    plot_equity(scaled_equity, selected)

    st.markdown("## Drawdown")
    plot_drawdown(drawdowns, selected)

    display_metrics = metrics.copy()
    for col in ["Total Return", "CAGR", "Annualized Volatility", "Max Drawdown", "Win Rate", "Best Month", "Worst Month", "Average Monthly Turnover"]:
        if col in display_metrics:
            display_metrics[col] = display_metrics[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    if "Sharpe Ratio" in display_metrics:
        display_metrics["Sharpe Ratio"] = display_metrics["Sharpe Ratio"].map(lambda x: f"{float(x):.2f}")
    st.dataframe(display_metrics, width='stretch')

    st.markdown("## Annual Returns")
    annual_display = annual.copy()
    annual_display.index = annual_display.index.astype(str)
    st.dataframe(annual_display.style.format("{:.2%}"), width='stretch')


with tab_lab:
    render_portfolio_lab()
    st.caption(
        "The hindsight Stress Test remains removed because applying a current target "
        "to past regimes is semantically misleading. The exploratory Signal Remix is "
        "isolated in ML Research Notes and cannot control this ticket."
    )


with tab_research:
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("## Model Drivers")
        plot_feature_importance(features)
    with right:
        st.markdown("## What Makes This Credible")
        st.markdown(
            """
            <div class="callout">
            <strong>No lookahead:</strong> each monthly rebalance is produced by models trained on prior months only.<br><br>
            <strong>Realistic frictions:</strong> the displayed strategy subtracts transaction costs from turnover.<br><br>
            <strong>Honest benchmarks:</strong> the terminal compares against SPY, equal-weight sector exposure,
            and a simple six-month momentum baseline.<br><br>
            <strong>Explainable allocation:</strong> every current position shows model forecast, momentum, and
            volatility-stability contributions.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("## Methodology")
        st.write(manifest["methodology"])
        st.write(f"Source data: adjusted ETF close prices from Yahoo Finance via `yfinance`.")
        st.write(f"Generated from git revision `{manifest.get('git_sha', 'unknown')}`.")

        st.markdown("## Responsible Use")
        st.markdown(
            """
            <div class="disclaimer">
            This app is designed to demonstrate research engineering, model validation, and portfolio analytics.
            It should not be used as a standalone investment decision system.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## ML Sandbox — exploratory, non-authoritative")
    st.caption(
        "Remix the existing ML artifact scores for visual research only. This sandbox "
        "does not write the authoritative workbench target, Portfolio Lab session "
        "state, or any rebalance ticket."
    )
    remix_cols = st.columns(5)
    forecast_mix = remix_cols[0].slider(
        "ML forecast mix",
        0.0,
        1.0,
        float(manifest["signal_weights"]["forecast"]),
        0.05,
        key="ml_sandbox_forecast",
    )
    momentum_mix = remix_cols[1].slider(
        "ML momentum mix",
        0.0,
        1.0,
        float(manifest["signal_weights"]["momentum"]),
        0.05,
        key="ml_sandbox_momentum",
    )
    stability_mix = remix_cols[2].slider(
        "ML stability mix",
        0.0,
        1.0,
        float(manifest["signal_weights"]["stability"]),
        0.05,
        key="ml_sandbox_stability",
    )
    remix_max = remix_cols[3].slider(
        "ML max sector",
        0.15,
        0.60,
        float(manifest["max_weight"]),
        0.05,
        key="ml_sandbox_max_sector",
    )
    remix_top_n = remix_cols[4].slider(
        "ML active sectors",
        2,
        len(sector_columns),
        int(manifest["top_n"]),
        1,
        key="ml_sandbox_top_n",
    )
    remix_weights = remix_allocation(
        allocation,
        forecast_mix,
        momentum_mix,
        stability_mix,
        remix_max,
        remix_top_n,
    )
    remix_df = allocation[["ticker", "sector"]].copy()
    remix_df["exploratory_weight"] = remix_df["ticker"].map(remix_weights).fillna(0.0)
    remix_df = remix_df.sort_values("exploratory_weight")
    remix_figure = go.Figure(
        go.Bar(
            x=remix_df["exploratory_weight"],
            y=remix_df["sector"] + " (" + remix_df["ticker"] + ")",
            orientation="h",
            marker=dict(color="#4b6f8f"),
            text=[pct(value) for value in remix_df["exploratory_weight"]],
            textposition="outside",
        )
    )
    remix_figure.update_layout(
        height=360,
        margin=dict(l=10, r=50, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,250,240,0.45)",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(remix_figure, width="stretch")
