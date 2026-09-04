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
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #17211f;
        --muted: #68736f;
        --canvas: #f7f7f4;
        --surface: #ffffff;
        --line: #d9dedb;
        --line-strong: #afb9b4;
        --green: #155b46;
        --green-soft: #e9f1ed;
        --amber: #9a632b;
        --red: #934238;
        --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        --mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    .stApp, .stApp p, .stApp label, .stApp button,
    .stApp input, .stApp textarea {
        font-family: var(--sans);
    }

    h1, h2, h3, h4 {
        font-family: var(--sans) !important;
        color: var(--ink);
        letter-spacing: -0.025em;
        font-weight: 650 !important;
    }

    h2 {
        margin-top: 2.2rem !important;
    }

    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    section[data-testid="stSidebar"] {
        background: #f1f3f0;
        border-right: 1px solid var(--line);
    }

    .hero {
        border-bottom: 1px solid var(--line-strong);
        padding: 0.35rem 0 1.2rem;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        font-family: var(--sans) !important;
        font-size: clamp(2rem, 4vw, 2.65rem);
        font-weight: 650;
        letter-spacing: -0.04em;
        line-height: 1;
        margin: 0;
        max-width: 760px;
    }

    .hero-copy {
        color: var(--muted);
        max-width: 720px;
        line-height: 1.55;
        font-size: 0.95rem;
        margin-top: 0.55rem;
    }

    .metric-card {
        border-top: 2px solid var(--line-strong);
        background: transparent;
        border-radius: 0;
        padding: 0.85rem 0.25rem 0.5rem;
        min-height: 94px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        font-family: var(--sans) !important;
        font-size: 1.8rem;
        font-variant-numeric: tabular-nums;
        font-weight: 550;
        color: var(--ink);
        line-height: 1;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.74rem;
        margin-top: 0.45rem;
    }

    .callout {
        border-left: 2px solid var(--green);
        background: transparent;
        border-radius: 0;
        padding: 0.2rem 0 0.2rem 1rem;
        color: var(--ink);
        font-size: 0.85rem;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        background: transparent;
        border: 0;
        border-top: 2px solid var(--line-strong);
        padding: 0.8rem 0.2rem 0.4rem;
        border-radius: 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background: transparent;
        border-bottom: 1px solid var(--line);
        padding: 0;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        padding: 0.8rem 0 0.7rem;
        color: var(--muted);
        font-size: 0.88rem;
        font-weight: 550;
    }

    .stTabs [aria-selected="true"] {
        background: transparent;
        color: var(--green);
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--green);
        height: 2px;
    }

    [data-testid="stAlert"] {
        border-radius: 3px;
        border-width: 0 0 0 3px;
        box-shadow: none;
    }

    [data-testid="stExpander"] {
        background: transparent;
        border-color: var(--line);
        border-radius: 3px;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 3px;
        overflow: hidden;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 3px;
        box-shadow: none;
        font-weight: 600;
    }

    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input {
        border-radius: 3px !important;
    }

    [data-baseweb="tag"] {
        border-radius: 2px !important;
    }

    hr {
        border-color: var(--line) !important;
    }

    code, pre, kbd {
        font-family: var(--mono) !important;
    }

    @media (max-width: 760px) {
        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 1rem;
        }

        .hero {
            padding-top: 0.15rem;
        }

        .hero-title {
            font-size: 2rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            overflow-x: auto;
        }
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
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#17211f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#e1e5e2", tickprefix="$", zeroline=False),
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
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#17211f"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#e1e5e2", tickformat=".0%", zeroline=False),
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
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#17211f"),
        xaxis=dict(tickformat=".0%", gridcolor="#e1e5e2", zeroline=False, range=[0, max(0.4, alloc["weight"].max() * 1.18)]),
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
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#17211f"),
        xaxis=dict(gridcolor="#e1e5e2", zeroline=False),
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
    st.markdown("### ML sector study")
    st.caption("These controls apply only to the two ML study tabs.")
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

    signal_weights = manifest.get("signal_weights", {})
    st.markdown("#### Model settings")
    st.caption(
        f"Forecast {signal_weights.get('forecast', 0):.0%} · "
        f"momentum {signal_weights.get('momentum', 0):.0%} · "
        f"stability {signal_weights.get('stability', 0):.0%}"
    )
    st.caption(
        f"Max sector {manifest.get('max_weight', 0):.0%} · "
        f"cost {manifest.get('transaction_cost_bps', 0):.0f} bps per trade"
    )

scaled_equity = equity / float(manifest.get("initial_value", 10000)) * initial_capital
primary = metrics.loc["ML Signal Blend"]
primary_final = scaled_equity["ML Signal Blend"].iloc[-1]

st.markdown(
    f"""
    <div class="hero">
        <h1 class="hero-title">ETF Research Studio</h1>
        <div class="hero-copy">
            Compare allocation rules and forecasts, inspect current targets, and build a
            clear rebalance plan.
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


with tab_research:
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("## Model Drivers")
        plot_feature_importance(features)
    with right:
        st.markdown("## Research Design")
        st.markdown(
            """
            <div class="callout">
            <strong>Monthly timing:</strong> each rebalance uses models trained on prior months.<br><br>
            <strong>Trading costs:</strong> the strategy subtracts estimated costs from monthly results.<br><br>
            <strong>Reference portfolios:</strong> the study compares SPY, equal-weight sector exposure,
            and a simple six-month momentum baseline.<br><br>
            <strong>Weight detail:</strong> each current position shows its forecast, momentum, and
            stability inputs.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("## Method Guide")
        st.markdown(
            "[Read the features, model, score, and portfolio rules]"
            "(https://github.com/sabdulmajid/ml-etf-rebalancer/blob/master/"
            "docs/research/ml-sector-study.md)."
        )

    st.markdown("## ML Signal Explorer")
    st.caption(
        "Change the forecast, momentum, and stability mix to see how the current "
        "sector ranking changes. This view stays separate from the workbench target "
        "and Portfolio Lab ticket."
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
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#17211f"),
        xaxis=dict(tickformat=".0%", gridcolor="#e1e5e2", zeroline=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(remix_figure, width="stretch")
