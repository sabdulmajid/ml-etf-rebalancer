import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.features import get_features, prepare_features_for_training
from portfolio.rebalance import compute_weights


SECTOR_ETFS = {
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}
BENCHMARK_TICKER = "SPY"
DEFAULT_START = "2010-01-01"


@dataclass(frozen=True)
class ResearchConfig:
    start: str = DEFAULT_START
    min_train_months: int = 48
    max_weight: float = 0.35
    top_n: int = 4
    transaction_cost_bps: float = 8.0
    initial_value: float = 10000.0
    artifact_dir: str = "artifacts/latest"
    forecast_weight: float = 0.40
    momentum_weight: float = 0.45
    stability_weight: float = 0.15


def fetch_adjusted_prices(start=DEFAULT_START, end=None):
    """Fetch adjusted ETF prices for the sector universe and SPY benchmark."""
    tickers = list(SECTOR_ETFS) + [BENCHMARK_TICKER]
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw.empty:
        raise RuntimeError("No price data returned from yfinance")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()

    prices = prices.reindex(columns=tickers).ffill().dropna(how="all")

    missing = [ticker for ticker in tickers if ticker not in prices or prices[ticker].dropna().empty]
    for ticker in missing:
        retry = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if retry.empty:
            continue
        if isinstance(retry.columns, pd.MultiIndex):
            retry_close = retry["Close"].iloc[:, 0]
        else:
            retry_close = retry["Close"] if "Close" in retry else retry.iloc[:, 0]
        prices[ticker] = retry_close

    prices = prices.reindex(columns=tickers).ffill().dropna(how="all")
    still_missing = [ticker for ticker in tickers if prices[ticker].dropna().empty]
    if still_missing:
        raise RuntimeError(f"Missing price history for: {', '.join(still_missing)}")

    prices.index = pd.to_datetime(prices.index)
    return prices


def monthly_returns(prices):
    month_end_prices = prices.resample("ME").last()
    return month_end_prices.pct_change().dropna(how="all")


def _new_model(seed=42):
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def _fit_sector_models(X_train, y_train):
    models = {}
    for i, ticker in enumerate(y_train.columns):
        model = _new_model(seed=42 + i)
        model.fit(X_train, y_train[ticker])
        models[ticker] = model
    return models


def _predict_one(models, X_row):
    return pd.Series(
        {ticker: model.predict(X_row)[0] for ticker, model in models.items()},
        dtype=float,
    )


def _model_importance(model, columns):
    ridge = model.named_steps["ridge"]
    return pd.Series(np.abs(ridge.coef_), index=columns)


def _momentum_weights(sector_returns, date, lookback_months=6, top_n=3):
    history = sector_returns.loc[:date].iloc[:-1]
    if len(history) < lookback_months:
        return pd.Series(1.0 / len(sector_returns.columns), index=sector_returns.columns)

    trailing = (1 + history.tail(lookback_months)).prod() - 1
    selected = trailing.nlargest(top_n).index
    weights = pd.Series(0.0, index=sector_returns.columns)
    weights.loc[selected] = 1.0 / len(selected)
    return weights


def _zscore(series):
    series = pd.Series(series, dtype=float)
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - series.mean()) / std


def _signal_scores(prediction, history, config):
    if len(history) >= 6:
        momentum = (1 + history.tail(6)).prod() - 1
    else:
        momentum = pd.Series(0.0, index=prediction.index)

    if len(history) >= 12:
        volatility = history.tail(12).std()
    else:
        volatility = history.std()

    stability = 1 / volatility.replace(0, np.nan)
    scores = pd.DataFrame(
        {
            "forecast_z": _zscore(prediction),
            "momentum_z": _zscore(momentum.reindex(prediction.index)),
            "stability_z": _zscore(stability.reindex(prediction.index)),
        }
    ).fillna(0.0)
    scores["composite_score"] = (
        config.forecast_weight * scores["forecast_z"]
        + config.momentum_weight * scores["momentum_z"]
        + config.stability_weight * scores["stability_z"]
    )
    return scores


def _portfolio_returns(weights, returns, transaction_cost_bps=0.0):
    rows = []
    prev_weights = pd.Series(0.0, index=weights.columns)

    for date, weight_row in weights.iterrows():
        weight_row = weight_row.astype(float).reindex(returns.columns, fill_value=0.0)
        period_returns = returns.loc[date].astype(float)
        gross_return = float((weight_row * period_returns).sum())
        turnover = float((weight_row - prev_weights).abs().sum())
        cost = turnover * transaction_cost_bps / 10000.0
        net_return = gross_return - cost
        rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "transaction_cost": cost,
                "turnover": turnover,
                "return": net_return,
            }
        )
        prev_weights = weight_row

    return pd.DataFrame(rows).set_index("date")


def _drawdown(return_series):
    curve = (1 + return_series).cumprod()
    peak = curve.cummax()
    return curve / peak - 1


def _metrics(return_series, turnover=None):
    returns = return_series.dropna().astype(float)
    periods = len(returns)
    if periods == 0:
        raise ValueError("Cannot calculate metrics for an empty return series")

    curve = (1 + returns).cumprod()
    total_return = curve.iloc[-1] - 1
    cagr = curve.iloc[-1] ** (12 / periods) - 1
    volatility = returns.std(ddof=0) * np.sqrt(12)
    sharpe = cagr / volatility if volatility > 0 else np.nan
    drawdown = _drawdown(returns)

    return {
        "Start": returns.index.min().strftime("%Y-%m-%d"),
        "End": returns.index.max().strftime("%Y-%m-%d"),
        "Months": periods,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": drawdown.min(),
        "Win Rate": float((returns > 0).mean()),
        "Best Month": returns.max(),
        "Worst Month": returns.min(),
        "Average Monthly Turnover": float(turnover.mean()) if turnover is not None else np.nan,
    }


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_research_artifacts(config=ResearchConfig()):
    """Run a walk-forward sector rotation study and write stable dashboard artifacts."""
    prices = fetch_adjusted_prices(start=config.start)
    sector_prices = prices[list(SECTOR_ETFS)]
    returns = monthly_returns(prices)
    sector_returns = returns[list(SECTOR_ETFS)].dropna()
    benchmark_returns = returns[BENCHMARK_TICKER].dropna()

    features = get_features(sector_prices, sector_returns)
    X, y = prepare_features_for_training(features, sector_returns)
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    predictions = []
    signal_scores = []
    weights = []

    for date in X.index[config.min_train_months:]:
        X_train = X.loc[X.index < date]
        y_train = y.loc[y.index < date]
        if len(X_train) < config.min_train_months:
            continue

        models = _fit_sector_models(X_train, y_train)
        pred = _predict_one(models, X.loc[[date]])
        scores = _signal_scores(pred, sector_returns.loc[:date].iloc[:-1], config)
        allocation_scores = scores["composite_score"] - scores["composite_score"].min()
        weight = compute_weights(
            allocation_scores,
            method="simple",
            min_weight=0.0,
            max_weight=config.max_weight,
            top_n=config.top_n,
        )
        predictions.append(pred.rename(date))
        signal_scores.append(scores["composite_score"].rename(date))
        weights.append(weight.rename(date))

    prediction_df = pd.DataFrame(predictions).sort_index()
    signal_score_df = pd.DataFrame(signal_scores).sort_index()
    weights_df = pd.DataFrame(weights).sort_index().reindex(columns=list(SECTOR_ETFS), fill_value=0.0)
    backtest_returns = y.loc[weights_df.index]

    strategy = _portfolio_returns(weights_df, backtest_returns, config.transaction_cost_bps)

    equal_weight = pd.DataFrame(
        1.0 / len(SECTOR_ETFS),
        index=weights_df.index,
        columns=list(SECTOR_ETFS),
    )
    equal_returns = _portfolio_returns(equal_weight, backtest_returns, 0.0)["return"]

    momentum_weight_rows = [
        _momentum_weights(sector_returns, date, top_n=3).rename(date)
        for date in weights_df.index
    ]
    momentum_weights = pd.DataFrame(momentum_weight_rows).reindex(columns=list(SECTOR_ETFS), fill_value=0.0)
    momentum_returns = _portfolio_returns(momentum_weights, backtest_returns, config.transaction_cost_bps)["return"]

    aligned_spy = benchmark_returns.reindex(weights_df.index).dropna()
    common_strategy_idx = weights_df.index.intersection(aligned_spy.index)

    returns_by_strategy = pd.DataFrame(
        {
            "ML Signal Blend": strategy.loc[common_strategy_idx, "return"],
            "Equal-Weight Sectors": equal_returns.loc[common_strategy_idx],
            "6M Momentum Top 3": momentum_returns.loc[common_strategy_idx],
            "SPY Buy & Hold": aligned_spy.loc[common_strategy_idx],
        }
    )

    equity_curves = (1 + returns_by_strategy).cumprod() * config.initial_value
    drawdowns = returns_by_strategy.apply(_drawdown)
    annual_returns = returns_by_strategy.groupby(returns_by_strategy.index.year).apply(
        lambda frame: (1 + frame).prod() - 1
    )

    metrics = pd.DataFrame(
        {
            "ML Signal Blend": _metrics(
                returns_by_strategy["ML Signal Blend"],
                turnover=strategy.loc[common_strategy_idx, "turnover"],
            ),
            "Equal-Weight Sectors": _metrics(returns_by_strategy["Equal-Weight Sectors"]),
            "6M Momentum Top 3": _metrics(
                returns_by_strategy["6M Momentum Top 3"],
                turnover=momentum_weights.diff().abs().sum(axis=1).loc[common_strategy_idx],
            ),
            "SPY Buy & Hold": _metrics(returns_by_strategy["SPY Buy & Hold"]),
        }
    ).T

    final_models = _fit_sector_models(X, y)
    latest_features = features.resample("ME").last().iloc[[-1]]
    latest_predictions = _predict_one(final_models, latest_features)
    latest_scores = _signal_scores(latest_predictions, sector_returns, config)
    latest_allocation_scores = latest_scores["composite_score"] - latest_scores["composite_score"].min()
    current_weights = compute_weights(
        latest_allocation_scores,
        method="simple",
        min_weight=0.0,
        max_weight=config.max_weight,
        top_n=config.top_n,
    )

    current_allocation = pd.DataFrame(
        {
            "ticker": current_weights.index,
            "sector": [SECTOR_ETFS[ticker] for ticker in current_weights.index],
            "weight": current_weights.values,
            "predicted_return": latest_predictions.reindex(current_weights.index).values,
            "forecast_score": latest_scores["forecast_z"].reindex(current_weights.index).values,
            "composite_score": latest_scores["composite_score"].reindex(current_weights.index).values,
            "momentum_score": latest_scores["momentum_z"].reindex(current_weights.index).values,
            "stability_score": latest_scores["stability_z"].reindex(current_weights.index).values,
        }
    ).sort_values("weight", ascending=False)

    importances = pd.DataFrame(
        {
            ticker: _model_importance(model, X.columns)
            for ticker, model in final_models.items()
        }
    )
    feature_importance = pd.DataFrame(
        {
            "feature": importances.mean(axis=1).sort_values(ascending=False).head(20).index,
            "importance": importances.mean(axis=1).sort_values(ascending=False).head(20).values,
        }
    )

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    prices.to_csv(artifact_dir / "prices.csv")
    sector_returns.to_csv(artifact_dir / "monthly_returns.csv")
    prediction_df.to_csv(artifact_dir / "predictions.csv")
    signal_score_df.to_csv(artifact_dir / "signal_scores.csv")
    weights_df.to_csv(artifact_dir / "weights.csv")
    strategy.to_csv(artifact_dir / "strategy_detail.csv")
    returns_by_strategy.to_csv(artifact_dir / "strategy_returns.csv")
    equity_curves.to_csv(artifact_dir / "equity_curves.csv")
    drawdowns.to_csv(artifact_dir / "drawdowns.csv")
    annual_returns.to_csv(artifact_dir / "annual_returns.csv")
    metrics.to_csv(artifact_dir / "metrics.csv")
    current_allocation.to_csv(artifact_dir / "current_allocation.csv", index=False)
    feature_importance.to_csv(artifact_dir / "feature_importance.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_start": prices.index.min().strftime("%Y-%m-%d"),
        "data_end": prices.index.max().strftime("%Y-%m-%d"),
        "backtest_start": returns_by_strategy.index.min().strftime("%Y-%m-%d"),
        "backtest_end": returns_by_strategy.index.max().strftime("%Y-%m-%d"),
        "latest_signal_date": latest_features.index[-1].strftime("%Y-%m-%d"),
        "universe": SECTOR_ETFS,
        "benchmark": BENCHMARK_TICKER,
        "methodology": "Walk-forward monthly retraining. Each rebalance uses only data available before the predicted month. Allocation blends model forecasts, trailing momentum, and volatility discipline.",
        "transaction_cost_bps": config.transaction_cost_bps,
        "max_weight": config.max_weight,
        "top_n": config.top_n,
        "signal_weights": {
            "forecast": config.forecast_weight,
            "momentum": config.momentum_weight,
            "stability": config.stability_weight,
        },
        "initial_value": config.initial_value,
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }

    with open(artifact_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return {
        "manifest": manifest,
        "metrics": metrics,
        "current_allocation": current_allocation,
        "artifact_dir": str(artifact_dir),
    }
