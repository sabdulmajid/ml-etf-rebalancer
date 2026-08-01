"""Portfolio-weight helpers and an auditable rebalance-ticket contract.

The legacy ML allocator at the top of this module is intentionally retained.
The workbench helpers below it never normalize user-entered weights and always
keep analytical cash separate from tradeable ETF orders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.engine import CASH_ASSET, WEIGHT_TOLERANCE

def _apply_max_weight(weights, max_weight):
    """Normalize long-only weights while respecting a per-asset cap."""
    if max_weight is None:
        return weights

    weights = weights.astype(float).copy()
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    if max_weight * len(weights) < 1 - 1e-12:
        raise ValueError("max_weight is too low to create a fully invested portfolio")

    fixed = pd.Series(False, index=weights.index)

    for _ in range(len(weights) + 1):
        over_cap = (weights > max_weight + 1e-12) & ~fixed
        if not over_cap.any():
            break

        fixed.loc[over_cap] = True
        weights.loc[over_cap] = max_weight

        remaining = ~fixed
        remaining_total = 1.0 - weights.loc[fixed].sum()
        if remaining.sum() == 0:
            break

        if remaining_total <= 0:
            weights.loc[remaining] = 0.0
            break

        remaining_sum = weights.loc[remaining].sum()
        if remaining_sum <= 0:
            weights.loc[remaining] = remaining_total / remaining.sum()
        else:
            weights.loc[remaining] = weights.loc[remaining] / remaining_sum * remaining_total

    total = weights.sum()
    if total <= 0:
        return pd.Series(1.0 / len(weights), index=weights.index)

    return weights / total


def compute_weights(predicted_returns, method='simple', min_weight=0.0, max_weight=None, top_n=None):
    """Convert predicted returns into portfolio allocation weights"""
    predicted_returns = pd.Series(predicted_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if predicted_returns.empty:
        raise ValueError("predicted_returns must contain at least one numeric value")

    if method == 'simple':
        # Keep only positive expected returns
        weights = predicted_returns.clip(lower=min_weight)

        if top_n is not None and top_n < len(weights):
            keep = weights.nlargest(top_n).index
            weights.loc[~weights.index.isin(keep)] = 0.0
        
        # Normalize to sum to 1.0
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # Equal weight if no positive predictions
            weights = pd.Series(1.0 / len(weights), index=weights.index)
    
    elif method == 'rank':
        # Simple rank-based weighting
        ranks = predicted_returns.rank()
        weights = ranks / ranks.sum()
    
    elif method == 'sharpe':
        # This would require volatility estimates
        raise NotImplementedError("Sharpe ratio weighting not implemented")
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Apply maximum weight constraint if specified
    weights = _apply_max_weight(weights, max_weight)
    
    return weights

def generate_rebalance_orders(current_weights, target_weights, min_trade_size=0.01):
    """Generate trading orders based on current vs target weights"""
    current_weights = pd.Series(current_weights, dtype=float)
    target_weights = pd.Series(target_weights, dtype=float)
    all_tickers = current_weights.index.union(target_weights.index)
    current_weights = current_weights.reindex(all_tickers, fill_value=0.0)
    target_weights = target_weights.reindex(all_tickers, fill_value=0.0)

    # Calculate weight differences
    trades = target_weights - current_weights
    
    # Filter out small trades
    trades = trades[abs(trades) >= min_trade_size]
    
    # Format as buy/sell orders with amounts
    orders = []
    for ticker, weight_change in trades.items():
        direction = "BUY" if weight_change > 0 else "SELL"
        orders.append({
            'ticker': ticker,
            'direction': direction,
            'weight_change': abs(weight_change)
        })
    
    return pd.DataFrame(orders)


def reconcile_current_weights(previous_selection, selected_etfs, weights):
    """Reconcile editor state after an ETF-selection change without normalizing.

    Removed ETF weight is explicitly moved to analytical cash. Newly selected
    ETFs start at zero. Existing weights, including an invalid total, are left
    untouched so the UI can explain the validation error rather than silently
    changing the user's portfolio.
    """
    previous = tuple(previous_selection)
    selected = tuple(selected_etfs)
    if len(set(previous)) != len(previous) or len(set(selected)) != len(selected):
        raise ValueError("ETF selections must be unique")
    if CASH_ASSET in previous or CASH_ASSET in selected:
        raise ValueError(f"{CASH_ASSET} is cash, not a selectable ETF")

    values = pd.Series(weights, dtype=float).copy()
    cash = float(values.get(CASH_ASSET, 0.0))
    for ticker in previous:
        if ticker not in selected:
            cash += float(values.get(ticker, 0.0))
    reconciled = pd.Series(
        {ticker: float(values.get(ticker, 0.0)) for ticker in selected},
        dtype=float,
    )
    reconciled.loc[CASH_ASSET] = cash
    return reconciled


def validate_portfolio_weights(weights, selected_etfs, *, name="current weights"):
    """Validate explicit, long-only ETF and cash weights; never normalize them."""
    selected = tuple(selected_etfs)
    expected = [*selected, CASH_ASSET]
    if not selected:
        raise ValueError("select at least one ETF")
    if len(set(selected)) != len(selected):
        raise ValueError("selected ETFs must be unique")
    if CASH_ASSET in selected:
        raise ValueError(f"{CASH_ASSET} is cash, not a selectable ETF")

    series = pd.Series(weights, dtype=float)
    if not series.index.is_unique:
        raise ValueError(f"{name} must have unique assets")
    missing = [asset for asset in expected if asset not in series.index]
    extra = [asset for asset in series.index if asset not in expected]
    if missing or extra:
        raise ValueError(
            f"{name} must contain exactly selected ETFs plus explicit cash; "
            f"missing={missing}, extra={extra}"
        )
    series = series.reindex(expected)
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values")
    if (series < 0.0).any() or (series > 1.0 + WEIGHT_TOLERANCE).any():
        raise ValueError(f"{name} must be between 0% and 100%")
    total = float(series.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=WEIGHT_TOLERANCE):
        raise ValueError(f"{name} must sum to 100%; entered total is {total:.2%}")
    return series


@dataclass(frozen=True)
class RebalanceTicket:
    """ETF security instructions plus a separate analytical cash balance row."""

    security_orders: pd.DataFrame
    cash_balance: pd.DataFrame
    turnover: float
    estimated_cost_rate: float
    estimated_cost_amount: float | None

    def download_frame(self):
        """Return a reconciled CSV-ready frame without making cash a security order."""
        securities = self.security_orders.copy()
        securities.insert(0, "row_type", "security_order")
        cash = self.cash_balance.copy()
        cash.insert(0, "row_type", "cash_balance")
        return pd.concat([securities, cash], ignore_index=True, sort=False)


def build_rebalance_ticket(
    current_weights,
    target_weights,
    selected_etfs,
    *,
    transaction_cost_bps=0.0,
    portfolio_value=None,
):
    """Build reconciled percentage-point and optional dollar rebalance tickets.

    Turnover follows the common engine convention: half the absolute change in
    all asset weights, including cash. ``CASH_ASSET`` is emitted only as a cash
    balance and can never become a BUY/SELL security order.
    """
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be finite and nonnegative")
    if transaction_cost_bps >= 10000.0:
        raise ValueError("transaction_cost_bps must be less than 10000")
    if portfolio_value is not None and (
        not np.isfinite(portfolio_value) or portfolio_value <= 0.0
    ):
        raise ValueError("portfolio_value must be finite and positive")

    selected = tuple(selected_etfs)
    current = validate_portfolio_weights(
        current_weights, selected, name="current weights"
    )
    target = validate_portfolio_weights(
        target_weights, selected, name="target weights"
    )
    changes = target - current
    turnover = 0.5 * float(changes.abs().sum())
    cost_rate = turnover * float(transaction_cost_bps) / 10000.0
    cost_amount = None if portfolio_value is None else cost_rate * float(portfolio_value)

    orders = pd.DataFrame(
        {
            "asset": selected,
            "current_weight": current.loc[list(selected)].to_numpy(),
            "target_weight": target.loc[list(selected)].to_numpy(),
            "percentage_point_change": (
                100.0 * changes.loc[list(selected)].to_numpy()
            ),
        }
    )
    orders["action"] = np.select(
        [
            orders["percentage_point_change"] > WEIGHT_TOLERANCE * 100.0,
            orders["percentage_point_change"] < -WEIGHT_TOLERANCE * 100.0,
        ],
        ["BUY", "SELL"],
        default="HOLD",
    )
    if portfolio_value is not None:
        orders["trade_amount"] = changes.loc[list(selected)].to_numpy() * float(
            portfolio_value
        )

    cash = pd.DataFrame(
        {
            "asset": [CASH_ASSET],
            "label": ["Cash — U.S. overnight-rate proxy"],
            "current_weight": [current.loc[CASH_ASSET]],
            "target_weight": [target.loc[CASH_ASSET]],
            "percentage_point_change": [100.0 * changes.loc[CASH_ASSET]],
        }
    )
    if portfolio_value is not None:
        cash["trade_amount"] = [changes.loc[CASH_ASSET] * float(portfolio_value)]

    if not np.isclose(
        float(changes.sum()), 0.0, rtol=0.0, atol=WEIGHT_TOLERANCE
    ):
        raise ArithmeticError("rebalance changes do not reconcile to zero")
    return RebalanceTicket(
        security_orders=orders,
        cash_balance=cash,
        turnover=turnover,
        estimated_cost_rate=cost_rate,
        estimated_cost_amount=cost_amount,
    )
