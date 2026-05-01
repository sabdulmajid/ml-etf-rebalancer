import pandas as pd
import numpy as np

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
