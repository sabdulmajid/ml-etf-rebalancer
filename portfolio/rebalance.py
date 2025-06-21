import pandas as pd
import numpy as np
import os
from datetime import datetime

def compute_weights(predicted_returns, method='simple', min_weight=0.0, max_weight=None):
    """Convert predicted returns into portfolio allocation weights"""
    if method == 'simple':
        # Keep only positive expected returns
        weights = predicted_returns.clip(lower=min_weight)
        
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
    if max_weight is not None:
        excess = weights[weights > max_weight].sum() - max_weight * (weights > max_weight).sum()
        if excess > 0:
            weights[weights > max_weight] = max_weight
            weights[weights <= max_weight] *= (1 + excess / weights[weights <= max_weight].sum())
    
    return weights

def generate_rebalance_orders(current_weights, target_weights, min_trade_size=0.01):
    """Generate trading orders based on current vs target weights"""
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

def save_allocation(weights, timestamp=None):
    """Save portfolio allocation weights to file"""
    os.makedirs("logs", exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    weights_df = pd.DataFrame(weights).T
    weights_df.columns = ['weight']
    weights_df.to_csv(f"logs/allocation_weights_{timestamp}.csv")
    
    # Also update the 'latest' allocation
    weights_df.to_csv("logs/allocation_weights_latest.csv")
    
    return weights_df

if __name__ == "__main__":
    from model.train_model import load_models, predict_returns
    from data.fetch_data import load_or_fetch_data
    from data.features import get_features, prepare_features_for_training
    
    # Load latest data
    prices, returns = load_or_fetch_data()
    features = get_features(prices)
    
    # Get the latest feature data point
    latest_features = features.iloc[[-1]]
    
    # Load trained models
    models = load_models()
    
    # Generate predictions
    preds = {}
    for ticker, model in models.items():
        preds[ticker] = model.predict(latest_features)[0]
    
    pred_series = pd.Series(preds)
    print("Predicted returns:")
    print(pred_series)
    
    # Compute allocation weights
    weights = compute_weights(pred_series, method='simple', min_weight=0, max_weight=0.3)
    print("\nPortfolio allocation:")
    print(weights)
    
    # Save allocation
    save_allocation(weights)
