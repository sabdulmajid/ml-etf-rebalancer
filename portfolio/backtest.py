import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def backtest(weights_df, returns_df):
    """Simulate portfolio performance using historical weights and returns"""
    # Ensure the dataframes have compatible indices
    common_idx = weights_df.index.intersection(returns_df.index)
    weights = weights_df.loc[common_idx]
    returns = returns_df.loc[common_idx]
    
    # Calculate period-by-period portfolio returns
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    
    for date in returns.index:
        if date in weights.index:
            # Use the weights from this date
            curr_weights = weights.loc[date]
            curr_returns = returns.loc[date]
            
            # Calculate weighted return
            portfolio_return = (curr_weights * curr_returns).sum()
            portfolio_returns[date] = portfolio_return
    
    # Calculate cumulative performance
    portfolio_value = (1 + portfolio_returns).cumprod()
    
    return portfolio_returns, portfolio_value

def compare_strategies(strategies, returns_df):
    """Compare multiple weighting strategies"""
    results = {}
    
    for name, weights_df in strategies.items():
        period_returns, cumulative_returns = backtest(weights_df, returns_df)
        results[name] = cumulative_returns
    
    # Convert to DataFrame for comparison
    results_df = pd.DataFrame(results)
    
    return results_df

def calculate_performance_metrics(returns_series):
    """Calculate key portfolio performance indicators"""
    # Convert to floating point to avoid precision issues
    returns = returns_series.astype(float)
    
    # Annualization factor (assuming monthly returns)
    ann_factor = 12
    
    # Total return
    total_return = (returns + 1).prod() - 1
    
    # Annualized return
    n_periods = len(returns)
    ann_return = (1 + total_return) ** (ann_factor / n_periods) - 1
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(ann_factor)
    
    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    sharpe = ann_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding().max()
    drawdown = (cum_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    metrics = {
        'Total Return': total_return,
        'Annualized Return': ann_return,
        'Annualized Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate
    }
    
    return pd.Series(metrics)

def save_backtest_results(returns, portfolio_value, strategy_name='ml_strategy'):
    """Save backtest results to disk"""
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    portfolio_value.to_csv(f"logs/portfolio_value_{strategy_name}_{timestamp}.csv")
    returns.to_csv(f"logs/portfolio_returns_{strategy_name}_{timestamp}.csv")
    
    # Update latest files
    portfolio_value.to_csv(f"logs/portfolio_value_{strategy_name}_latest.csv")
    returns.to_csv(f"logs/portfolio_returns_{strategy_name}_latest.csv")
    
    # Calculate and save metrics
    metrics = calculate_performance_metrics(returns)
    metrics.to_csv(f"logs/performance_metrics_{strategy_name}_latest.csv")
    
    return metrics

if __name__ == "__main__":
    from data.fetch_data import load_or_fetch_data
    from model.train_model import load_models, predict_returns
    from data.features import get_features, prepare_features_for_training
    
    # Load data
    prices, returns = load_or_fetch_data()
    features = get_features(prices, returns)
    X, y = prepare_features_for_training(features, returns)
    
    # Load models
    try:
        models = load_models()
    except FileNotFoundError:
        from model.train_model import train_all_models
        models = train_all_models(X, y)
    
    # Generate predictions for each period
    predictions = predict_returns(models, X)
    
    # Compute portfolio weights
    from portfolio.rebalance import compute_weights
    
    # Simple strategy: invest in ETFs with positive predicted returns
    weights_simple = predictions.apply(
        lambda x: compute_weights(x, method='simple', min_weight=0),
        axis=1
    )
    
    # Equal weight benchmark
    n_assets = len(returns.columns)
    equal_weight = pd.DataFrame(
        1.0 / n_assets,
        index=returns.index,
        columns=returns.columns
    )
    
    # Run backtest
    strategies = {
        'ML Strategy': weights_simple,
        'Equal Weight': equal_weight
    }
    
    results = compare_strategies(strategies, returns)
    
    # Print performance summary
    for name in strategies:
        strategy_returns = results[name].pct_change().dropna()
        metrics = calculate_performance_metrics(strategy_returns)
        print(f"\n{name} Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Save ML strategy results
    ml_returns = results['ML Strategy'].pct_change().dropna()
    save_backtest_results(ml_returns, results['ML Strategy'], 'ml_strategy')
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    results.plot()
    plt.title('Strategy Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.savefig('logs/strategy_comparison.png')
    plt.close()
