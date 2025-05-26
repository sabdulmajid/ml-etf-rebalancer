import os
import pandas as pd
from datetime import datetime

def setup_directories():
    """Create project directories if they don't exist"""
    dirs = ["data/raw", "data/processed", "model/trained", "model/predictions", 
            "model/evaluation", "logs", "dashboard"]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def run_pipeline(backtest=True, save_artifacts=True):
    """Execute the full ETF rebalancing pipeline"""
    print("----- Starting ETF rebalancer pipeline -----")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Import functions at runtime to ensure we use the latest versions
    from data.fetch_data import load_or_fetch_data
    from data.features import get_features, prepare_features_for_training
    from model.train_model import train_all_models, predict_returns, evaluate_models
    from portfolio.rebalance import compute_weights, save_allocation
    
    # Step 1: Load or fetch the latest data
    print("\nStep 1: Loading market data...")
    prices, returns = load_or_fetch_data(force_refresh=True)
    print(f"Data loaded: {len(prices)} days of price data for {len(prices.columns)} ETFs")
    
    # Step 2: Generate features
    print("\nStep 2: Generating features...")
    features = get_features(prices, returns)
    X, y = prepare_features_for_training(features, returns)
    print(f"Features generated: {X.shape[1]} features for {X.shape[0]} time periods")
    
    # Step 3: Train models
    print("\nStep 3: Training ML models...")
    models = train_all_models(X, y)
    
    # Step 4: Evaluate models
    print("\nStep 4: Evaluating models...")
    metrics = evaluate_models(models, X, y)
    print("\nModel performance metrics:")
    print(metrics)
    
    # Step 5: Generate predictions for the most recent period
    print("\nStep 5: Generating predictions for current allocation...")
    latest_features = features.iloc[[-1]]
    latest_preds = {}
    
    for ticker, model in models.items():
        latest_preds[ticker] = model.predict(latest_features)[0]
    
    latest_predictions = pd.Series(latest_preds)
    
    # Step 6: Compute portfolio weights
    print("\nStep 6: Computing portfolio allocation...")
    weights = compute_weights(latest_predictions, method='simple', min_weight=0, max_weight=0.3)
    print("Current allocation:")
    for ticker, weight in weights.items():
        print(f"  {ticker}: {weight:.2%}")
    
    # Save allocation to disk
    if save_artifacts:
        save_allocation(weights)
    
    # Step 7: Backtest (optional)
    if backtest:
        print("\nStep 7: Running backtest simulation...")
        from portfolio.backtest import backtest, calculate_performance_metrics, save_backtest_results
        
        # Generate predictions for historical periods
        all_predictions = predict_returns(models, X)
        
        # Compute weights for each period
        from portfolio.rebalance import compute_weights
        weights_df = pd.DataFrame(index=all_predictions.index, columns=all_predictions.columns)
        
        for date in all_predictions.index:
            date_preds = all_predictions.loc[date]
            weights = compute_weights(date_preds, method='simple', min_weight=0)
            for ticker in weights.index:
                weights_df.loc[date, ticker] = weights[ticker]
        
        # Run backtest
        returns_series, portfolio_value = backtest(weights_df, y)
        
        # Calculate and save performance metrics
        metrics = calculate_performance_metrics(returns_series)
        print("\nBacktest performance metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save backtest results
        if save_artifacts:
            save_backtest_results(returns_series, portfolio_value)
    
    print("\n----- Pipeline completed successfully -----")
    return weights

if __name__ == "__main__":
    setup_directories()
    weights = run_pipeline()
    print("\nReady for deployment to dashboard")
