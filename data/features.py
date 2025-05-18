import pandas as pd
import numpy as np
import os

def get_features(prices, returns=None):
    """Generate technical indicators for ETF trading strategy"""
    # Momentum indicators across multiple timeframes (trading days)
    mom1 = prices.pct_change(20).shift(1)
    mom3 = prices.pct_change(60).shift(1)
    mom6 = prices.pct_change(125).shift(1)
    mom12 = prices.pct_change(250).shift(1)
    
    # Volatility metrics
    vol1m = prices.pct_change().rolling(20).std().shift(1)
    vol3m = prices.pct_change().rolling(60).std().shift(1)
    
    # Moving average crossover signals
    ma50 = prices.rolling(50).mean().shift(1)
    ma200 = prices.rolling(200).mean().shift(1)
    ma_ratio = ma50 / ma200
    
    # Relative strength calculation (sector vs market)
    if returns is not None:
        prices_monthly = prices.resample('ME').last()  # End of month frequency
        rs6m = pd.DataFrame(index=prices_monthly.index, columns=prices_monthly.columns)
        
        for col in prices_monthly.columns:
            ret6m = prices_monthly[col].pct_change(6)
            market6m = prices_monthly.mean(axis=1).pct_change(6)
            rs6m[col] = ret6m / market6m
            
        rs6m = rs6m.reindex(prices.index, method='ffill').shift(1)
    else:
        rs6m = pd.DataFrame(index=prices.index, columns=prices.columns)
    
    # Collect all feature sets with descriptive prefixes
    feature_dfs = {
        'mom1m': mom1,
        'mom3m': mom3,
        'mom6m': mom6,
        'mom12m': mom12,
        'vol1m': vol1m,
        'vol3m': vol3m,
        'ma_ratio': ma_ratio,
        'rs6m': rs6m
    }
    
    all_features = pd.DataFrame(index=prices.index)
    
    for prefix, df in feature_dfs.items():
        if df.empty:
            continue
            
        df = df.copy()
        df.columns = [f"{prefix}_{col}" for col in df.columns]
        all_features = all_features.join(df)
    
    all_features = all_features.dropna()
    
    os.makedirs("data/processed", exist_ok=True)
    all_features.to_csv("data/processed/features.csv")
    
    return all_features

def add_macro_features(features, start_date=None, end_date=None):
    """Future extension point for macroeconomic indicators like rates, inflation, etc."""
    # Placeholder for future implementation
    return features

def prepare_features_for_training(features, returns, freq='M'):
    """Align features and returns for predictive modeling"""
    # Resample to target frequency
    if freq == 'M':
        features_resampled = features.resample('ME').last()  # End of month frequency
    elif freq == 'W':
        features_resampled = features.resample('W-FRI').last()
    else:
        raise ValueError(f"Unsupported frequency: {freq}")
    
    # Time-shifted alignment for prediction (use previous period features to predict current returns)
    X = features_resampled.shift(1)
    y = returns
    
    # Ensure X and y have matching timestamps
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx].dropna()
    y = y.loc[X.index]
    
    return X, y

if __name__ == "__main__":
    from fetch_data import load_or_fetch_data
    
    prices, returns = load_or_fetch_data()
    features = get_features(prices, returns)
    
    print(f"Generated features shape: {features.shape}")
    print(f"Sample features:\n{features.tail(3)}")
    
    X, y = prepare_features_for_training(features, returns)
    print(f"Training data shapes: X = {X.shape}, y = {y.shape}")
