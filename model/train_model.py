import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
import datetime

# NOTE: We're using sklearn estimators instead of XGBoost for compatibility
# In a production environment, you would use XGBoost as it generally performs better

def train_model(X_train, y_train, ticker, model_type='gbm'):
    """Train a machine learning model for ETF return prediction"""
    print(f"Training model for {ticker}...")
    
    # Choose model type (using sklearn models for compatibility)
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=42
        )
    else:  # default to GBM
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        model, 
        X_train, 
        y_train, 
        cv=tscv,
        scoring='neg_mean_squared_error'
    )
    
    print(f"CV MSE scores for {ticker}: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    model.fit(X_train, y_train)
    
    return model

def train_all_models(X, y, model_type='gbm'):
    """Train separate ML models for each ETF sector"""
    models = {}
    feature_importances = {}
    
    for ticker in y.columns:
        model = train_model(X, y[ticker], ticker, model_type)
        models[ticker] = model
        
        importances = pd.Series(model.feature_importances_, index=X.columns)
        feature_importances[ticker] = importances.sort_values(ascending=False)
    
    os.makedirs("model/trained", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_filepath = f"model/trained/sector_models_{timestamp}.joblib"
    joblib.dump(models, model_filepath)
    
    importances_df = pd.DataFrame(feature_importances)
    importances_df.to_csv(f"model/trained/feature_importances_{timestamp}.csv")
    
    latest_symlink = "model/trained/sector_models_latest.joblib"
    if os.path.exists(latest_symlink) or os.path.islink(latest_symlink):
        if os.path.islink(latest_symlink):
            os.unlink(latest_symlink)
        else:
            os.remove(latest_symlink)
    os.symlink(os.path.abspath(model_filepath), latest_symlink)
    
    return models

def predict_returns(models, X):
    """Generate future ETF return predictions from trained models"""
    predictions = {}
    
    for ticker, model in models.items():
        predictions[ticker] = model.predict(X)
    
    pred_df = pd.DataFrame(predictions, index=X.index)
    
    os.makedirs("model/predictions", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_df.to_csv(f"model/predictions/predicted_returns_{timestamp}.csv")
    
    return pred_df

def evaluate_models(models, X, y):
    """Calculate performance metrics for each sector model"""
    metrics = []
    
    for ticker, model in models.items():
        preds = model.predict(X)
        mse = mean_squared_error(y[ticker], preds)
        r2 = r2_score(y[ticker], preds)
        
        metrics.append({
            'ticker': ticker,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        })
    
    metrics_df = pd.DataFrame(metrics).set_index('ticker')
    
    os.makedirs("model/evaluation", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_df.to_csv(f"model/evaluation/model_metrics_{timestamp}.csv")
    
    return metrics_df

def load_models(model_path=None):
    """Load previously trained sector models"""
    if model_path is None:
        model_path = "model/trained/sector_models_latest.joblib"
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)

if __name__ == "__main__":
    from data.fetch_data import load_or_fetch_data
    from data.features import get_features, prepare_features_for_training
    
    prices, returns = load_or_fetch_data()
    features = get_features(prices, returns)
    
    X, y = prepare_features_for_training(features, returns)
    
    models = train_all_models(X, y)
    
    metrics = evaluate_models(models, X, y)
    print("Model performance metrics:")
    print(metrics)
    
    predictions = predict_returns(models, X)
    print("Sample predictions:")
    print(predictions.tail(3))
