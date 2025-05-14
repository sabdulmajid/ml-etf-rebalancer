import yfinance as yf
import pandas as pd
import os
from datetime import datetime

SECTORS = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLI', 'XLU', 'XLB', 'XLP']

def get_price_data(start="2015-01-01", end=None):
    """Fetch historical price data for sector ETFs"""
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
        
    print(f"Fetching price data from {start} to {end}")
    data = yf.download(SECTORS, start=start, end=end)
    
    # Handle different column structures that yfinance might return
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-level columns, try to get Adj Close or Close
        if 'Adj Close' in data.columns.levels[0]:
            prices = data['Adj Close']
        else:
            print("'Adj Close' not available, using 'Close' instead")
            prices = data['Close']
    else:
        # Single-level columns (for single ticker)
        prices = data
        
    prices = prices.ffill()
    
    os.makedirs("data/raw", exist_ok=True)
    prices.to_csv("data/raw/sector_etf_prices.csv")
    
    return prices

def compute_monthly_returns(prices):
    """Transform daily prices into monthly returns"""
    monthly = prices.resample("ME").last()  # End of month frequency
    monthly_returns = monthly.pct_change().dropna()
    
    os.makedirs("data/processed", exist_ok=True)
    monthly_returns.to_csv("data/processed/monthly_returns.csv")
    
    return monthly_returns

def compute_weekly_returns(prices):
    """Transform daily prices into weekly returns (Friday-to-Friday)"""
    weekly = prices.resample("W-FRI").last()
    weekly_returns = weekly.pct_change().dropna()
    
    weekly_returns.to_csv("data/processed/weekly_returns.csv")
    
    return weekly_returns

def load_or_fetch_data(start="2015-01-01", end=None, force_refresh=False):
    """Smart data loader with incremental updates"""
    prices_path = "data/raw/sector_etf_prices.csv"
    
    if not force_refresh and os.path.exists(prices_path):
        print("Loading data from disk...")
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        
        # Update dataset with latest prices if needed
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')
            
        last_date = prices.index[-1].strftime('%Y-%m-%d')
        
        if last_date < end:
            print(f"Updating data from {last_date} to {end}")
            new_prices = get_price_data(start=last_date, end=end)
            prices = pd.concat([prices, new_prices]).drop_duplicates()
            prices.to_csv(prices_path)
    else:
        prices = get_price_data(start=start, end=end)
    
    monthly_returns = compute_monthly_returns(prices)
    
    return prices, monthly_returns

if __name__ == "__main__":
    prices, returns = load_or_fetch_data()
    print(f"Fetched price data shape: {prices.shape}")
    print(f"Monthly returns shape: {returns.shape}")
    print(f"Sample monthly returns:\n{returns.tail(3)}")
