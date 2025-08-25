# ML-Powered ETF Rebalancer

A machine learning system for sector ETF allocation and automated portfolio rebalancing. You can access the live website [here](https://etf-rebalancer.streamlit.app/)!

## Overview

This project uses machine learning to predict sector ETF performance and automatically rebalance a portfolio based on those predictions. The system is hosted on [Streamlit](https://etf-rebalancer.streamlit.app/).

## Features

- Data collection from Yahoo Finance
- Technical feature engineering for ETF prediction
- XGBoost-based machine learning models
- Portfolio allocation optimization
- Performance backtesting
- Interactive dashboard
- Automated monthly rebalancing

## Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Data | yfinance | Historical market data |
| ML & Backend | Python, xgboost, scikit-learn | Predictive modeling |
| Visualization | Streamlit, Plotly | Interactive dashboard |
| Storage | CSV/Parquet | Flat file storage |

## Project Structure

```
ml-etf-rebalancer/
├── data/              # Data collection and processing
│   ├── fetch_data.py  # Download ETF data
│   └── features.py    # Feature engineering
├── model/             # ML models
│   └── train_model.py # Model training
├── portfolio/         # Portfolio logic
│   ├── rebalance.py   # Allocation strategy
│   └── backtest.py    # Performance testing
├── dashboard/         # Web interface
│   └── app.py         # Streamlit dashboard
├── logs/              # Results and metrics
├── run_pipeline.py    # Main execution script
└── requirements.txt   # Dependencies
```

## Getting Started

### Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/ml-etf-rebalancer.git
cd ml-etf-rebalancer
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

### Running Locally

1. Run the full pipeline
```bash
python run_pipeline.py
```

2. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

## Disclaimer

This software is for educational purposes only. It is not financial advice and should not be used to make investment decisions. Past performance is not indicative of future results.
