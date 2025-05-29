import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Configure the app
st.set_page_config(page_title="ML ETF Rebalancer", layout="wide")

# Define demo data
def generate_demo_data():
    sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLI', 'XLU', 'XLB', 'XLP']
    
    # Portfolio allocation
    weights = {
        'XLF': 0.15, 'XLK': 0.20, 'XLE': 0.05, 'XLV': 0.15, 
        'XLY': 0.10, 'XLI': 0.10, 'XLU': 0.10, 'XLB': 0.05, 'XLP': 0.10
    }
    
    # Portfolio performance
    dates = pd.date_range(start='2015-01-01', end='2025-06-01', freq='M')
    
    # Simulate portfolio growth with some volatility
    np.random.seed(42)
    returns = np.random.normal(0.008, 0.04, size=len(dates))  # Monthly returns with positive drift
    portfolio_value = 1000 * (1 + returns).cumprod()
    
    # Create comparison with a benchmark that performs slightly worse
    benchmark = 1000 * (1 + returns * 0.8).cumprod()
    
    # Model evaluation metrics
    metrics = {
        'Total Return': 1.86,  # 186% over the period
        'Annualized Return': 0.108, # 10.8% annually
        'Sharpe Ratio': 1.27,
        'Max Drawdown': -0.23, # 23% max drawdown
        'Win Rate': 0.67 # 67% winning months
    }
    
    # Feature importances
    features = [
        "mom1m", "mom3m", "mom6m", "mom12m", 
        "vol1m", "vol3m", "ma_ratio", "rs6m"
    ]
    
    importances = {}
    for sector in sectors:
        # Different sectors are driven by different factors
        imp = np.random.random(size=len(features))
        imp = imp / imp.sum()  # Normalize to sum to 1
        importances[sector] = pd.Series(imp, index=features)
    
    return {
        'sectors': sectors,
        'weights': weights,
        'dates': dates,
        'portfolio_value': portfolio_value,
        'benchmark': benchmark,
        'metrics': metrics,
        'importances': importances
    }

# Generate demo data
data = generate_demo_data()

# App title and header
st.title("ML-Powered ETF Sector Rebalancer")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

# Sidebar for settings
with st.sidebar:
    st.header("ETF Rebalancer")
    st.write("This dashboard displays the portfolio allocation and performance of an ML-powered ETF sector rotation strategy.")
    
    # Add option to run pipeline (in deployed app this would be triggered by scheduler)
    if st.button("Run Rebalance"):
        with st.spinner("Running rebalance pipeline..."):
            st.success("Portfolio rebalanced successfully!")
    
    # Show data sources
    st.subheader("Data Sources")
    st.markdown("""
    - Historical ETF prices: yfinance
    - Allocation weights: ML prediction
    """)

# Main content layout with tabs
tab1, tab2, tab3 = st.tabs(["Portfolio Allocation", "Performance", "Model Insights"])

# Tab 1: Portfolio Allocation
with tab1:
    st.header("Current Portfolio Allocation")
    
    # Format allocation for display
    allocation_df = pd.DataFrame({
        'Sector ETF': list(data['weights'].keys()),
        'Weight': list(data['weights'].values())
    })
    
    # Display allocation table
    st.dataframe(
        allocation_df,
        column_config={"Weight": st.column_config.ProgressColumn("Weight", format="%0.2f")},
        hide_index=True
    )
    
    # Create pie chart of allocation
    fig = px.pie(
        allocation_df, 
        values='Weight', 
        names='Sector ETF',
        title='Portfolio Allocation by Sector',
        hole=0.3,
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Performance
with tab2:
    st.header("Strategy Performance")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{data['metrics']['Total Return']:.2%}")
        
    with col2:
        st.metric("Annual Return", f"{data['metrics']['Annualized Return']:.2%}")
        
    with col3:
        st.metric("Sharpe Ratio", f"{data['metrics']['Sharpe Ratio']:.2f}")
        
    with col4:
        st.metric("Max Drawdown", f"{data['metrics']['Max Drawdown']:.2%}")
    
    # Plot portfolio value over time
    performance_df = pd.DataFrame({
        'Date': data['dates'],
        'ML Strategy': data['portfolio_value'],
        'Equal Weight': data['benchmark']
    }).set_index('Date')
    
    fig = px.line(
        performance_df, 
        title='Portfolio Value Over Time ($1,000 Initial Investment)',
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Model Insights
with tab3:
    st.header("Model Insights")
    
    # Select sector for feature importance
    selected_sector = st.selectbox("Select Sector ETF", data['sectors'])
    
    # Get features for the selected sector
    sector_importances = data['importances'][selected_sector].sort_values(ascending=False)
    
    # Create a DataFrame for the plot
    importance_df = pd.DataFrame({
        'Feature': sector_importances.index,
        'Importance': sector_importances.values
    })
    
    # Plot feature importance
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title=f'Feature Importance for {selected_sector}',
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model evaluation metrics
    st.header("Model Performance by Sector")
    
    # Create a demo metrics table
    model_metrics = pd.DataFrame({
        'Sector': data['sectors'],
        'RMSE': np.random.uniform(0.01, 0.05, len(data['sectors'])),
        'R²': np.random.uniform(0.1, 0.4, len(data['sectors']))
    }).set_index('Sector')
    
    st.dataframe(model_metrics)

# Footer
st.markdown("---")
st.caption("ML-Powered ETF Rebalancer | Demo Version | Data shown is simulated for demonstration purposes")
