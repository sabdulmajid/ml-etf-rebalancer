import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import sys

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# App title and configuration
st.set_page_config(page_title="ML ETF Rebalancer", layout="wide")
st.title("ML-Powered ETF Sector Rebalancer")

# Display last update time
try:
    allocation_path = "../logs/allocation_weights_latest.csv"
    if os.path.exists(allocation_path):
        allocation_mod_time = os.path.getmtime(allocation_path)
        st.caption(f"Last updated: {datetime.fromtimestamp(allocation_mod_time).strftime('%Y-%m-%d %H:%M:%S')}")
except:
    pass

# Sidebar for settings
with st.sidebar:
    st.header("ETF Rebalancer")
    st.write("This dashboard displays the portfolio allocation and performance of an ML-powered ETF sector rotation strategy.")
    
    # Add option to run pipeline (in deployed app this would be triggered by scheduler)
    if st.button("Run Rebalance"):
        with st.spinner("Running rebalance pipeline..."):
            try:
                from run_pipeline import run_pipeline
                weights = run_pipeline()
                st.success("Portfolio rebalanced successfully!")
            except Exception as e:
                st.error(f"Error running pipeline: {str(e)}")
    
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
    
    try:
        # Load latest allocation
        allocation = pd.read_csv("../logs/allocation_weights_latest.csv", index_col=0)
        
        # Format allocation for display
        allocation = allocation.reset_index()
        allocation.columns = ["Sector ETF", "Weight"]
        allocation["Weight"] = allocation["Weight"].apply(lambda x: f"{x:.2%}")
        allocation["Weight_Float"] = allocation["Weight"].apply(lambda x: float(x.strip('%')) / 100)
        
        # Display allocation table
        st.dataframe(
            allocation[["Sector ETF", "Weight"]],
            column_config={"Weight": st.column_config.ProgressColumn("Weight", format="%0.2f%%")},
            hide_index=True
        )
        
        # Create pie chart of allocation
        fig = px.pie(
            allocation, 
            values='Weight_Float', 
            names='Sector ETF',
            title='Portfolio Allocation by Sector',
            hole=0.3,
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning("No allocation data available yet. Run the pipeline to generate allocation.")
        st.error(f"Error: {str(e)}")

# Tab 2: Performance
with tab2:
    st.header("Strategy Performance")
    
    try:
        # Load portfolio value history
        portfolio_value = pd.read_csv("../logs/portfolio_value_ml_strategy_latest.csv", 
                                     index_col=0, parse_dates=True)
        
        # Performance metrics
        metrics = pd.read_csv("../logs/performance_metrics_ml_strategy_latest.csv", index_col=0)
        
        # Create performance metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics.loc['Total Return', '0']:.2%}")
            
        with col2:
            st.metric("Annual Return", f"{metrics.loc['Annualized Return', '0']:.2%}")
            
        with col3:
            st.metric("Sharpe Ratio", f"{metrics.loc['Sharpe Ratio', '0']:.2f}")
            
        with col4:
            st.metric("Max Drawdown", f"{metrics.loc['Max Drawdown', '0']:.2%}")
        
        # Plot portfolio value over time
        fig = px.line(
            portfolio_value, 
            title='Portfolio Value Over Time',
            labels={'value': 'Portfolio Value', 'index': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Try to load comparison data if available
        try:
            # Check if strategy comparison image exists
            comparison_img = "../logs/strategy_comparison.png"
            if os.path.exists(comparison_img):
                st.header("Strategy Comparison")
                st.image(comparison_img)
        except:
            pass
            
    except Exception as e:
        st.warning("No performance data available yet. Run a backtest to generate performance metrics.")
        st.error(f"Error: {str(e)}")

# Tab 3: Model Insights
with tab3:
    st.header("Model Insights")
    
    try:
        # Load feature importances
        feature_files = [f for f in os.listdir("../model/trained") if f.startswith("feature_importances")]
        
        if feature_files:
            latest_file = max(feature_files, key=lambda x: os.path.getctime(f"../model/trained/{x}"))
            importances = pd.read_csv(f"../model/trained/{latest_file}")
            
            # Select sector for feature importance
            sectors = importances.columns.tolist()
            selected_sector = st.selectbox("Select Sector ETF", sectors)
            
            # Get top 10 features for the selected sector
            top_features = importances.sort_values(by=selected_sector, ascending=False).head(10)
            
            # Plot feature importance
            fig = px.bar(
                top_features, 
                x=selected_sector, 
                y=top_features.index,
                orientation='h',
                title=f'Top Features for {selected_sector}',
                labels={selected_sector: 'Importance', 'index': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model evaluation metrics
            try:
                eval_files = [f for f in os.listdir("../model/evaluation") if f.startswith("model_metrics")]
                if eval_files:
                    latest_eval = max(eval_files, key=lambda x: os.path.getctime(f"../model/evaluation/{x}"))
                    metrics = pd.read_csv(f"../model/evaluation/{latest_eval}")
                    
                    st.header("Model Performance by Sector")
                    st.dataframe(metrics)
                    
                    # Plot RMSE comparison
                    fig = px.bar(
                        metrics,
                        x=metrics.index,
                        y='rmse',
                        title='Model Error by Sector (RMSE)',
                        labels={'rmse': 'RMSE', 'index': 'Sector ETF'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except:
                pass
                
    except Exception as e:
        st.warning("No model insights available yet. Train models to generate feature importance.")
        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("ML-Powered ETF Rebalancer | Data as of: " + datetime.now().strftime("%Y-%m-%d"))
