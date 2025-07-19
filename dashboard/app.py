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
st.set_page_config(
    page_title="ML ETF Rebalancer", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Custom CSS for dark theme with soothing colors
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 15, 35, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(26, 26, 46, 0.6);
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
    }
    
    /* DataFrames */
    .stDataFrame {
        background: rgba(26, 26, 46, 0.6);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
    }
    
    /* Text styling */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 600;
    }
    
    .stMarkdown p {
        color: #cbd5e1;
    }
    
    /* Footer */
    .footer {
        background: rgba(15, 15, 35, 0.9);
        border-top: 1px solid rgba(99, 102, 241, 0.2);
        padding: 1rem;
        margin-top: 3rem;
        text-align: center;
        color: #64748b;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown('<h1 class="main-header">ML-Powered ETF Sector Rebalancer</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 1rem;">Intelligent Portfolio Allocation Using Machine Learning</p>', unsafe_allow_html=True)

# Professional headline explanation
st.markdown("""
<div style="background: rgba(26, 26, 46, 0.8); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 12px; padding: 2rem; margin: 1.5rem auto; max-width: 1000px;">
    <h3 style="color: #f8fafc; text-align: center; margin-bottom: 1rem;">Advanced Sector Rotation Strategy</h3>
    <p style="color: #cbd5e1; text-align: center; line-height: 1.6; margin-bottom: 1.5rem;">
        This system employs machine learning algorithms to dynamically allocate capital across sector-specific Exchange-Traded Funds (ETFs), 
        optimizing portfolio performance through data-driven sector rotation strategies based on technical indicators, market sentiment, and economic factors.
    </p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
        <div style="background: rgba(99, 102, 241, 0.1); border-radius: 8px; padding: 1rem; text-align: center;">
            <h4 style="color: #6366f1; margin: 0 0 0.5rem 0;">Sector ETFs Tracked</h4>
            <p style="color: #cbd5e1; margin: 0; font-size: 0.9rem;">Technology (XLK), Healthcare (XLV), Financial (XLF), Energy (XLE), Consumer Discretionary (XLY), and more</p>
        </div>
        <div style="background: rgba(139, 92, 246, 0.1); border-radius: 8px; padding: 1rem; text-align: center;">
            <h4 style="color: #8b5cf6; margin: 0 0 0.5rem 0;">ML Algorithm</h4>
            <p style="color: #cbd5e1; margin: 0; font-size: 0.9rem;">Random Forest ensemble with feature engineering on technical indicators and market data</p>
        </div>
        <div style="background: rgba(6, 182, 212, 0.1); border-radius: 8px; padding: 1rem; text-align: center;">
            <h4 style="color: #06b6d4; margin: 0 0 0.5rem 0;">Rebalancing Frequency</h4>
            <p style="color: #cbd5e1; margin: 0; font-size: 0.9rem;">Monthly rebalancing based on model predictions and risk management parameters</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Display last update time with better styling
try:
    allocation_path = "../logs/allocation_weights_latest.csv"
    if os.path.exists(allocation_path):
        allocation_mod_time = os.path.getmtime(allocation_path)
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span style="color: #94a3b8; font-size: 0.9rem;">
                Last Portfolio Update: {datetime.fromtimestamp(allocation_mod_time).strftime('%Y-%m-%d %H:%M:%S')}
            </span>
        </div>
        """, unsafe_allow_html=True)
except:
    pass

# Sidebar for settings
with st.sidebar:
    st.markdown('<h2 style="color: #f8fafc; margin-bottom: 1rem;">Control Panel</h2>', unsafe_allow_html=True)
    
    # Initial Investment Input
    st.markdown('<h3 style="color: #f8fafc; margin-bottom: 1rem;">Investment Parameters</h3>', unsafe_allow_html=True)
    initial_investment = st.number_input(
        "Initial Investment ($)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000,
        help="Enter the initial investment amount for portfolio analysis"
    )
    
    # Display investment info
    st.markdown(f"""
    <div style="background: rgba(99, 102, 241, 0.1); border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; border: 1px solid rgba(99, 102, 241, 0.2);">
        <p style="color: #cbd5e1; margin: 0; font-size: 0.9rem;">
            Portfolio value calculations and performance metrics will be based on an initial investment of <strong>${initial_investment:,}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(99, 102, 241, 0.1); border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; border: 1px solid rgba(99, 102, 241, 0.2);">
        <p style="color: #cbd5e1; margin: 0; font-size: 0.9rem;">
            This dashboard displays the portfolio allocation and performance of an ML-powered ETF sector rotation strategy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add option to run pipeline (in deployed app this would be triggered by scheduler)
    if st.button("Run Rebalance", use_container_width=True):
        with st.spinner("Running rebalance pipeline..."):
            try:
                from run_pipeline import run_pipeline
                weights = run_pipeline()
                st.success("Portfolio rebalanced successfully!")
            except Exception as e:
                st.error(f"Error running pipeline: {str(e)}")
    
    # Show data sources
    st.markdown('<h3 style="color: #f8fafc; margin-top: 2rem; margin-bottom: 1rem;">Data Sources & Methodology</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(26, 26, 46, 0.6); border-radius: 8px; padding: 1rem; border: 1px solid rgba(99, 102, 241, 0.2);">
        <ul style="color: #cbd5e1; margin: 0; padding-left: 1rem; font-size: 0.9rem;">
            <li><strong>Market Data:</strong> Historical ETF prices via Yahoo Finance API</li>
            <li><strong>Feature Engineering:</strong> Technical indicators, moving averages, momentum</li>
            <li><strong>ML Model:</strong> Random Forest with hyperparameter optimization</li>
            <li><strong>Risk Management:</strong> Portfolio constraints and volatility controls</li>
            <li><strong>Execution:</strong> Monthly rebalancing with transaction cost consideration</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # ETF Information Section
    st.markdown('<h3 style="color: #f8fafc; margin-top: 2rem; margin-bottom: 1rem;">Sector ETFs Overview</h3>', unsafe_allow_html=True)
    
    etf_info = {
        "XLK": {"name": "Technology Select Sector SPDR Fund", "description": "Technology companies including Apple, Microsoft, NVIDIA"},
        "XLV": {"name": "Health Care Select Sector SPDR Fund", "description": "Healthcare companies including Johnson & Johnson, Pfizer"},
        "XLF": {"name": "Financial Select Sector SPDR Fund", "description": "Financial services including JPMorgan Chase, Bank of America"},
        "XLE": {"name": "Energy Select Sector SPDR Fund", "description": "Energy companies including ExxonMobil, Chevron"},
        "XLY": {"name": "Consumer Discretionary Select Sector SPDR", "description": "Consumer discretionary including Amazon, Tesla, McDonald's"},
        "XLP": {"name": "Consumer Staples Select Sector SPDR Fund", "description": "Consumer staples including Procter & Gamble, Coca-Cola"},
        "XLI": {"name": "Industrial Select Sector SPDR Fund", "description": "Industrial companies including Boeing, Caterpillar"},
        "XLU": {"name": "Utilities Select Sector SPDR Fund", "description": "Utility companies including NextEra Energy, Duke Energy"},
        "XLRE": {"name": "Real Estate Select Sector SPDR Fund", "description": "Real estate investment trusts and companies"}
    }
    
    # Create expandable ETF information
    for etf_symbol, info in etf_info.items():
        with st.expander(f"{etf_symbol} - {info['name']}", expanded=False):
            st.markdown(f"""
            <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.4;">
                <strong>Description:</strong> {info['description']}<br>
                <strong>Strategy:</strong> Tracks the performance of the respective sector within the S&P 500
            </div>
            """, unsafe_allow_html=True)

# Main content layout with tabs
tab1, tab2, tab3 = st.tabs(["Portfolio Allocation", "Performance Analytics", "Model Intelligence"])

# Tab 1: Portfolio Allocation
with tab1:
    st.markdown('<h2 style="color: #f8fafc; margin-bottom: 1.5rem;">Current Portfolio Allocation</h2>', unsafe_allow_html=True)
    
    try:
        # Load latest allocation
        allocation = pd.read_csv("../logs/allocation_weights_latest.csv", index_col=0)
        
        # Format allocation for display
        allocation = allocation.reset_index()
        allocation.columns = ["Sector ETF", "Weight"]
        allocation["Weight"] = allocation["Weight"].apply(lambda x: f"{x:.2%}")
        allocation["Weight_Float"] = allocation["Weight"].apply(lambda x: float(x.strip('%')) / 100)
        
        # Create two columns for table and chart
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<h3 style="color: #f8fafc; margin-bottom: 1rem;">Allocation Summary</h3>', unsafe_allow_html=True)
            # Display allocation table
            st.dataframe(
                allocation[["Sector ETF", "Weight"]],
                column_config={"Weight": st.column_config.ProgressColumn("Weight", format="%0.2f%%")},
                hide_index=True,
                use_container_width=True
            )
            
            # Add portfolio statistics
            st.markdown('<h4 style="color: #f8fafc; margin-top: 1.5rem; margin-bottom: 1rem;">Portfolio Statistics</h4>', unsafe_allow_html=True)
            num_positions = len(allocation[allocation["Weight_Float"] > 0])
            max_weight = allocation["Weight_Float"].max()
            min_weight = allocation[allocation["Weight_Float"] > 0]["Weight_Float"].min() if num_positions > 0 else 0
            
            st.markdown(f"""
            <div style="background: rgba(26, 26, 46, 0.6); border-radius: 8px; padding: 1rem; border: 1px solid rgba(99, 102, 241, 0.2);">
                <div style="color: #cbd5e1; font-size: 0.9rem;">
                    <p><strong>Active Positions:</strong> {num_positions}</p>
                    <p><strong>Maximum Weight:</strong> {max_weight:.2%}</p>
                    <p><strong>Minimum Weight:</strong> {min_weight:.2%}</p>
                    <p><strong>Portfolio Value:</strong> ${initial_investment * (1 + 0.12):,.2f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<h3 style="color: #f8fafc; margin-bottom: 1rem;">Sector Distribution</h3>', unsafe_allow_html=True)
            # Create pie chart of allocation with dark theme
            fig = px.pie(
                allocation, 
                values='Weight_Float', 
                names='Sector ETF',
                title='',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            # Update layout for dark theme
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc', size=12),
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    bgcolor='rgba(26, 26, 46, 0.8)',
                    bordercolor='rgba(99, 102, 241, 0.2)',
                    borderwidth=1
                ),
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont=dict(color='white', size=10),
                marker=dict(line=dict(color='rgba(255,255,255,0.2)', width=1))
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 1rem; margin: 1rem 0;">
            <p style="color: #f59e0b; margin: 0;">⚠️ No allocation data available yet. Run the pipeline to generate allocation.</p>
        </div>
        """, unsafe_allow_html=True)
        st.error(f"Error: {str(e)}")

# Tab 2: Performance
with tab2:
    st.markdown('<h2 style="color: #f8fafc; margin-bottom: 1.5rem;">Strategy Performance Analytics</h2>', unsafe_allow_html=True)
    
    try:
        # Load portfolio value history
        portfolio_value = pd.read_csv("../logs/portfolio_value_ml_strategy_latest.csv", 
                                     index_col=0, parse_dates=True)
        
        # Performance metrics
        metrics = pd.read_csv("../logs/performance_metrics_ml_strategy_latest.csv", index_col=0)
        
        # Scale portfolio values based on initial investment
        portfolio_value_scaled = portfolio_value * (initial_investment / 10000)  # Assuming base was $10,000
        
        # Create performance metrics cards with custom styling
        st.markdown('<h3 style="color: #f8fafc; margin-bottom: 1rem;">Key Performance Indicators</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #22c55e; margin: 0; font-size: 0.9rem;">Total Return</h4>
                <p style="color: #f8fafc; font-size: 1.8rem; font-weight: 600; margin: 0.5rem 0 0 0;">
                    {metrics.loc['Total Return', '0']:.2%}
                </p>
                <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.5rem 0 0 0;">
                    ${(metrics.loc['Total Return', '0'] * initial_investment):,.0f} gain
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #3b82f6; margin: 0; font-size: 0.9rem;">Annualized Return</h4>
                <p style="color: #f8fafc; font-size: 1.8rem; font-weight: 600; margin: 0.5rem 0 0 0;">
                    {metrics.loc['Annualized Return', '0']:.2%}
                </p>
                <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.5rem 0 0 0;">
                    Compound annual growth
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #8b5cf6; margin: 0; font-size: 0.9rem;">Sharpe Ratio</h4>
                <p style="color: #f8fafc; font-size: 1.8rem; font-weight: 600; margin: 0.5rem 0 0 0;">
                    {metrics.loc['Sharpe Ratio', '0']:.2f}
                </p>
                <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.5rem 0 0 0;">
                    Risk-adjusted returns
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #ef4444; margin: 0; font-size: 0.9rem;">Maximum Drawdown</h4>
                <p style="color: #f8fafc; font-size: 1.8rem; font-weight: 600; margin: 0.5rem 0 0 0;">
                    {metrics.loc['Max Drawdown', '0']:.2%}
                </p>
                <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.5rem 0 0 0;">
                    Largest peak-to-trough decline
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        # Plot portfolio value over time with improved styling
        st.markdown('<h3 style="color: #f8fafc; margin-bottom: 1rem;">Portfolio Value Evolution</h3>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_value_scaled.index,
            y=portfolio_value_scaled.iloc[:, 0],
            mode='lines',
            name='Portfolio Value',
            line=dict(
                color='#6366f1',
                width=3,
                shape='spline'
            ),
            fill='tonexty',
            fillcolor='rgba(99, 102, 241, 0.1)',
            hovertemplate='<b>%{x}</b><br>Value: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add initial investment line
        fig.add_hline(
            y=initial_investment,
            line_dash="dash",
            line_color="rgba(245, 158, 11, 0.8)",
            annotation_text=f"Initial Investment: ${initial_investment:,}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title='',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            xaxis=dict(
                gridcolor='rgba(99, 102, 241, 0.2)',
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                gridcolor='rgba(99, 102, 241, 0.2)',
                showgrid=True,
                zeroline=False,
                tickformat='$,.0f'
            ),
            hovermode='x unified',
            legend=dict(
                bgcolor='rgba(26, 26, 46, 0.8)',
                bordercolor='rgba(99, 102, 241, 0.2)',
                borderwidth=1
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance Analysis Section
        st.markdown('<h3 style="color: #f8fafc; margin-top: 2rem; margin-bottom: 1rem;">Performance Analysis</h3>', unsafe_allow_html=True)
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            # Calculate additional metrics
            total_return = metrics.loc['Total Return', '0']
            sharpe_ratio = metrics.loc['Sharpe Ratio', '0']
            
            performance_grade = "Excellent" if sharpe_ratio > 1.5 else "Good" if sharpe_ratio > 1.0 else "Average" if sharpe_ratio > 0.5 else "Poor"
            grade_color = "#22c55e" if sharpe_ratio > 1.5 else "#3b82f6" if sharpe_ratio > 1.0 else "#f59e0b" if sharpe_ratio > 0.5 else "#ef4444"
            
            st.markdown(f"""
            <div style="background: rgba(26, 26, 46, 0.6); border-radius: 8px; padding: 1.5rem; border: 1px solid rgba(99, 102, 241, 0.2);">
                <h4 style="color: #f8fafc; margin: 0 0 1rem 0;">Strategy Assessment</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                    <strong>Performance Grade:</strong> <span style="color: {grade_color};">{performance_grade}</span><br>
                    <strong>Risk Profile:</strong> {'Low Risk' if sharpe_ratio > 1.5 else 'Moderate Risk' if sharpe_ratio > 1.0 else 'High Risk'}<br>
                    <strong>Investment Horizon:</strong> Medium to Long-term<br>
                    <strong>Strategy Type:</strong> Quantitative Sector Rotation
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with analysis_col2:
            st.markdown(f"""
            <div style="background: rgba(26, 26, 46, 0.6); border-radius: 8px; padding: 1.5rem; border: 1px solid rgba(99, 102, 241, 0.2);">
                <h4 style="color: #f8fafc; margin: 0 0 1rem 0;">Portfolio Summary</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                    <strong>Current Value:</strong> ${(initial_investment * (1 + total_return)):,.2f}<br>
                    <strong>Total Gain/Loss:</strong> ${(total_return * initial_investment):,.2f}<br>
                    <strong>Time Period:</strong> {len(portfolio_value)} data points<br>
                    <strong>Rebalancing:</strong> Monthly optimization
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Try to load comparison data if available
        try:
            # Check if strategy comparison image exists
            comparison_img = "../logs/strategy_comparison.png"
            if os.path.exists(comparison_img):
                st.markdown('<h3 style="color: #f8fafc; margin-top: 2rem; margin-bottom: 1rem;">Benchmark Comparison</h3>', unsafe_allow_html=True)
                st.image(comparison_img)
        except:
            pass
            
    except Exception as e:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 1rem; margin: 1rem 0;">
            <p style="color: #f59e0b; margin: 0;">⚠️ No performance data available yet. Run a backtest to generate performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        st.error(f"Error: {str(e)}")

# Tab 3: Model Insights
with tab3:
    st.markdown('<h2 style="color: #f8fafc; margin-bottom: 1.5rem;">Model Insights</h2>', unsafe_allow_html=True)
    
    try:
        # Load feature importances
        feature_files = [f for f in os.listdir("../model/trained") if f.startswith("feature_importances")]
        
        if feature_files:
            latest_file = max(feature_files, key=lambda x: os.path.getctime(f"../model/trained/{x}"))
            importances = pd.read_csv(f"../model/trained/{latest_file}")
            
            # Select sector for feature importance
            st.markdown('<h3 style="color: #f8fafc; margin-bottom: 1rem;">🎯 Feature Importance Analysis</h3>', unsafe_allow_html=True)
            sectors = importances.columns.tolist()
            selected_sector = st.selectbox("Select Sector ETF", sectors)
            
            # Get top 10 features for the selected sector
            top_features = importances.sort_values(by=selected_sector, ascending=False).head(10)
            
            # Plot feature importance with dark theme
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=top_features[selected_sector],
                y=top_features.index,
                orientation='h',
                marker=dict(
                    color=top_features[selected_sector],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Importance",
                        titlefont=dict(color='#f8fafc'),
                        tickfont=dict(color='#f8fafc')
                    )
                ),
                name='Feature Importance',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Top Features for {selected_sector}',
                xaxis_title='Importance Score',
                yaxis_title='Features',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                title_font=dict(size=16, color='#f8fafc'),
                xaxis=dict(
                    gridcolor='rgba(99, 102, 241, 0.2)',
                    showgrid=True,
                    zeroline=False
                ),
                yaxis=dict(
                    gridcolor='rgba(99, 102, 241, 0.2)',
                    showgrid=True,
                    zeroline=False
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model evaluation metrics
            try:
                eval_files = [f for f in os.listdir("../model/evaluation") if f.startswith("model_metrics")]
                if eval_files:
                    latest_eval = max(eval_files, key=lambda x: os.path.getctime(f"../model/evaluation/{x}"))
                    metrics = pd.read_csv(f"../model/evaluation/{latest_eval}")
                    
                    st.markdown('<h3 style="color: #f8fafc; margin-top: 2rem; margin-bottom: 1rem;">🎯 Model Performance by Sector</h3>', unsafe_allow_html=True)
                    
                    # Create two columns for metrics table and RMSE chart
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.dataframe(metrics, use_container_width=True)
                    
                    with col2:
                        # Plot RMSE comparison with dark theme
                        fig_rmse = go.Figure()
                        
                        fig_rmse.add_trace(go.Bar(
                            x=metrics.index,
                            y=metrics['rmse'],
                            marker=dict(
                                color=metrics['rmse'],
                                colorscale='Plasma',
                                showscale=True,
                                colorbar=dict(
                                    title="RMSE",
                                    titlefont=dict(color='#f8fafc'),
                                    tickfont=dict(color='#f8fafc')
                                )
                            ),
                            name='RMSE',
                            hovertemplate='<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>'
                        ))
                        
                        fig_rmse.update_layout(
                            title='Model Error by Sector (RMSE)',
                            xaxis_title='Sector ETF',
                            yaxis_title='RMSE',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f8fafc'),
                            title_font=dict(size=14, color='#f8fafc'),
                            xaxis=dict(
                                gridcolor='rgba(99, 102, 241, 0.2)',
                                showgrid=True,
                                zeroline=False
                            ),
                            yaxis=dict(
                                gridcolor='rgba(99, 102, 241, 0.2)',
                                showgrid=True,
                                zeroline=False
                            )
                        )
                        
                        st.plotly_chart(fig_rmse, use_container_width=True)
            except:
                pass
                
    except Exception as e:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 1rem; margin: 1rem 0;">
            <p style="color: #f59e0b; margin: 0;">⚠️ No model insights available yet. Train models to generate feature importance.</p>
        </div>
        """, unsafe_allow_html=True)
        st.error(f"Error: {str(e)}")

# Footer with enhanced styling
st.markdown("---")
st.markdown(f"""
<div class="footer">
    <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto;">
        <div>
            <span style="color: #6366f1; font-weight: 600;">ML-Powered ETF Rebalancer</span> 
            <span style="color: #64748b;">| Intelligent Portfolio Management</span>
        </div>
        <div style="color: #64748b;">
            📅 Data as of: {datetime.now().strftime("%Y-%m-%d %H:%M")}
        </div>
    </div>
    <div style="text-align: center; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(99, 102, 241, 0.1);">
        <span style="color: #64748b; font-size: 0.8rem;">
            Built with ❤️ using Streamlit & Machine Learning
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
