"""
Frontend Streamlit Dashboard - User Interface for Financial Analysis
This module provides the web interface for the financial dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Any, Optional
import time

# Import the backend engine
from financial_engine import (
    financial_processor,
    get_stock_analysis,
    get_financial_statements,
    calculate_ratios,
    get_historical_data,
    validate_symbol,
    clear_cache,
    get_cache_stats
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border: none;
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border: none;
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border: none;
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stTab > div:first-child > div:first-child {
        gap: 0.5rem;
    }
    
    .stTab [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTab [data-baseweb="tab"] {
        padding: 1rem 2rem;
        border-radius: 8px;
        background-color: #f8f9fa;
    }
    
    .highlight-row {
        background: linear-gradient(90deg, #ffeaa7 0%, #fab1a0 100%);
        color: #2d3436;
        border-radius: 5px;
        padding: 0.2rem;
    }
    
    .chart-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class DashboardUI:
    """Frontend UI class for the financial dashboard"""
    
    def __init__(self):
        self.processor = financial_processor
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'current_symbol' not in st.session_state:
            st.session_state.current_symbol = None
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = None
        if 'selected_rows' not in st.session_state:
            st.session_state.selected_rows = {}
        if 'chart_preferences' not in st.session_state:
            st.session_state.chart_preferences = {
                'chart_type': 'line',
                'theme': 'plotly_white',
                'show_grid': True
            }
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<div class="main-header">📈 Advanced Financial Dashboard</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        with st.sidebar:
            st.header("🎛️ Dashboard Controls")
            
            # Symbol input section
            st.subheader("📊 Stock Selection")
            symbol_input = st.text_input(
                "Enter Stock Symbol:",
                value="AAPL",
                help="Enter a valid stock symbol (e.g., AAPL, MSFT, GOOGL)",
                placeholder="e.g., AAPL"
            )
            
            # Popular stocks dropdown
            popular_stocks = {
                "": "",
                "Apple Inc.": "AAPL",
                "Microsoft Corp.": "MSFT",
                "Alphabet Inc.": "GOOGL",
                "Amazon.com Inc.": "AMZN",
                "Tesla Inc.": "TSLA",
                "Meta Platforms": "META",
                "Netflix Inc.": "NFLX",
                "NVIDIA Corp.": "NVDA",
                "Berkshire Hathaway": "BRK-B",
                "Johnson & Johnson": "JNJ"
            }
            
            selected_popular = st.selectbox(
                "Or select from popular stocks:",
                list(popular_stocks.keys()),
                help="Quick selection of popular stocks"
            )
            
            if selected_popular:
                symbol_input = popular_stocks[selected_popular]
            
            st.divider()
            
            # Chart preferences
            st.subheader("📈 Chart Preferences")
            chart_type = st.selectbox(
                "Default Chart Type:",
                ["line", "bar", "area"],
                index=0,
                help="Choose the default chart type for row plotting"
            )
            
            chart_theme = st.selectbox(
                "Chart Theme:",
                ["plotly_white", "plotly_dark", "simple_white", "ggplot2"],
                index=0,
                help="Select chart theme"
            )
            
            show_grid = st.checkbox("Show Grid Lines", value=True)
            
            # Update session state
            st.session_state.chart_preferences = {
                'chart_type': chart_type,
                'theme': chart_theme,
                'show_grid': show_grid
            }
            
            st.divider()
            
            # Historical data preferences
            st.subheader("📅 Historical Data")
            hist_period = st.selectbox(
                "Historical Period:",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
                index=5,
                help="Select time period for historical data"
            )
            
            st.divider()
            
            # Action buttons
            st.subheader("⚡ Actions")
            fetch_data = st.button("🔍 Fetch Data", use_container_width=True, type="primary")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Cache", use_container_width=True):
                    clear_cache()
                    st.success("Cache cleared!")
            
            with col2:
                if st.button("📊 Cache Stats", use_container_width=True):
                    stats = get_cache_stats()
                    st.json(stats)
            
            return symbol_input, fetch_data, hist_period
    
    def create_metric_cards(self, analysis_data: Dict[str, Any]):
        """Create metric cards for key financial data"""
        basic_info = analysis_data['basic_info']
        valuation = analysis_data['valuation_metrics']
        profitability = analysis_data['profitability_metrics']
        
        # First row - Basic info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${basic_info.get('current_price', 'N/A'):.2f}" if basic_info.get('current_price') else "N/A",
                delta=f"{basic_info.get('change_percent', 0):.2f}%"
            )
        
        with col2:
            st.metric(
                "Market Cap",
                f"${basic_info.get('market_cap', 0) / 1e9:.2f}B" if basic_info.get('market_cap') else "N/A",
                help="Market Capitalization in billions"
            )
        
        with col3:
            st.metric(
                "P/E Ratio",
                f"{valuation.get('pe_ratio', 'N/A'):.2f}" if valuation.get('pe_ratio') else "N/A",
                help="Price-to-Earnings Ratio"
            )
        
        with col4:
            st.metric(
                "ROE",
                f"{profitability.get('roe', 'N/A'):.2f}%" if profitability.get('roe') else "N/A",
                help="Return on Equity"
            )
        
        # Second row - Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "52W High",
                f"${basic_info.get('fifty_two_week_high', 'N/A'):.2f}" if basic_info.get('fifty_two_week_high') else "N/A",
                help="52-Week High"
            )
        
        with col2:
            st.metric(
                "52W Low",
                f"${basic_info.get('fifty_two_week_low', 'N/A'):.2f}" if basic_info.get('fifty_two_week_low') else "N/A",
                help="52-Week Low"
            )
        
        with col3:
            st.metric(
                "Volume",
                f"{basic_info.get('volume', 0) / 1e6:.2f}M" if basic_info.get('volume') else "N/A",
                help="Trading Volume in millions"
            )
        
        with col4:
            st.metric(
                "Dividend Yield",
                f"{basic_info.get('dividend_yield', 'N/A'):.2f}%" if basic_info.get('dividend_yield') else "N/A",
                help="Dividend Yield"
            )
    
    def create_financial_charts(self, analysis_data: Dict[str, Any]):
        """Create financial charts"""
        st.subheader("📊 Financial Charts")
        
        # Price chart
        historical_data = analysis_data.get('historical_data')
        if historical_data is not None and not historical_data.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=f"Price History - {st.session_state.current_symbol}",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template=st.session_state.chart_preferences['theme'],
                showlegend=True,
                height=400
            )
            
            if st.session_state.chart_preferences['show_grid']:
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        if historical_data is not None and not historical_data.empty:
            fig_volume = go.Figure()
            
            fig_volume.add_trace(go.Bar(
                x=historical_data.index,
                y=historical_data['Volume'],
                name='Volume',
                marker_color='rgba(55, 83, 109, 0.6)'
            ))
            
            fig_volume.update_layout(
                title=f"Trading Volume - {st.session_state.current_symbol}",
                xaxis_title="Date",
                yaxis_title="Volume",
                template=st.session_state.chart_preferences['theme'],
                height=300
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
    
    def create_financial_statements_tables(self, analysis_data: Dict[str, Any]):
        """Create financial statements tables"""
        statements = analysis_data.get('financial_statements', {})
        
        # Create tabs for different statements
        tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
        
        with tab1:
            income_statement = statements.get('income_statement')
            if income_statement is not None and not income_statement.empty:
                st.subheader("📈 Income Statement")
                
                # Display as interactive table
                selected_rows = st.multiselect(
                    "Select rows to highlight:",
                    income_statement.index.tolist(),
                    key="income_statement_rows"
                )
                
                # Style the dataframe
                styled_df = income_statement.style.format("{:,.0f}")
                if selected_rows:
                    styled_df = styled_df.apply(
                        lambda x: ['background-color: #ffeaa7' if x.name in selected_rows else '' for i in x],
                        axis=1
                    )
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Plot selected rows
                if selected_rows:
                    self.plot_selected_rows(income_statement, selected_rows, "Income Statement")
            else:
                st.warning("Income statement data not available")
        
        with tab2:
            balance_sheet = statements.get('balance_sheet')
            if balance_sheet is not None and not balance_sheet.empty:
                st.subheader("📊 Balance Sheet")
                
                selected_rows = st.multiselect(
                    "Select rows to highlight:",
                    balance_sheet.index.tolist(),
                    key="balance_sheet_rows"
                )
                
                styled_df = balance_sheet.style.format("{:,.0f}")
                if selected_rows:
                    styled_df = styled_df.apply(
                        lambda x: ['background-color: #ffeaa7' if x.name in selected_rows else '' for i in x],
                        axis=1
                    )
                
                st.dataframe(styled_df, use_container_width=True)
                
                if selected_rows:
                    self.plot_selected_rows(balance_sheet, selected_rows, "Balance Sheet")
            else:
                st.warning("Balance sheet data not available")
        
        with tab3:
            cash_flow = statements.get('cash_flow')
            if cash_flow is not None and not cash_flow.empty:
                st.subheader("💰 Cash Flow Statement")
                
                selected_rows = st.multiselect(
                    "Select rows to highlight:",
                    cash_flow.index.tolist(),
                    key="cash_flow_rows"
                )
                
                styled_df = cash_flow.style.format("{:,.0f}")
                if selected_rows:
                    styled_df = styled_df.apply(
                        lambda x: ['background-color: #ffeaa7' if x.name in selected_rows else '' for i in x],
                        axis=1
                    )
                
                st.dataframe(styled_df, use_container_width=True)
                
                if selected_rows:
                    self.plot_selected_rows(cash_flow, selected_rows, "Cash Flow Statement")
            else:
                st.warning("Cash flow statement data not available")
    
    def plot_selected_rows(self, df: pd.DataFrame, selected_rows: List[str], title: str):
        """Plot selected rows from financial statements"""
        st.subheader(f"📊 {title} - Selected Rows Visualization")
        
        # Create subplot for each selected row
        fig = make_subplots(
            rows=len(selected_rows),
            cols=1,
            subplot_titles=selected_rows,
            vertical_spacing=0.1
        )
        
        for i, row in enumerate(selected_rows):
            row_data = df.loc[row]
            
            if st.session_state.chart_preferences['chart_type'] == 'line':
                fig.add_trace(
                    go.Scatter(
                        x=row_data.index,
                        y=row_data.values,
                        mode='lines+markers',
                        name=row,
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )
            elif st.session_state.chart_preferences['chart_type'] == 'bar':
                fig.add_trace(
                    go.Bar(
                        x=row_data.index,
                        y=row_data.values,
                        name=row,
                        opacity=0.8
                    ),
                    row=i+1, col=1
                )
            elif st.session_state.chart_preferences['chart_type'] == 'area':
                fig.add_trace(
                    go.Scatter(
                        x=row_data.index,
                        y=row_data.values,
                        mode='lines',
                        name=row,
                        fill='tozeroy',
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            height=300 * len(selected_rows),
            template=st.session_state.chart_preferences['theme'],
            showlegend=False
        )
        
        if st.session_state.chart_preferences['show_grid']:
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_ratio_analysis(self, analysis_data: Dict[str, Any]):
        """Create ratio analysis section"""
        st.subheader("🔢 Financial Ratios Analysis")
        
        ratios = analysis_data.get('ratios', {})
        if not ratios:
            st.warning("Ratio data not available")
            return
        
        # Group ratios by category
        ratio_categories = {
            "Profitability": ['gross_profit_margin', 'operating_margin', 'net_margin', 'roe', 'roa'],
            "Liquidity": ['current_ratio', 'quick_ratio', 'cash_ratio'],
            "Leverage": ['debt_to_equity', 'debt_ratio', 'interest_coverage'],
            "Efficiency": ['asset_turnover', 'inventory_turnover', 'receivables_turnover'],
            "Valuation": ['pe_ratio', 'pb_ratio', 'ev_ebitda', 'price_to_sales']
        }
        
        # Create tabs for different ratio categories
        tabs = st.tabs(list(ratio_categories.keys()))
        
        for tab, (category, ratio_list) in zip(tabs, ratio_categories.items()):
            with tab:
                st.subheader(f"{category} Ratios")
                
                # Create metrics for each ratio
                cols = st.columns(min(3, len(ratio_list)))
                for i, ratio in enumerate(ratio_list):
                    if ratio in ratios:
                        with cols[i % 3]:
                            value = ratios[ratio]
                            if isinstance(value, (int, float)) and not np.isnan(value):
                                st.metric(
                                    ratio.replace('_', ' ').title(),
                                    f"{value:.2f}",
                                    help=f"Current {ratio.replace('_', ' ')} ratio"
                                )
                            else:
                                st.metric(
                                    ratio.replace('_', ' ').title(),
                                    "N/A",
                                    help=f"Data not available for {ratio.replace('_', ' ')}"
                                )
                
                # Create visualization for available ratios
                available_ratios = {k: v for k, v in ratios.items() if k in ratio_list and isinstance(v, (int, float)) and not np.isnan(v)}
                
                if available_ratios:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(available_ratios.keys()),
                            y=list(available_ratios.values()),
                            marker_color='rgba(55, 83, 109, 0.6)'
                        )
                    ])
                    
                    fig.update_layout(
                        title=f"{category} Ratios Overview",
                        xaxis_title="Ratio",
                        yaxis_title="Value",
                        template=st.session_state.chart_preferences['theme'],
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def create_summary_report(self, analysis_data: Dict[str, Any]):
        """Create summary report section"""
        st.subheader("📋 Executive Summary")
        
        basic_info = analysis_data.get('basic_info', {})
        
        # Company information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Company:** {basic_info.get('company_name', 'N/A')}
            **Symbol:** {st.session_state.current_symbol}
            **Sector:** {basic_info.get('sector', 'N/A')}
            **Industry:** {basic_info.get('industry', 'N/A')}
            """)
        
        with col2:
            st.markdown(f"""
            **Market Cap:** ${basic_info.get('market_cap', 0) / 1e9:.2f}B
            **Employees:** {basic_info.get('employees', 'N/A'):,}
            **Founded:** {basic_info.get('founded', 'N/A')}
            **Exchange:** {basic_info.get('exchange', 'N/A')}
            """)
        
        # Key insights
        st.markdown("### 🔍 Key Insights")
        
        insights = []
        
        # Price performance
        current_price = basic_info.get('current_price', 0)
        week_52_high = basic_info.get('fifty_two_week_high', 0)
        week_52_low = basic_info.get('fifty_two_week_low', 0)
        
        if current_price and week_52_high and week_52_low:
            high_ratio = (current_price / week_52_high) * 100
            low_ratio = (current_price / week_52_low) * 100
            
            if high_ratio > 90:
                insights.append("🟢 Stock is trading near its 52-week high")
            elif high_ratio < 50:
                insights.append("🔴 Stock is trading significantly below its 52-week high")
            
            if low_ratio > 150:
                insights.append("🟢 Stock has shown strong recovery from its 52-week low")
        
        # Valuation insights
        valuation = analysis_data.get('valuation_metrics', {})
        pe_ratio = valuation.get('pe_ratio')
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < 15:
                insights.append("🟢 Stock appears undervalued based on P/E ratio")
            elif pe_ratio > 25:
                insights.append("🔴 Stock appears overvalued based on P/E ratio")
            else:
                insights.append("🟡 Stock is fairly valued based on P/E ratio")
        
        # Profitability insights
        profitability = analysis_data.get('profitability_metrics', {})
        roe = profitability.get('roe')
        if roe and roe > 0:
            if roe > 15:
                insights.append("🟢 Company shows strong return on equity")
            elif roe < 5:
                insights.append("🔴 Company shows weak return on equity")
        
        # Display insights
        if insights:
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.markdown("• No specific insights available with current data")
    
    def run(self):
        """Main method to run the dashboard"""
        self.render_header()
        
        # Get sidebar inputs
        symbol_input, fetch_data, hist_period = self.render_sidebar()
        
        # Main content area
        if fetch_data or symbol_input != st.session_state.current_symbol:
            if symbol_input:
                # Validate symbol
                if validate_symbol(symbol_input):
                    st.session_state.current_symbol = symbol_input.upper()
                    
                    # Show loading spinner
                    with st.spinner(f"Fetching data for {symbol_input.upper()}..."):
                        try:
                            # Fetch analysis data
                            analysis_data = get_stock_analysis(symbol_input.upper(), period=hist_period)
                            st.session_state.analysis_data = analysis_data
                            
                            st.success(f"✅ Data loaded successfully for {symbol_input.upper()}")
                            
                        except Exception as e:
                            st.error(f"❌ Error fetching data: {str(e)}")
                            st.session_state.analysis_data = None
                else:
                    st.error(f"❌ Invalid symbol: {symbol_input}")
        
        # Display analysis if data is available
        if st.session_state.analysis_data:
            analysis_data = st.session_state.analysis_data
            
            # Create main tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview", 
                "📈 Charts", 
                "📋 Statements", 
                "🔢 Ratios", 
                "📝 Summary"
            ])
            
            with tab1:
                st.subheader(f"📊 {st.session_state.current_symbol} - Overview")
                self.create_metric_cards(analysis_data)
            
            with tab2:
                self.create_financial_charts(analysis_data)
            
            with tab3:
                self.create_financial_statements_tables(analysis_data)
            
            with tab4:
                self.create_ratio_analysis(analysis_data)
            
            with tab5:
                self.create_summary_report(analysis_data)
        
        else:
            # Welcome message
            st.markdown("""
            ## 👋 Welcome to the Advanced Financial Dashboard!
            
            This dashboard provides comprehensive financial analysis for public companies. 
            
            **Features:**
            - Real-time stock data and metrics
            - Interactive financial charts
            - Complete financial statements
            - Ratio analysis across multiple categories
            - Executive summary with key insights
            
            **To get started:**
            1. Enter a stock symbol in the sidebar (e.g., AAPL, MSFT, GOOGL)
            2. Click "🔍 Fetch Data" to load the analysis
            3. Explore the different tabs for detailed insights
            
            **Tips:**
            - Use the popular stocks dropdown for quick selection
            - Customize chart preferences in the sidebar
            - Select multiple rows in financial statements for comparison
            - Clear cache if you encounter data issues
            """)
            
            # Show sample symbols
            st.markdown("### 📈 Popular Stocks to Try:")
            sample_cols = st.columns(5)
            sample_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            
            for i, symbol in enumerate(sample_symbols):
                with sample_cols[i]:
                    if st.button(f"📊 {symbol}", use_container_width=True):
                        st.session_state.current_symbol = symbol
                        st.experimental_rerun()

# Main execution
if __name__ == "__main__":
    dashboard = DashboardUI()
    dashboard.run()
