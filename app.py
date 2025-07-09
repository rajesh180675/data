# /mount/src/data/app.py

"""
Frontend Streamlit Dashboard - User Interface for Financial Analysis
This module provides the web interface for the financial dashboard with interactive charting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any

# Import the robust backend engine
from financial_engine import (
    get_stock_analysis,
    validate_symbol,
    clear_cache,
    get_cache_stats
)

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
        font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center;
        margin-bottom: 2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stTab [data-baseweb="tab-list"] { gap: 2rem; }
    .stTab [data-baseweb="tab"] { padding: 1rem 2rem; border-radius: 8px; background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)


def _style_selected_rows(row: pd.Series, selected_rows: List[str]) -> List[str]:
    """Helper function to apply background style to selected DataFrame rows."""
    highlight_style = 'background-color: #ffeaa7'
    return [highlight_style if row.name in selected_rows else '' for _ in row]


class DashboardUI:
    """Frontend UI class for the financial dashboard"""

    def __init__(self):
        """Constructor for the UI class."""
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables on the first run."""
        defaults = {
            'current_symbol': None,
            'analysis_data': None,
            'hist_period': '1y',
            'chart_preferences': {'chart_type': 'line', 'theme': 'plotly_white'},
            'user_symbol_input': 'AAPL'
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render_header(self):
        """Render the main header of the application."""
        st.markdown('<div class="main-header">📈 Advanced Financial Dashboard</div>', unsafe_allow_html=True)
        st.markdown("---")

    def render_sidebar(self):
        """Render the sidebar with all user controls."""
        with st.sidebar:
            st.header("🎛️ Dashboard Controls")
            st.subheader("📊 Stock Selection")

            symbol_input_from_user = st.text_input(
                "Enter Stock Symbol:",
                value=st.session_state.user_symbol_input,
                help="Enter a valid stock symbol (e.g., AAPL, MSFT, BRK-B)"
            )
            st.session_state.user_symbol_input = symbol_input_from_user

            st.write("Or select from popular stocks:")
            pop_cols = st.columns(3)
            popular_stocks = {"AAPL": pop_cols[0], "MSFT": pop_cols[1], "GOOGL": pop_cols[2]}
            for symbol, col in popular_stocks.items():
                if col.button(symbol, use_container_width=True):
                    st.session_state.user_symbol_input = symbol
                    st.rerun()

            st.divider()
            st.subheader("📈 Chart Preferences")
            chart_type = st.selectbox("Chart Type for Statements", ["line", "bar"], help="Select the chart type for visualizing statement rows.")
            chart_theme = st.selectbox("Chart Theme", ["plotly_white", "plotly_dark", "simple_white", "ggplot2"])
            st.session_state.chart_preferences = {'chart_type': chart_type, 'theme': chart_theme}
            
            st.subheader("📅 Historical Data")
            hist_period = st.selectbox("Historical Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=5)
            
            st.divider()
            st.subheader("⚡ Actions")
            fetch_data = st.button("🔍 Fetch Data", use_container_width=True, type="primary")

            col1, col2 = st.columns(2)
            if col1.button("🗑️ Clear Cache", use_container_width=True):
                clear_cache(); st.success("Cache cleared!")
            if col2.button("📊 Cache Stats", use_container_width=True):
                with st.spinner("Fetching cache stats..."):
                    st.json(get_cache_stats())

            return st.session_state.user_symbol_input, fetch_data, hist_period

    def create_metric_cards(self, analysis_data: Dict[str, Any]):
        """Create metric cards for key financial data."""
        basic_info = analysis_data.get('basic_info', {})
        valuation_metrics = analysis_data.get('valuation_metrics', {})
        profitability_metrics = analysis_data.get('profitability_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            price = basic_info.get('current_price')
            change = basic_info.get('change_percent')
            st.metric("Current Price", f"${price:.2f}" if isinstance(price, (int, float)) else "N/A", f"{change:.2f}%" if isinstance(change, (int, float)) else None)
        with col2:
            mkt_cap = basic_info.get('market_cap')
            st.metric("Market Cap", f"${mkt_cap / 1e9:.2f}B" if isinstance(mkt_cap, (int, float)) else "N/A", help="Market Capitalization")
        with col3:
            pe = valuation_metrics.get('pe_ratio')
            st.metric("P/E Ratio", f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A", help="Price-to-Earnings (TTM)")
        with col4:
            roe = profitability_metrics.get('roe')
            st.metric("ROE", f"{roe*100:.2f}%" if isinstance(roe, (int, float)) else "N/A", help="Return on Equity")


    def create_financial_charts(self, analysis_data: Dict[str, Any]):
        """Create combined financial price and volume charts."""
        historical_data = analysis_data.get('historical_data')
        if historical_data is not None and not historical_data.empty:
            st.subheader("📈 Price & Volume History")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Close'], name='Close Price', line=dict(color='#1f77b4')), row=1, col=1)
            fig.add_trace(go.Bar(x=historical_data['Date'], y=historical_data['Volume'], name='Volume', marker_color='rgba(55, 83, 109, 0.6)'), row=2, col=1)
            fig.update_layout(
                title_text=f"Price and Volume for {st.session_state.current_symbol}",
                template=st.session_state.chart_preferences['theme'],
                height=500,
                showlegend=False,
                xaxis_rangeslider_visible=False
            )
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

    # FEATURE ENHANCEMENT: New plotting function
    def plot_selected_rows(self, df: pd.DataFrame, selected_rows: List[str], title: str):
        """
        Plots multiple selected rows from a financial statement DataFrame on a single chart
        for correlation and comparison analysis.
        """
        if not selected_rows:
            return

        # Prepare data for plotting
        plot_df = df.loc[selected_rows].T
        # Convert index to numeric for proper sorting on x-axis if it's years
        plot_df.index = pd.to_numeric(plot_df.index)
        plot_df = plot_df.sort_index()
        plot_df.index = plot_df.index.astype(str) # Convert back to string for display

        # Get chart preferences from sidebar
        chart_prefs = st.session_state.chart_preferences
        chart_type = chart_prefs.get('chart_type', 'line')
        theme = chart_prefs.get('theme', 'plotly_white')

        st.markdown("---")
        st.subheader(f"📊 Chart for Selected Rows in {title}")

        if chart_type == 'bar':
            fig = px.bar(plot_df, x=plot_df.index, y=plot_df.columns, title=f"Comparison: {', '.join(selected_rows)}", barmode='group')
        else: # Default to line chart
            fig = px.line(plot_df, x=plot_df.index, y=plot_df.columns, title=f"Trend Analysis: {', '.join(selected_rows)}", markers=True)

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Amount",
            legend_title="Metrics",
            template=theme
        )
        st.plotly_chart(fig, use_container_width=True)


    # FEATURE ENHANCEMENT: Updated this function to include multiselect and call the plotter
    def create_financial_statements_tables(self, analysis_data: Dict[str, Any]):
        """Create financial statements tables with interactive row selection for charting."""
        statements = analysis_data.get('statements', {})
        tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])

        def display_statement_with_charting(statement_df, title, key_prefix):
            if statement_df is not None and not statement_df.empty:
                df_display = statement_df.set_index('Year').T
                
                # FEATURE: Add multiselect widget to choose rows
                st.markdown(f"**Select rows from the {title} below to visualize them.**")
                selected_rows = st.multiselect(
                    f"Select rows to chart:",
                    options=df_display.index.tolist(),
                    key=f"{key_prefix}_multiselect"
                )

                # Apply styling to highlight selected rows
                styled_df = df_display.style.format("{:,.0f}", na_rep="N/A")
                if selected_rows:
                    styled_df = styled_df.apply(_style_selected_rows, selected_rows=selected_rows, axis=1)
                
                st.dataframe(styled_df, use_container_width=True)

                # FEATURE: Call the plotting function if rows are selected
                self.plot_selected_rows(df_display, selected_rows, title)
            else:
                st.warning(f"{title} data not available.")

        with tab1:
            display_statement_with_charting(statements.get('income_statement'), "Income Statement", "income")
        with tab2:
            display_statement_with_charting(statements.get('balance_sheet'), "Balance Sheet", "balance")
        with tab3:
            display_statement_with_charting(statements.get('cash_flow'), "Cash Flow", "cashflow")

    def create_ratio_analysis(self, analysis_data: Dict[str, Any]):
        """Create ratio analysis section."""
        ratios_df = analysis_data.get('ratios')
        if ratios_df is None or ratios_df.empty:
            st.warning("Ratio data not available.")
            return
        
        st.subheader("🔢 Financial Ratios Over Time")
        st.dataframe(ratios_df.set_index('Year'), use_container_width=True)

        # Plot key ratios over time for visual analysis
        key_ratios_to_plot = ['Net Profit Margin (%)', 'ROE (%)', 'Debt to Equity', 'Current Ratio']
        plot_ratios = [r for r in key_ratios_to_plot if r in ratios_df.columns]
        if plot_ratios:
            fig = px.line(ratios_df, x='Year', y=plot_ratios, title="Key Financial Ratio Trends", markers=True)
            fig.update_layout(template=st.session_state.chart_preferences['theme'])
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main method to run the dashboard application."""
        self.render_header()
        symbol_input, fetch_data, hist_period = self.render_sidebar()

        needs_fetch = (
            fetch_data or
            (symbol_input and symbol_input != st.session_state.current_symbol) or
            (hist_period and hist_period != st.session_state.hist_period)
        )

        if needs_fetch and symbol_input:
            is_valid, formatted_symbol = validate_symbol(symbol_input)
            if is_valid:
                with st.spinner(f"Fetching data for {formatted_symbol}..."):
                    try:
                        analysis_data = get_stock_analysis(formatted_symbol, period=hist_period)
                        if analysis_data:
                            st.session_state.analysis_data = analysis_data
                            st.session_state.current_symbol = formatted_symbol
                            st.session_state.hist_period = hist_period
                            st.success(f"✅ Data loaded for {formatted_symbol}")
                        else:
                            st.error(f"❌ Could not retrieve data for {formatted_symbol}.")
                            st.session_state.analysis_data = None
                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")
                        st.session_state.analysis_data = None
            else:
                st.error(f"❌ Invalid symbol format: {symbol_input}")

        if st.session_state.analysis_data:
            analysis_data = st.session_state.analysis_data
            tab1, tab2 = st.tabs(["📊 Overview", "📈 Statements & Ratios"])

            with tab1:
                self.create_metric_cards(analysis_data)
                self.create_financial_charts(analysis_data)
            with tab2:
                self.create_financial_statements_tables(analysis_data)
                st.markdown("---")
                self.create_ratio_analysis(analysis_data)
        else:
            st.markdown("## 👋 Welcome to the Advanced Financial Dashboard!")
            st.info("Enter a stock symbol and click 'Fetch Data' to begin.")


if __name__ == "__main__":
    dashboard = DashboardUI()
    dashboard.run()
