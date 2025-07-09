# /mount/src/data/app.py

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
from typing import List, Dict, Any

# Import the robust backend engine
from financial_engine import (
    get_stock_analysis,
    validate_symbol,
    clear_cache,
    get_cache_stats
)

# Configure Streamlit page
st.set_page_config(page_title="Advanced Financial Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""<style>/* ... your CSS ... */</style>""", unsafe_allow_html=True)


def _style_selected_rows(row: pd.Series, selected_rows: List[str]) -> List[str]:
    highlight_style = 'background-color: #ffeaa7'
    return [highlight_style if row.name in selected_rows else '' for _ in row]


class DashboardUI:
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        defaults = {
            'current_symbol': None,
            'analysis_data': None,
            'hist_period': '1y',
            'chart_preferences': {'chart_type': 'line', 'theme': 'plotly_white', 'show_grid': True},
            # FRONTEND FIX: Use a different key for the user's text input to avoid conflict
            'user_symbol_input': 'AAPL'
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render_sidebar(self):
        with st.sidebar:
            st.header("🎛️ Dashboard Controls")
            st.subheader("📊 Stock Selection")
            
            # FRONTEND FIX: This is the user's input, which drives the process
            symbol_input = st.text_input(
                "Enter Stock Symbol:",
                value=st.session_state.user_symbol_input,
                key="symbol_input_widget", # Use a unique key
                help="Enter a valid stock symbol (e.g., AAPL, MSFT, BRK-B)"
            )
            # Update the session state based on this widget's value
            st.session_state.user_symbol_input = symbol_input

            # Action buttons for popular stocks that set the state for the next run
            st.write("Or select from popular stocks:")
            pop_cols = st.columns(3)
            popular_stocks = {"AAPL": pop_cols[0], "MSFT": pop_cols[1], "GOOGL": pop_cols[2]}
            for symbol, col in popular_stocks.items():
                if col.button(symbol, use_container_width=True):
                    # Set the input for the next run and rerun
                    st.session_state.user_symbol_input = symbol
                    st.rerun()

            st.divider()
            # ... rest of the sidebar ...
            st.subheader("📈 Chart Preferences")
            chart_type = st.selectbox("Default Chart Type:", ["line", "bar", "area"])
            chart_theme = st.selectbox("Chart Theme:", ["plotly_white", "plotly_dark", "simple_white", "ggplot2"])
            st.session_state.chart_preferences = {'chart_type': chart_type, 'theme': chart_theme}
            
            st.subheader("📅 Historical Data")
            hist_period = st.selectbox("Historical Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=5)
            
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
        basic_info = analysis_data.get('basic_info', {})
        # ... (This function remains the same as the previous corrected version) ...

    def create_financial_charts(self, analysis_data: Dict[str, Any]):
        historical_data = analysis_data.get('historical_data')
        # ... (This function remains the same as the previous corrected version) ...

    def create_financial_statements_tables(self, analysis_data: Dict[str, Any]):
        statements = analysis_data.get('statements', {})
        # ... (This function remains the same as the previous corrected version) ...

    # ... other class methods (plot_selected_rows, create_ratio_analysis, etc.) remain the same ...

    def run(self):
        self.render_header()
        symbol_input, fetch_data, hist_period = self.render_sidebar()

        # Determine if a data fetch is needed
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

        # Display content if data is available
        if st.session_state.analysis_data:
            analysis_data = st.session_state.analysis_data
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Statements & Ratios", "📝 Summary"])

            with tab1:
                self.create_metric_cards(analysis_data)
                self.create_financial_charts(analysis_data)
            # ... rest of the tabs ...
        else:
            st.markdown("## 👋 Welcome to the Advanced Financial Dashboard!")
            st.info("Enter a stock symbol and click 'Fetch Data' to begin.")

if __name__ == "__main__":
    dashboard = DashboardUI()
    dashboard.run()
