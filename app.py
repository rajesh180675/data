# /mount/src/data/app.py

"""
Frontend Streamlit Dashboard - User Interface for Financial Analysis
This module provides the web interface for the financial dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

# Custom CSS for enhanced styling (unchanged)
st.markdown("""
<style>
    /* ... your CSS remains the same ... */
</style>
""", unsafe_allow_html=True)


def _style_selected_rows(row: pd.Series, selected_rows: List[str]) -> List[str]:
    """Helper function to apply background style to selected DataFrame rows."""
    highlight_style = 'background-color: #ffeaa7'
    return [highlight_style if row.name in selected_rows else '' for _ in row]


class DashboardUI:
    """Frontend UI class for the financial dashboard"""

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'current_symbol' not in st.session_state:
            st.session_state.current_symbol = None
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = None
        # ROBUSTNESS: Store the period to detect changes
        if 'hist_period' not in st.session_state:
            st.session_state.hist_period = "1y"
        if 'chart_preferences' not in st.session_state:
            st.session_state.chart_preferences = {
                'chart_type': 'line',
                'theme': 'plotly_white',
                'show_grid': True
            }
        # UI/UX FIX: State for linked inputs
        if 'symbol_text_input' not in st.session_state:
            st.session_state.symbol_text_input = "AAPL"

    def _update_symbol_from_selectbox(self):
        """Callback to link popular stock selectbox to the text input."""
        popular_stocks = {
            "Apple Inc.": "AAPL", "Microsoft Corp.": "MSFT", "Alphabet Inc.": "GOOGL",
            "Amazon.com Inc.": "AMZN", "Tesla Inc.": "TSLA", "Meta Platforms": "META",
            "NVIDIA Corp.": "NVDA"
        }
        selected_company = st.session_state.get("popular_stock_selector", "")
        if selected_company and selected_company in popular_stocks:
            st.session_state.symbol_text_input = popular_stocks[selected_company]

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
                key="symbol_text_input",
                help="Enter a valid stock symbol (e.g., AAPL, MSFT, BRK-B)",
                placeholder="e.g., AAPL"
            )

            popular_stocks_list = ["", "Apple Inc.", "Microsoft Corp.", "Alphabet Inc.", "Amazon.com Inc.", "Tesla Inc.", "Meta Platforms", "NVIDIA Corp."]
            st.selectbox(
                "Or select from popular stocks:",
                popular_stocks_list,
                key="popular_stock_selector",
                on_change=self._update_symbol_from_selectbox,
                help="Quick selection of popular stocks"
            )

            st.divider()

            # Chart preferences
            st.subheader("📈 Chart Preferences")
            chart_type = st.selectbox("Default Chart Type:", ["line", "bar", "area"], index=0, help="Choose the default chart type for row plotting")
            chart_theme = st.selectbox("Chart Theme:", ["plotly_white", "plotly_dark", "simple_white", "ggplot2"], index=0, help="Select chart theme")
            show_grid = st.checkbox("Show Grid Lines", value=True)

            st.session_state.chart_preferences = {'chart_type': chart_type, 'theme': chart_theme, 'show_grid': show_grid}

            st.divider()

            # Historical data preferences
            st.subheader("📅 Historical Data")
            hist_period = st.selectbox("Historical Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=5, help="Select time period for historical data")

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
                    with st.spinner("Fetching cache stats..."):
                        stats = get_cache_stats()
                        st.json(stats)

            return symbol_input, fetch_data, hist_period

    def create_metric_cards(self, analysis_data: Dict[str, Any]):
        """Create metric cards for key financial data with robust error handling"""
        basic_info = analysis_data.get('basic_info', {})
        valuation = analysis_data.get('valuation_metrics', {})
        profitability = analysis_data.get('profitability_metrics', {})

        col1, col2, col3, col4 = st.columns(4)

        # ROBUSTNESS: Check for numeric types before formatting to prevent errors
        with col1:
            price = basic_info.get('current_price')
            change = basic_info.get('change_percent')
            price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "N/A"
            change_str = f"{change:.2f}%" if isinstance(change, (int, float)) else None
            st.metric("Current Price", price_str, delta=change_str)

        with col2:
            mkt_cap = basic_info.get('market_cap')
            mkt_cap_str = f"${mkt_cap / 1e9:.2f}B" if isinstance(mkt_cap, (int, float)) else "N/A"
            st.metric("Market Cap", mkt_cap_str, help="Market Capitalization in billions")

        with col3:
            pe = valuation.get('pe_ratio')
            pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) and pe is not None else "N/A"
            st.metric("P/E Ratio", pe_str, help="Price-to-Earnings Ratio")

        with col4:
            roe = profitability.get('roe')
            roe_str = f"{roe * 100:.2f}%" if isinstance(roe, (int, float)) and roe is not None else "N/A"
            st.metric("ROE", roe_str, help="Return on Equity")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            high = basic_info.get('fifty_two_week_high')
            st.metric("52W High", f"${high:.2f}" if isinstance(high, (int, float)) else "N/A")
        with col2:
            low = basic_info.get('fifty_two_week_low')
            st.metric("52W Low", f"${low:.2f}" if isinstance(low, (int, float)) else "N/A")
        with col3:
            volume = basic_info.get('volume')
            vol_str = f"{volume / 1e6:.2f}M" if isinstance(volume, (int, float)) else "N/A"
            st.metric("Volume", vol_str, help="Trading Volume in millions")
        with col4:
            div_yield = basic_info.get('dividend_yield')
            div_str = f"{div_yield * 100:.2f}%" if isinstance(div_yield, (int, float)) and div_yield is not None else "N/A"
            st.metric("Dividend Yield", div_str)

    def create_financial_charts(self, analysis_data: Dict[str, Any]):
        """Create financial charts"""
        historical_data = analysis_data.get('historical_data')
        if historical_data is not None and not historical_data.empty:
            st.subheader("📈 Price & Volume History")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                row_heights=[0.7, 0.3])

            fig.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Close'], mode='lines', name='Close Price', line=dict(color='#1f77b4', width=2)), row=1, col=1)
            fig.add_trace(go.Bar(x=historical_data['Date'], y=historical_data['Volume'], name='Volume', marker_color='rgba(55, 83, 109, 0.6)'), row=2, col=1)

            fig.update_layout(
                title_text=f"Price and Volume for {st.session_state.current_symbol}",
                template=st.session_state.chart_preferences['theme'],
                height=500,
                xaxis_rangeslider_visible=False,
                showlegend=False
            )
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Historical data not available.")

    def create_financial_statements_tables(self, analysis_data: Dict[str, Any]):
        """Create financial statements tables"""
        statements = analysis_data.get('statements', {})
        tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])

        def display_statement(statement_df, title, key_prefix):
            if statement_df is not None and not statement_df.empty:
                st.subheader(f"📈 {title}")
                # Set 'Year' as the index for better display and processing
                statement_df = statement_df.set_index('Year').T
                
                selected_rows = st.multiselect("Select rows to chart:", statement_df.index.tolist(), key=f"{key_prefix}_rows")
                
                # CODE QUALITY: Use helper function for styling
                styled_df = statement_df.style.format("{:,.0f}", na_rep="N/A")
                if selected_rows:
                    styled_df = styled_df.apply(_style_selected_rows, selected_rows=selected_rows, axis=1)

                st.dataframe(styled_df, use_container_width=True)

                if selected_rows:
                    self.plot_selected_rows(statement_df, selected_rows, title)
            else:
                st.warning(f"{title} data not available.")

        with tab1:
            display_statement(statements.get('income_statement'), "Income Statement", "income")
        with tab2:
            display_statement(statements.get('balance_sheet'), "Balance Sheet", "balance")
        with tab3:
            display_statement(statements.get('cash_flow'), "Cash Flow", "cashflow")

    def plot_selected_rows(self, df: pd.DataFrame, selected_rows: List[str], title: str):
        """Plot selected rows from financial statements"""
        chart_prefs = st.session_state.chart_preferences
        chart_type = chart_prefs['chart_type']
        
        # Ensure we're plotting numeric data and handle potential non-numeric dtypes
        plot_df = df.loc[selected_rows].apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

        if plot_df.empty:
            st.warning("Selected rows contain no plot-able numeric data.")
            return

        fig = px.line(plot_df.T, title=f"Trend for Selected Items in {title}", labels={'index': 'Year', 'value': 'Amount'})
        
        if chart_type == 'bar':
            fig = px.bar(plot_df.T, title=f"Trend for Selected Items in {title}", labels={'index': 'Year', 'value': 'Amount'}, barmode='group')
        elif chart_type == 'area':
            fig = px.area(plot_df.T, title=f"Trend for Selected Items in {title}", labels={'index': 'Year', 'value': 'Amount'})

        fig.update_layout(template=chart_prefs['theme'], height=400 * (1 + len(selected_rows) // 3))
        st.plotly_chart(fig, use_container_width=True)


    def create_ratio_analysis(self, analysis_data: Dict[str, Any]):
        """Create ratio analysis section"""
        ratios_df = analysis_data.get('ratios')
        if ratios_df is None or ratios_df.empty:
            st.warning("Ratio data not available.")
            return
        
        st.subheader("🔢 Financial Ratios Over Time")
        st.dataframe(ratios_df.set_index('Year'), use_container_width=True)

        # Plot key ratios over time
        st.subheader("📊 Key Ratio Trends")
        key_ratios_to_plot = ['Net Profit Margin (%)', 'ROE (%)', 'Debt to Equity', 'Current Ratio']
        plot_ratios = [r for r in key_ratios_to_plot if r in ratios_df.columns]

        if plot_ratios:
            fig = px.line(ratios_df, x='Year', y=plot_ratios, title="Key Financial Ratio Trends", markers=True)
            fig.update_layout(template=st.session_state.chart_preferences['theme'], height=500)
            st.plotly_chart(fig, use_container_width=True)

    def create_summary_report(self, analysis_data: Dict[str, Any]):
        """Create summary report section"""
        st.subheader("📋 Executive Summary")
        basic_info = analysis_data.get('basic_info', {})
        st.markdown(f"### {basic_info.get('company_name', 'N/A')} ({st.session_state.current_symbol})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Sector:** {basic_info.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {basic_info.get('industry', 'N/A')}")
        with col2:
            market_cap_str = basic_info.get('market_cap_formatted', 'N/A')
            st.markdown(f"**Market Cap:** {market_cap_str}")
            st.markdown(f"**Website:** [{basic_info.get('website', 'N/A')}]({basic_info.get('website')})")

        st.markdown("**Business Summary:**")
        st.info(basic_info.get('summary', 'No summary available.'))

    def run(self):
        """Main method to run the dashboard"""
        self.render_header()
        symbol_input, fetch_data, hist_period = self.render_sidebar()

        # CRITICAL FIX: Trigger data fetch on symbol OR period change
        if fetch_data or symbol_input != st.session_state.current_symbol or hist_period != st.session_state.hist_period:
            if symbol_input:
                is_valid, formatted_symbol = validate_symbol(symbol_input)
                if is_valid:
                    st.session_state.current_symbol = formatted_symbol
                    st.session_state.hist_period = hist_period  # Store the period used for this fetch

                    with st.spinner(f"Fetching data for {formatted_symbol}..."):
                        try:
                            # Pass the period to the backend
                            analysis_data = get_stock_analysis(formatted_symbol, period=hist_period)
                            if analysis_data:
                                st.session_state.analysis_data = analysis_data
                                st.success(f"✅ Data loaded successfully for {formatted_symbol}")
                            else:
                                st.error(f"❌ Could not retrieve data for {formatted_symbol}. The symbol may be invalid or delisted.")
                                st.session_state.analysis_data = None
                        except Exception as e:
                            st.error(f"❌ An error occurred: {str(e)}")
                            st.session_state.analysis_data = None
                else:
                    st.error(f"❌ Invalid symbol format: {formatted_symbol}")
        
        if st.session_state.analysis_data:
            analysis_data = st.session_state.analysis_data
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Statements & Ratios", "📝 Summary", "⚙️ Raw Data", "ℹ️ About"])

            with tab1:
                self.create_metric_cards(analysis_data)
                self.create_financial_charts(analysis_data)
            with tab2:
                self.create_financial_statements_tables(analysis_data)
                self.create_ratio_analysis(analysis_data)
            with tab3:
                self.create_summary_report(analysis_data)
            with tab4:
                st.subheader("Raw Analysis Data")
                st.json(analysis_data, expanded=False)
            with tab5:
                st.info("Dashboard created by [Your Name]. Data provided by Yahoo Finance.")
        else:
            st.markdown("## 👋 Welcome to the Advanced Financial Dashboard!")
            st.info("Enter a stock symbol in the sidebar (e.g., AAPL, MSFT, GOOGL) and click 'Fetch Data' to begin.")
            sample_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            st.markdown("### 📈 Popular Stocks to Try:")
            sample_cols = st.columns(len(sample_symbols))
            for i, symbol in enumerate(sample_symbols):
                with sample_cols[i]:
                    if st.button(f"📊 {symbol}", use_container_width=True):
                        st.session_state.symbol_text_input = symbol
                        # BUG FIX: Use st.rerun() instead of deprecated version
                        st.rerun()

if __name__ == "__main__":
    dashboard = DashboardUI()
    dashboard.run()
