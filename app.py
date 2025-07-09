import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set page configuration for a wide layout and custom title
st.set_page_config(page_title="Financial Dashboard", layout="wide")

# Add a title and brief description
st.title("📈 Real-Time Financial Dashboard")
st.write("Enter a stock symbol to get the latest financial data and visualizations.")

# Let users choose how to input the stock symbol
option = st.radio("Choose input method:", ["Enter Symbol", "Select Popular Stock"])
if option == "Enter Symbol":
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", value="AAPL")
else:
    popular_stocks = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Tesla": "TSLA"
    }
    selected_stock = st.selectbox("Select a popular stock:", list(popular_stocks.keys()))
    symbol = popular_stocks[selected_stock]

# Cache the data fetching to improve performance
@st.cache_data
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    # Fetch financial statements
    financials = stock.financials if not stock.financials.empty else None
    balance_sheet = stock.balance_sheet if not stock.balance_sheet.empty else None
    cashflow = stock.cashflow if not stock.cashflow.empty else None
    # Return a dictionary of serializable data
    return {
        "info": info,
        "financials": financials.to_dict() if financials is not None else None,
        "balance_sheet": balance_sheet.to_dict() if balance_sheet is not None else None,
        "cashflow": cashflow.to_dict() if cashflow is not None else None
    }

# Button to trigger data fetching
if st.button("Get Data"):
    with st.spinner("Fetching data..."):
        try:
            data = get_stock_data(symbol)
            if not data["info"]:
                st.error("Invalid symbol or no data available.")
            else:
                st.success(f"Data for {symbol} fetched successfully!")
                
                # Recreate the Ticker object if needed
                stock = yf.Ticker(symbol)
                
                # Section: Company Information
                with st.expander("Company Information", expanded=True):
                    st.write(f"**Name:** {data['info'].get('longName', 'N/A')}")
                    st.write(f"**Sector:** {data['info'].get('sector', 'N/A')}")
                    st.write(f"**Industry:** {data['info'].get('industry', 'N/A')}")
                    st.write(f"**Market Cap:** {data['info'].get('marketCap', 'N/A')}")
                
                # Section: Key Metrics
                with st.expander("Key Metrics"):
                    metrics = ["trailingPE", "forwardPE", "priceToBook", "debtToEquity"]
                    for metric in metrics:
                        value = data['info'].get(metric, 'N/A')
                        st.write(f"**{metric}:** {value}")
                
                # Section: Historical Price Data
                with st.expander("Historical Price Data"):
                    period = st.selectbox("Select Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
                    hist = stock.history(period=period)
                    if not hist.empty:
                        st.line_chart(hist['Close'])
                    else:
                        st.warning("No historical data available for the selected period.")
                
                # Section: Financial Statements
                with st.expander("Financial Statements"):
                    stmt_type = st.selectbox("Select Statement:", ["Income Statement", "Balance Sheet", "Cash Flow"])
                    if stmt_type == "Income Statement" and data["financials"]:
                        stmt = pd.DataFrame(data["financials"])
                    elif stmt_type == "Balance Sheet" and data["balance_sheet"]:
                        stmt = pd.DataFrame(data["balance_sheet"])
                    elif stmt_type == "Cash Flow" and data["cashflow"]:
                        stmt = pd.DataFrame(data["cashflow"])
                    else:
                        stmt = None
                    if stmt is not None:
                        st.dataframe(stmt)
                    else:
                        st.warning(f"No {stmt_type.lower()} data available.")
                
                # Section: Recent News
                with st.expander("Recent News"):
                    try:
                        news = stock.news
                        if news:
                            for item in news:
                                st.write(f"**{item['title']}**")
                                st.write(item['summary'])
                                st.write(f"[Read more]({item['url']})")
                                st.markdown("---")
                        else:
                            st.info("No recent news available.")
                    except Exception as e:
                        st.warning(f"Could not fetch news: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer with timestamp
st.markdown("---")
st.write(f"Data fetched on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
