"""
Backend Financial Engine - Core Logic for Financial Data Processing
This module handles all data fetching, processing, and analysis operations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union
import time
import re
import logging
from functools import wraps
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class FinancialDataCache:
    """Simple in-memory cache for financial data"""
    def __init__(self, ttl_minutes: int = 15):
        self.cache = {}
        self.ttl_minutes = ttl_minutes
    
    def _get_cache_key(self, symbol: str, data_type: str) -> str:
        """Generate cache key"""
        return f"{symbol}_{data_type}_{datetime.now().strftime('%Y%m%d_%H')}"
    
    def get(self, symbol: str, data_type: str) -> Optional[Any]:
        """Get cached data"""
        key = self._get_cache_key(symbol, data_type)
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.ttl_minutes * 60:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, symbol: str, data_type: str, data: Any) -> None:
        """Set cache data"""
        key = self._get_cache_key(symbol, data_type)
        self.cache[key] = (data, datetime.now())
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()

class FinancialDataProcessor:
    """Core financial data processing engine"""
    
    def __init__(self, cache_ttl_minutes: int = 15):
        self.cache = FinancialDataCache(cache_ttl_minutes)
        self.setup_plotting_style()
        self.CRORE = 1_00_00_000
        self.MILLION = 1_000_000
        self.BILLION = 1_000_000_000
        
    def setup_plotting_style(self):
        """Configure matplotlib and seaborn styles"""
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.titlesize': 16,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'sans-serif'
        })

    def log_operation(func):
        """Decorator to log operations"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                end_time = time.time()
                logger.info(f"{func.__name__} completed in {end_time - start_time:.2f}s")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper

    def validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """Validate and format stock symbol"""
        if not symbol or not isinstance(symbol, str):
            return False, "Invalid symbol format"
        
        # Clean and format symbol
        symbol = symbol.strip().upper()
        
        # Remove any extra spaces or special characters
        symbol = re.sub(r'[^A-Z0-9.]', '', symbol)
        
        # Basic validation - allow letters, numbers, and dots
        if not re.match(r'^[A-Z0-9]{1,10}(\.[A-Z0-9]{1,5})?$', symbol):
            return False, "Symbol should contain only letters, numbers, and optional exchange suffix"
        
        return True, symbol

    def safe_division(self, numerator: Union[pd.Series, float], denominator: Union[pd.Series, float]) -> Union[pd.Series, float]:
        """Perform safe division with proper error handling"""
        try:
            if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
                num_aligned, den_aligned = numerator.align(denominator, fill_value=0)
                den_aligned = den_aligned.replace(0, np.nan)
                result = num_aligned / den_aligned
                return result.fillna(0)
            else:
                return numerator / denominator if denominator != 0 else 0
        except Exception as e:
            logger.error(f"Error in safe division: {str(e)}")
            return 0 if not isinstance(numerator, pd.Series) else pd.Series(0)

    @log_operation
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch basic stock information with caching"""
        # Check cache first
        cached_data = self.cache.get(symbol, 'stock_info')
        if cached_data:
            return cached_data
        
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info or len(info) < 5:
                logger.warning(f"No valid info found for symbol: {symbol}")
                return None
            
            # Validate essential fields
            required_fields = ['symbol', 'longName', 'currentPrice', 'regularMarketPrice']
            if not any(field in info for field in required_fields):
                logger.warning(f"Missing essential fields for symbol: {symbol}")
                return None
            
            # Cache the result
            self.cache.set(symbol, 'stock_info', info)
            return info
            
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return None

    @log_operation
    def get_financial_statements(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch all financial statements with robust error handling"""
        # Check cache first
        cached_data = self.cache.get(symbol, 'financial_statements')
        if cached_data:
            return cached_data
        
        try:
            stock = yf.Ticker(symbol)
            statements = {}
            
            # Define statement types and their methods
            statement_types = {
                'income_statement': ('get_financials', 'Income Statement'),
                'balance_sheet': ('get_balance_sheet', 'Balance Sheet'),
                'cash_flow': ('get_cashflow', 'Cash Flow Statement')
            }
            
            for stmt_key, (method_name, display_name) in statement_types.items():
                try:
                    method = getattr(stock, method_name)
                    stmt_data = method(freq='yearly').transpose()
                    
                    if not stmt_data.empty:
                        stmt_data.reset_index(inplace=True)
                        stmt_data.rename(columns={'index': 'Year'}, inplace=True)
                        
                        # Convert datetime to year string
                        stmt_data['Year'] = pd.to_datetime(stmt_data['Year']).dt.strftime('%Y')
                        
                        # Sort by year in descending order (most recent first)
                        stmt_data = stmt_data.sort_values('Year', ascending=False)
                        
                        # Clean column names
                        stmt_data.columns = [col.strip() if isinstance(col, str) else col for col in stmt_data.columns]
                        
                        statements[stmt_key] = stmt_data
                        logger.info(f"Successfully fetched {display_name} for {symbol}")
                    else:
                        logger.warning(f"Empty {display_name} for {symbol}")
                        statements[stmt_key] = pd.DataFrame()
                        
                except Exception as e:
                    logger.error(f"Error fetching {display_name} for {symbol}: {str(e)}")
                    statements[stmt_key] = pd.DataFrame()
            
            # Cache the result if we have at least one valid statement
            if any(not df.empty for df in statements.values()):
                self.cache.set(symbol, 'financial_statements', statements)
                return statements
            else:
                logger.warning(f"No financial statements available for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching financial statements for {symbol}: {str(e)}")
            return None

    @log_operation
    def calculate_financial_ratios(self, statements: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate comprehensive financial ratios"""
        try:
            income_stmt = statements.get('income_statement', pd.DataFrame())
            balance_sheet = statements.get('balance_sheet', pd.DataFrame())
            cash_flow = statements.get('cash_flow', pd.DataFrame())
            
            if income_stmt.empty or balance_sheet.empty:
                logger.warning("Income statement or balance sheet is empty")
                return pd.DataFrame()
            
            # Get common years
            income_years = set(income_stmt['Year'].tolist())
            balance_years = set(balance_sheet['Year'].tolist())
            common_years = sorted(list(income_years & balance_years), reverse=True)
            
            if not common_years:
                logger.warning("No common years found between statements")
                return pd.DataFrame()
            
            ratios_data = []
            
            for year in common_years:
                try:
                    income_year = income_stmt[income_stmt['Year'] == year]
                    balance_year = balance_sheet[balance_sheet['Year'] == year]
                    cash_year = cash_flow[cash_flow['Year'] == year] if not cash_flow.empty else pd.DataFrame()
                    
                    if income_year.empty or balance_year.empty:
                        continue
                    
                    # Helper function to safely extract values
                    def safe_get(df, columns, default=0):
                        """Try multiple column name variations"""
                        if isinstance(columns, str):
                            columns = [columns]
                        
                        for col in columns:
                            if col in df.columns:
                                value = df[col].iloc[0]
                                return value if pd.notna(value) else default
                        return default
                    
                    # Extract financial metrics with multiple column name variations
                    net_income = safe_get(income_year, ['Net Income', 'Net Income Common Stockholders', 'Net Income Continuous Operations'])
                    total_revenue = safe_get(income_year, ['Total Revenue', 'Revenue', 'Net Revenue'])
                    total_assets = safe_get(balance_year, ['Total Assets', 'Total Assets'])
                    stockholder_equity = safe_get(balance_year, ['Stockholder Equity', 'Total Stockholder Equity', 'Shareholders Equity'])
                    current_assets = safe_get(balance_year, ['Current Assets', 'Total Current Assets'])
                    current_liabilities = safe_get(balance_year, ['Current Liabilities', 'Total Current Liabilities'])
                    total_liabilities = safe_get(balance_year, ['Total Liabilities Net Minority Interest', 'Total Liabilities', 'Total Debt'])
                    
                    # Calculate ratios with proper error handling
                    ratio_data = {
                        'Year': year,
                        'Net Profit Margin (%)': self.safe_division(net_income, total_revenue) * 100,
                        'ROE (%)': self.safe_division(net_income, stockholder_equity) * 100,
                        'ROA (%)': self.safe_division(net_income, total_assets) * 100,
                        'Current Ratio': self.safe_division(current_assets, current_liabilities),
                        'Debt to Equity': self.safe_division(total_liabilities, stockholder_equity),
                        'Asset Turnover': self.safe_division(total_revenue, total_assets),
                        'Equity Multiplier': self.safe_division(total_assets, stockholder_equity),
                    }
                    
                    # Add additional ratios if data is available
                    if not cash_year.empty:
                        operating_cash_flow = safe_get(cash_year, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
                        ratio_data['Operating Cash Flow Ratio'] = self.safe_division(operating_cash_flow, current_liabilities)
                    
                    ratios_data.append(ratio_data)
                    
                except Exception as e:
                    logger.error(f"Error calculating ratios for year {year}: {str(e)}")
                    continue
            
            if ratios_data:
                ratios_df = pd.DataFrame(ratios_data)
                # Round all numeric columns to 2 decimal places
                numeric_columns = ratios_df.select_dtypes(include=[np.number]).columns
                ratios_df[numeric_columns] = ratios_df[numeric_columns].round(2)
                return ratios_df
            else:
                logger.warning("No ratio data could be calculated")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {str(e)}")
            return pd.DataFrame()

    @log_operation
    def get_historical_data(self, symbol: str, period: str = '1y') -> Optional[pd.DataFrame]:
        """Fetch historical price data with caching"""
        cache_key = f"historical_{period}"
        cached_data = self.cache.get(symbol, cache_key)
        if cached_data:
            return cached_data
        
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"No historical data found for {symbol} with period {period}")
                return None
            
            hist.reset_index(inplace=True)
            
            # Add technical indicators
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['Daily_Return'] = hist['Close'].pct_change()
            hist['Volatility'] = hist['Daily_Return'].rolling(window=20).std()
            
            # Cache the result
            self.cache.set(symbol, cache_key, hist)
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None

    def format_large_number(self, value: float) -> str:
        """Format large numbers with appropriate suffixes"""
        if pd.isna(value) or value == 0:
            return "0"
        
        abs_value = abs(value)
        if abs_value >= self.BILLION:
            return f"{value/self.BILLION:.2f}B"
        elif abs_value >= self.MILLION:
            return f"{value/self.MILLION:.2f}M"
        elif abs_value >= 1000:
            return f"{value/1000:.2f}K"
        else:
            return f"{value:.2f}"

    def get_company_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive company analysis"""
        try:
            stock_info = self.get_stock_info(symbol)
            if not stock_info:
                return None
            
            statements = self.get_financial_statements(symbol)
            if not statements:
                return None
            
            ratios = self.calculate_financial_ratios(statements)
            
            # Calculate additional metrics
            current_price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 0))
            market_cap = stock_info.get('marketCap', 0)
            
            analysis = {
                'basic_info': {
                    'symbol': symbol,
                    'company_name': stock_info.get('longName', 'N/A'),
                    'sector': stock_info.get('sector', 'N/A'),
                    'industry': stock_info.get('industry', 'N/A'),
                    'current_price': current_price,
                    'market_cap': market_cap,
                    'market_cap_formatted': self.format_large_number(market_cap),
                },
                'valuation_metrics': {
                    'pe_ratio': stock_info.get('trailingPE', 0),
                    'forward_pe': stock_info.get('forwardPE', 0),
                    'pb_ratio': stock_info.get('priceToBook', 0),
                    'ps_ratio': stock_info.get('priceToSalesTrailing12Months', 0),
                    'peg_ratio': stock_info.get('pegRatio', 0),
                },
                'profitability_metrics': {
                    'roe': stock_info.get('returnOnEquity', 0),
                    'roa': stock_info.get('returnOnAssets', 0),
                    'gross_margin': stock_info.get('grossMargins', 0),
                    'profit_margin': stock_info.get('profitMargins', 0),
                },
                'financial_health': {
                    'debt_to_equity': stock_info.get('debtToEquity', 0),
                    'current_ratio': stock_info.get('currentRatio', 0),
                    'quick_ratio': stock_info.get('quickRatio', 0),
                    'free_cash_flow': stock_info.get('freeCashflow', 0),
                },
                'price_metrics': {
                    'fifty_two_week_high': stock_info.get('fiftyTwoWeekHigh', 0),
                    'fifty_two_week_low': stock_info.get('fiftyTwoWeekLow', 0),
                    'beta': stock_info.get('beta', 0),
                    'dividend_yield': stock_info.get('dividendYield', 0),
                },
                'statements': statements,
                'ratios': ratios
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in company analysis for {symbol}: {str(e)}")
            return None

    def get_peer_comparison(self, symbol: str, peer_symbols: List[str]) -> Optional[pd.DataFrame]:
        """Get peer comparison data"""
        try:
            comparison_data = []
            
            # Add main company
            all_symbols = [symbol] + peer_symbols
            
            for sym in all_symbols:
                info = self.get_stock_info(sym)
                if info:
                    comparison_data.append({
                        'Symbol': sym,
                        'Company': info.get('longName', 'N/A'),
                        'Market Cap': info.get('marketCap', 0),
                        'P/E Ratio': info.get('trailingPE', 0),
                        'P/B Ratio': info.get('priceToBook', 0),
                        'ROE (%)': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                        'Debt/Equity': info.get('debtToEquity', 0),
                        'Dividend Yield (%)': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                    })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                # Format market cap
                df['Market Cap Formatted'] = df['Market Cap'].apply(self.format_large_number)
                return df
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in peer comparison: {str(e)}")
            return None

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cache cleared successfully")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache.cache),
            'cache_keys': list(self.cache.cache.keys()),
            'ttl_minutes': self.cache.ttl_minutes
        }

# Create a singleton instance
financial_processor = FinancialDataProcessor()

# Convenience functions for external use
def get_stock_analysis(symbol: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive stock analysis"""
    return financial_processor.get_company_analysis(symbol)

def get_financial_statements(symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Get financial statements for a symbol"""
    return financial_processor.get_financial_statements(symbol)

def calculate_ratios(statements: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate financial ratios from statements"""
    return financial_processor.calculate_financial_ratios(statements)

def get_historical_data(symbol: str, period: str = '1y') -> Optional[pd.DataFrame]:
    """Get historical price data"""
    return financial_processor.get_historical_data(symbol, period)

def validate_symbol(symbol: str) -> Tuple[bool, str]:
    """Validate stock symbol"""
    return financial_processor.validate_symbol(symbol)

def clear_cache():
    """Clear all cached data"""
    financial_processor.clear_cache()

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return financial_processor.get_cache_stats()
