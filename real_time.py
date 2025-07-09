import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any

class RealTimeFinancialDashboard:
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'figure.figsize': (12, 6), 'axes.titlesize': 16, 'axes.labelsize': 12,
            'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10
        })

    def _safe_division(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        if not isinstance(numerator, pd.Series):
            numerator = pd.Series(numerator)
        if not isinstance(denominator, pd.Series):
            denominator = pd.Series(denominator)
        num_aligned, den_aligned = numerator.align(denominator, fill_value=0)
        den_aligned = den_aligned.replace(0, np.nan)
        return num_aligned / den_aligned

    def get_screener_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        stock_symbol = symbol.upper() + '.NS' if not symbol.upper().endswith('.NS') else symbol.upper()
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        
        if not info or ('regularMarketPrice' not in info and 'currentPrice' not in info):
            return None

        try:
            # Fetch ALL years, you miserable fools
            income_statement = stock.get_financials(freq='yearly').transpose().reset_index()
            balance_sheet = stock.get_balance_sheet(freq='yearly').transpose().reset_index()
            cash_flow = stock.get_cashflow(freq='yearly').transpose().reset_index()
            if income_statement.empty or balance_sheet.empty:
                return None
        except Exception:
            return None

        # Rename 'index' to 'years' and format it, because chaos needs order
        for df in [income_statement, balance_sheet, cash_flow]:
            df.rename(columns={'index': 'years'}, inplace=True)
            df['years'] = pd.to_datetime(df['years']).dt.strftime('%Y')

        # Helper to grab a column as a Series, no mercy for missing data
        def get_series(df: pd.DataFrame, col_name: str) -> pd.Series:
            return df.set_index('years')[col_name].fillna(0) if col_name in df.columns else pd.Series(0, index=df['years'], name=col_name)

        ratios = pd.DataFrame({'years': income_statement['years']})
        net_income = get_series(income_statement, 'Net Income')
        total_revenue = get_series(income_statement, 'Total Revenue')
        stockholder_equity = get_series(balance_sheet, 'Total Stockholder Equity')
        total_assets = get_series(balance_sheet, 'Total Assets')
        current_assets = get_series(balance_sheet, 'Total Current Assets')
        current_liabilities = get_series(balance_sheet, 'Total Current Liabilities')
        total_liabilities = get_series(balance_sheet, 'Total Liab')
        
        # Ratios for every year, no holding back
        ratios['Net Profit Margin'] = self._safe_division(net_income, total_revenue)
        ratios['Return on Equity (ROE)'] = self._safe_division(net_income, stockholder_equity)
        ratios['Return on Assets (ROA)'] = self._safe_division(net_income, total_assets)
        ratios['Current Ratio'] = self._safe_division(current_assets, current_liabilities)
        ratios['Debt to Equity'] = self._safe_division(total_liabilities, stockholder_equity)
        
        for df in [ratios, income_statement, balance_sheet, cash_flow]:
            df.fillna(0, inplace=True)

        # Rest of the data grabbing stays the same—focus on the years, damn it
        ebitda = info.get('ebitda', 0)
        enterprise_value = info.get('enterpriseValue', 0)
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        prev_close = info.get('previousClose', current_price)

        data = {
            "symbol": stock_symbol, 
            "company_name": info.get('longName', 'N/A'),
            "current_price": current_price, 
            "change": current_price - prev_close,
            "change_percent": ((current_price / prev_close) - 1) * 100 if prev_close else 0,
            "market_cap": info.get('marketCap', 0), 
            "book_value": info.get('bookValue', 0),
            "dividend_yield": info.get('dividendYield', 0), 
            "pe_ratio": info.get('trailingPE', 0),
            " financials": {
                "ratios": ratios.to_dict('records'), 
                "income_statement": income_statement.to_dict('records'),
                "balance_sheet": balance_sheet.to_dict('records'), 
                "cash_flow": cash_flow.to_dict('records'),
            },
            "real_time_metrics": {
                "return_on_equity": info.get('returnOnEquity', 0), 
                "debt_to_equity": info.get('debtToEquity', 0),
                "free_cashflow": info.get('freeCashflow', 0), 
                "enterprise_value": enterprise_value,
                "ev_to_ebitda": enterprise_value / ebitda if ebitda else 0,
                "fifty_two_week_high": info.get('fiftyTwoWeekHigh', 0), 
                "fifty_two_week_low": info.get('fiftyTwoWeekLow', 0),
            }
        }
        return data

    def create_comprehensive_dashboard(self, symbol: str) -> Optional[plt.Figure]:
        data = self.get_screener_data(symbol)
        if not data: 
            return None
            
        CRORE = 1_00_00_000 
        df_income = pd.DataFrame(data['financials']['income_statement']).set_index('years')
        df_bs = pd.DataFrame(data['financials']['balance_sheet']).set_index('years')
        df_cf = pd.DataFrame(data['financials']['cash_flow']).set_index('years')
        df_ratios = pd.DataFrame(data['financials']['ratios']).set_index('years')

        # Sort by year so we see the full history, you blind bat
        for df in [df_income, df_bs, df_cf, df_ratios]: 
            df.sort_index(inplace=True)

        def get_plot_series(df: pd.DataFrame, col_name: str) -> pd.Series:
            return df[col_name].fillna(0) if col_name in df.columns else pd.Series(0, index=df.index, name=col_name)

        # Plot every damn year—revenue, income, assets, liabilities, cash flow, ratios
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Financial Health of {data["company_name"]}', fontsize=20, y=1.02)
        
        revenue_series = get_plot_series(df_income, 'Total Revenue') / CRORE
        net_income_series = get_plot_series(df_income, 'Net Income') / CRORE
        axes[0, 0].bar(df_income.index, revenue_series, label='Total Revenue (Cr)', color='skyblue')
        axes[0, 0].plot(df_income.index, net_income_series, label='Net Income (Cr)', marker='o', color='crimson', linewidth=2.5)
        axes[0, 0].set_title("Revenue & Net Income Trend")
        axes[0, 0].set_ylabel("Amount (in Cr)")
        axes[0, 0].legend()

        assets_series = get_plot_series(df_bs, 'Total Assets') / CRORE
        liabilities_series = get_plot_series(df_bs, 'Total Liab') / CRORE
        axes[0, 1].plot(df_bs.index, assets_series, label='Total Assets (Cr)', marker='o', color='darkgreen')
        axes[0, 1].plot(df_bs.index, liabilities_series, label='Total Liabilities (Cr)', marker='^', color='orangered')
        axes[0, 1].set_title("Assets vs. Liabilities")
        axes[0, 1].set_ylabel("Amount (in Cr)")
        axes[0, 1].legend()

        op_cash_flow = get_plot_series(df_cf, 'Total Cash From Operating Activities') / CRORE
        colors = ['salmon' if val < 0 else 'seagreen' for val in op_cash_flow]
        axes[1, 0].bar(op_cash_flow.index, op_cash_flow, label='Operating Cash Flow (Cr)', color=colors)
        axes[1, 0].set_title("Operating Cash Flow")
        axes[1, 0].set_ylabel("Amount (in Cr)")
        axes[1, 0].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[1, 0].legend()

        debt_equity = get_plot_series(df_ratios, 'Debt to Equity')
        axes[1, 1].plot(df_ratios.index, debt_equity, label='Debt-to-Equity Ratio', marker='s', color='purple')
        axes[1, 1].set_title("Debt-to-Equity Ratio")
        axes[1, 1].set_ylabel("Ratio")
        axes[1, 1].legend()

        for ax in axes.flat: 
            ax.tick_params(axis='x', rotation=45)
            
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return fig
