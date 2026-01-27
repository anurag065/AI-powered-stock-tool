import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
from streamlit_searchbox import st_searchbox

# Disable tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from modules.data_module.data_fetcher import DataFetcher
from modules.indicator_dashboard.technical_analysis import TechnicalAnalysis
from modules.sentiment_module.sentiment_analyzer import SentimentAnalyzer
from modules.chatbot_module.rag_chatbot import EnhancedOpenAIChatbot
from modules.utils.finnhub_client import get_finnhub_client
from utils.database import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="AI Stock Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def search_ticker_symbols_realtime(searchterm: str) -> list:
    """
    Real-time search for ticker symbols using Finnhub API
    Returns ALL matching tickers from the market as you type
    """
    if not searchterm or len(searchterm) < 2:
        return []

    try:
        client = get_finnhub_client()
        result = client.symbol_lookup(searchterm)

        if result and 'result' in result and result.get('count', 0) > 0:
            suggestions = []

            for item in result['result']:
                symbol = item.get('symbol', '')
                description = item.get('description', '')
                ticker_type = item.get('type', '')
                display_symbol = item.get('displaySymbol', symbol)

                if ticker_type == 'Common Stock':
                    is_us_stock = '.' not in symbol
                    display_text = f"{description} ({display_symbol})"
                    suggestions.append((display_text, is_us_stock))

            suggestions.sort(key=lambda x: (not x[1], x[0]))
            return [text for text, _ in suggestions[:20]]

        return []

    except Exception as e:
        return []

def main():
    # Header
    st.markdown('<div class="main-header">📈 AI-Powered Stock Analysis Platform</div>', unsafe_allow_html=True)
    
    # Sidebar for ticker input
    with st.sidebar:
        st.title("🔧 Configuration")

        # Stock ticker search with real-time autocomplete
        st.markdown("### 🔍 Search Stock")

        selected = st_searchbox(
            search_ticker_symbols_realtime,
            placeholder="Type to search stocks (e.g., AAPL, Tesla, Microsoft...)",
            label="Search by company name or ticker symbol",
            default="Apple Inc (AAPL)",
            clear_on_submit=False,
            key="ticker_searchbox"
        )

        # Extract ticker symbol from selected option
        if selected:
            if '(' in selected and ')' in selected:
                ticker = selected.split('(')[-1].split(')')[0].strip()
                st.markdown(f"**✓ Selected:** `{ticker}`")
            else:
                ticker = selected
                st.markdown(f"**✓ Selected:** `{ticker}`")
        else:
            ticker = "AAPL"
            st.info("👆 Start typing to search for stocks")

        st.divider()

        # Chart settings
        st.subheader("📊 Chart Settings")

        analysis_period = st.selectbox(
            "Time Range",
            options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
            index=2,
            help="📈 Select how much historical data to display on charts.\n\n"
                 "• 1d = Last 1 day (intraday)\n"
                 "• 5d = Last 5 days\n"
                 "• 1mo = Last month\n"
                 "• 3mo = Last 3 months\n"
                 "• 6mo = Last 6 months\n"
                 "• 1y = Last year",
            key="time_range"
        )

        # Advanced settings in expander
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.caption("Technical settings for data management")

            data_days = st.selectbox(
                "Data Retention",
                options=[1, 2, 7, 30],
                index=1,
                help="🗄️ How long to keep cached data on disk.\n\n"
                     "Lower values save storage space but require more API calls.\n"
                     "Higher values cache more data for faster access.\n\n"
                     "⚠️ Note: Must be ≥ selected time range to avoid missing data.",
                key="data_retention"
            )

            # Validation warning
            period_days_map = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
            required_days = period_days_map.get(analysis_period, 30)

            if data_days < required_days:
                st.warning(
                    f"⚠️ Data retention ({data_days} days) is less than time range "
                    f"({analysis_period} = ~{required_days} days). Some data may be missing.",
                    icon="⚠️"
                )

        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared! Data will refresh on next load.")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Technical Analysis", "📰 Sentiment Analysis", "🤖 AI Chatbot"])
    
    if ticker:
        # Initialize session state for quick period selector
        if 'quick_period' not in st.session_state:
            st.session_state.quick_period = analysis_period

        try:
            # Initialize components
            data_fetcher = DataFetcher()

            with tab1:
                st.header("📊 Real-Time Data & Fundamentals")
                display_data_overview(ticker, st.session_state.quick_period, data_days)
              
            
            with tab2:
                st.header("📈 Technical Indicators Dashboard")
                display_technical_analysis(ticker, analysis_period)
            
            with tab3:
                st.header("📰 Sentiment Analysis")
                display_sentiment_analysis(ticker)
            
            with tab4:
                st.header("🤖 AI-Powered Q&A Chatbot")
                display_chatbot(ticker)

            # Handle chat input outside of tabs (required by Streamlit)
            handle_chat_input(ticker)

        except Exception as e:
            st.error(f"Error loading data for {ticker}: {str(e)}")
            st.info("Please check if the ticker symbol is valid.")

# def display_data_overview(data_fetcher, ticker, period):
#     """Display real-time data and fundamental metrics"""
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.subheader("Price Chart")
#         try:
#             # Fetch and display price data
#             stock_data = data_fetcher.get_stock_data(ticker, period)
#             if not stock_data.empty:
#                 st.line_chart(stock_data['Close'])
#             else:
#                 st.warning("No data available for this ticker.")
#         except Exception as e:
#             st.error(f"Error fetching stock data: {str(e)}")
    
#     with col2:
#         st.subheader("Key Metrics")
#         try:
#             # Fetch fundamental data
#             fundamentals = data_fetcher.get_fundamentals(ticker)
#             if fundamentals:
#                 for key, value in fundamentals.items():
#                     st.metric(key, value)
#             else:
#                 st.info("Fundamental data not available.")
#         except Exception as e:
#             st.error(f"Error fetching fundamentals: {str(e)}")


def display_data_overview(selected_ticker, selected_period, data_retention_days):
    """
    Display real-time data and fundamentals with FIXED function signature
    Args:
        selected_ticker: Stock ticker symbol
        selected_period: Analysis period (1d, 5d, 1mo, etc.)
        data_retention_days: Cache retention setting
    """
    # Quick time range selector buttons
    st.markdown("##### ⚡ Quick Time Range")
    col_buttons = st.columns([1, 1, 1, 1, 1, 1, 2])

    periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y"]
    period_labels = ["1D", "5D", "1M", "3M", "6M", "1Y"]

    for i, (period, label) in enumerate(zip(periods, period_labels)):
        with col_buttons[i]:
            button_type = "primary" if st.session_state.quick_period == period else "secondary"
            if st.button(label, key=f"period_btn_tab1_{period}", type=button_type, use_container_width=True):
                st.session_state.quick_period = period
                st.rerun()

    st.markdown("---")

    try:
        from modules.data_module.data_fetcher import DataFetcher

        # Use the data_retention_days parameter
        data_fetcher = DataFetcher(cache_days=data_retention_days)

        # Fetch stock data
        with st.spinner(f"Fetching data for {selected_ticker}..."):
            stock_data = data_fetcher.get_stock_data(selected_ticker, selected_period)
            fundamentals = data_fetcher.get_fundamentals(selected_ticker)
            realtime_price = data_fetcher.get_realtime_price(selected_ticker)

        if not stock_data.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Price Chart")

                # Create FIXED price chart with OHLCV hover tooltips
                import plotly.graph_objects as go

                fig = go.Figure()

                # Determine if this is intraday data (1d or 5d periods)
                is_intraday = selected_period in ['1d', '5d']

                # Create custom hover text with all OHLCV data
                hover_text = []
                for i in range(len(stock_data)):
                    # Use different date format for intraday vs daily data
                    if is_intraday:
                        date_str = stock_data.index[i].strftime('%Y-%m-%d %H:%M')
                        date_label = "Time"
                    else:
                        date_str = stock_data.index[i].strftime('%Y-%m-%d')
                        date_label = "Date"

                    hover_text.append(
                        f"<b>{date_label}:</b> {date_str}<br>"
                        f"<b>Open:</b> ${stock_data['Open'].iloc[i]:.2f}<br>"
                        f"<b>High:</b> ${stock_data['High'].iloc[i]:.2f}<br>"
                        f"<b>Low:</b> ${stock_data['Low'].iloc[i]:.2f}<br>"
                        f"<b>Close:</b> ${stock_data['Close'].iloc[i]:.2f}<br>"
                        f"<b>Volume:</b> {stock_data['Volume'].iloc[i]:,.0f}"
                    )

                # Add line chart for all periods
                if len(stock_data) > 1:
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name=f'{selected_ticker} Price',
                        line=dict(color='#00d4aa', width=2),
                        hovertext=hover_text,
                        hoverinfo='text'
                    ))
                    
                
                # Update layout with proper scaling
                fig.update_layout(
                    title=f'{selected_ticker} Stock Price ({selected_period.upper()})',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    template='plotly_dark',
                    height=400,
                    showlegend=False,
                    hovermode='x',  # Show tooltip for nearest x-value
                    hoverdistance=-1,  # Show tooltip at any vertical distance
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        showspikes=True,  # Show vertical spike line on hover
                        spikecolor='#00d4aa',
                        spikethickness=1,
                        spikedash='dot',
                        spikemode='across'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(128, 128, 128, 0.2)',
                        autorange=True,
                        fixedrange=False,
                        showspikes=True,  # Show horizontal spike line on hover
                        spikecolor='#00d4aa',
                        spikethickness=1,
                        spikedash='dot',
                        spikemode='across'
                    )
                )
                
                # FIXED: Ensure proper price scaling
                if len(stock_data) > 0:
                    min_price = stock_data['Close'].min()
                    max_price = stock_data['Close'].max()
                    latest_price = stock_data['Close'].iloc[-1]
                    
                    # Force Y-axis range to show price movement
                    price_range = max_price - min_price
                    if price_range > 0:
                        margin = price_range * 0.05  # 5% margin
                        fig.update_layout(
                            yaxis=dict(
                                range=[min_price - margin, max_price + margin]
                            )
                        )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Debug info (can remove later if needed)
                if st.checkbox("🔍 Show Debug Info"):
                    st.write(f"**Data Points:** {len(stock_data)}")
                    if len(stock_data) > 0:
                        st.write(f"**Price Range:** ${min_price:.2f} - ${max_price:.2f}")
                        st.write(f"**Latest Price:** ${latest_price:.2f}")
                        st.write(f"**Date Range:** {stock_data.index.min().date()} to {stock_data.index.max().date()}")
                        st.write(f"**Cache Days:** {data_retention_days}")
                    
                    # Show sample data
                    st.dataframe(stock_data.head(3))
            
            with col2:
                st.subheader("Key Metrics")
                
                # Display fundamentals
                if fundamentals:
                    pe_ratio = fundamentals.get('P/E Ratio', 'N/A')
                    eps = fundamentals.get('EPS', 'N/A')
                    roe = fundamentals.get('ROE', 'N/A')
                    debt_equity = fundamentals.get('Debt/Equity', 'N/A')
                    market_cap = fundamentals.get('Market Cap', 'N/A')
                    
                    # Format market cap
                    if isinstance(market_cap, (int, float)) and market_cap != 'N/A':
                        if market_cap >= 1e12:
                            market_cap_formatted = f"{market_cap/1e12:.1f}T"
                        elif market_cap >= 1e9:
                            market_cap_formatted = f"{market_cap/1e9:.1f}B"
                        elif market_cap >= 1e6:
                            market_cap_formatted = f"{market_cap/1e6:.1f}M"
                        else:
                            market_cap_formatted = f"{market_cap:,.0f}"
                    else:
                        market_cap_formatted = str(market_cap)
                    
                    # Display metrics in a clean grid
                    st.metric("P/E Ratio", pe_ratio)
                    st.metric("EPS", f"${eps}" if isinstance(eps, (int, float)) and eps != 'N/A' else eps)
                    st.metric("ROE", f"{roe:.2f}%" if isinstance(roe, (int, float)) and roe != 'N/A' else roe)
                    st.metric("Debt/Equity", debt_equity)
                    st.metric("Market Cap", market_cap_formatted)
                
                # Real-time price info
                if realtime_price:
                    st.markdown("---")
                    st.subheader("Live Price")
                    
                    price = realtime_price.get('price', 0)
                    change = realtime_price.get('change', 0)
                    change_percent = realtime_price.get('change_percent', 0)
                    
                    st.metric(
                        "Current Price",
                        f"${price:.2f}",
                        f"{change:+.2f} ({change_percent:+.2f}%)"
                    )
                else:
                    st.info("Real-time price data not available")
        
        else:
            st.error(f"No data available for ticker '{selected_ticker}'. Please check the ticker symbol and try again.")
            st.info("""
            **Common issues:**
            - Ticker might be delisted or invalid
            - Market might be closed
            - API rate limits reached
            
            **Try:**
            - Different ticker (AAPL, MSFT, GOOGL)
            - Different time period
            - Refresh the data
            """)
            
    except Exception as e:
        st.error(f"Error loading data for {selected_ticker}: {str(e)}")
        
        # Show helpful debug info
        st.info(f"""
        **Troubleshooting:**
        - Check if ticker '{selected_ticker}' is valid
        - Verify internet connection  
        - Try a different analysis period
        - Current period: {selected_period}
        - Cache setting: {data_retention_days} days
        """)

# def display_technical_analysis(ticker, period):
#     """Display technical analysis indicators"""
#     try:
#         technical_analyzer = TechnicalAnalysis()
#         indicators = technical_analyzer.calculate_indicators(ticker, period)
        
#         if indicators:
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("Moving Averages")
#                 if 'SMA_50' in indicators and 'SMA_200' in indicators:
#                     st.plotly_chart(indicators['ma_chart'], use_container_width=True)
            
#             with col2:
#                 st.subheader("RSI & MACD")
#                 if 'RSI' in indicators:
#                     st.plotly_chart(indicators['rsi_chart'], use_container_width=True)
#         else:
#             st.warning("Unable to calculate technical indicators for this ticker.")
            
#     except Exception as e:
#         st.error(f"Error in technical analysis: {str(e)}")

def display_technical_analysis(ticker, period):
    """Display technical analysis with real SEC financial data - CORRECTED IMPORTS"""
    try:
        technical_analyzer = TechnicalAnalysis()
        indicators = technical_analyzer.calculate_indicators(ticker, period)
        
        if indicators:
            # Create only 2 tabs as requested
            tech_tab1, tech_tab2 = st.tabs(["📈 Technical Indicators", "📄 SEC Financial Data"])
            
            with tech_tab1:
                # Technical indicators charts (same as before)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Moving Averages")
                    if 'ma_chart' in indicators:
                        st.plotly_chart(indicators['ma_chart'], use_container_width=True)
                
                with col2:
                    st.subheader("RSI")
                    if 'rsi_chart' in indicators:
                        st.plotly_chart(indicators['rsi_chart'], use_container_width=True)
                
                # MACD and Bollinger Bands
                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("MACD")
                    if 'macd_chart' in indicators:
                        st.plotly_chart(indicators['macd_chart'], use_container_width=True)
                
                with col4:
                    st.subheader("Bollinger Bands")
                    if 'bollinger_chart' in indicators:
                        st.plotly_chart(indicators['bollinger_chart'], use_container_width=True)
                
                # Quick signals summary
                st.markdown("---")
                st.subheader("📊 Quick Signals Summary")
                
                signals = technical_analyzer.get_signal_analysis(ticker, period)
                if signals:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        ma_signal = signals.get('MA_Signal', 'N/A')
                        if "Bullish" in ma_signal:
                            st.success(f"🟢 **MA:** Bullish")
                        elif "Bearish" in ma_signal:
                            st.error(f"🔴 **MA:** Bearish")
                        else:
                            st.info(f"ℹ️ **MA:** {ma_signal}")
                    
                    with col2:
                        rsi_signal = signals.get('RSI_Signal', 'N/A')
                        if "Overbought" in rsi_signal:
                            st.warning(f"🟡 **RSI:** Overbought")
                        elif "Oversold" in rsi_signal:
                            st.success(f"🟢 **RSI:** Oversold")
                        else:
                            st.info(f"ℹ️ **RSI:** Neutral")
                    
                    with col3:
                        macd_signal = signals.get('MACD_Signal', 'N/A')
                        if "Bullish" in macd_signal:
                            st.success(f"🟢 **MACD:** Bullish")
                        elif "Bearish" in macd_signal:
                            st.error(f"🔴 **MACD:** Bearish")
                        else:
                            st.info(f"ℹ️ **MACD:** {macd_signal}")
                    
                    with col4:
                        bb_signal = signals.get('BB_Signal', 'N/A')
                        if "Overbought" in bb_signal:
                            st.warning(f"🟡 **BB:** Overbought")
                        elif "Oversold" in bb_signal:
                            st.success(f"🟢 **BB:** Oversold")
                        else:
                            st.info(f"ℹ️ **BB:** Normal")
            
            with tech_tab2:
                # Real SEC Financial Data section
                st.subheader("📄 Real SEC Financial Data")
                
                # Fixed import path based on your file structure
                try:
                    import sys
                    from pathlib import Path
                    
                    # Add the correct path where sec_real_data_fetcher.py is located
                    current_dir = Path(__file__).parent
                    modules_utils_path = current_dir / "modules" / "utils"
                    
                    # Try different possible paths
                    possible_paths = [
                        modules_utils_path,
                        current_dir / "utils",
                        Path(__file__).parent / "modules" / "utils",
                        Path(__file__).parent.parent / "modules" / "utils"
                    ]
                    
                    sec_integration = None
                    for path in possible_paths:
                        if str(path) not in sys.path:
                            sys.path.append(str(path))
                        
                        try:
                            from sec_real_data_fetcher import RealSECIntegration
                            sec_integration = RealSECIntegration()
                            break
                        except ImportError:
                            continue
                    
                    if sec_integration is None:
                        # Try direct import assuming it's in the same directory
                        exec(open('modules/utils/sec_real_data_fetcher.py').read(), globals())
                        sec_integration = RealSECIntegration()
                    
                    # Fetch real financial data
                    with st.spinner(f"Fetching real SEC financial data for {ticker}..."):
                        sec_data = sec_integration.get_real_sec_data(ticker)
                    
                    if sec_data and sec_data.get('has_data'):
                        # Display annual data
                        annual_data = sec_data.get('annual_summary')
                        quarterly_data = sec_data.get('quarterly_summary')
                        
                        if annual_data:
                            st.markdown("**📋 10-K Annual Report (Real Data)**")
                            st.success(f"""
                            **Fiscal Year:** {annual_data['fiscal_year']}
                            **Filed:** {annual_data['filed_date']}
                            **Revenue:** {annual_data['revenue']} (Growth: {annual_data['revenue_growth']})
                            **Net Income:** {annual_data['net_income']} | **EPS:** {annual_data['eps']}
                            """)
                            
                            # Key financial metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                revenue_value = annual_data['revenue']
                                growth_value = annual_data['revenue_growth']
                                st.metric("Annual Revenue", revenue_value, growth_value)
                            
                            with col2:
                                net_income = annual_data['net_income']
                                st.metric("Net Income", net_income)
                            
                            with col3:
                                total_assets = annual_data['total_assets']
                                st.metric("Total Assets", total_assets)
                            
                            with col4:
                                cash = annual_data['cash']
                                st.metric("Cash & Equivalents", cash)
                            
                            # Additional metrics
                            col5, col6 = st.columns(2)
                            with col5:
                                eps = annual_data['eps']
                                st.metric("Earnings Per Share", eps)
                            
                            with col6:
                                debt_to_equity = annual_data['debt_to_equity']
                                st.metric("Debt-to-Equity Ratio", debt_to_equity)
                        
                        # Display quarterly data
                        if quarterly_data:
                            st.markdown("---")
                            st.markdown("**📊 10-Q Quarterly Report (Real Data)**")
                            st.info(f"""
                            **Period:** {quarterly_data['fiscal_period']} {quarterly_data['fiscal_year']}
                            **Filed:** {quarterly_data['filed_date']}
                            **Quarterly Revenue:** {quarterly_data['revenue']} (YoY Growth: {quarterly_data['revenue_growth_yoy']})
                            **Quarterly Net Income:** {quarterly_data['net_income']} | **EPS:** {quarterly_data['eps']}
                            """)
                            
                            # Quarterly metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                q_revenue = quarterly_data['revenue']
                                q_growth = quarterly_data['revenue_growth_yoy']
                                st.metric("Quarterly Revenue", q_revenue, q_growth)
                            
                            with col2:
                                q_net_income = quarterly_data['net_income']
                                st.metric("Quarterly Net Income", q_net_income)
                            
                            with col3:
                                q_eps = quarterly_data['eps']
                                st.metric("Quarterly EPS", q_eps)
                        
                        # Data source information
                        st.markdown("---")
                        st.success("✅ **Data Source:** Real SEC EDGAR API")
                        st.markdown("**📊 Key Financial Insights:**")
                        
                        # Create insights based on real data
                        insights_col1, insights_col2, insights_col3 = st.columns(3)
                        
                        with insights_col1:
                            st.metric("Latest Filing", "10-K Annual", "Current Year")
                        
                        with insights_col2:
                            if annual_data and quarterly_data:
                                st.metric("Data Coverage", "Annual + Quarterly", "✅")
                            elif annual_data:
                                st.metric("Data Coverage", "Annual Only", "📋")
                            else:
                                st.metric("Data Coverage", "Limited", "⚠️")
                        
                        with insights_col3:
                            st.metric("Data Freshness", "Real-time", "🔄")
                        
                        # Additional information
                        with st.expander("📈 About This Real Financial Data", expanded=False):
                            st.markdown("""
                            **Data Source:** Official SEC EDGAR API
                            
                            **What You're Seeing:**
                            - **Revenue & Growth:** Actual reported revenue with year-over-year growth calculations
                            - **Profitability:** Real net income and earnings per share from official filings
                            - **Financial Position:** Total assets, cash position, and debt ratios
                            - **Quarterly Trends:** Most recent quarterly performance vs. same quarter previous year
                            
                            **Key Metrics Explained:**
                            - **Revenue Growth:** Percentage change from previous period
                            - **EPS:** Earnings Per Share (Net Income ÷ Shares Outstanding)
                            - **Debt-to-Equity:** Total Liabilities ÷ Stockholders' Equity
                            - **Total Assets:** Sum of all company assets from balance sheet
                            
                            **Note:** All financial data is extracted directly from SEC EDGAR filings (10-K annual reports and 10-Q quarterly reports) and automatically calculated without hardcoded values.
                            """)
                    
                    elif sec_data and not sec_data.get('has_data'):
                        st.warning("⚠️ Unable to fetch real SEC financial data")
                        error_msg = sec_data.get('error', 'Unknown error')
                        st.error(f"**Error:** {error_msg}")
                        
                        st.info(f"""
                        **Possible reasons:**
                        - **Ticker not found:** {ticker} may not be in SEC database
                        - **No recent filings:** Company may not have filed 10-K/10-Q recently
                        - **API rate limiting:** SEC API may be temporarily rate-limited
                        - **Data format changes:** SEC may have updated their data structure
                        
                        **Supported companies:** Major US public companies that file with SEC
                        **Try these tickers:** AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA
                        """)
                    
                    else:
                        st.info("🔄 Loading real SEC financial data...")
                        st.markdown("*This may take a few moments for the first request...*")
                
                except Exception as e:
                    st.error(f"❌ Error loading SEC data module: {str(e)}")
                    st.info(f"""
                    **Debug Info:** 
                    - **Error Type:** {type(e).__name__}
                    - **File Location:** Looking for sec_real_data_fetcher.py in modules/utils/
                    - **Current Working Directory:** {Path.cwd()}
                    
                    **Quick Fix:** 
                    Make sure `sec_real_data_fetcher.py` is in the `modules/utils/` directory and restart the app.
                    """)
                    
        else:
            st.warning("Unable to calculate technical indicators for this ticker.")
            
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")

def display_sentiment_analysis(ticker):
    """Enhanced sentiment analysis display with time-weighted scoring and better UI"""
    try:
        from modules.sentiment_module.sentiment_analyzer import SentimentAnalyzer
        
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_data = sentiment_analyzer.analyze_ticker_sentiment(ticker)
        
        if sentiment_data:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📊 Overall Sentiment")
                
                weighted_sentiment = sentiment_data.get('weighted_sentiment', 0)
                traditional_sentiment = sentiment_data.get('overall_sentiment', 0)
                sentiment_label = sentiment_data.get('sentiment_label', 'Neutral')
                
                st.metric(
                    "Current Sentiment Score", 
                    f"{weighted_sentiment:.3f}",
                    help="Time-weighted sentiment (recent news has more impact)"
                )
                
                if 'Very Positive' in sentiment_label:
                    st.success(f"🟢 **{sentiment_label}**")
                elif 'Slightly Positive' in sentiment_label:
                    st.success(f"🟡 **{sentiment_label}**")
                elif 'Neutral' in sentiment_label:
                    st.info(f"⚪ **{sentiment_label}**")
                elif 'Slightly Negative' in sentiment_label:
                    st.warning(f"🟠 **{sentiment_label}**")
                elif 'Very Negative' in sentiment_label:
                    st.error(f"🔴 **{sentiment_label}**")
                
                st.caption(f"Traditional Score: {traditional_sentiment:.3f}")
                
                st.markdown("---")
                st.subheader("📈 Sentiment Breakdown")
                
                very_positive = sentiment_data.get('very_positive_count', 0)
                slightly_positive = sentiment_data.get('slightly_positive_count', 0)
                neutral = sentiment_data.get('neutral_count', 0)
                slightly_negative = sentiment_data.get('slightly_negative_count', 0)
                very_negative = sentiment_data.get('very_negative_count', 0)
                total = sentiment_data.get('total_articles', 1)
                
                col_metrics = st.columns(3)
                with col_metrics[0]:
                    st.metric("Positive", very_positive + slightly_positive, f"{((very_positive + slightly_positive)/total)*100:.0f}%")
                with col_metrics[1]:
                    st.metric("Neutral", neutral, f"{(neutral/total)*100:.0f}%")
                with col_metrics[2]:
                    st.metric("Negative", slightly_negative + very_negative, f"{((slightly_negative + very_negative)/total)*100:.0f}%")
                
                with st.expander("🕒 Time Weighting Info", expanded=False):
                    st.markdown("""
                    **How Time Weighting Works:**
                    - Today's news: **100%** weight
                    - Yesterday's news: **80%** weight  
                    - Last week's news: **30%** weight
                    - Last month's news: **10%** weight
                    - Older news: **5%** weight
                    
                    This ensures recent market sentiment is prioritized over older news.
                    """)
            
            with col2:
                st.subheader("📰 Recent News Headlines")
                
                headlines = sentiment_data.get('headlines', [])
                if headlines:
                    st.markdown("**Filter by Sentiment:**")
                    filter_col1, filter_col2 = st.columns(2)
                    
                    with filter_col1:
                        show_positive = st.checkbox("Positive News", value=True)
                        show_neutral = st.checkbox("Neutral News", value=True)
                    
                    with filter_col2:
                        show_negative = st.checkbox("Negative News", value=True)
                        sort_by_time = st.checkbox("Sort by Time Weight", value=True)
                    
                    filtered_headlines = []
                    for headline in headlines:
                        sentiment_label = headline.get('sentiment_label', 'Neutral').lower()
                        
                        include = False
                        if show_positive and ('positive' in sentiment_label):
                            include = True
                        elif show_neutral and ('neutral' in sentiment_label):
                            include = True
                        elif show_negative and ('negative' in sentiment_label):
                            include = True
                        
                        if include:
                            filtered_headlines.append(headline)
                    
                    if sort_by_time:
                        filtered_headlines.sort(key=lambda x: x.get('time_weight', 0), reverse=True)
                    else:
                        filtered_headlines.sort(key=lambda x: x.get('published', datetime.min), reverse=True)
                    
                    for i, headline in enumerate(filtered_headlines[:50], 1):
                        formatted_article = sentiment_analyzer.format_article_with_link(headline)
                        
                        with st.expander(f"{i}. {formatted_article['title'][:80]}..." if len(formatted_article['title']) > 80 else f"{i}. {formatted_article['title']}"):
                            
                            detail_col1, detail_col2 = st.columns(2)
                            
                            with detail_col1:
                                sentiment_label = formatted_article['sentiment_label']
                                if 'Very Positive' in sentiment_label:
                                    st.markdown(f"**Sentiment:** :green[{sentiment_label}] ")
                                elif 'Slightly Positive' in sentiment_label:
                                    st.markdown(f"**Sentiment:** :green[{sentiment_label}] ")
                                elif 'Neutral' in sentiment_label:
                                    st.markdown(f"**Sentiment:** :gray[{sentiment_label}] ")
                                elif 'Slightly Negative' in sentiment_label:
                                    st.markdown(f"**Sentiment:** :orange[{sentiment_label}] ")
                                elif 'Very Negative' in sentiment_label:
                                    st.markdown(f"**Sentiment:** :red[{sentiment_label}] ")
                                
                                st.markdown(f"**Source:** {formatted_article['source']}")
                            
                            with detail_col2:
                                st.markdown(f"**Published:** {formatted_article['published']}")
                                st.markdown(f"**Time Weight:** {formatted_article['time_weight']}")
                            
                            st.markdown("---")
                            st.markdown(formatted_article['read_full_link'], unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.info(f"📊 Showing {len(filtered_headlines)} of {len(headlines)} articles • Time-weighted sentiment analysis active")
                else:
                    st.warning("No recent news headlines found for sentiment analysis.")
                    st.info("""
                    **Possible reasons:**
                    - No recent news available for this ticker
                    - News APIs temporarily unavailable
                    - Ticker symbol might not have recent coverage
                    
                    **Try:**
                    - Different ticker symbol (AAPL, MSFT, GOOGL)
                    - Refresh the page
                    - Check back later for updated news
                    """)
        else:
            st.warning(f"No sentiment data available for {ticker}")
            st.info("""
            **Unable to analyze sentiment for this ticker.**
            
            This could be because:
            - No recent news articles found
            - Ticker symbol not recognized
            - Sentiment analysis APIs temporarily unavailable
            
            **Try these popular tickers:**
            AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, NFLX
            """)
            
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")

def display_chatbot(ticker):
    """Display AI chatbot interface"""
    try:
        # Initialize chatbot
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = EnhancedOpenAIChatbot()
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

    except Exception as e:
        st.error(f"Error in chatbot: {str(e)}")


def handle_chat_input(ticker):
    """Handle chat input separately (must be outside tabs)"""
    try:
        # Chat input (must be at root level, not in tabs)
        user_input = st.chat_input("Ask anything about the stock analysis...")
        
        if user_input:
            # Initialize chatbot if not exists
            if 'chatbot' not in st.session_state:
                st.session_state.chatbot = EnhancedOpenAIChatbot()

            # Initialize chat history if not exists
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get bot response
            with st.spinner("Analyzing..."):
                response = st.session_state.chatbot.get_response(user_input, ticker)
            
            # Add bot response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to display new messages
            st.rerun()
            
    except Exception as e:
        st.error(f"Error in chatbot: {str(e)}")

if __name__ == "__main__":
    main()