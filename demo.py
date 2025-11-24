import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import sys
from pathlib import Path
from streamlit_searchbox import st_searchbox

# Add modules to path
sys.path.append(str(Path(__file__).parent))
from modules.utils.finnhub_client import get_finnhub_client

# Page configuration
st.set_page_config(
    page_title="AI Stock Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_demo_data(ticker, period_days=30):
    """Generate demo stock data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=period_days, freq='D')
    
    # Generate realistic price data
    base_price = 150 + np.random.randn() * 20
    returns = np.random.randn(period_days) * 0.02
    prices = base_price * np.exp(returns.cumsum())
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(period_days) * 0.01),
        'High': prices * (1 + np.abs(np.random.randn(period_days)) * 0.02),
        'Low': prices * (1 - np.abs(np.random.randn(period_days)) * 0.02),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, period_days)
    }, index=dates)
    
    return data

def generate_demo_fundamentals(ticker):
    """Generate demo fundamental data"""
    return {
        'P/E Ratio': round(np.random.uniform(15, 35), 2),
        'EPS': round(np.random.uniform(3, 15), 2),
        'ROE': round(np.random.uniform(0.1, 0.3), 3),
        'Debt/Equity': round(np.random.uniform(0.2, 1.5), 2),
        'Market Cap': f"${np.random.randint(100, 3000)}B"
    }

def search_ticker_symbols_realtime(searchterm: str) -> list:
    """
    Real-time search for ticker symbols using Finnhub API
    Returns ALL matching tickers from the market as you type
    Used with st_searchbox for instant autocomplete

    Args:
        searchterm: Search query (company name or ticker)

    Returns:
        list: List of display strings in format "Company Name (SYMBOL)"
    """
    if not searchterm or len(searchterm) < 2:
        # Return empty for short queries
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

                # Filter for common stocks only
                if ticker_type == 'Common Stock':
                    # Prefer US stocks (no dots in symbol) but show international if needed
                    is_us_stock = '.' not in symbol

                    display_text = f"{description} ({display_symbol})"
                    suggestions.append((display_text, is_us_stock))

            # Sort to prioritize US stocks first
            suggestions.sort(key=lambda x: (not x[1], x[0]))  # US stocks first, then alphabetically

            # Return just the display text
            return [text for text, _ in suggestions[:20]]  # Show top 20

        return []

    except Exception as e:
        # Return empty on error
        return []

def main():
    # Header
    st.markdown('<div style="font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;">📈 AI-Powered Stock Analysis Platform (Demo Mode)</div>', unsafe_allow_html=True)
    
    # Notice about demo mode
    st.info("🔧 **Demo Mode Active**: Using simulated data due to Yahoo Finance API rate limits. Wait 10 minutes and restart for real data.")
    
    # Sidebar
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
            # Format is "Company Name (SYMBOL)" so extract SYMBOL
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

        st.subheader("📊 Chart Settings")

        # Main time range selector with clear explanation
        analysis_period = st.selectbox(
            "Time Range",
            options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
            index=2,  # Default to 1 month
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
                index=1,  # Default to 2 days
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

        if st.button("🔄 Generate New Demo Data", type="primary", use_container_width=True):
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Technical Analysis", "📰 Sentiment Analysis"])
    
    if ticker:
        # Initialize session state for quick period selector
        if 'quick_period' not in st.session_state:
            st.session_state.quick_period = analysis_period

        # Generate demo data
        with st.spinner("Generating demo data..."):
            period_map = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "max": 365}
            demo_data = generate_demo_data(ticker, period_map.get(st.session_state.quick_period, 30))
            demo_fundamentals = generate_demo_fundamentals(ticker)

        with tab1:
            st.header("📊 Real-Time Data & Fundamentals (Demo)")

            # Quick time range selector buttons
            st.markdown("##### ⚡ Quick Time Range")
            col_buttons = st.columns([1, 1, 1, 1, 1, 1, 1, 2])

            periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "max"]
            period_labels = ["1D", "5D", "1M", "3M", "6M", "1Y", "Max"]

            for i, (period, label) in enumerate(zip(periods, period_labels)):
                with col_buttons[i]:
                    # Highlight active button
                    button_type = "primary" if st.session_state.quick_period == period else "secondary"
                    if st.button(label, key=f"period_btn_{period}", type=button_type, use_container_width=True):
                        st.session_state.quick_period = period
                        st.rerun()

            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Price Chart")
                fig = go.Figure()

                # Create custom hover text with all OHLCV data
                hover_text = []
                for i in range(len(demo_data)):
                    hover_text.append(
                        f"<b>Date:</b> {demo_data.index[i].strftime('%Y-%m-%d')}<br>"
                        f"<b>Open:</b> ${demo_data['Open'].iloc[i]:.2f}<br>"
                        f"<b>High:</b> ${demo_data['High'].iloc[i]:.2f}<br>"
                        f"<b>Low:</b> ${demo_data['Low'].iloc[i]:.2f}<br>"
                        f"<b>Close:</b> ${demo_data['Close'].iloc[i]:.2f}<br>"
                        f"<b>Volume:</b> {demo_data['Volume'].iloc[i]:,.0f}"
                    )

                fig.add_trace(go.Scatter(
                    x=demo_data.index,
                    y=demo_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2),
                    hovertext=hover_text,
                    hoverinfo='text'
                ))
                fig.update_layout(
                    title=f'{ticker} Stock Price',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Key Metrics")
                for key, value in demo_fundamentals.items():
                    st.metric(key, value)
        
        with tab2:
            st.header("📈 Technical Analysis (Demo)")

            # Quick time range selector buttons
            st.markdown("##### ⚡ Quick Time Range")
            col_buttons2 = st.columns([1, 1, 1, 1, 1, 1, 1, 2])

            for i, (period, label) in enumerate(zip(periods, period_labels)):
                with col_buttons2[i]:
                    button_type = "primary" if st.session_state.quick_period == period else "secondary"
                    if st.button(label, key=f"period_btn_tab2_{period}", type=button_type, use_container_width=True):
                        st.session_state.quick_period = period
                        st.rerun()

            st.markdown("---")

            # Calculate simple moving averages
            demo_data['SMA_20'] = demo_data['Close'].rolling(20).mean()
            demo_data['SMA_50'] = demo_data['Close'].rolling(50).mean()

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Moving Averages")
                fig = go.Figure()

                # Add Close price with detailed hover
                close_hover = []
                for i in range(len(demo_data)):
                    close_hover.append(
                        f"<b>Date:</b> {demo_data.index[i].strftime('%Y-%m-%d')}<br>"
                        f"<b>Close:</b> ${demo_data['Close'].iloc[i]:.2f}<br>"
                        f"<b>Volume:</b> {demo_data['Volume'].iloc[i]:,.0f}"
                    )

                fig.add_trace(go.Scatter(
                    x=demo_data.index,
                    y=demo_data['Close'],
                    name='Close',
                    line=dict(color='blue'),
                    hovertext=close_hover,
                    hoverinfo='text'
                ))

                # Add SMA 20
                sma20_hover = [f"<b>Date:</b> {demo_data.index[i].strftime('%Y-%m-%d')}<br>"
                              f"<b>SMA 20:</b> ${demo_data['SMA_20'].iloc[i]:.2f}"
                              if not pd.isna(demo_data['SMA_20'].iloc[i]) else "No data"
                              for i in range(len(demo_data))]

                fig.add_trace(go.Scatter(
                    x=demo_data.index,
                    y=demo_data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange'),
                    hovertext=sma20_hover,
                    hoverinfo='text'
                ))

                # Add SMA 50
                sma50_hover = [f"<b>Date:</b> {demo_data.index[i].strftime('%Y-%m-%d')}<br>"
                              f"<b>SMA 50:</b> ${demo_data['SMA_50'].iloc[i]:.2f}"
                              if not pd.isna(demo_data['SMA_50'].iloc[i]) else "No data"
                              for i in range(len(demo_data))]

                fig.add_trace(go.Scatter(
                    x=demo_data.index,
                    y=demo_data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='red'),
                    hovertext=sma50_hover,
                    hoverinfo='text'
                ))

                fig.update_layout(
                    title=f'{ticker} - Moving Averages',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Technical Signals")
                current_price = demo_data['Close'].iloc[-1]
                sma_20 = demo_data['SMA_20'].iloc[-1] if not pd.isna(demo_data['SMA_20'].iloc[-1]) else current_price
                sma_50 = demo_data['SMA_50'].iloc[-1] if not pd.isna(demo_data['SMA_50'].iloc[-1]) else current_price
                
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("SMA 20", f"${sma_20:.2f}")
                st.metric("SMA 50", f"${sma_50:.2f}")
                
                # Simple signal
                if current_price > sma_20 > sma_50:
                    st.success("📈 Bullish Signal")
                elif current_price < sma_20 < sma_50:
                    st.error("📉 Bearish Signal")
                else:
                    st.warning("➡️ Neutral Signal")
        
        with tab3:
            st.header("📰 Sentiment Analysis (Demo)")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Overall Sentiment")
                demo_sentiment = np.random.uniform(-1, 1)
                sentiment_label = "Positive" if demo_sentiment > 0.1 else "Negative" if demo_sentiment < -0.1 else "Neutral"
                
                st.metric("Sentiment Score", f"{demo_sentiment:.3f}")
                st.metric("Sentiment", sentiment_label)
            
            with col2:
                st.subheader("Demo News Headlines")
                demo_headlines = [
                    f"{ticker} reports strong quarterly earnings",
                    f"Analysts upgrade {ticker} price target",
                    f"{ticker} announces new product launch",
                    f"Market volatility affects {ticker} trading",
                    f"{ticker} CEO discusses future strategy"
                ]
                
                for i, headline in enumerate(demo_headlines):
                    sentiment_score = np.random.uniform(-0.5, 0.5)
                    with st.expander(headline):
                        st.write(f"**Sentiment:** {sentiment_score:.3f}")
                        st.write(f"**Source:** Demo News {i+1}")

if __name__ == "__main__":
    main()