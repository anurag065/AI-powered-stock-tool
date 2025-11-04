import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

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

def main():
    # Header
    st.markdown('<div style="font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem;">📈 AI-Powered Stock Analysis Platform (Demo Mode)</div>', unsafe_allow_html=True)
    
    # Notice about demo mode
    st.info("🔧 **Demo Mode Active**: Using simulated data due to Yahoo Finance API rate limits. Wait 10 minutes and restart for real data.")
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 Configuration")
        ticker = st.text_input("Enter Stock Ticker", value="AAPL", help="Demo mode - any ticker will work")
        
        st.subheader("Data Settings")
        data_days = st.selectbox("Data Retention (Days)", [1, 2], index=1)
        analysis_period = st.selectbox("Analysis Period", ["1d", "5d", "1mo", "3mo"], index=2)
        
        if st.button("🔄 Generate New Demo Data", type="primary"):
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Technical Analysis", "📰 Sentiment Analysis"])
    
    if ticker:
        # Generate demo data
        with st.spinner("Generating demo data..."):
            period_map = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90}
            demo_data = generate_demo_data(ticker, period_map.get(analysis_period, 30))
            demo_fundamentals = generate_demo_fundamentals(ticker)
        
        with tab1:
            st.header("📊 Real-Time Data & Fundamentals (Demo)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=demo_data.index,
                    y=demo_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title=f'{ticker} Stock Price',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Key Metrics")
                for key, value in demo_fundamentals.items():
                    st.metric(key, value)
        
        with tab2:
            st.header("📈 Technical Analysis (Demo)")
            
            # Calculate simple moving averages
            demo_data['SMA_20'] = demo_data['Close'].rolling(20).mean()
            demo_data['SMA_50'] = demo_data['Close'].rolling(50).mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Moving Averages")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=demo_data.index, y=demo_data['Close'], name='Close', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=demo_data.index, y=demo_data['SMA_20'], name='SMA 20', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=demo_data.index, y=demo_data['SMA_50'], name='SMA 50', line=dict(color='red')))
                fig.update_layout(title=f'{ticker} - Moving Averages', height=400)
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