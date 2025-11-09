import streamlit as st
import sys
import os
from pathlib import Path

# Disable tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from modules.data_module.data_fetcher import DataFetcher
from modules.indicator_dashboard.technical_analysis import TechnicalAnalysis
from modules.sentiment_module.sentiment_analyzer import SentimentAnalyzer
from modules.chatbot_module.rag_chatbot import RAGChatbot
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

def main():
    # Header
    st.markdown('<div class="main-header">📈 AI-Powered Stock Analysis Platform</div>', unsafe_allow_html=True)
    
    # Sidebar for ticker input
    with st.sidebar:
        st.title("🔧 Configuration")
        ticker = st.text_input("Enter Stock Ticker", value="AAPL", help="Enter a valid stock ticker symbol")
        
        # Data retention settings
        st.subheader("Data Settings")
        data_days = st.selectbox("Data Retention (Days)", [1, 2], index=1, 
                                help="Number of days of stock data to store")
        
        # Analysis period
        analysis_period = st.selectbox("Analysis Period", 
                                     ["1d", "5d", "1mo", "3mo", "6mo", "1y"], 
                                     index=2)
        
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.success("Cache cleared! Data will refresh on next load.")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Technical Analysis", "📰 Sentiment Analysis", "🤖 AI Chatbot"])
    
    if ticker:
        try:
            # Initialize components
            data_fetcher = DataFetcher()
            
            with tab1:
                st.header("📊 Real-Time Data & Fundamentals")
                display_data_overview(data_fetcher, ticker, analysis_period)
            
            with tab2:
                st.header("📈 Technical Indicators Dashboard")
                display_technical_analysis(ticker, analysis_period)
            
            with tab3:
                st.header("📰 Sentiment Analysis")
                display_sentiment_analysis(ticker)
            
            with tab4:
                st.header("🤖 AI-Powered Q&A Chatbot")
                display_chatbot(ticker)
                
        except Exception as e:
            st.error(f"Error loading data for {ticker}: {str(e)}")
            st.info("Please check if the ticker symbol is valid.")

def display_data_overview(data_fetcher, ticker, period):
    """Display real-time data and fundamental metrics"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Price Chart")
        try:
            # Fetch and display price data
            stock_data = data_fetcher.get_stock_data(ticker, period)
            if not stock_data.empty:
                st.line_chart(stock_data['Close'])
            else:
                st.warning("No data available for this ticker.")
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
    
    with col2:
        st.subheader("Key Metrics")
        try:
            # Fetch fundamental data
            fundamentals = data_fetcher.get_fundamentals(ticker)
            if fundamentals:
                for key, value in fundamentals.items():
                    st.metric(key, value)
            else:
                st.info("Fundamental data not available.")
        except Exception as e:
            st.error(f"Error fetching fundamentals: {str(e)}")

def display_technical_analysis(ticker, period):
    """Display technical analysis indicators"""
    try:
        technical_analyzer = TechnicalAnalysis()
        indicators = technical_analyzer.calculate_indicators(ticker, period)
        
        if indicators:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Moving Averages")
                if 'SMA_50' in indicators and 'SMA_200' in indicators:
                    st.plotly_chart(indicators['ma_chart'], use_container_width=True)
            
            with col2:
                st.subheader("RSI & MACD")
                if 'RSI' in indicators:
                    st.plotly_chart(indicators['rsi_chart'], use_container_width=True)
        else:
            st.warning("Unable to calculate technical indicators for this ticker.")
            
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")

def display_sentiment_analysis(ticker):
    """Display sentiment analysis results"""
    try:
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_data = sentiment_analyzer.analyze_ticker_sentiment(ticker)
        
        if sentiment_data:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Overall Sentiment")
                sentiment_score = sentiment_data.get('overall_sentiment', 0)
                sentiment_label = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"
                
                st.metric("Sentiment Score", f"{sentiment_score:.3f}")
                st.metric("Sentiment", sentiment_label)
            
            with col2:
                st.subheader("Recent News Headlines")
                headlines = sentiment_data.get('headlines', [])
                for headline in headlines[:10]:
                    with st.expander(headline['title'][:100] + "..."):
                        st.write(f"**Sentiment:** {headline['sentiment']:.3f}")
                        st.write(f"**Source:** {headline['source']}")
                        st.write(f"**Published:** {headline['published']}")
        else:
            st.info("No recent news found for sentiment analysis.")
            
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")

def display_chatbot(ticker):
    """Display AI chatbot interface"""
    try:
        # Initialize chatbot
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = RAGChatbot()
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask anything about the stock analysis...")
        
        if user_input:
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