import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
SEC_FILINGS_DIR = DATA_DIR / "sec_filings"

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, VECTOR_STORE_DIR, SEC_FILINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys (set these in your .env file)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Data retention settings
DATA_RETENTION_DAYS = 2  # Store only 1-2 days of data as requested
SENTIMENT_CACHE_HOURS = 4  # Cache sentiment data for 4 hours
VECTOR_STORE_UPDATE_HOURS = 6  # Update vector store every 6 hours

# Technical analysis settings
TECHNICAL_INDICATORS = {
    'SMA_WINDOWS': [50, 200],
    'RSI_WINDOW': 14,
    'BOLLINGER_WINDOW': 20,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9
}

# Sentiment analysis settings
SENTIMENT_CONFIG = {
    'MODEL_NAME': 'ProsusAI/finbert',
    'FALLBACK_MODEL': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'MAX_ARTICLES_PER_TICKER': 20,
    'NEWS_LOOKBACK_DAYS': 7
}

# Vector store settings
VECTOR_STORE_CONFIG = {
    'MODEL_NAME': 'all-MiniLM-L6-v2',
    'EMBEDDING_DIM': 384,
    'SIMILARITY_THRESHOLD': 0.3,
    'MAX_SEARCH_RESULTS': 5
}

# Database settings
DATABASE_CONFIG = {
    'STOCK_DATA_DB': 'stock_data.db',
    'SENTIMENT_DB': 'sentiment_data.db',
    'CHATBOT_DB': 'chatbot_knowledge.db',
    'CLEANUP_INTERVAL_HOURS': 24
}

# Streamlit settings
STREAMLIT_CONFIG = {
    'PAGE_TITLE': 'AI Stock Analysis Platform',
    'PAGE_ICON': '📈',
    'LAYOUT': 'wide',
    'INITIAL_SIDEBAR_STATE': 'expanded'
}

# Yahoo Finance settings
YAHOO_FINANCE_CONFIG = {
    'DEFAULT_PERIOD': '3mo',
    'AVAILABLE_PERIODS': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
    'DEFAULT_INTERVAL': '1d',
    'REALTIME_INTERVAL': '1m'
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': str(PROJECT_ROOT / 'logs' / 'app.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Create logs directory
(PROJECT_ROOT / 'logs').mkdir(exist_ok=True)

# RSS Feed URLs for news sentiment
NEWS_SOURCES = {
    'google_news': 'https://news.google.com/rss/search?q={ticker}+stock+financial&hl=en-US&gl=US&ceid=US:en',
    'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US',
    'marketwatch': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
    'reuters_business': 'https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best'
}

# Error messages
ERROR_MESSAGES = {
    'INVALID_TICKER': 'Invalid ticker symbol. Please enter a valid stock ticker.',
    'NO_DATA': 'No data available for this ticker. Please try a different symbol.',
    'API_ERROR': 'Error fetching data from external API. Please try again later.',
    'NETWORK_ERROR': 'Network connection error. Please check your internet connection.',
    'MODEL_ERROR': 'Error loading AI model. Some features may be unavailable.',
    'DATABASE_ERROR': 'Database error. Data may not be saved properly.'
}

# Default tickers for demo/testing
DEFAULT_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']

# Feature flags
FEATURES = {
    'ENABLE_SEC_FILINGS': False,  # Set to True when SEC filing feature is implemented
    'ENABLE_REAL_LLM': False,     # Set to True when connecting to actual LLM API
    'ENABLE_ADVANCED_CHARTS': True,
    'ENABLE_EXPORT': True,
    'ENABLE_ALERTS': False        # Future feature for price alerts
}

# Rate limiting (to avoid overwhelming APIs)
RATE_LIMITS = {
    'YAHOO_FINANCE_REQUESTS_PER_MINUTE': 60,
    'NEWS_REQUESTS_PER_HOUR': 100,
    'LLM_REQUESTS_PER_MINUTE': 10
}