"""
Configuration Management Module
Centralizes all application configuration and constants
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
_env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=_env_path)


class Config:
    """Application configuration class"""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    CACHE_DIR = DATA_DIR / "cache"

    # API Configuration
    FINNHUB_API_KEY: Optional[str] = os.getenv('FINNHUB_API_KEY')
    TWELVEDATA_API_KEY: Optional[str] = os.getenv('TWELVEDATA_API_KEY')
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')

    # Feature flags
    USE_FINNHUB: bool = os.getenv('USE_FINNHUB', 'true').lower() == 'true'

    # Rate limiting configuration
    RATE_LIMIT_CALLS_PER_SECOND: int = int(os.getenv('RATE_LIMIT_CALLS_PER_SECOND', '30'))
    RATE_LIMIT_CALLS_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_CALLS_PER_MINUTE', '60'))

    # Cache configuration
    CACHE_TTL_MINUTES: int = 15
    CACHE_DAYS: int = 2
    TICKER_CACHE_TTL_HOURS: int = int(os.getenv('TICKER_CACHE_TTL_HOURS', '1'))
    SENTIMENT_CACHE_HOURS: int = 4
    FUNDAMENTALS_CACHE_HOURS: int = 24

    # Database configuration
    DB_FILE_NAME: str = "stock_data.db"
    SENTIMENT_DB_FILE_NAME: str = "sentiment_data.db"

    # API defaults
    DEFAULT_PERIOD: str = "1mo"
    DEFAULT_INTERVAL: str = "1day"
    MAX_NEWS_ARTICLES: int = 20
    NEWS_RETENTION_DAYS: int = 7

    # Technical analysis parameters
    SMA_SHORT_WINDOW: int = 50
    SMA_LONG_WINDOW: int = 200
    RSI_WINDOW: int = 14
    RSI_OVERBOUGHT: int = 70
    RSI_OVERSOLD: int = 30
    BOLLINGER_WINDOW: int = 20
    BOLLINGER_STD: int = 2

    # Sentiment analysis configuration
    SENTIMENT_MODEL_NAME: str = "ProsusAI/finbert"
    SENTIMENT_FALLBACK_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    MAX_TEXT_LENGTH: int = 512

    # Tokenizers configuration
    TOKENIZERS_PARALLELISM: str = "false"

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_db_path(cls, db_name: str) -> Path:
        """Get full path to database file"""
        return cls.CACHE_DIR / db_name

    @classmethod
    def is_finnhub_configured(cls) -> bool:
        """Check if Finnhub API key is properly configured"""
        return bool(cls.FINNHUB_API_KEY and cls.FINNHUB_API_KEY != 'your_finnhub_api_key_here')

    @classmethod
    def is_twelvedata_configured(cls) -> bool:
        """Check if Twelve Data API key is properly configured"""
        return bool(cls.TWELVEDATA_API_KEY and cls.TWELVEDATA_API_KEY != 'your_twelvedata_api_key_here')

    @classmethod
    def validate_configuration(cls) -> list[str]:
        """
        Validate configuration and return list of warnings

        Returns:
            List of warning messages for missing or invalid configuration
        """
        warnings = []

        if not cls.is_finnhub_configured():
            warnings.append("Finnhub API key not configured. Some features may not work.")

        if not cls.is_twelvedata_configured():
            warnings.append("Twelve Data API key not configured. Historical charts may not work.")

        if cls.RATE_LIMIT_CALLS_PER_SECOND > 30:
            warnings.append("Rate limit calls per second exceeds Finnhub free tier limit (30/sec).")

        if cls.RATE_LIMIT_CALLS_PER_MINUTE > 60:
            warnings.append("Rate limit calls per minute exceeds Finnhub free tier limit (60/min).")

        return warnings


# Ensure directories exist on import
Config.ensure_directories()


# API endpoint constants
class APIEndpoints:
    """API endpoint constants"""

    # Finnhub endpoints (base URL is handled by SDK)
    FINNHUB_QUOTE = "quote"
    FINNHUB_CANDLES = "stock/candle"
    FINNHUB_PROFILE = "stock/profile2"
    FINNHUB_FINANCIALS = "stock/metric"
    FINNHUB_NEWS = "company-news"

    # Twelve Data endpoints (base URL is handled by SDK)
    TWELVEDATA_TIMESERIES = "time_series"
    TWELVEDATA_QUOTE = "quote"


# Error messages
class ErrorMessages:
    """Standardized error messages"""

    API_KEY_MISSING = "{api_name} API key not found in .env file. {feature} may not work."
    API_CALL_FAILED = "{api_name} API call failed for {symbol}: {error}"
    NO_DATA_FOUND = "No data found for ticker {symbol}"
    INVALID_PERIOD = "Invalid period format: {period}"
    RATE_LIMIT_EXCEEDED = "Rate limit exceeded. Please wait before making more requests."
    DATABASE_ERROR = "Database error: {error}"
    INVALID_TICKER = "Invalid ticker symbol: {symbol}"


# Success messages
class SuccessMessages:
    """Standardized success messages"""

    CLIENT_INITIALIZED = "{client_name} client initialized successfully"
    DATA_RETRIEVED = "Retrieved {count} data points from {source} for {symbol}"
    CACHE_HIT = "Cache hit for {key}"
    CACHE_CLEARED = "{client_name} client cache cleared"
