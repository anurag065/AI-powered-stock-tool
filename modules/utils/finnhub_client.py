"""
Finnhub API Client Wrapper
Provides centralized access to Finnhub API with rate limiting and error handling
"""

import finnhub
import os
import logging
from datetime import datetime, timedelta
from threading import Lock
from pathlib import Path
from dotenv import load_dotenv
from .ticker_cache import get_finnhub_rate_limiter, retry_with_backoff

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


class FinnhubClient:
    """
    Singleton wrapper for Finnhub API client
    Provides rate-limited access to Finnhub data with caching and error handling
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # Get API key from environment
        self.api_key = os.getenv('FINNHUB_API_KEY')
        if not self.api_key or self.api_key == 'your_finnhub_api_key_here':
            logger.warning("Finnhub API key not found in .env file. Some features may not work.")
            self.client = None
        else:
            self.client = finnhub.Client(api_key=self.api_key)
            logger.info("Finnhub client initialized successfully")

        # Get rate limiter
        self.rate_limiter = get_finnhub_rate_limiter()

        # Cache for API responses
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = timedelta(minutes=15)  # Cache responses for 15 minutes

    def _check_client(self):
        """Check if client is initialized"""
        if not self.client:
            raise ValueError("Finnhub client not initialized. Please set FINNHUB_API_KEY in .env file")

    def _get_cache_key(self, method, *args, **kwargs):
        """Generate cache key from method name and parameters"""
        key_parts = [method] + [str(arg) for arg in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
        return ":".join(key_parts)

    def _get_cached(self, cache_key):
        """Get cached response if valid"""
        if cache_key in self.cache:
            cache_time = self.cache_timestamps[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return self.cache[cache_key]
        return None

    def _set_cache(self, cache_key, value):
        """Set cached response"""
        self.cache[cache_key] = value
        self.cache_timestamps[cache_key] = datetime.now()

    @retry_with_backoff(max_retries=3, initial_delay=2)
    def _api_call(self, method, *args, **kwargs):
        """
        Make rate-limited API call with retry logic

        Args:
            method: Name of the Finnhub client method to call
            *args, **kwargs: Arguments to pass to the method
        """
        self._check_client()

        # Apply rate limiting
        self.rate_limiter.wait_if_needed()

        # Make the API call
        try:
            func = getattr(self.client, method)
            result = func(*args, **kwargs)
            logger.debug(f"Finnhub API call successful: {method}")
            return result
        except Exception as e:
            logger.error(f"Finnhub API call failed: {method} - {str(e)}")
            raise

    # Stock Data Methods

    def quote(self, symbol):
        """
        Get real-time quote data

        Args:
            symbol: Stock ticker symbol

        Returns:
            dict: Quote data with current price, change, etc.
        """
        cache_key = self._get_cache_key('quote', symbol)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = self._api_call('quote', symbol)
        self._set_cache(cache_key, result)
        return result

    def stock_candles(self, symbol, resolution, start_timestamp, end_timestamp):
        """
        Get candlestick data (OHLCV)

        Args:
            symbol: Stock ticker symbol
            resolution: Candle resolution (1, 5, 15, 30, 60, D, W, M)
            start_timestamp: Start time (Unix timestamp)
            end_timestamp: End time (Unix timestamp)

        Returns:
            dict: Candlestick data
        """
        cache_key = self._get_cache_key('stock_candles', symbol, resolution, start_timestamp, end_timestamp)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = self._api_call('stock_candles', symbol, resolution, start_timestamp, end_timestamp)
        self._set_cache(cache_key, result)
        return result

    def company_profile2(self, symbol):
        """
        Get company profile information

        Args:
            symbol: Stock ticker symbol

        Returns:
            dict: Company profile data
        """
        cache_key = self._get_cache_key('company_profile2', symbol)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = self._api_call('company_profile2', symbol=symbol)
        self._set_cache(cache_key, result)
        return result

    # Fundamental Data Methods

    def company_basic_financials(self, symbol, metric='all'):
        """
        Get company basic financials

        Args:
            symbol: Stock ticker symbol
            metric: Metric type (all, margin, price, valuation, etc.)

        Returns:
            dict: Financial metrics
        """
        cache_key = self._get_cache_key('company_basic_financials', symbol, metric)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = self._api_call('company_basic_financials', symbol, metric)
        self._set_cache(cache_key, result)
        return result

    def company_earnings(self, symbol, limit=10):
        """
        Get company earnings data

        Args:
            symbol: Stock ticker symbol
            limit: Number of results to return

        Returns:
            list: Earnings data
        """
        cache_key = self._get_cache_key('company_earnings', symbol, limit)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = self._api_call('company_earnings', symbol, limit=limit)
        self._set_cache(cache_key, result)
        return result

    # Technical Analysis Methods

    def technical_indicator(self, symbol, resolution, start_timestamp, end_timestamp, indicator, **indicator_params):
        """
        Get technical indicator data

        Args:
            symbol: Stock ticker symbol
            resolution: Candle resolution
            start_timestamp: Start time (Unix timestamp)
            end_timestamp: End time (Unix timestamp)
            indicator: Indicator name (e.g., 'rsi', 'macd', 'sma')
            **indicator_params: Indicator-specific parameters

        Returns:
            dict: Technical indicator data
        """
        cache_key = self._get_cache_key('technical_indicator', symbol, resolution, start_timestamp,
                                       end_timestamp, indicator, **indicator_params)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = self._api_call('technical_indicator', symbol, resolution, start_timestamp,
                               end_timestamp, indicator, **indicator_params)
        self._set_cache(cache_key, result)
        return result

    def aggregate_indicator(self, symbol, resolution):
        """
        Get aggregate technical indicator signals (BUY/SELL/HOLD)
        Combines multiple indicators into overall signal

        Args:
            symbol: Stock ticker symbol
            resolution: Resolution (1, 5, 15, 30, 60, D, W, M)

        Returns:
            dict: Aggregate signals
        """
        cache_key = self._get_cache_key('aggregate_indicator', symbol, resolution)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = self._api_call('aggregate_indicator', symbol, resolution)
        self._set_cache(cache_key, result)
        return result

    # News & Sentiment Methods

    def company_news(self, symbol, _from, to):
        """
        Get company news

        Args:
            symbol: Stock ticker symbol
            _from: Start date (YYYY-MM-DD)
            to: End date (YYYY-MM-DD)

        Returns:
            list: News articles
        """
        cache_key = self._get_cache_key('company_news', symbol, _from, to)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = self._api_call('company_news', symbol, _from=_from, to=to)
        self._set_cache(cache_key, result)
        return result

    def news_sentiment(self, symbol):
        """
        Get news sentiment

        Args:
            symbol: Stock ticker symbol

        Returns:
            dict: Sentiment data
        """
        cache_key = self._get_cache_key('news_sentiment', symbol)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = self._api_call('news_sentiment', symbol)
        self._set_cache(cache_key, result)
        return result

    # Alternative Data Methods

    def recommendation_trends(self, symbol):
        """
        Get recommendation trends from analysts

        Args:
            symbol: Stock ticker symbol

        Returns:
            list: Recommendation data
        """
        cache_key = self._get_cache_key('recommendation_trends', symbol)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = self._api_call('recommendation_trends', symbol)
        self._set_cache(cache_key, result)
        return result

    def insider_transactions(self, symbol, _from=None, to=None):
        """
        Get insider transactions

        Args:
            symbol: Stock ticker symbol
            _from: Start date (YYYY-MM-DD)
            to: End date (YYYY-MM-DD)

        Returns:
            dict: Insider transaction data
        """
        cache_key = self._get_cache_key('insider_transactions', symbol, _from, to)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        result = self._api_call('stock_insider_transactions', symbol, _from=_from, to=to)
        self._set_cache(cache_key, result)
        return result

    def clear_cache(self):
        """Clear all cached responses"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Finnhub client cache cleared")


# Global singleton instance
_finnhub_client_instance = None


def get_finnhub_client():
    """
    Get the global FinnhubClient instance

    Returns:
        FinnhubClient singleton instance
    """
    global _finnhub_client_instance
    if _finnhub_client_instance is None:
        _finnhub_client_instance = FinnhubClient()
    return _finnhub_client_instance
