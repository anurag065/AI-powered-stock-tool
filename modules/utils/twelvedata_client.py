"""
Twelve Data API Client Wrapper
Provides centralized access to Twelve Data API for historical price charts
Free tier: 800 API calls per day
"""

from twelvedata import TDClient
import pandas as pd
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


class TwelveDataClient:
    """
    Singleton wrapper for Twelve Data API client
    Provides rate-limited access to historical price data with caching
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
        self.api_key = os.getenv('TWELVEDATA_API_KEY')
        if not self.api_key or self.api_key == 'your_twelvedata_api_key_here':
            logger.warning("Twelve Data API key not found in .env file. Historical charts may not work.")
            self.client = None
        else:
            self.client = TDClient(apikey=self.api_key)
            logger.info("Twelve Data client initialized successfully")

        # Use the same rate limiter as Finnhub (conservative approach)
        # Twelve Data free tier: 8 calls/min, so we'll be well under that
        self.rate_limiter = get_finnhub_rate_limiter()

        # Cache for API responses
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = timedelta(minutes=15)  # Cache responses for 15 minutes

    def _check_client(self):
        """Check if client is initialized"""
        if not self.client:
            raise ValueError("Twelve Data client not initialized. Please set TWELVEDATA_API_KEY in .env file")

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

    def _period_to_outputsize(self, period):
        """
        Convert period string to outputsize for Twelve Data

        Args:
            period: Period string like '1mo', '3mo', '6mo', '1y'

        Returns:
            int: Number of data points to request
        """
        try:
            if period.endswith('y'):
                years = int(period[:-1])
                return min(years * 252, 5000)  # 252 trading days per year, max 5000
            elif period.endswith('mo'):
                months = int(period[:-2])
                return min(months * 21, 5000)  # ~21 trading days per month
            elif period.endswith('d'):
                days = int(period[:-1])
                return min(days, 5000)
            else:
                return 30  # Default to 1 month
        except:
            return 30

    @retry_with_backoff(max_retries=3, initial_delay=2)
    def get_time_series(self, symbol, interval='1day', outputsize=30, period=None):
        """
        Get historical OHLCV data for a stock

        Args:
            symbol: Stock ticker symbol
            interval: Time interval (1min, 5min, 15min, 30min, 1h, 1day, 1week, 1month)
            outputsize: Number of data points to return (max 5000)
            period: Optional period string like '1mo', '3mo' (will override outputsize)

        Returns:
            pandas.DataFrame: Historical price data with OHLCV columns
        """
        self._check_client()

        # Convert period to outputsize if provided
        if period:
            outputsize = self._period_to_outputsize(period)

        # Check cache first
        cache_key = self._get_cache_key('time_series', symbol, interval, outputsize)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Apply rate limiting
        self.rate_limiter.wait_if_needed()

        # Make the API call
        try:
            logger.info(f"Fetching Twelve Data time series for {symbol} (interval={interval}, size={outputsize})")

            ts = self.client.time_series(
                symbol=symbol,
                interval=interval,
                outputsize=outputsize
            )

            # Convert to pandas DataFrame
            df = ts.as_pandas()

            if df.empty:
                logger.warning(f"Twelve Data returned empty data for {symbol}")
                return pd.DataFrame()

            # Rename columns to match yfinance format for consistency
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Sort by date (oldest to newest)
            df = df.sort_index()

            # Cache the result
            self._set_cache(cache_key, df)

            logger.info(f"Retrieved {len(df)} data points from Twelve Data for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Twelve Data API call failed for {symbol}: {str(e)}")
            raise

    def get_quote(self, symbol):
        """
        Get real-time quote (Note: Twelve Data free tier has limited real-time data)

        Args:
            symbol: Stock ticker symbol

        Returns:
            dict: Quote data
        """
        self._check_client()

        cache_key = self._get_cache_key('quote', symbol)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Apply rate limiting
        self.rate_limiter.wait_if_needed()

        try:
            logger.info(f"Fetching Twelve Data quote for {symbol}")
            quote = self.client.quote(symbol=symbol)

            if quote:
                result = quote.as_json()
                self._set_cache(cache_key, result)
                return result
            return None

        except Exception as e:
            logger.error(f"Twelve Data quote failed for {symbol}: {str(e)}")
            raise

    def clear_cache(self):
        """Clear all cached responses"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Twelve Data client cache cleared")


# Global singleton instance
_twelvedata_client_instance = None


def get_twelvedata_client():
    """
    Get the global TwelveDataClient instance

    Returns:
        TwelveDataClient singleton instance
    """
    global _twelvedata_client_instance
    if _twelvedata_client_instance is None:
        _twelvedata_client_instance = TwelveDataClient()
    return _twelvedata_client_instance
