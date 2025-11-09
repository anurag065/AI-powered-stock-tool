"""
Centralized ticker cache with dual rate limiting for Finnhub API
Implements singleton pattern to ensure one cache instance across all modules
Supports both 30 calls/second and 60 calls/minute limits
"""

import yfinance as yf
import time
import logging
from threading import Lock
from collections import deque
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Callable, Any, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import Config
    from exceptions import RateLimitError, YFinanceAPIError
except ImportError:
    from ..config import Config
    from ..exceptions import RateLimitError, YFinanceAPIError

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter to prevent hitting API limits
    Implements token bucket algorithm with sliding window
    """

    def __init__(self, max_calls: int = 60, period: int = 60) -> None:
        """
        Initialize rate limiter

        Args:
            max_calls: Maximum number of calls allowed per period
            period: Time period in seconds
        """
        self.max_calls: int = max_calls
        self.period: int = period
        self.calls: deque = deque()
        self.lock: Lock = Lock()

    def wait_if_needed(self) -> None:
        """Manually check and wait if rate limit would be exceeded"""
        with self.lock:
            now: float = time.time()

            # Remove old calls
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()

            # If at limit, wait
            if len(self.calls) >= self.max_calls:
                sleep_time: float = self.period - (now - self.calls[0]) + 0.1
                logger.debug(f"Rate limit check: waiting {sleep_time:.2f} seconds")
                time.sleep(sleep_time)

                # Clean up again
                now = time.time()
                while self.calls and self.calls[0] < now - self.period:
                    self.calls.popleft()

            # Record the call
            self.calls.append(now)


class DualRateLimiter:
    """
    Dual rate limiter to handle both per-second and per-minute limits
    For Finnhub: 30 calls/second AND 60 calls/minute
    """

    def __init__(
        self,
        calls_per_second: int = Config.RATE_LIMIT_CALLS_PER_SECOND,
        calls_per_minute: int = Config.RATE_LIMIT_CALLS_PER_MINUTE
    ) -> None:
        """
        Initialize dual rate limiter

        Args:
            calls_per_second: Maximum calls allowed per second
            calls_per_minute: Maximum calls allowed per minute
        """
        self.second_limiter: RateLimiter = RateLimiter(max_calls=calls_per_second, period=1)
        self.minute_limiter: RateLimiter = RateLimiter(max_calls=calls_per_minute, period=60)
        logger.info(f"DualRateLimiter initialized: {calls_per_second}/sec, {calls_per_minute}/min")

    def wait_if_needed(self) -> None:
        """
        Check both rate limits and wait if either would be exceeded
        This is NOT thread-safe across both limiters, but each limiter is thread-safe
        """
        # Check second-level rate limit first (stricter for burst)
        self.second_limiter.wait_if_needed()
        # Check minute-level rate limit
        self.minute_limiter.wait_if_needed()


def retry_with_backoff(max_retries: int = 3, initial_delay: int = 2) -> Callable:
    """
    Decorator to retry failed API calls with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (doubles with each retry)

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay: int = initial_delay
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg: str = str(e)

                    # Check if it's a rate limit error
                    if '429' in error_msg or 'Too Many Requests' in error_msg or 'rate limit' in error_msg.lower():
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Rate limit hit (attempt {attempt + 1}/{max_retries}), "
                                f"retrying in {delay} seconds: {error_msg}"
                            )
                            time.sleep(delay)
                            delay *= 2  # Exponential backoff
                        else:
                            logger.error(f"Rate limit hit after {max_retries} retries: {error_msg}")
                            raise RateLimitError("API", retry_after=delay)
                    # Check for JSON parsing errors (often caused by empty responses)
                    elif 'Expecting value' in error_msg or 'JSONDecodeError' in error_msg:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"JSON parse error (attempt {attempt + 1}/{max_retries}), "
                                f"retrying in {delay} seconds"
                            )
                            time.sleep(delay)
                            delay *= 2
                        else:
                            logger.error(f"JSON parse error after {max_retries} retries")
                            raise YFinanceAPIError(f"JSON parse error after {max_retries} retries")
                    else:
                        # For other errors, don't retry
                        raise

            # If we get here, all retries failed
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class TickerCache:
    """
    Singleton cache for yfinance Ticker objects
    Prevents redundant API calls by reusing ticker objects
    Note: This is kept for backwards compatibility and fallback to yfinance
    """

    _instance: Optional['TickerCache'] = None
    _lock: Lock = Lock()

    def __new__(cls) -> 'TickerCache':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized: bool = True
        self.cache: Dict[str, yf.Ticker] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl: timedelta = timedelta(hours=Config.TICKER_CACHE_TTL_HOURS)
        # Use dual rate limiter from config
        self.rate_limiter: DualRateLimiter = DualRateLimiter(
            calls_per_second=Config.RATE_LIMIT_CALLS_PER_SECOND,
            calls_per_minute=Config.RATE_LIMIT_CALLS_PER_MINUTE
        )
        logger.info(
            f"TickerCache initialized with dual rate limiting "
            f"({Config.RATE_LIMIT_CALLS_PER_SECOND}/sec, {Config.RATE_LIMIT_CALLS_PER_MINUTE}/min)"
        )

    @retry_with_backoff(max_retries=3, initial_delay=2)
    def get_ticker(self, ticker_symbol: str) -> yf.Ticker:
        """
        Get a cached or new yfinance Ticker object with rate limiting

        Args:
            ticker_symbol: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            yf.Ticker object

        Raises:
            YFinanceAPIError: If ticker creation fails after retries
        """
        ticker_symbol = ticker_symbol.upper()
        now: datetime = datetime.now()

        # Check if we have a valid cached ticker
        if ticker_symbol in self.cache:
            cache_age: timedelta = now - self.cache_timestamps[ticker_symbol]
            if cache_age < self.cache_ttl:
                logger.debug(f"Using cached ticker for {ticker_symbol} (age: {cache_age.seconds}s)")
                return self.cache[ticker_symbol]
            else:
                logger.debug(f"Cached ticker for {ticker_symbol} expired, refreshing")

        # Apply rate limiting before creating new ticker
        self.rate_limiter.wait_if_needed()

        # Create new ticker object
        logger.info(f"Creating new Ticker object for {ticker_symbol}")
        try:
            ticker: yf.Ticker = yf.Ticker(ticker_symbol)

            # Cache it
            self.cache[ticker_symbol] = ticker
            self.cache_timestamps[ticker_symbol] = now

            return ticker
        except Exception as e:
            error_msg = f"Error creating Ticker object for {ticker_symbol}: {str(e)}"
            logger.error(error_msg)
            raise YFinanceAPIError(error_msg) from e

    def clear_cache(self, ticker_symbol: Optional[str] = None) -> None:
        """
        Clear cached ticker(s)

        Args:
            ticker_symbol: If provided, clear only this ticker. Otherwise clear all.
        """
        if ticker_symbol:
            ticker_symbol = ticker_symbol.upper()
            if ticker_symbol in self.cache:
                del self.cache[ticker_symbol]
                del self.cache_timestamps[ticker_symbol]
                logger.info(f"Cleared cache for {ticker_symbol}")
        else:
            self.cache.clear()
            self.cache_timestamps.clear()
            logger.info("Cleared entire ticker cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache

        Returns:
            Dictionary with cache statistics
        """
        return {
            'cached_tickers': list(self.cache.keys()),
            'cache_size': len(self.cache),
            'cache_timestamps': {k: v.isoformat() for k, v in self.cache_timestamps.items()}
        }


# Global singleton instances
_ticker_cache_instance: Optional[TickerCache] = None
_finnhub_rate_limiter: Optional[DualRateLimiter] = None


def get_ticker_cache() -> TickerCache:
    """
    Get the global TickerCache instance

    Returns:
        TickerCache singleton instance
    """
    global _ticker_cache_instance
    if _ticker_cache_instance is None:
        _ticker_cache_instance = TickerCache()
    return _ticker_cache_instance


def get_finnhub_rate_limiter() -> DualRateLimiter:
    """
    Get the global DualRateLimiter instance for Finnhub API

    Returns:
        DualRateLimiter singleton instance
    """
    global _finnhub_rate_limiter
    if _finnhub_rate_limiter is None:
        _finnhub_rate_limiter = DualRateLimiter(
            calls_per_second=Config.RATE_LIMIT_CALLS_PER_SECOND,
            calls_per_minute=Config.RATE_LIMIT_CALLS_PER_MINUTE
        )
    return _finnhub_rate_limiter
