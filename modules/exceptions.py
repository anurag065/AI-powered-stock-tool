"""
Custom Exceptions Module
Defines custom exception classes for better error handling
"""


class StockAnalysisError(Exception):
    """Base exception for all stock analysis errors"""
    pass


class APIError(StockAnalysisError):
    """Base class for API-related errors"""
    pass


class FinnhubAPIError(APIError):
    """Raised when Finnhub API calls fail"""
    pass


class TwelveDataAPIError(APIError):
    """Raised when Twelve Data API calls fail"""
    pass


class YFinanceAPIError(APIError):
    """Raised when yfinance API calls fail"""
    pass


class ConfigurationError(StockAnalysisError):
    """Raised when configuration is invalid or missing"""
    pass


class APIKeyMissingError(ConfigurationError):
    """Raised when required API key is missing"""
    def __init__(self, api_name: str):
        self.api_name = api_name
        super().__init__(f"{api_name} API key is not configured")


class DataNotFoundError(StockAnalysisError):
    """Raised when requested data cannot be found"""
    def __init__(self, ticker: str, data_type: str = "data"):
        self.ticker = ticker
        self.data_type = data_type
        super().__init__(f"No {data_type} found for ticker {ticker}")


class InvalidTickerError(StockAnalysisError):
    """Raised when ticker symbol is invalid"""
    def __init__(self, ticker: str):
        self.ticker = ticker
        super().__init__(f"Invalid ticker symbol: {ticker}")


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded"""
    def __init__(self, api_name: str, retry_after: int = None):
        self.api_name = api_name
        self.retry_after = retry_after
        message = f"{api_name} rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message)


class CacheError(StockAnalysisError):
    """Raised when cache operations fail"""
    pass


class DatabaseError(StockAnalysisError):
    """Raised when database operations fail"""
    pass


class ModelLoadError(StockAnalysisError):
    """Raised when ML model fails to load"""
    def __init__(self, model_name: str, error: Exception):
        self.model_name = model_name
        self.original_error = error
        super().__init__(f"Failed to load model {model_name}: {str(error)}")


class InvalidPeriodError(StockAnalysisError):
    """Raised when period format is invalid"""
    def __init__(self, period: str):
        self.period = period
        super().__init__(f"Invalid period format: {period}")
