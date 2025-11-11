# import yfinance as yf
# import pandas as pd
# import sqlite3
# from datetime import datetime, timedelta
# import os
# from pathlib import Path
# import logging
# import sys

# # Add parent directory to path for imports
# sys.path.append(str(Path(__file__).parent.parent))
# try:
#     from utils.finnhub_client import get_finnhub_client
#     from utils.ticker_cache import get_ticker_cache
#     from utils.twelvedata_client import get_twelvedata_client
# except ImportError:
#     # Fallback for when running as script
#     from ..utils.finnhub_client import get_finnhub_client
#     from ..utils.ticker_cache import get_ticker_cache
#     from ..utils.twelvedata_client import get_twelvedata_client

# logger = logging.getLogger(__name__)


# class DataFetcher:
#     """
#     Data module for fetching stock data from Finnhub API with yfinance fallback
#     Uses Twelve Data for historical price charts, Finnhub for fundamentals/quotes
#     Implements caching strategy to store only 1-2 days of data as requested
#     """

#     def __init__(self, cache_days=2):
#         self.cache_days = cache_days
#         self.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
#         self.db_path = self.cache_dir / "stock_data.db"

#         # Initialize Finnhub client for fundamentals, quotes, company info
#         try:
#             self.finnhub = get_finnhub_client()
#             self.use_finnhub = True
#             logger.info("DataFetcher initialized with Finnhub API")
#         except Exception as e:
#             logger.warning(f"Failed to initialize Finnhub client: {e}. Using yfinance only.")
#             self.finnhub = None
#             self.use_finnhub = False

#         # Initialize Twelve Data client for historical price charts
#         try:
#             self.twelvedata = get_twelvedata_client()
#             self.use_twelvedata = True
#             logger.info("DataFetcher initialized with Twelve Data API")
#         except Exception as e:
#             logger.warning(f"Failed to initialize Twelve Data client: {e}. Using yfinance fallback.")
#             self.twelvedata = None
#             self.use_twelvedata = False

#         # Initialize yfinance ticker cache for historical data
#         self.ticker_cache = get_ticker_cache()

#         self._init_database()

#     def _init_database(self):
#         """Initialize SQLite database for caching"""
#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()

#         # Create tables
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS stock_prices (
#                 ticker TEXT,
#                 date TEXT,
#                 open REAL,
#                 high REAL,
#                 low REAL,
#                 close REAL,
#                 volume INTEGER,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 PRIMARY KEY (ticker, date)
#             )
#         ''')

#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS fundamentals (
#                 ticker TEXT PRIMARY KEY,
#                 pe_ratio REAL,
#                 eps REAL,
#                 roe REAL,
#                 debt_to_equity REAL,
#                 market_cap REAL,
#                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#         ''')

#         conn.commit()
#         conn.close()

#     def _clean_old_data(self):
#         """Remove data older than cache_days"""
#         cutoff_date = datetime.now() - timedelta(days=self.cache_days)

#         conn = sqlite3.connect(self.db_path)
#         cursor = conn.cursor()

#         cursor.execute('''
#             DELETE FROM stock_prices
#             WHERE created_at < ?
#         ''', (cutoff_date,))

#         cursor.execute('''
#             DELETE FROM fundamentals
#             WHERE updated_at < ?
#         ''', (cutoff_date,))

#         conn.commit()
#         conn.close()

#     def _period_exceeds_one_year(self, period):
#         """
#         Check if a period string exceeds 1 year

#         Args:
#             period: Period string like '1mo', '3mo', '6mo', '1y', '2y', etc.

#         Returns:
#             bool: True if period exceeds 1 year
#         """
#         try:
#             if period.endswith('y'):
#                 years = int(period[:-1])
#                 return years > 1
#             elif period.endswith('mo'):
#                 months = int(period[:-2])
#                 return months > 12
#             elif period.endswith('d'):
#                 days = int(period[:-1])
#                 return days > 365
#             else:
#                 # Default to False for unknown formats
#                 return False
#         except:
#             return False

#     def _period_to_days(self, period):
#         """
#         Convert period string to number of days

#         Args:
#             period: Period string like '1mo', '3mo', '6mo', '1y'

#         Returns:
#             int: Number of days
#         """
#         try:
#             if period.endswith('y'):
#                 return int(period[:-1]) * 365
#             elif period.endswith('mo'):
#                 return int(period[:-2]) * 30
#             elif period.endswith('d'):
#                 return int(period[:-1])
#             else:
#                 return 30  # Default to 1 month
#         except:
#             return 30

#     def _get_finnhub_candles(self, ticker, period):
#         """
#         Get candlestick data from Finnhub

#         Args:
#             ticker: Stock ticker symbol
#             period: Period string

#         Returns:
#             pandas.DataFrame: Stock data
#         """
#         try:
#             # Calculate timestamps
#             days = self._period_to_days(period)
#             end_time = datetime.now()
#             start_time = end_time - timedelta(days=days)

#             # Convert to Unix timestamps
#             end_timestamp = int(end_time.timestamp())
#             start_timestamp = int(start_time.timestamp())

#             # Get data from Finnhub (D = daily resolution)
#             logger.info(f"Fetching Finnhub candles for {ticker} ({period})")
#             candles = self.finnhub.stock_candles(ticker, 'D', start_timestamp, end_timestamp)

#             if candles.get('s') == 'ok':
#                 # Convert to DataFrame
#                 df = pd.DataFrame({
#                     'Open': candles['o'],
#                     'High': candles['h'],
#                     'Low': candles['l'],
#                     'Close': candles['c'],
#                     'Volume': candles['v']
#                 })

#                 # Convert timestamps to datetime index
#                 df.index = pd.to_datetime(candles['t'], unit='s')
#                 df.index.name = 'Date'

#                 return df
#             else:
#                 logger.warning(f"Finnhub returned no data for {ticker}")
#                 return pd.DataFrame()

#         except Exception as e:
#             logger.error(f"Error fetching Finnhub candles for {ticker}: {str(e)}")
#             return pd.DataFrame()

#     def get_stock_data(self, ticker, period="1mo"):
#         """
#         Fetch stock price data with intelligent caching
#         Uses Twelve Data for historical charts, falls back to yfinance if needed
#         """
#         try:
#             # Clean old data first
#             self._clean_old_data()

#             # Check cache first
#             conn = sqlite3.connect(self.db_path)

#             # Get cached data
#             cached_data = pd.read_sql_query('''
#                 SELECT date, open, high, low, close, volume
#                 FROM stock_prices
#                 WHERE ticker = ?
#                 ORDER BY date
#             ''', conn, params=(ticker,))

#             conn.close()

#             # If we have recent data, return it
#             if not cached_data.empty:
#                 cached_data['date'] = pd.to_datetime(cached_data['date'])
#                 cached_data.set_index('date', inplace=True)

#                 # Check if data is recent enough
#                 latest_date = cached_data.index.max()
#                 if latest_date.date() >= (datetime.now() - timedelta(days=1)).date():
#                     logger.info(f"Returning cached data for {ticker}")
#                     return cached_data

#             # Try Twelve Data first for historical price charts
#             if self.use_twelvedata:
#                 try:
#                     logger.info(f"Fetching Twelve Data historical data for {ticker} (period: {period})")
#                     data = self.twelvedata.get_time_series(ticker, interval='1day', period=period)

#                     if not data.empty:
#                         logger.info(f"Retrieved {len(data)} data points from Twelve Data for {ticker}")
#                         # Store in cache (only last few days based on cache_days)
#                         recent_data = data.tail(self.cache_days * 5)  # Approximate for trading days
#                         self._cache_stock_data(ticker, recent_data)
#                         return data
#                     else:
#                         logger.warning(f"Twelve Data returned empty data for {ticker}, falling back to yfinance")
#                 except Exception as e:
#                     logger.warning(f"Twelve Data failed for {ticker}: {e}, falling back to yfinance")

#             # Fallback to yfinance for historical price data
#             logger.info(f"Fetching yfinance historical data for {ticker} (period: {period})")
#             stock = self.ticker_cache.get_ticker(ticker)
#             data = stock.history(period=period)

#             if data.empty:
#                 logger.warning(f"No data found for ticker {ticker}")
#                 return pd.DataFrame()

#             # Store in cache (only last few days based on cache_days)
#             recent_data = data.tail(self.cache_days * 5)  # Approximate for trading days
#             self._cache_stock_data(ticker, recent_data)

#             return data

#         except Exception as e:
#             logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
#             return pd.DataFrame()

#     def _cache_stock_data(self, ticker, data):
#         """Cache stock data in database"""
#         conn = sqlite3.connect(self.db_path)

#         for date, row in data.iterrows():
#             conn.execute('''
#                 INSERT OR REPLACE INTO stock_prices
#                 (ticker, date, open, high, low, close, volume)
#                 VALUES (?, ?, ?, ?, ?, ?, ?)
#             ''', (
#                 ticker,
#                 date.strftime('%Y-%m-%d'),
#                 row['Open'],
#                 row['High'],
#                 row['Low'],
#                 row['Close'],
#                 int(row['Volume'])
#             ))

#         conn.commit()
#         conn.close()

#     def get_fundamentals(self, ticker):
#         """
#         Fetch fundamental data for a stock
#         Uses Finnhub company_basic_financials, falls back to yfinance
#         Returns key metrics like P/E, EPS, ROE, Debt-to-Equity
#         """
#         try:
#             # Check cache first
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()

#             cursor.execute('''
#                 SELECT pe_ratio, eps, roe, debt_to_equity, market_cap, updated_at
#                 FROM fundamentals
#                 WHERE ticker = ?
#             ''', (ticker,))

#             result = cursor.fetchone()

#             # If we have recent data (less than 1 day old), return it
#             if result:
#                 updated_at = datetime.fromisoformat(result[5])
#                 if datetime.now() - updated_at < timedelta(hours=24):
#                     conn.close()
#                     return {
#                         'P/E Ratio': result[0] if result[0] else 'N/A',
#                         'EPS': result[1] if result[1] else 'N/A',
#                         'ROE': result[2] if result[2] else 'N/A',
#                         'Debt/Equity': result[3] if result[3] else 'N/A',
#                         'Market Cap': result[4] if result[4] else 'N/A'
#                     }

#             conn.close()

#             # Try Finnhub first
#             if self.use_finnhub:
#                 try:
#                     logger.info(f"Fetching Finnhub fundamentals for {ticker}")
#                     financials = self.finnhub.company_basic_financials(ticker, 'all')

#                     if financials and 'metric' in financials:
#                         metrics = financials['metric']
#                         fundamentals = {
#                             'P/E Ratio': metrics.get('peNormalizedAnnual', 'N/A'),
#                             'EPS': metrics.get('epsBasic', 'N/A'),
#                             'ROE': metrics.get('roeRfy', 'N/A'),
#                             'Debt/Equity': metrics.get('totalDebt/totalEquityAnnual', 'N/A'),
#                             'Market Cap': metrics.get('marketCapitalization', 'N/A')
#                         }

#                         # Cache and return
#                         self._cache_fundamentals(ticker, fundamentals)
#                         return fundamentals
#                 except Exception as e:
#                     logger.warning(f"Finnhub fundamentals failed for {ticker}: {e}, falling back to yfinance")

#             # Fallback to yfinance
#             logger.info(f"Fetching yfinance fundamentals for {ticker}")
#             stock = self.ticker_cache.get_ticker(ticker)
#             info = stock.info

#             fundamentals = {
#                 'P/E Ratio': info.get('trailingPE', 'N/A'),
#                 'EPS': info.get('trailingEps', 'N/A'),
#                 'ROE': info.get('returnOnEquity', 'N/A'),
#                 'Debt/Equity': info.get('debtToEquity', 'N/A'),
#                 'Market Cap': info.get('marketCap', 'N/A')
#             }

#             # Cache the fundamentals
#             self._cache_fundamentals(ticker, fundamentals)

#             return fundamentals

#         except Exception as e:
#             logger.error(f"Error fetching fundamentals for {ticker}: {str(e)}")
#             return {}

#     def _cache_fundamentals(self, ticker, fundamentals):
#         """Cache fundamental data"""
#         conn = sqlite3.connect(self.db_path)

#         # Convert 'N/A' to None for database storage
#         pe_ratio = fundamentals['P/E Ratio'] if fundamentals['P/E Ratio'] != 'N/A' else None
#         eps = fundamentals['EPS'] if fundamentals['EPS'] != 'N/A' else None
#         roe = fundamentals['ROE'] if fundamentals['ROE'] != 'N/A' else None
#         debt_eq = fundamentals['Debt/Equity'] if fundamentals['Debt/Equity'] != 'N/A' else None
#         market_cap = fundamentals['Market Cap'] if fundamentals['Market Cap'] != 'N/A' else None

#         conn.execute('''
#             INSERT OR REPLACE INTO fundamentals
#             (ticker, pe_ratio, eps, roe, debt_to_equity, market_cap)
#             VALUES (?, ?, ?, ?, ?, ?)
#         ''', (ticker, pe_ratio, eps, roe, debt_eq, market_cap))

#         conn.commit()
#         conn.close()

#     def get_realtime_price(self, ticker):
#         """
#         Get current price and basic info
#         Uses Finnhub quote(), falls back to yfinance
#         """
#         try:
#             # Try Finnhub first
#             if self.use_finnhub:
#                 try:
#                     logger.info(f"Fetching Finnhub quote for {ticker}")
#                     quote = self.finnhub.quote(ticker)

#                     if quote and quote.get('c'):  # 'c' is current price
#                         return {
#                             'price': quote['c'],  # Current price
#                             'change': quote['d'],  # Change
#                             'change_percent': quote['dp'],  # Percent change
#                             'volume': quote.get('v', 0),  # Volume
#                             'timestamp': datetime.fromtimestamp(quote['t']) if quote.get('t') else datetime.now()
#                         }
#                 except Exception as e:
#                     logger.warning(f"Finnhub quote failed for {ticker}: {e}, falling back to yfinance")

#             # Fallback to yfinance
#             logger.info(f"Fetching yfinance realtime price for {ticker}")
#             stock = self.ticker_cache.get_ticker(ticker)
#             data = stock.history(period="1d", interval="1m")

#             if data.empty:
#                 return None

#             latest = data.iloc[-1]
#             info = stock.info

#             return {
#                 'price': latest['Close'],
#                 'change': info.get('regularMarketChange', 0),
#                 'change_percent': info.get('regularMarketChangePercent', 0),
#                 'volume': latest['Volume'],
#                 'timestamp': data.index[-1]
#             }

#         except Exception as e:
#             logger.error(f"Error fetching realtime price for {ticker}: {str(e)}")
#             return None

#     def get_stock_info(self, ticker):
#         """
#         Get comprehensive stock information
#         Uses Finnhub company_profile2(), falls back to yfinance
#         """
#         try:
#             # Try Finnhub first
#             if self.use_finnhub:
#                 try:
#                     logger.info(f"Fetching Finnhub company profile for {ticker}")
#                     profile = self.finnhub.company_profile2(ticker)

#                     if profile:
#                         return {
#                             'company_name': profile.get('name', ticker),
#                             'sector': profile.get('finnhubIndustry', 'N/A'),
#                             'industry': profile.get('finnhubIndustry', 'N/A'),
#                             'website': profile.get('weburl', 'N/A'),
#                             'description': f"{profile.get('name', ticker)} operates in the {profile.get('finnhubIndustry', 'N/A')} sector. Exchange: {profile.get('exchange', 'N/A')}. Country: {profile.get('country', 'N/A')}."
#                         }
#                 except Exception as e:
#                     logger.warning(f"Finnhub company profile failed for {ticker}: {e}, falling back to yfinance")

#             # Fallback to yfinance
#             logger.info(f"Fetching yfinance stock info for {ticker}")
#             stock = self.ticker_cache.get_ticker(ticker)
#             info = stock.info

#             return {
#                 'company_name': info.get('longName', ticker),
#                 'sector': info.get('sector', 'N/A'),
#                 'industry': info.get('industry', 'N/A'),
#                 'website': info.get('website', 'N/A'),
#                 'description': info.get('longBusinessSummary', 'N/A')
#             }

#         except Exception as e:
#             logger.error(f"Error fetching stock info for {ticker}: {str(e)}")
#             return {}


import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
try:
    from utils.finnhub_client import get_finnhub_client
    from utils.ticker_cache import get_ticker_cache
    from utils.twelvedata_client import get_twelvedata_client
except ImportError:
    # Fallback for when running as script
    from ..utils.finnhub_client import get_finnhub_client
    from ..utils.ticker_cache import get_ticker_cache
    from ..utils.twelvedata_client import get_twelvedata_client

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Data module for fetching stock data from Finnhub API with yfinance fallback
    Uses Twelve Data for historical price charts, Finnhub for fundamentals/quotes
    Implements caching strategy to store only 1-2 days of data as requested
    """

    def __init__(self, cache_days=2):
        self.cache_days = cache_days
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "stock_data.db"

        # Initialize Finnhub client for fundamentals, quotes, company info
        try:
            self.finnhub = get_finnhub_client()
            self.use_finnhub = True
            logger.info("DataFetcher initialized with Finnhub API")
        except Exception as e:
            logger.warning(f"Failed to initialize Finnhub client: {e}. Using yfinance only.")
            self.finnhub = None
            self.use_finnhub = False

        # Initialize Twelve Data client for historical price charts
        try:
            self.twelvedata = get_twelvedata_client()
            self.use_twelvedata = True
            logger.info("DataFetcher initialized with Twelve Data API")
        except Exception as e:
            logger.warning(f"Failed to initialize Twelve Data client: {e}. Using yfinance fallback.")
            self.twelvedata = None
            self.use_twelvedata = False

        # Initialize yfinance ticker cache for historical data
        self.ticker_cache = get_ticker_cache()

        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for caching"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundamentals (
                ticker TEXT PRIMARY KEY,
                pe_ratio REAL,
                eps REAL,
                roe REAL,
                debt_to_equity REAL,
                market_cap REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def _clean_old_data(self):
        """Remove data older than cache_days"""
        cutoff_date = datetime.now() - timedelta(days=self.cache_days)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            DELETE FROM stock_prices
            WHERE created_at < ?
        ''', (cutoff_date,))

        cursor.execute('''
            DELETE FROM fundamentals
            WHERE updated_at < ?
        ''', (cutoff_date,))

        conn.commit()
        conn.close()

    def _period_exceeds_one_year(self, period):
        """
        Check if a period string exceeds 1 year

        Args:
            period: Period string like '1mo', '3mo', '6mo', '1y', '2y', etc.

        Returns:
            bool: True if period exceeds 1 year
        """
        try:
            if period.endswith('y'):
                years = int(period[:-1])
                return years > 1
            elif period.endswith('mo'):
                months = int(period[:-2])
                return months > 12
            elif period.endswith('d'):
                days = int(period[:-1])
                return days > 365
            else:
                # Default to False for unknown formats
                return False
        except:
            return False

    def _period_to_days(self, period):
        """
        Convert period string to number of days

        Args:
            period: Period string like '1mo', '3mo', '6mo', '1y'

        Returns:
            int: Number of days
        """
        try:
            if period.endswith('y'):
                return int(period[:-1]) * 365
            elif period.endswith('mo'):
                return int(period[:-2]) * 30
            elif period.endswith('d'):
                return int(period[:-1])
            else:
                return 30  # Default to 1 month
        except:
            return 30

    def _standardize_dataframe(self, df, ticker):
        """
        Standardize DataFrame columns and handle missing data
        FIXES: KeyError for missing 'Close' column
        """
        try:
            if df.empty:
                return df
            
            # Handle different column name variations
            column_mapping = {
                'open': 'Open', 'Open': 'Open', 'OPEN': 'Open',
                'high': 'High', 'High': 'High', 'HIGH': 'High',
                'low': 'Low', 'Low': 'Low', 'LOW': 'Low',
                'close': 'Close', 'Close': 'Close', 'CLOSE': 'Close',
                'adj close': 'Close', 'Adj Close': 'Close', 'ADJ_CLOSE': 'Close',
                'volume': 'Volume', 'Volume': 'Volume', 'VOLUME': 'Volume'
            }
            
            # Rename columns
            df_columns = {col: column_mapping.get(col, col) for col in df.columns}
            df = df.rename(columns=df_columns)
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            for col in required_columns:
                if col not in df.columns:
                    if col == 'Close' and 'Adj Close' in df.columns:
                        df['Close'] = df['Adj Close']
                    elif col == 'Volume' and col not in df.columns:
                        df['Volume'] = 0
                    else:
                        logger.warning(f"Missing column {col} for {ticker}")
                        if col in ['Open', 'High', 'Low', 'Close'] and 'Close' in df.columns:
                            df[col] = df['Close']
                        elif col in ['Open', 'High', 'Low', 'Close']:
                            df[col] = 100
                        else:
                            df[col] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error standardizing DataFrame for {ticker}: {str(e)}")
            return df

    def _get_finnhub_candles(self, ticker, period):
        """
        Get candlestick data from Finnhub

        Args:
            ticker: Stock ticker symbol
            period: Period string

        Returns:
            pandas.DataFrame: Stock data
        """
        try:
            # Calculate timestamps
            days = self._period_to_days(period)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            # Convert to Unix timestamps
            end_timestamp = int(end_time.timestamp())
            start_timestamp = int(start_time.timestamp())

            # Get data from Finnhub (D = daily resolution)
            logger.info(f"Fetching Finnhub candles for {ticker} ({period})")
            candles = self.finnhub.stock_candles(ticker, 'D', start_timestamp, end_timestamp)

            if candles.get('s') == 'ok':
                # Convert to DataFrame
                df = pd.DataFrame({
                    'Open': candles['o'],
                    'High': candles['h'],
                    'Low': candles['l'],
                    'Close': candles['c'],
                    'Volume': candles['v']
                })

                # Convert timestamps to datetime index
                df.index = pd.to_datetime(candles['t'], unit='s')
                df.index.name = 'Date'

                return df
            else:
                logger.warning(f"Finnhub returned no data for {ticker}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching Finnhub candles for {ticker}: {str(e)}")
            return pd.DataFrame()

    def get_stock_data(self, ticker, period="1mo"):
        """
        Fetch stock price data with intelligent caching
        Uses Twelve Data for historical charts, falls back to yfinance if needed
        """
        try:
            # Clean old data first
            self._clean_old_data()

            # Check cache first
            conn = sqlite3.connect(self.db_path)

            # Get cached data
            cached_data = pd.read_sql_query('''
                SELECT date, open, high, low, close, volume
                FROM stock_prices
                WHERE ticker = ?
                ORDER BY date
            ''', conn, params=(ticker,))

            conn.close()

            # If we have recent data, return it
            if not cached_data.empty:
                cached_data['date'] = pd.to_datetime(cached_data['date'])
                cached_data.set_index('date', inplace=True)
                cached_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

                # Check if data is recent enough
                latest_date = cached_data.index.max()
                if latest_date.date() >= (datetime.now() - timedelta(days=1)).date():
                    logger.info(f"Returning cached data for {ticker}")
                    return cached_data

            # Try Twelve Data first for historical price charts
            if self.use_twelvedata:
                try:
                    logger.info(f"Fetching Twelve Data historical data for {ticker} (period: {period})")
                    data = self.twelvedata.get_time_series(ticker, interval='1day', period=period)

                    if not data.empty:
                        # Apply standardization fix
                        data = self._standardize_dataframe(data, ticker)
                        if not data.empty:
                            logger.info(f"Retrieved {len(data)} data points from Twelve Data for {ticker}")
                            # Store in cache (only last few days based on cache_days)
                            recent_data = data.tail(self.cache_days * 5)  # Approximate for trading days
                            self._cache_stock_data(ticker, recent_data)
                            return data
                    else:
                        logger.warning(f"Twelve Data returned empty data for {ticker}, falling back to yfinance")
                except Exception as e:
                    logger.warning(f"Twelve Data failed for {ticker}: {e}, falling back to yfinance")

            # Fallback to yfinance for historical price data
            logger.info(f"Fetching yfinance historical data for {ticker} (period: {period})")
            stock = self.ticker_cache.get_ticker(ticker)
            data = stock.history(period=period)

            # Apply standardization fix to prevent KeyError
            data = self._standardize_dataframe(data, ticker)

            if data.empty:
                logger.warning(f"No data found for ticker {ticker}")
                return pd.DataFrame()

            # Store in cache (only last few days based on cache_days)
            recent_data = data.tail(self.cache_days * 5)  # Approximate for trading days
            self._cache_stock_data(ticker, recent_data)

            return data

        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _cache_stock_data(self, ticker, data):
        """Cache stock data in database"""
        conn = sqlite3.connect(self.db_path)

        for date, row in data.iterrows():
            conn.execute('''
                INSERT OR REPLACE INTO stock_prices
                (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                date.strftime('%Y-%m-%d'),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                int(row['Volume'])
            ))

        conn.commit()
        conn.close()

    def get_fundamentals(self, ticker):
        """
        Fetch fundamental data for a stock
        Uses Finnhub company_basic_financials, falls back to yfinance
        Returns key metrics like P/E, EPS, ROE, Debt-to-Equity
        """
        try:
            # Check cache first
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT pe_ratio, eps, roe, debt_to_equity, market_cap, updated_at
                FROM fundamentals
                WHERE ticker = ?
            ''', (ticker,))

            result = cursor.fetchone()

            # If we have recent data (less than 1 day old), return it
            if result:
                updated_at = datetime.fromisoformat(result[5])
                if datetime.now() - updated_at < timedelta(hours=24):
                    conn.close()
                    return {
                        'P/E Ratio': result[0] if result[0] else 'N/A',
                        'EPS': result[1] if result[1] else 'N/A',
                        'ROE': result[2] if result[2] else 'N/A',
                        'Debt/Equity': result[3] if result[3] else 'N/A',
                        'Market Cap': result[4] if result[4] else 'N/A'
                    }

            conn.close()

            # Try Finnhub first
            if self.use_finnhub:
                try:
                    logger.info(f"Fetching Finnhub fundamentals for {ticker}")
                    financials = self.finnhub.company_basic_financials(ticker, 'all')

                    if financials and 'metric' in financials:
                        metrics = financials['metric']
                        fundamentals = {
                            'P/E Ratio': metrics.get('peNormalizedAnnual', 'N/A'),
                            'EPS': metrics.get('epsBasic', 'N/A'),
                            'ROE': metrics.get('roeRfy', 'N/A'),
                            'Debt/Equity': metrics.get('totalDebt/totalEquityAnnual', 'N/A'),
                            'Market Cap': metrics.get('marketCapitalization', 'N/A')
                        }

                        # Cache and return
                        self._cache_fundamentals(ticker, fundamentals)
                        return fundamentals
                except Exception as e:
                    logger.warning(f"Finnhub fundamentals failed for {ticker}: {e}, falling back to yfinance")

            # Fallback to yfinance
            logger.info(f"Fetching yfinance fundamentals for {ticker}")
            stock = self.ticker_cache.get_ticker(ticker)
            info = stock.info

            fundamentals = {
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                'EPS': info.get('trailingEps', 'N/A'),
                'ROE': info.get('returnOnEquity', 'N/A'),
                'Debt/Equity': info.get('debtToEquity', 'N/A'),
                'Market Cap': info.get('marketCap', 'N/A')
            }

            # Cache the fundamentals
            self._cache_fundamentals(ticker, fundamentals)

            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {str(e)}")
            return {}

    def _cache_fundamentals(self, ticker, fundamentals):
        """Cache fundamental data"""
        conn = sqlite3.connect(self.db_path)

        # Convert 'N/A' to None for database storage
        pe_ratio = fundamentals['P/E Ratio'] if fundamentals['P/E Ratio'] != 'N/A' else None
        eps = fundamentals['EPS'] if fundamentals['EPS'] != 'N/A' else None
        roe = fundamentals['ROE'] if fundamentals['ROE'] != 'N/A' else None
        debt_eq = fundamentals['Debt/Equity'] if fundamentals['Debt/Equity'] != 'N/A' else None
        market_cap = fundamentals['Market Cap'] if fundamentals['Market Cap'] != 'N/A' else None

        conn.execute('''
            INSERT OR REPLACE INTO fundamentals
            (ticker, pe_ratio, eps, roe, debt_to_equity, market_cap)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ticker, pe_ratio, eps, roe, debt_eq, market_cap))

        conn.commit()
        conn.close()

    def get_realtime_price(self, ticker):
        """
        Get current price and basic info
        Uses Finnhub quote(), falls back to yfinance
        """
        try:
            # Try Finnhub first
            if self.use_finnhub:
                try:
                    logger.info(f"Fetching Finnhub quote for {ticker}")
                    quote = self.finnhub.quote(ticker)

                    if quote and quote.get('c'):  # 'c' is current price
                        return {
                            'price': quote['c'],  # Current price
                            'change': quote['d'],  # Change
                            'change_percent': quote['dp'],  # Percent change
                            'volume': quote.get('v', 0),  # Volume
                            'timestamp': datetime.fromtimestamp(quote['t']) if quote.get('t') else datetime.now()
                        }
                except Exception as e:
                    logger.warning(f"Finnhub quote failed for {ticker}: {e}, falling back to yfinance")

            # Fallback to yfinance
            logger.info(f"Fetching yfinance realtime price for {ticker}")
            stock = self.ticker_cache.get_ticker(ticker)
            data = stock.history(period="1d", interval="1m")

            if data.empty:
                return None

            latest = data.iloc[-1]
            info = stock.info

            return {
                'price': latest['Close'],
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': latest['Volume'],
                'timestamp': data.index[-1]
            }

        except Exception as e:
            logger.error(f"Error fetching realtime price for {ticker}: {str(e)}")
            return None

    def get_stock_info(self, ticker):
        """
        Get comprehensive stock information
        Uses Finnhub company_profile2(), falls back to yfinance
        """
        try:
            # Try Finnhub first
            if self.use_finnhub:
                try:
                    logger.info(f"Fetching Finnhub company profile for {ticker}")
                    profile = self.finnhub.company_profile2(ticker)

                    if profile:
                        return {
                            'company_name': profile.get('name', ticker),
                            'sector': profile.get('finnhubIndustry', 'N/A'),
                            'industry': profile.get('finnhubIndustry', 'N/A'),
                            'website': profile.get('weburl', 'N/A'),
                            'description': f"{profile.get('name', ticker)} operates in the {profile.get('finnhubIndustry', 'N/A')} sector. Exchange: {profile.get('exchange', 'N/A')}. Country: {profile.get('country', 'N/A')}."
                        }
                except Exception as e:
                    logger.warning(f"Finnhub company profile failed for {ticker}: {e}, falling back to yfinance")

            # Fallback to yfinance
            logger.info(f"Fetching yfinance stock info for {ticker}")
            stock = self.ticker_cache.get_ticker(ticker)
            info = stock.info

            return {
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'website': info.get('website', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A')
            }

        except Exception as e:
            logger.error(f"Error fetching stock info for {ticker}: {str(e)}")
            return {}