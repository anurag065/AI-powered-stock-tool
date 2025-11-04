import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Data module for fetching stock data from Yahoo Finance API
    Implements caching strategy to store only 1-2 days of data as requested
    """
    
    def __init__(self, cache_days=2):
        self.cache_days = cache_days
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "stock_data.db"
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
    
    def get_stock_data(self, ticker, period="1mo"):
        """
        Fetch stock price data with intelligent caching
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
                
                # Check if data is recent enough
                latest_date = cached_data.index.max()
                if latest_date.date() >= (datetime.now() - timedelta(days=1)).date():
                    logger.info(f"Returning cached data for {ticker}")
                    return cached_data
            
            # Fetch new data from Yahoo Finance
            logger.info(f"Fetching fresh data for {ticker}")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
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
                row['Volume']
            ))
        
        conn.commit()
        conn.close()
    
    def get_fundamentals(self, ticker):
        """
        Fetch fundamental data for a stock
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
                        'P/E Ratio': result[0],
                        'EPS': result[1], 
                        'ROE': result[2],
                        'Debt/Equity': result[3],
                        'Market Cap': result[4]
                    }
            
            conn.close()
            
            # Fetch fresh data
            stock = yf.Ticker(ticker)
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
        """Get current price and basic info"""
        try:
            stock = yf.Ticker(ticker)
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
        """Get comprehensive stock information"""
        try:
            stock = yf.Ticker(ticker)
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