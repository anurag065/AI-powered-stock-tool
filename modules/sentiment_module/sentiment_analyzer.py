import feedparser
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import sqlite3
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
try:
    from utils.finnhub_client import get_finnhub_client
    from data_module.data_fetcher import DataFetcher
except ImportError:
    from ..utils.finnhub_client import get_finnhub_client
    from ..data_module.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment Analysis module for analyzing financial news sentiment
    Uses FinBERT for financial sentiment analysis and Finnhub for news data
    """

    def __init__(self):
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "sentiment_data.db"

        # Initialize Finnhub client for news
        try:
            self.finnhub = get_finnhub_client()
            self.use_finnhub = True
            logger.info("SentimentAnalyzer initialized with Finnhub news API")
        except Exception as e:
            logger.warning(f"Failed to initialize Finnhub client: {e}. Using RSS feeds fallback.")
            self.finnhub = None
            self.use_finnhub = False

        # Initialize DataFetcher for company info
        self.data_fetcher = DataFetcher()

        self._init_database()
        self._init_sentiment_model()
    
    def _init_database(self):
        """Initialize database for caching sentiment data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                title TEXT,
                summary TEXT,
                url TEXT,
                source TEXT,
                published TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, url)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ticker_sentiment_summary (
                ticker TEXT PRIMARY KEY,
                overall_sentiment REAL,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                total_articles INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_sentiment_model(self):
        """Initialize FinBERT model for financial sentiment analysis"""
        try:
            # Use a lightweight financial sentiment model
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=None
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT model: {str(e)}")
            # Fallback to a simpler model
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    top_k=None
                )
                logger.info("Using fallback sentiment model")
            except Exception as e2:
                logger.error(f"Failed to load any sentiment model: {str(e2)}")
                self.sentiment_pipeline = None
    
    def get_news_urls_for_ticker(self, ticker):
        """
        Generate RSS feed URLs for financial news about a specific ticker
        (Fallback method when Finnhub news is not available)
        """
        # Get company name for better search results
        try:
            stock_info = self.data_fetcher.get_stock_info(ticker)
            company_name = stock_info.get('company_name', ticker)
            search_terms = [ticker, company_name.split()[0] if company_name != ticker else ticker]
        except:
            search_terms = [ticker]

        urls = []

        # Google News RSS feeds
        for term in search_terms[:2]:  # Limit to avoid too many requests
            google_news_url = f"https://news.google.com/rss/search?q={term}+stock+financial&hl=en-US&gl=US&ceid=US:en"
            urls.append(('Google News', google_news_url))

        # Yahoo Finance RSS (if available)
        yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        urls.append(('Yahoo Finance', yahoo_url))

        return urls
    
    def fetch_news_headlines(self, ticker, max_articles=20):
        """
        Fetch recent news headlines for a ticker from Finnhub (primary) or RSS feeds (fallback)
        """
        try:
            # Clean old data first (keep only last 2 days)
            self._clean_old_sentiment_data()

            # Check if we have recent cached data
            cached_news = self._get_cached_news(ticker)
            if cached_news:
                logger.info(f"Returning cached news for {ticker}")
                return cached_news

            all_articles = []

            # Try Finnhub news first
            if self.use_finnhub:
                try:
                    logger.info(f"Fetching Finnhub company news for {ticker}")
                    # Get news from last 7 days
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)

                    news_items = self.finnhub.company_news(
                        ticker,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )

                    if news_items:
                        for item in news_items[:max_articles]:
                            article = {
                                'title': item.get('headline', ''),
                                'summary': item.get('summary', ''),
                                'url': item.get('url', ''),
                                'source': item.get('source', 'Finnhub'),
                                'published': datetime.fromtimestamp(item.get('datetime', 0)),
                                'ticker': ticker
                            }
                            all_articles.append(article)

                        logger.info(f"Retrieved {len(all_articles)} news articles from Finnhub for {ticker}")
                        # Sort by date
                        all_articles.sort(key=lambda x: x['published'], reverse=True)
                        return all_articles

                except Exception as e:
                    logger.warning(f"Finnhub news failed for {ticker}: {e}, falling back to RSS feeds")

            # Fallback to RSS feeds
            logger.info(f"Fetching RSS feed news for {ticker}")
            news_urls = self.get_news_urls_for_ticker(ticker)

            for source_name, url in news_urls:
                try:
                    feed = feedparser.parse(url)

                    for entry in feed.entries[:max_articles//len(news_urls)]:
                        article = {
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', entry.get('description', '')),
                            'url': entry.get('link', ''),
                            'source': source_name,
                            'published': self._parse_date(entry.get('published', '')),
                            'ticker': ticker
                        }

                        # Only include articles from last 7 days
                        if article['published'] and (datetime.now() - article['published']).days <= 7:
                            all_articles.append(article)

                except Exception as e:
                    logger.warning(f"Error fetching from {source_name}: {str(e)}")
                    continue

            # Sort by date and limit
            all_articles.sort(key=lambda x: x['published'] or datetime.min, reverse=True)
            return all_articles[:max_articles]

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return []
    
    def _get_cached_news(self, ticker):
        """Get cached news if it's recent (less than 4 hours old)"""
        conn = sqlite3.connect(self.db_path)
        
        # Check for recent data
        cutoff_time = datetime.now() - timedelta(hours=4)
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT title, summary, url, source, published, sentiment_score, sentiment_label
            FROM news_sentiment 
            WHERE ticker = ? AND created_at > ?
            ORDER BY published DESC
        ''', (ticker, cutoff_time))
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            cached_articles = []
            for row in results:
                article = {
                    'title': row[0],
                    'summary': row[1],
                    'url': row[2],
                    'source': row[3],
                    'published': datetime.fromisoformat(row[4]) if row[4] else None,
                    'sentiment': row[5],
                    'sentiment_label': row[6],
                    'ticker': ticker
                }
                cached_articles.append(article)
            return cached_articles
        
        return None
    
    def _parse_date(self, date_str):
        """Parse date string from RSS feed"""
        if not date_str:
            return None

        try:
            # Try different date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S'
            ]

            parsed_date = None
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue

            if parsed_date is None:
                # If all formats fail, return current time
                logger.warning(f"Could not parse date: {date_str}")
                return datetime.now()

            # Remove timezone info to make it timezone-naive
            if parsed_date.tzinfo is not None:
                parsed_date = parsed_date.replace(tzinfo=None)

            return parsed_date

        except Exception as e:
            logger.warning(f"Error parsing date {date_str}: {str(e)}")
            return datetime.now()
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a text using FinBERT or fallback model
        Returns sentiment score (-1 to 1) and label
        """
        if not self.sentiment_pipeline:
            return 0.0, 'neutral'
        
        try:
            # Truncate text if too long
            text = text[:512]
            
            results = self.sentiment_pipeline(text)
            
            # Process results based on model type
            if isinstance(results[0], list):
                results = results[0]
            
            # Convert to standardized format
            sentiment_score = 0.0
            sentiment_label = 'neutral'
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label or 'bullish' in label:
                    sentiment_score += score
                    sentiment_label = 'positive' if score > 0.5 else sentiment_label
                elif 'negative' in label or 'bearish' in label:
                    sentiment_score -= score
                    sentiment_label = 'negative' if score > 0.5 else sentiment_label
            
            # Normalize score to -1 to 1 range
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            return sentiment_score, sentiment_label
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return 0.0, 'neutral'
    
    def analyze_ticker_sentiment(self, ticker):
        """
        Analyze overall sentiment for a ticker based on recent news
        """
        try:
            # Fetch news headlines
            articles = self.fetch_news_headlines(ticker)
            
            if not articles:
                return None
            
            # If articles don't have sentiment yet, analyze them
            if 'sentiment' not in articles[0]:
                analyzed_articles = []
                for article in articles:
                    # Combine title and summary for analysis
                    text = f"{article['title']} {article['summary']}"
                    sentiment_score, sentiment_label = self.analyze_sentiment(text)
                    
                    article['sentiment'] = sentiment_score
                    article['sentiment_label'] = sentiment_label
                    analyzed_articles.append(article)
                    
                    # Cache the sentiment data
                    self._cache_sentiment_data(article)
                
                articles = analyzed_articles
            
            # Calculate overall sentiment
            sentiment_scores = [article['sentiment'] for article in articles]
            overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            
            # Count sentiment categories
            positive_count = sum(1 for score in sentiment_scores if score > 0.1)
            negative_count = sum(1 for score in sentiment_scores if score < -0.1)
            neutral_count = len(sentiment_scores) - positive_count - negative_count
            
            # Cache the summary
            self._cache_sentiment_summary(ticker, overall_sentiment, positive_count, negative_count, neutral_count, len(articles))
            
            return {
                'ticker': ticker,
                'overall_sentiment': overall_sentiment,
                'sentiment_label': self._get_sentiment_label(overall_sentiment),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'total_articles': len(articles),
                'headlines': articles
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {ticker}: {str(e)}")
            return None
    
    def _get_sentiment_label(self, score):
        """Convert sentiment score to label"""
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    def _cache_sentiment_data(self, article):
        """Cache individual article sentiment data"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT OR REPLACE INTO news_sentiment 
            (ticker, title, summary, url, source, published, sentiment_score, sentiment_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article['ticker'],
            article['title'],
            article['summary'],
            article['url'],
            article['source'],
            article['published'].isoformat() if article['published'] else None,
            article['sentiment'],
            article['sentiment_label']
        ))
        
        conn.commit()
        conn.close()
    
    def _cache_sentiment_summary(self, ticker, overall_sentiment, positive_count, negative_count, neutral_count, total_articles):
        """Cache ticker sentiment summary"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT OR REPLACE INTO ticker_sentiment_summary 
            (ticker, overall_sentiment, positive_count, negative_count, neutral_count, total_articles)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ticker, overall_sentiment, positive_count, negative_count, neutral_count, total_articles))
        
        conn.commit()
        conn.close()
    
    def _clean_old_sentiment_data(self):
        """Remove sentiment data older than 2 days"""
        cutoff_date = datetime.now() - timedelta(days=2)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM news_sentiment WHERE created_at < ?', (cutoff_date,))
        cursor.execute('DELETE FROM ticker_sentiment_summary WHERE updated_at < ?', (cutoff_date,))
        
        conn.commit()
        conn.close()
    
    def get_sentiment_trends(self, ticker, days=7):
        """
        Get sentiment trends over the past few days
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT DATE(created_at) as date, AVG(sentiment_score) as avg_sentiment, COUNT(*) as article_count
                FROM news_sentiment 
                WHERE ticker = ? AND created_at >= date('now', '-{} days')
                GROUP BY DATE(created_at)
                ORDER BY date
            '''.format(days)
            
            df = pd.read_sql_query(query, conn, params=(ticker,))
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting sentiment trends for {ticker}: {str(e)}")
            return pd.DataFrame()