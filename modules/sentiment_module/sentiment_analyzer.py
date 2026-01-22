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
    Enhanced Sentiment Analysis module for analyzing financial news sentiment
    Features:
    - Time-weighted sentiment scoring (recent news gets more weight)
    - Descriptive sentiment labels (Very Positive, Slightly Positive, etc.)
    - "Read full news" links that open in new tabs
    - Uses FinBERT for financial sentiment analysis and Finnhub for news data
    """

    def __init__(self):
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "sentiment_data.db"

        try:
            self.finnhub = get_finnhub_client()
            self.use_finnhub = True
            logger.info("SentimentAnalyzer initialized with Finnhub news API")
        except Exception as e:
            logger.warning(f"Failed to initialize Finnhub client: {e}. Using RSS feeds fallback.")
            self.finnhub = None
            self.use_finnhub = False

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
                time_weight REAL,
                weighted_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, url)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ticker_sentiment_summary (
                ticker TEXT PRIMARY KEY,
                overall_sentiment REAL,
                weighted_sentiment REAL,
                sentiment_label TEXT,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                very_positive_count INTEGER,
                slightly_positive_count INTEGER,
                slightly_negative_count INTEGER,
                very_negative_count INTEGER,
                total_articles INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_sentiment_model(self):
        """Initialize FinBERT model for financial sentiment analysis"""
        try:
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
    
    def calculate_time_weight(self, published_date):
        """
        Calculate time-based weight for news articles
        Recent news gets higher weight:
        - Today: 100% (1.0)
        - Yesterday: 80% (0.8) 
        - Last week: 30% (0.3)
        - Last month: 10% (0.1)
        - Older than month: 5% (0.05)
        """
        if not published_date:
            return 0.1  
        
        now = datetime.now()
        time_diff = now - published_date
        days_old = time_diff.days
        hours_old = time_diff.total_seconds() / 3600
        
        if hours_old <= 24:  
            return 1.0
        elif days_old <= 1:   
            return 0.8
        elif days_old <= 7:  
            return 0.3
        elif days_old <= 30:  
            return 0.1
        else:  
            return 0.05
    
    def get_enhanced_sentiment_label(self, sentiment_score):
        """
        Convert sentiment score to descriptive labels
        More granular than just positive/negative/neutral
        """
        if sentiment_score >= 0.4:
            return "Very Positive"
        elif sentiment_score >= 0.15:
            return "Slightly Positive"
        elif sentiment_score > -0.15:
            return "Neutral"
        elif sentiment_score > -0.4:
            return "Slightly Negative"
        else:
            return "Very Negative"
    
    def get_news_urls_for_ticker(self, ticker):
        """
        Generate RSS feed URLs for financial news about a specific ticker
        (Fallback method when Finnhub news is not available)
        """
        try:
            stock_info = self.data_fetcher.get_stock_info(ticker)
            company_name = stock_info.get('company_name', ticker)
            search_terms = [ticker, company_name.split()[0] if company_name != ticker else ticker]
        except:
            search_terms = [ticker]

        urls = []

        for term in search_terms[:2]:  
            google_news_url = f"https://news.google.com/rss/search?q={term}+stock+financial&hl=en-US&gl=US&ceid=US:en"
            urls.append(('Google News', google_news_url))

        yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        urls.append(('Yahoo Finance', yahoo_url))

        return urls
    
    def fetch_news_headlines(self, ticker, max_articles=50):
        """
        Fetch recent news headlines for a ticker from Finnhub (primary) or RSS feeds (fallback)
        """
        try:
            self._clean_old_sentiment_data()

            cached_news = self._get_cached_news(ticker)
            if cached_news:
                logger.info(f"Returning cached news for {ticker}")
                return cached_news

            all_articles = []

            if self.use_finnhub:
                try:
                    logger.info(f"Fetching Finnhub company news for {ticker}")
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=10)

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
                        all_articles.sort(key=lambda x: x['published'], reverse=True)
                        return all_articles

                except Exception as e:
                    logger.warning(f"Finnhub news failed for {ticker}: {e}, falling back to RSS feeds")

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

                        if article['published'] and (datetime.now() - article['published']).days <= 7:
                            all_articles.append(article)

                except Exception as e:
                    logger.warning(f"Error fetching from {source_name}: {str(e)}")
                    continue

            all_articles.sort(key=lambda x: x['published'] or datetime.min, reverse=True)
            return all_articles[:max_articles]

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return []
    
    def _get_cached_news(self, ticker):
        """Get cached news if it's recent (less than 4 hours old)"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_time = datetime.now() - timedelta(hours=4)
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT title, summary, url, source, published, sentiment_score, sentiment_label, time_weight, weighted_score
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
                    'time_weight': row[7],
                    'weighted_score': row[8],
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
                logger.warning(f"Could not parse date: {date_str}")
                return datetime.now()

            if parsed_date.tzinfo is not None:
                parsed_date = parsed_date.replace(tzinfo=None)

            return parsed_date

        except Exception as e:
            logger.warning(f"Error parsing date {date_str}: {str(e)}")
            return datetime.now()
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a text using FinBERT or fallback model
        Returns sentiment score (-1 to 1) and enhanced label
        """
        if not self.sentiment_pipeline:
            return 0.0, 'Neutral'
        
        try:
            text = text[:512]
            
            results = self.sentiment_pipeline(text)
            
            if isinstance(results[0], list):
                results = results[0]
            
            sentiment_score = 0.0
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label or 'bullish' in label:
                    sentiment_score += score
                elif 'negative' in label or 'bearish' in label:
                    sentiment_score -= score
            
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            sentiment_label = self.get_enhanced_sentiment_label(sentiment_score)
            
            return sentiment_score, sentiment_label
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return 0.0, 'Neutral'
    
    def analyze_ticker_sentiment(self, ticker):
        """
        Analyze overall sentiment for a ticker based on recent news
        Uses time-weighted scoring for more accurate current sentiment
        """
        try:
            articles = self.fetch_news_headlines(ticker)
            
            if not articles:
                return None
            
            if 'sentiment' not in articles[0]:
                analyzed_articles = []
                total_weighted_score = 0.0
                total_weights = 0.0
                
                for article in articles:
                    text = f"{article['title']} {article['summary']}"
                    sentiment_score, sentiment_label = self.analyze_sentiment(text)
                    
                    time_weight = self.calculate_time_weight(article['published'])
                    weighted_score = sentiment_score * time_weight
                    
                    article['sentiment'] = sentiment_score
                    article['sentiment_label'] = sentiment_label
                    article['time_weight'] = time_weight
                    article['weighted_score'] = weighted_score
                    
                    total_weighted_score += weighted_score
                    total_weights += time_weight
                    
                    analyzed_articles.append(article)
                    
                    self._cache_sentiment_data(article)
                
                articles = analyzed_articles
            else:
                total_weighted_score = sum(article.get('weighted_score', 0) for article in articles)
                total_weights = sum(article.get('time_weight', 1) for article in articles)
            
            sentiment_scores = [article['sentiment'] for article in articles]
            overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            
            weighted_sentiment = total_weighted_score / total_weights if total_weights > 0 else 0.0
            
            sentiment_counts = self._count_enhanced_sentiment_categories(articles)
            
            overall_label = self.get_enhanced_sentiment_label(weighted_sentiment)
            
            self._cache_enhanced_sentiment_summary(
                ticker, overall_sentiment, weighted_sentiment, overall_label, 
                sentiment_counts, len(articles)
            )
            
            return {
                'ticker': ticker,
                'overall_sentiment': overall_sentiment,  
                'weighted_sentiment': weighted_sentiment, 
                'sentiment_label': overall_label,
                'positive_count': sentiment_counts['positive'],
                'negative_count': sentiment_counts['negative'], 
                'neutral_count': sentiment_counts['neutral'],
                'very_positive_count': sentiment_counts['very_positive'],
                'slightly_positive_count': sentiment_counts['slightly_positive'],
                'slightly_negative_count': sentiment_counts['slightly_negative'],
                'very_negative_count': sentiment_counts['very_negative'],
                'total_articles': len(articles),
                'headlines': articles
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {ticker}: {str(e)}")
            return None
    
    def _count_enhanced_sentiment_categories(self, articles):
        """Count articles by enhanced sentiment categories"""
        counts = {
            'very_positive': 0,
            'slightly_positive': 0,
            'neutral': 0,
            'slightly_negative': 0,
            'very_negative': 0,
            'positive': 0,  
            'negative': 0   
        }
        
        for article in articles:
            label = article.get('sentiment_label', 'Neutral').lower()
            
            if 'very positive' in label:
                counts['very_positive'] += 1
                counts['positive'] += 1
            elif 'slightly positive' in label:
                counts['slightly_positive'] += 1
                counts['positive'] += 1
            elif 'neutral' in label:
                counts['neutral'] += 1
            elif 'slightly negative' in label:
                counts['slightly_negative'] += 1
                counts['negative'] += 1
            elif 'very negative' in label:
                counts['very_negative'] += 1
                counts['negative'] += 1
        
        return counts
    
    def _cache_sentiment_data(self, article):
        """Cache individual article sentiment data with enhanced fields"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT OR REPLACE INTO news_sentiment 
            (ticker, title, summary, url, source, published, sentiment_score, sentiment_label, time_weight, weighted_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article['ticker'],
            article['title'],
            article['summary'],
            article['url'],
            article['source'],
            article['published'].isoformat() if article['published'] else None,
            article['sentiment'],
            article['sentiment_label'],
            article.get('time_weight', 1.0),
            article.get('weighted_score', article['sentiment'])
        ))
        
        conn.commit()
        conn.close()
    
    def _cache_enhanced_sentiment_summary(self, ticker, overall_sentiment, weighted_sentiment, 
                                        sentiment_label, sentiment_counts, total_articles):
        """Cache ticker sentiment summary with enhanced data"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT OR REPLACE INTO ticker_sentiment_summary 
            (ticker, overall_sentiment, weighted_sentiment, sentiment_label, positive_count, negative_count, 
             neutral_count, very_positive_count, slightly_positive_count, slightly_negative_count, 
             very_negative_count, total_articles)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker, overall_sentiment, weighted_sentiment, sentiment_label,
            sentiment_counts['positive'], sentiment_counts['negative'], sentiment_counts['neutral'],
            sentiment_counts['very_positive'], sentiment_counts['slightly_positive'],
            sentiment_counts['slightly_negative'], sentiment_counts['very_negative'],
            total_articles
        ))
        
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
        Now includes both traditional and time-weighted trends
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT DATE(created_at) as date, 
                       AVG(sentiment_score) as avg_sentiment,
                       AVG(weighted_score) as avg_weighted_sentiment,
                       COUNT(*) as article_count
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
    
    def format_article_with_link(self, article):
        """
        Format article with 'Read full news' link that opens in new tab
        Returns formatted string for display
        """
        title = article.get('title', 'No title')
        sentiment_label = article.get('sentiment_label', 'Neutral')
        sentiment_score = article.get('sentiment_score', 0)
        source = article.get('source', 'Unknown')
        published = article.get('published')
        url = article.get('url', '')
        time_weight = article.get('time_weight', 1.0)
        
        if published:
            if isinstance(published, str):
                try:
                    published = datetime.fromisoformat(published)
                except:
                    pass
            
            if isinstance(published, datetime):
                published_str = published.strftime('%Y-%m-%d %H:%M:%S')
            else:
                published_str = str(published)
        else:
            published_str = 'Unknown'
        
        formatted_article = {
            'title': title,
            'sentiment': f"{sentiment_score:.3f}",
            'sentiment_label': sentiment_label,
            'source': source,
            'published': published_str,
            'time_weight': f"{time_weight:.1%}", 
            'read_full_link': f'<a href="{url}" target="_blank" rel="noopener noreferrer">Read full news</a>' if url else 'No link available'
        }
        
        return formatted_article