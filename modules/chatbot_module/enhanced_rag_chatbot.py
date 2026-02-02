"""
Enhanced RAG Chatbot for AI-Powered Stock Analysis Platform
- Fast response times (< 3 seconds target)
- Streaming responses for better UX
- Unified data retrieval from all sources
- Smart caching and optimized vector search
- Comprehensive context from SEC, sentiment, technical, fundamentals
"""

import os
import sqlite3
import json
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
import time

import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..data_module.data_fetcher import DataFetcher
from ..indicator_dashboard.technical_analysis import TechnicalAnalysis
from ..sentiment_module.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: List[str] = field(default_factory=list)
    query_type: str = ""


@dataclass
class SearchResult:
    """Represents a search result from vector store"""
    text: str
    score: float
    source_type: str
    metadata: Dict[str, Any]


class LRUCache:
    """Thread-safe LRU Cache for embeddings and responses"""

    def __init__(self, maxsize: int = 100):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.maxsize:
                    self.cache.popitem(last=False)
            self.cache[key] = {
                'value': value,
                'expires': datetime.now() + timedelta(seconds=ttl_seconds)
            }

    def is_valid(self, key: str) -> bool:
        with self.lock:
            if key not in self.cache:
                return False
            entry = self.cache[key]
            if datetime.now() > entry['expires']:
                del self.cache[key]
                return False
            return True

    def get_value(self, key: str) -> Optional[Any]:
        if self.is_valid(key):
            return self.cache[key]['value']
        return None


class UnifiedDataRetriever:
    """
    Unified data retriever that efficiently fetches and caches data from all sources:
    - Stock prices and fundamentals
    - Technical indicators
    - Sentiment analysis
    - SEC filings
    """

    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalyzer()

        self.data_cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        self.cache = LRUCache(maxsize=50)

        # Cache TTLs in seconds
        self.ttl_realtime = 60  # 1 minute for real-time data
        self.ttl_fundamentals = 3600  # 1 hour for fundamentals
        self.ttl_technical = 300  # 5 minutes for technical
        self.ttl_sentiment = 1800  # 30 minutes for sentiment
        self.ttl_sec = 86400  # 24 hours for SEC data

    def get_comprehensive_data(self, ticker: str) -> Dict[str, Any]:
        """Get all available data for a ticker efficiently"""
        cache_key = f"comprehensive_{ticker}"
        cached = self.cache.get_value(cache_key)
        if cached:
            return cached

        data = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'realtime': self._get_realtime_data(ticker),
            'fundamentals': self._get_fundamentals_data(ticker),
            'technical': self._get_technical_data(ticker),
            'sentiment': self._get_sentiment_data(ticker),
            'sec_data': self._get_sec_data(ticker),
            'stock_info': self._get_stock_info(ticker)
        }

        # Cache for 2 minutes (shortest TTL component)
        self.cache.set(cache_key, data, ttl_seconds=120)
        return data

    def _get_realtime_data(self, ticker: str) -> Optional[Dict]:
        """Get real-time price data"""
        try:
            return self.data_fetcher.get_realtime_price(ticker)
        except Exception as e:
            logger.warning(f"Error fetching realtime data for {ticker}: {e}")
            return None

    def _get_fundamentals_data(self, ticker: str) -> Optional[Dict]:
        """Get fundamental metrics"""
        try:
            return self.data_fetcher.get_fundamentals(ticker)
        except Exception as e:
            logger.warning(f"Error fetching fundamentals for {ticker}: {e}")
            return None

    def _get_stock_info(self, ticker: str) -> Optional[Dict]:
        """Get company info"""
        try:
            return self.data_fetcher.get_stock_info(ticker)
        except Exception as e:
            logger.warning(f"Error fetching stock info for {ticker}: {e}")
            return None

    def _get_technical_data(self, ticker: str) -> Optional[Dict]:
        """Get technical indicators and signals"""
        try:
            signals = self.technical_analyzer.get_signal_analysis(ticker)
            # Use calculate_indicators which returns indicator values
            indicators = self.technical_analyzer.calculate_indicators(ticker)
            return {
                'signals': signals,
                'indicators': indicators
            }
        except Exception as e:
            logger.warning(f"Error fetching technical data for {ticker}: {e}")
            return None

    def _get_sentiment_data(self, ticker: str) -> Optional[Dict]:
        """Get sentiment analysis from cache or fresh"""
        try:
            # First try cached sentiment from database
            cached = self._get_cached_sentiment(ticker)
            if cached:
                return cached

            # Fall back to fresh analysis
            sentiment = self.sentiment_analyzer.analyze_ticker_sentiment(ticker)
            return sentiment
        except Exception as e:
            logger.warning(f"Error fetching sentiment for {ticker}: {e}")
            return None

    def _get_cached_sentiment(self, ticker: str) -> Optional[Dict]:
        """Get sentiment from SQLite cache"""
        try:
            db_path = self.data_cache_dir / "sentiment_data.db"
            if not db_path.exists():
                return None

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get summary
            cursor.execute("""
                SELECT overall_sentiment, weighted_sentiment, sentiment_label,
                       positive_count, negative_count, neutral_count, total_articles
                FROM ticker_sentiment_summary
                WHERE ticker = ?
            """, (ticker,))
            summary = cursor.fetchone()

            # Get recent headlines
            cursor.execute("""
                SELECT title, sentiment_score, sentiment_label, source, published, time_weight, url
                FROM news_sentiment
                WHERE ticker = ?
                ORDER BY published DESC
                LIMIT 10
            """, (ticker,))
            headlines = cursor.fetchall()

            conn.close()

            if not summary:
                return None

            return {
                'overall_sentiment': summary[0],
                'weighted_sentiment': summary[1],
                'sentiment_label': summary[2],
                'positive_count': summary[3],
                'negative_count': summary[4],
                'neutral_count': summary[5],
                'total_articles': summary[6],
                'headlines': [
                    {
                        'title': h[0],
                        'sentiment_score': h[1],
                        'sentiment_label': h[2],
                        'source': h[3],
                        'published': h[4],
                        'time_weight': h[5],
                        'url': h[6]
                    } for h in headlines
                ]
            }
        except Exception as e:
            logger.warning(f"Error reading cached sentiment: {e}")
            return None

    def _get_sec_data(self, ticker: str) -> Optional[Dict]:
        """Get SEC filing data from cache"""
        try:
            sec_data = {}

            # Try SEC filings database
            sec_filings_path = self.data_cache_dir / "sec_filings.db"
            if sec_filings_path.exists():
                conn = sqlite3.connect(sec_filings_path)
                cursor = conn.cursor()

                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [t[0] for t in cursor.fetchall()]

                for table in tables:
                    try:
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = [col[1] for col in cursor.fetchall()]

                        if 'ticker' in columns:
                            cursor.execute(f"SELECT * FROM {table} WHERE ticker = ? LIMIT 5", (ticker,))
                        else:
                            continue

                        rows = cursor.fetchall()
                        if rows:
                            sec_data[table] = [dict(zip(columns, row)) for row in rows]
                    except:
                        continue

                conn.close()

            # Try SEC financial data database
            sec_financial_path = self.data_cache_dir / "sec_financial_data.db"
            if sec_financial_path.exists():
                conn = sqlite3.connect(sec_financial_path)
                cursor = conn.cursor()

                try:
                    cursor.execute("""
                        SELECT fiscal_year, fiscal_period, revenue, net_income,
                               total_assets, eps, filing_date
                        FROM financial_data
                        WHERE ticker = ?
                        ORDER BY fiscal_year DESC, fiscal_period DESC
                        LIMIT 5
                    """, (ticker,))
                    rows = cursor.fetchall()
                    if rows:
                        sec_data['financial_data'] = [
                            {
                                'fiscal_year': r[0],
                                'fiscal_period': r[1],
                                'revenue': r[2],
                                'net_income': r[3],
                                'total_assets': r[4],
                                'eps': r[5],
                                'filing_date': r[6]
                            } for r in rows
                        ]
                except:
                    pass

                conn.close()

            return sec_data if sec_data else None

        except Exception as e:
            logger.warning(f"Error fetching SEC data for {ticker}: {e}")
            return None


class OptimizedVectorStore:
    """
    Optimized vector store with:
    - Lazy loading of indices
    - Batch indexing
    - Smart caching
    - Fast similarity search
    """

    def __init__(self):
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "vector_store"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = None
        self.embedding_dim = 384
        self.indices = {}
        self.documents = {}
        self.metadata = {}

        self.embedding_cache = LRUCache(maxsize=500)
        self.indexed_tickers = set()

        self.index_types = ['fundamentals', 'technical', 'sentiment', 'news', 'sec_filings', 'sec_financial']

        self._lazy_init = False

    def _ensure_initialized(self):
        """Lazy initialization of embedder and indices"""
        if self._lazy_init:
            return True

        if not SENTENCE_TRANSFORMERS_AVAILABLE or not FAISS_AVAILABLE:
            logger.warning("SentenceTransformers or FAISS not available")
            return False

        try:
            # Load embedder (this is the slow part)
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

            # Initialize indices
            for idx_type in self.index_types:
                self.indices[idx_type] = faiss.IndexFlatIP(self.embedding_dim)
                self.documents[idx_type] = []
                self.metadata[idx_type] = []

            # Load existing indices
            self._load_indices()

            self._lazy_init = True
            logger.info("Vector store initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return False

    def _load_indices(self):
        """Load existing indices from disk"""
        try:
            for idx_type in self.index_types:
                index_path = self.cache_dir / f"faiss_{idx_type}_index.bin"
                docs_path = self.cache_dir / f"{idx_type}_documents.pkl"
                meta_path = self.cache_dir / f"{idx_type}_metadata.pkl"

                if all(p.exists() for p in [index_path, docs_path, meta_path]):
                    import pickle

                    self.indices[idx_type] = faiss.read_index(str(index_path))

                    with open(docs_path, 'rb') as f:
                        self.documents[idx_type] = pickle.load(f)

                    with open(meta_path, 'rb') as f:
                        self.metadata[idx_type] = pickle.load(f)

                    # Track indexed tickers
                    for meta in self.metadata[idx_type]:
                        if 'ticker' in meta:
                            self.indexed_tickers.add(meta['ticker'].upper())

                    logger.debug(f"Loaded {idx_type} index with {len(self.documents[idx_type])} docs")
        except Exception as e:
            logger.warning(f"Could not load indices: {e}")

    def _save_indices(self):
        """Save indices to disk"""
        try:
            import pickle

            for idx_type in self.index_types:
                if self.indices[idx_type].ntotal > 0:
                    index_path = self.cache_dir / f"faiss_{idx_type}_index.bin"
                    docs_path = self.cache_dir / f"{idx_type}_documents.pkl"
                    meta_path = self.cache_dir / f"{idx_type}_metadata.pkl"

                    faiss.write_index(self.indices[idx_type], str(index_path))

                    with open(docs_path, 'wb') as f:
                        pickle.dump(self.documents[idx_type], f)

                    with open(meta_path, 'wb') as f:
                        pickle.dump(self.metadata[idx_type], f)
        except Exception as e:
            logger.error(f"Error saving indices: {e}")

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding with caching"""
        if not self._ensure_initialized():
            return None

        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cached = self.embedding_cache.get_value(cache_key)
        if cached is not None:
            return cached

        # Generate embedding
        try:
            embedding = self.embedder.encode([text])
            faiss.normalize_L2(embedding)

            # Cache for 1 hour
            self.embedding_cache.set(cache_key, embedding, ttl_seconds=3600)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def index_data(self, ticker: str, data: Dict[str, Any]) -> int:
        """Index all data for a ticker"""
        if not self._ensure_initialized():
            return 0

        indexed_count = 0
        ticker = ticker.upper()

        # Skip if already indexed recently
        if ticker in self.indexed_tickers:
            return 0

        try:
            # Index fundamentals
            if data.get('fundamentals'):
                text = self._create_fundamentals_text(ticker, data['fundamentals'], data.get('stock_info'))
                if self._add_to_index('fundamentals', text, ticker, 'fundamentals'):
                    indexed_count += 1

            # Index technical
            if data.get('technical') and data['technical'].get('signals'):
                text = self._create_technical_text(ticker, data['technical'])
                if self._add_to_index('technical', text, ticker, 'technical'):
                    indexed_count += 1

            # Index sentiment
            if data.get('sentiment'):
                text = self._create_sentiment_text(ticker, data['sentiment'])
                if self._add_to_index('sentiment', text, ticker, 'sentiment'):
                    indexed_count += 1

                # Index news headlines
                headlines = data['sentiment'].get('headlines', [])
                for headline in headlines[:5]:
                    news_text = f"News for {ticker}: {headline.get('title', '')} - Sentiment: {headline.get('sentiment_label', 'Unknown')}"
                    if self._add_to_index('news', news_text, ticker, 'news', source=headline.get('source', 'Unknown')):
                        indexed_count += 1

            # Index SEC data
            if data.get('sec_data'):
                for sec_type, sec_records in data['sec_data'].items():
                    if sec_records:
                        text = self._create_sec_text(ticker, sec_type, sec_records)
                        idx_type = 'sec_financial' if 'financial' in sec_type else 'sec_filings'
                        if self._add_to_index(idx_type, text, ticker, sec_type):
                            indexed_count += 1

            if indexed_count > 0:
                self.indexed_tickers.add(ticker)
                self._save_indices()
                logger.info(f"Indexed {indexed_count} documents for {ticker}")

        except Exception as e:
            logger.error(f"Error indexing data for {ticker}: {e}")

        return indexed_count

    def _add_to_index(self, index_type: str, text: str, ticker: str,
                      data_type: str, source: str = None) -> bool:
        """Add document to specific index"""
        try:
            embedding = self._get_embedding(text)
            if embedding is None:
                return False

            self.indices[index_type].add(embedding)
            self.documents[index_type].append(text)
            self.metadata[index_type].append({
                'ticker': ticker,
                'type': data_type,
                'source': source or data_type,
                'timestamp': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Error adding to {index_type} index: {e}")
            return False

    def search(self, query: str, ticker: str = None, top_k: int = 3) -> Dict[str, List[SearchResult]]:
        """Search across all indices"""
        if not self._ensure_initialized():
            return {}

        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return {}

        results = {}

        for idx_type, index in self.indices.items():
            if index.ntotal == 0:
                continue

            try:
                k = min(top_k, index.ntotal)
                scores, indices = index.search(query_embedding, k)

                idx_results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self.documents[idx_type]):
                        continue

                    meta = self.metadata[idx_type][idx]

                    # Filter by ticker if specified
                    if ticker and meta.get('ticker', '').upper() != ticker.upper():
                        continue

                    # Only include relevant results
                    if score > 0.15:
                        idx_results.append(SearchResult(
                            text=self.documents[idx_type][idx],
                            score=float(score),
                            source_type=idx_type,
                            metadata=meta
                        ))

                if idx_results:
                    idx_results.sort(key=lambda x: x.score, reverse=True)
                    results[idx_type] = idx_results

            except Exception as e:
                logger.warning(f"Error searching {idx_type}: {e}")

        return results

    def _create_fundamentals_text(self, ticker: str, fundamentals: Dict, stock_info: Dict = None) -> str:
        """Create text for fundamentals data"""
        parts = [f"Fundamental analysis for {ticker}:"]

        if stock_info:
            if stock_info.get('company_name'):
                parts.append(f"Company: {stock_info['company_name']}")
            if stock_info.get('sector'):
                parts.append(f"Sector: {stock_info['sector']}")
            if stock_info.get('industry'):
                parts.append(f"Industry: {stock_info['industry']}")

        for key, value in fundamentals.items():
            if value and value != 'N/A':
                parts.append(f"{key}: {value}")

        return " | ".join(parts)

    def _create_technical_text(self, ticker: str, technical: Dict) -> str:
        """Create text for technical data"""
        parts = [f"Technical analysis for {ticker}:"]

        signals = technical.get('signals', {})
        for signal_type, signal_value in signals.items():
            parts.append(f"{signal_type.replace('_', ' ')}: {signal_value}")

        return " | ".join(parts)

    def _create_sentiment_text(self, ticker: str, sentiment: Dict) -> str:
        """Create text for sentiment data"""
        parts = [f"Sentiment analysis for {ticker}:"]
        parts.append(f"Overall sentiment: {sentiment.get('sentiment_label', 'Unknown')} ({sentiment.get('overall_sentiment', 0):.3f})")
        parts.append(f"Articles analyzed: {sentiment.get('total_articles', 0)}")
        parts.append(f"Positive: {sentiment.get('positive_count', 0)}")
        parts.append(f"Negative: {sentiment.get('negative_count', 0)}")
        parts.append(f"Neutral: {sentiment.get('neutral_count', 0)}")

        return " | ".join(parts)

    def _create_sec_text(self, ticker: str, sec_type: str, records: List[Dict]) -> str:
        """Create text for SEC data"""
        parts = [f"SEC {sec_type} data for {ticker}:"]

        if records:
            latest = records[0]
            for key, value in latest.items():
                if key not in ['ticker', 'id', 'cik', 'raw_data'] and value:
                    # Format large numbers
                    if isinstance(value, (int, float)) and abs(value) > 1000000:
                        value = f"${value/1000000:.1f}M"
                    parts.append(f"{key}: {value}")

        return " | ".join(parts)


class StreamingOpenAIService:
    """
    OpenAI service with streaming support for better UX
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        self.model = "gpt-4o-mini"  # Fast, cost-effective model

        if self.api_key and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"OpenAI init error: {e}")

    def generate_response(self, query: str, context: str, ticker: str = None) -> str:
        """Generate non-streaming response"""
        if not self.client:
            return self._generate_fallback_response(query, ticker, context)

        try:
            prompt = self._build_prompt(query, context, ticker)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return self._generate_fallback_response(query, ticker, context)

    def generate_streaming_response(self, query: str, context: str, ticker: str = None) -> Generator[str, None, None]:
        """Generate streaming response for real-time display"""
        if not self.client:
            yield self._generate_fallback_response(query, ticker, context)
            return

        try:
            prompt = self._build_prompt(query, context, ticker)

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield self._generate_fallback_response(query, ticker, context)

    def _get_system_prompt(self) -> str:
        return """You are an expert stock analyst assistant. Provide clear, concise, and actionable analysis.

Guidelines:
- Use bullet points for readability
- Include specific numbers and data points
- Cite data sources when available
- Be direct and avoid unnecessary jargon
- If data is missing, acknowledge it
- For price-related queries, always include current price if available
- For technical queries, explain what the signals mean
- For sentiment queries, summarize the market mood
- For SEC queries, highlight key financial metrics

Format your response with clear sections using **bold headers** when appropriate."""

    def _build_prompt(self, query: str, context: str, ticker: str) -> str:
        return f"""Stock Analysis Request for {ticker or 'General'}:

**User Query:** {query}

**Available Data:**
{context[:3000]}

Please provide a comprehensive but concise analysis addressing the user's query. Include:
1. Direct answer to the question
2. Key supporting data points
3. Any relevant insights or warnings
4. Data sources used"""

    def _generate_fallback_response(self, query: str, ticker: str, context: str) -> str:
        """Generate response when API is unavailable"""
        ticker_str = ticker or "the requested stock"

        # Parse context for key data points
        response_parts = [f"**Analysis for {ticker_str}:**\n"]

        if context:
            response_parts.append("**Available Data:**")

            # Extract key information from context
            if "price" in context.lower() or "Current price" in context:
                response_parts.append("- Real-time price data available")
            if "technical" in context.lower():
                response_parts.append("- Technical indicators calculated (RSI, MACD, Moving Averages)")
            if "sentiment" in context.lower():
                response_parts.append("- News sentiment analysis available")
            if "fundamental" in context.lower():
                response_parts.append("- Fundamental metrics available (P/E, EPS, ROE)")
            if "sec" in context.lower():
                response_parts.append("- SEC filing data available")

            response_parts.append(f"\n**Your Query:** {query}")
            response_parts.append("\n**Note:** AI-enhanced analysis temporarily unavailable. Raw data shown above.")
        else:
            response_parts.append("No data currently available. Please ensure the ticker is valid and try again.")

        return "\n".join(response_parts)


class EnhancedRAGChatbot:
    """
    Enhanced RAG Chatbot with:
    - Fast response times (< 3 seconds target)
    - Streaming response support
    - Unified data retrieval
    - Smart caching
    - Comprehensive context from all sources
    """

    def __init__(self, openai_api_key: str = None):
        self.data_retriever = UnifiedDataRetriever()
        self.vector_store = OptimizedVectorStore()
        self.llm_service = StreamingOpenAIService(openai_api_key)

        self.response_cache = LRUCache(maxsize=100)

        # Knowledge base
        self.db_path = Path(__file__).parent.parent.parent / "data" / "vector_store" / "chatbot_knowledge.db"
        self._init_database()

        # Conversation context
        self.conversation_history: List[ChatMessage] = []
        self.max_history = 10

    def _init_database(self):
        """Initialize conversation database"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    user_query TEXT,
                    bot_response TEXT,
                    query_type TEXT,
                    sources TEXT,
                    response_time_ms INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database init error: {e}")

    def get_response(self, query: str, ticker: str = None) -> str:
        """
        Get chatbot response (non-streaming)
        Target: < 3 seconds response time
        """
        start_time = time.time()

        try:
            # Check response cache
            cache_key = f"{ticker}:{query}"
            cached_response = self.response_cache.get_value(cache_key)
            if cached_response:
                return cached_response

            # Get comprehensive data
            data = self.data_retriever.get_comprehensive_data(ticker) if ticker else {}

            # Index new data
            if ticker and data:
                self.vector_store.index_data(ticker, data)

            # Build context
            context = self._build_context(query, ticker, data)

            # Classify query
            query_type = self._classify_query(query)

            # Generate response
            response = self.llm_service.generate_response(query, context, ticker)

            # Cache response (5 minutes TTL)
            self.response_cache.set(cache_key, response, ttl_seconds=300)

            # Store conversation
            response_time = int((time.time() - start_time) * 1000)
            self._store_conversation(ticker, query, response, query_type, response_time)

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error. Please try again."

    def get_streaming_response(self, query: str, ticker: str = None) -> Generator[str, None, None]:
        """
        Get streaming chatbot response for real-time display
        """
        start_time = time.time()

        try:
            # Get comprehensive data
            data = self.data_retriever.get_comprehensive_data(ticker) if ticker else {}

            # Index new data
            if ticker and data:
                self.vector_store.index_data(ticker, data)

            # Build context
            context = self._build_context(query, ticker, data)

            # Classify query
            query_type = self._classify_query(query)

            # Stream response
            full_response = []
            for chunk in self.llm_service.generate_streaming_response(query, context, ticker):
                full_response.append(chunk)
                yield chunk

            # Store conversation
            response_text = "".join(full_response)
            response_time = int((time.time() - start_time) * 1000)
            self._store_conversation(ticker, query, response_text, query_type, response_time)

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield "I apologize, but I encountered an error. Please try again."

    def _build_context(self, query: str, ticker: str, data: Dict) -> str:
        """Build comprehensive context from all data sources"""
        context_parts = []
        sources = []

        # Add real-time price data (highest priority for price queries)
        if data.get('realtime'):
            rt = data['realtime']
            context_parts.append(f"**Current Price Data:**")
            context_parts.append(f"- Price: ${rt.get('price', 'N/A')}")
            context_parts.append(f"- Change: {rt.get('change', 0):+.2f} ({rt.get('change_percent', 0):+.2f}%)")
            context_parts.append(f"- Volume: {rt.get('volume', 'N/A'):,}" if isinstance(rt.get('volume'), (int, float)) else "")
            sources.append("Real-time Market Data")

        # Add stock info
        if data.get('stock_info'):
            info = data['stock_info']
            context_parts.append(f"\n**Company Info:**")
            context_parts.append(f"- Name: {info.get('company_name', 'N/A')}")
            context_parts.append(f"- Sector: {info.get('sector', 'N/A')}")
            context_parts.append(f"- Industry: {info.get('industry', 'N/A')}")

        # Add fundamentals
        if data.get('fundamentals'):
            fund = data['fundamentals']
            context_parts.append(f"\n**Fundamental Metrics:**")
            for key, value in fund.items():
                if value and value != 'N/A':
                    context_parts.append(f"- {key}: {value}")
            sources.append("Fundamental Data")

        # Add technical signals
        if data.get('technical') and data['technical'].get('signals'):
            signals = data['technical']['signals']
            context_parts.append(f"\n**Technical Signals:**")
            for signal, value in signals.items():
                context_parts.append(f"- {signal.replace('_', ' ').title()}: {value}")
            sources.append("Technical Analysis")

        # Add sentiment
        if data.get('sentiment'):
            sent = data['sentiment']
            context_parts.append(f"\n**Market Sentiment:**")
            context_parts.append(f"- Overall: {sent.get('sentiment_label', 'Unknown')} ({sent.get('overall_sentiment', 0):.2f})")
            context_parts.append(f"- Based on {sent.get('total_articles', 0)} articles")
            context_parts.append(f"- Positive: {sent.get('positive_count', 0)} | Negative: {sent.get('negative_count', 0)} | Neutral: {sent.get('neutral_count', 0)}")

            # Add top headlines
            headlines = sent.get('headlines', [])[:3]
            if headlines:
                context_parts.append(f"\n**Recent Headlines:**")
                for h in headlines:
                    context_parts.append(f"- [{h.get('sentiment_label', 'N/A')}] {h.get('title', 'No title')[:80]}")
            sources.append("News Sentiment Analysis")

        # Add SEC data
        if data.get('sec_data'):
            context_parts.append(f"\n**SEC Filing Data:**")
            for sec_type, records in data['sec_data'].items():
                if records:
                    latest = records[0]
                    context_parts.append(f"- {sec_type}:")
                    for key, value in latest.items():
                        if key not in ['ticker', 'id', 'cik', 'raw_data'] and value:
                            if isinstance(value, (int, float)) and abs(value) > 1000000:
                                context_parts.append(f"  - {key}: ${value/1000000:.1f}M")
                            else:
                                context_parts.append(f"  - {key}: {value}")
            sources.append("SEC EDGAR Filings")

        # Add vector search results for additional context
        search_results = self.vector_store.search(query, ticker, top_k=2)
        for idx_type, results in search_results.items():
            if results and idx_type not in ['news']:  # Avoid duplicate news
                for result in results[:1]:
                    if result.score > 0.3:  # High relevance only
                        context_parts.append(f"\n**Additional {idx_type.title()} Context:**")
                        context_parts.append(f"- {result.text[:200]}...")

        # Add sources
        if sources:
            context_parts.append(f"\n**Data Sources:** {', '.join(sources)}")

        return "\n".join(context_parts)

    def _classify_query(self, query: str) -> str:
        """Classify query type for better response handling"""
        query_lower = query.lower()

        keywords = {
            'price': ['price', 'cost', 'worth', 'value', 'trading', 'quote'],
            'technical': ['technical', 'indicator', 'rsi', 'macd', 'moving average', 'bollinger', 'signal', 'chart'],
            'fundamental': ['fundamental', 'pe ratio', 'p/e', 'eps', 'roe', 'debt', 'earnings', 'revenue', 'profit'],
            'sentiment': ['sentiment', 'news', 'opinion', 'mood', 'feeling', 'outlook'],
            'sec': ['sec', 'filing', '10-k', '10-q', 'annual report', 'quarterly'],
            'comparison': ['compare', 'versus', 'vs', 'better', 'worse', 'difference']
        }

        for query_type, words in keywords.items():
            if any(word in query_lower for word in words):
                return query_type

        return 'general'

    def _store_conversation(self, ticker: str, query: str, response: str,
                           query_type: str, response_time: int):
        """Store conversation for history"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO conversations (ticker, user_query, bot_response, query_type, response_time_ms)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, query, response, query_type, response_time))
            conn.commit()
            conn.close()

            # Update in-memory history
            self.conversation_history.append(ChatMessage(
                role='user',
                content=query,
                query_type=query_type
            ))
            self.conversation_history.append(ChatMessage(
                role='assistant',
                content=response
            ))

            # Trim history
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-(self.max_history * 2):]

        except Exception as e:
            logger.warning(f"Error storing conversation: {e}")

    def get_conversation_history(self, ticker: str = None, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        try:
            conn = sqlite3.connect(self.db_path)

            if ticker:
                query = """
                    SELECT user_query, bot_response, query_type, response_time_ms, timestamp
                    FROM conversations
                    WHERE ticker = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                cursor = conn.execute(query, (ticker, limit))
            else:
                query = """
                    SELECT user_query, bot_response, query_type, response_time_ms, timestamp
                    FROM conversations
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                cursor = conn.execute(query, (limit,))

            results = cursor.fetchall()
            conn.close()

            return [
                {
                    'query': r[0],
                    'response': r[1],
                    'type': r[2],
                    'response_time_ms': r[3],
                    'timestamp': r[4]
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_suggested_questions(self, ticker: str) -> List[str]:
        """Get suggested questions based on available data"""
        suggestions = []

        if ticker:
            suggestions = [
                f"What's the current price and trend for {ticker}?",
                f"Show me the technical analysis for {ticker}",
                f"What's the market sentiment for {ticker}?",
                f"Give me the fundamental metrics for {ticker}",
                f"What are the recent SEC filings for {ticker}?",
                f"Provide a comprehensive analysis of {ticker}"
            ]
        else:
            suggestions = [
                "Search for a stock ticker to get started",
                "Try asking about AAPL, MSFT, GOOGL, or TSLA"
            ]

        return suggestions
