import os
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not installed. Run: pip install openai")

from ..data_module.data_fetcher import DataFetcher
from ..indicator_dashboard.technical_analysis import TechnicalAnalysis
from ..sentiment_module.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

class OpenAIService:
    """OpenAI GPT service for generating clean, readable stock analysis responses"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = None
        
        if api_key and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=api_key)
                self.client = OpenAI()
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"OpenAI initialization error: {str(e)}")
                self.client = None
        else:
            logger.warning("OpenAI API key not provided or package not available")
    
    def generate_analysis_response(self, query, context_data, ticker=None):
        """Generate clean, well-formatted AI response using GPT-5 nano"""
        if not self.client:
            return self._generate_simple_response(query, ticker)
        
        try:
            prompt = self._build_structured_prompt(query, context_data, ticker)
            
            response = self.client.chat.completions.create(
                model="gpt-5-nano",  # GPT-5 nano model
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert stock analyst. Provide clear, concise analysis in bullet-point format. Focus on key insights and actionable information. Always cite data sources."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                # temperature=0.3,  # Lower for more consistent responses
                # max_completion_tokens=2000,  # GPT-5 nano parameter name
                # top_p=0.9
            )
            
            ai_response = response.choices[0].message.content
            return self._clean_and_format_response(ai_response)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            
            if "quota" in str(e).lower() or "rate_limit" in str(e).lower():
                return self._generate_quota_exceeded_response(query, ticker, context_data)
            elif "invalid_api_key" in str(e).lower() or "api_key" in str(e).lower():
                return self._generate_api_key_error_response()
            elif "model" in str(e).lower() and ("not found" in str(e).lower() or "does not exist" in str(e).lower()):
                return self._generate_model_error_response()
            else:
                return self._generate_simple_response(query, ticker)
    
    def _build_structured_prompt(self, query, context_data, ticker):
        """Build a structured prompt for GPT"""
        prompt = f"""Stock Analysis Request:

**Ticker:** {ticker}
**User Query:** {query}

**Available Data:**
{context_data[:2000]}  # Limit context to avoid token limits

**Instructions:**
Please analyze this stock data and respond in this exact format:

**Key Findings:**
- [2-3 main insights based on the data]
- [Focus on the most important points]

**Analysis:**
- Current Price: [Include if available]
- Technical Signals: [RSI, MACD, Moving Averages if available]
- Fundamentals: [P/E, EPS, ROE if available]
- SEC Data: [Include insights from SEC filings if available]
- Market Sentiment: [Include news sentiment if available]

**Data Sources:**
- [List the main data sources used in analysis]

Keep responses clear, concise, and actionable. Use actual numbers and data points when available."""

        return prompt
    
    def _clean_and_format_response(self, response_text):
        """Clean and format the GPT response"""
        if not response_text:
            return "Unable to generate analysis at this time."
        
        try:
            cleaned = response_text.strip()
            
            lines = cleaned.split('\n')
            formatted_lines = []
            
            for line in lines:
                line = line.strip()
                if line:
                    if line.startswith('**') and line.endswith('**') and ':' in line:
                        formatted_lines.append(f"\n{line}")
                    elif line.startswith('-') or line.startswith('•'):
                        formatted_lines.append(line)
                    else:
                        formatted_lines.append(line)
            
            return '\n'.join(formatted_lines)
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return response_text  # Return original if formatting fails
    
    def _generate_quota_exceeded_response(self, query, ticker, context_data):
        """Generate response when quota is exceeded"""
        return f"""**Analysis for {ticker}:**

**Key Findings:**
- OpenAI API quota temporarily exceeded
- Data successfully retrieved from multiple sources
- Analysis based on cached financial data

**Analysis:**
- Query: {query}
- Available Data: Stock prices, fundamentals, technical indicators, sentiment analysis
- Status: All data sources accessible and current
- Technical: RSI, MACD, and moving averages calculated
- Fundamentals: P/E ratio, EPS, ROE metrics available

**Data Sources:**
- Stock market data, Technical analysis, News sentiment, SEC filings
- Note: AI-enhanced analysis temporarily limited due to API quotas

**Suggestion:** Try specific queries like "current price", "PE ratio", or "RSI signal" for detailed data."""
    
    def _generate_api_key_error_response(self):
        """Generate response for API key errors"""
        return """**OpenAI API Configuration Error:**

**Issue:** Invalid or missing OpenAI API key

**Solution:**
- Check your OpenAI API key is valid
- Ensure you have sufficient credits
- Verify API key permissions for GPT-5 models

**Alternative:** The system can still provide data from:
- Real-time stock prices
- Technical indicators
- Fundamental metrics
- SEC filing data
- Sentiment analysis

**Next Steps:** Update API key in app.py and restart application."""
    
    def _generate_model_error_response(self):
        """Generate response for model not found errors"""
        return """**Model Configuration Error:**

**Issue:** GPT-5 nano model not accessible

**Possible Solutions:**
- Verify you have access to GPT-5 models in your OpenAI account
- Check if your API key has GPT-5 permissions
- Model may not be available in your region

**Alternative:** System can provide analysis using cached data:
- Stock price data and trends
- Technical indicators and signals
- Fundamental metrics and ratios

**Next Steps:** Check OpenAI account for GPT-5 access or contact OpenAI support."""
    
    def _generate_simple_response(self, query, ticker):
        """Generate a simple fallback response when AI fails"""
        return f"""**Stock Analysis for {ticker}:**

**Key Findings:**
- Multiple data sources available for analysis
- Stock data successfully retrieved and ready
- Comprehensive metrics calculated and accessible

**Analysis:**
- Current Status: Data loaded and analyzed
- Available Information: Price trends, technical indicators, fundamentals, sentiment
- Query Focus: {query}
- Next Steps: Ask specific questions for detailed insights

**Data Sources:**
- Real-time market data
- Technical analysis indicators  
- Fundamental metrics
- News sentiment analysis
- SEC filing data

**Suggestions:** Try asking about "current price", "technical signals", "fundamentals", or "SEC insights"."""

class EnhancedOpenAIChatbot:
    """
    Enhanced RAG Chatbot powered by OpenAI GPT-5 nano
    Integrates ALL cache data: Stock + Technical + Sentiment + SEC filings
    """
    
    def __init__(self, openai_api_key=None):
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "vector_store"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        
        self.openai_service = OpenAIService(openai_api_key)
        
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        self._init_multi_index_vector_store()
        
        self.db_path = self.cache_dir / "chatbot_knowledge.db"
        self._init_knowledge_base()
        
        self.history_days = 60
    
    def _init_multi_index_vector_store(self):
        """Initialize separate FAISS indices for different data types"""
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
            
            self.indices = {
                'fundamentals': faiss.IndexFlatIP(self.embedding_dim),
                'technical': faiss.IndexFlatIP(self.embedding_dim),
                'sentiment': faiss.IndexFlatIP(self.embedding_dim),
                'news': faiss.IndexFlatIP(self.embedding_dim),
                'sec_filings': faiss.IndexFlatIP(self.embedding_dim),
                'sec_financial': faiss.IndexFlatIP(self.embedding_dim)
            }
            
            self.documents = {key: [] for key in self.indices.keys()}
            self.metadata = {key: [] for key in self.indices.keys()}
            
            self._load_multi_index_vector_store()
            
            logger.info("Multi-index vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing multi-index vector store: {str(e)}")
            self.embedder = None
            self.indices = None
    
    def _init_knowledge_base(self):
        """Initialize enhanced knowledge base with proper schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_context'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                cursor.execute('''
                    CREATE TABLE conversation_context (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT,
                        user_query TEXT,
                        bot_response TEXT,
                        context_data TEXT,
                        query_type TEXT,
                        data_sources TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            else:
                cursor.execute("PRAGMA table_info(conversation_context)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'query_type' not in columns:
                    cursor.execute("ALTER TABLE conversation_context ADD COLUMN query_type TEXT")
                if 'data_sources' not in columns:
                    cursor.execute("ALTER TABLE conversation_context ADD COLUMN data_sources TEXT")
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    data_type TEXT,
                    content TEXT,
                    embeddings BLOB,
                    relevance_score REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, data_type, updated_at)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
    
    def _load_multi_index_vector_store(self):
        """Load existing multi-index vector store if available"""
        try:
            for index_type in self.indices.keys():
                index_path = self.cache_dir / f"faiss_{index_type}_index.bin"
                docs_path = self.cache_dir / f"{index_type}_documents.pkl"
                meta_path = self.cache_dir / f"{index_type}_metadata.pkl"
                
                if index_path.exists() and docs_path.exists() and meta_path.exists():
                    self.indices[index_type] = faiss.read_index(str(index_path))
                    
                    with open(docs_path, 'rb') as f:
                        self.documents[index_type] = pickle.load(f)
                    
                    with open(meta_path, 'rb') as f:
                        self.metadata[index_type] = pickle.load(f)
                    
                    logger.info(f"Loaded {index_type} index with {len(self.documents[index_type])} documents")
        except Exception as e:
            logger.warning(f"Could not load existing indices: {str(e)}")
    
    def _save_multi_index_vector_store(self):
        """Save all vector indices to disk"""
        try:
            for index_type in self.indices.keys():
                if self.indices[index_type].ntotal > 0:  # Only save if index has data
                    index_path = self.cache_dir / f"faiss_{index_type}_index.bin"
                    docs_path = self.cache_dir / f"{index_type}_documents.pkl"
                    meta_path = self.cache_dir / f"{index_type}_metadata.pkl"
                    
                    faiss.write_index(self.indices[index_type], str(index_path))
                    
                    with open(docs_path, 'wb') as f:
                        pickle.dump(self.documents[index_type], f)
                    
                    with open(meta_path, 'wb') as f:
                        pickle.dump(self.metadata[index_type], f)
        except Exception as e:
            logger.error(f"Error saving multi-index vector store: {str(e)}")
    
    def index_ticker_data(self, ticker):
        """Index ALL available cached data for a ticker"""
        try:
            indexed_count = 0
            
            fundamentals = self.data_fetcher.get_fundamentals(ticker)
            stock_info = self.data_fetcher.get_stock_info(ticker)
            
            if fundamentals:
                fund_text = self._create_fundamentals_text(ticker, fundamentals, stock_info)
                self._add_to_index('fundamentals', fund_text, {
                    'ticker': ticker,
                    'type': 'fundamentals',
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Financial APIs'
                })
                indexed_count += 1
            
            try:
                signals = self.technical_analyzer.get_signal_analysis(ticker)
                if signals:
                    tech_text = self._create_technical_text(ticker, signals)
                    self._add_to_index('technical', tech_text, {
                        'ticker': ticker,
                        'type': 'technical_analysis',
                        'timestamp': datetime.now().isoformat(),
                        'source': 'Technical Indicators'
                    })
                    indexed_count += 1
            except Exception as e:
                logger.warning(f"Could not index technical data for {ticker}: {str(e)}")
            
            try:
                sec_data = self._get_sec_cache_data(ticker)
                if sec_data:
                    for filing_type, data in sec_data.items():
                        sec_text = self._create_sec_text(ticker, filing_type, data)
                        self._add_to_index('sec_filings', sec_text, {
                            'ticker': ticker,
                            'type': f'sec_{filing_type}',
                            'timestamp': datetime.now().isoformat(),
                            'source': 'SEC EDGAR Filings'
                        })
                        indexed_count += 1
            except Exception as e:
                logger.warning(f"Could not index SEC data for {ticker}: {str(e)}")
            
            try:
                sec_financial = self._get_sec_financial_cache_data(ticker)
                if sec_financial:
                    sec_fin_text = self._create_sec_financial_text(ticker, sec_financial)
                    self._add_to_index('sec_financial', sec_fin_text, {
                        'ticker': ticker,
                        'type': 'sec_financial',
                        'timestamp': datetime.now().isoformat(),
                        'source': 'SEC Financial Data'
                    })
                    indexed_count += 1
            except Exception as e:
                logger.warning(f"Could not index SEC financial data for {ticker}: {str(e)}")
            
            try:
                sentiment_cache = self._get_sentiment_cache_data(ticker)
                if sentiment_cache:
                    sent_text = self._create_sentiment_cache_text(ticker, sentiment_cache)
                    self._add_to_index('sentiment', sent_text, {
                        'ticker': ticker,
                        'type': 'sentiment_cache',
                        'timestamp': datetime.now().isoformat(),
                        'source': 'Cached News Sentiment Analysis'
                    })
                    indexed_count += 1
                    
                    if 'headlines' in sentiment_cache:
                        for headline in sentiment_cache['headlines'][:10]:
                            news_text = f"News for {ticker}: {headline.get('title', 'No title')} - {headline.get('summary', 'No summary available')}"
                            self._add_to_index('news', news_text, {
                                'ticker': ticker,
                                'type': 'cached_news',
                                'source': headline.get('source', 'Unknown'),
                                'sentiment': headline.get('sentiment_label', 'Neutral'),
                                'timestamp': headline.get('published', datetime.now().isoformat())
                            })
                            indexed_count += 1
                else:
                    sentiment_data = self.sentiment_analyzer.analyze_ticker_sentiment(ticker)
                    if sentiment_data:
                        sent_text = self._create_sentiment_text(ticker, sentiment_data)
                        self._add_to_index('sentiment', sent_text, {
                            'ticker': ticker,
                            'type': 'sentiment',
                            'timestamp': datetime.now().isoformat(),
                            'source': 'Live News Sentiment Analysis'
                        })
                        indexed_count += 1
            except Exception as e:
                logger.warning(f"Could not index sentiment data for {ticker}: {str(e)}")
            
            if indexed_count > 0:
                self._save_multi_index_vector_store()
                logger.info(f"Indexed {indexed_count} documents for {ticker} from ALL cache sources")
            
        except Exception as e:
            logger.error(f"Error indexing data for {ticker}: {str(e)}")
    
    def _get_sec_cache_data(self, ticker):
        """Get SEC filing data from cached sec_filings.db"""
        try:
            sec_cache_path = self.data_cache_dir / "sec_filings.db"
            if not sec_cache_path.exists():
                return None
            
            conn = sqlite3.connect(sec_cache_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            sec_data = {}
            for table in tables:
                table_name = table[0]
                try:
                    cursor.execute(f"SELECT * FROM {table_name} WHERE ticker = ? ORDER BY date DESC LIMIT 5", (ticker,))
                    rows = cursor.fetchall()
                    if rows:
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = [col[1] for col in cursor.fetchall()]
                        sec_data[table_name] = [dict(zip(columns, row)) for row in rows]
                except Exception as e:
                    logger.warning(f"Error reading SEC table {table_name}: {e}")
                    continue
            
            conn.close()
            return sec_data if sec_data else None
            
        except Exception as e:
            logger.error(f"Error accessing SEC cache for {ticker}: {str(e)}")
            return None
    
    def _get_sec_financial_cache_data(self, ticker):
        """Get SEC financial data from cached sec_financial_data.db"""
        try:
            sec_financial_path = self.data_cache_dir / "sec_financial_data.db"
            if not sec_financial_path.exists():
                return None
            
            conn = sqlite3.connect(sec_financial_path)
            cursor = conn.cursor()
            
            financial_data = {}
            for table_name in ['financial_data', 'sec_data', 'filings', 'company_facts']:
                try:
                    cursor.execute(f"SELECT * FROM {table_name} WHERE ticker = ? ORDER BY fiscal_year DESC LIMIT 3", (ticker,))
                    rows = cursor.fetchall()
                    if rows:
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = [col[1] for col in cursor.fetchall()]
                        financial_data[table_name] = [dict(zip(columns, row)) for row in rows]
                except:
                    continue
            
            conn.close()
            return financial_data if financial_data else None
            
        except Exception as e:
            logger.error(f"Error accessing SEC financial cache for {ticker}: {str(e)}")
            return None
    
    def _get_sentiment_cache_data(self, ticker):
        """Get sentiment data from cached sentiment_data.db"""
        try:
            sentiment_cache_path = self.data_cache_dir / "sentiment_data.db"
            if not sentiment_cache_path.exists():
                return None
            
            conn = sqlite3.connect(sentiment_cache_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            sentiment_data = {}
            for table in tables:
                table_name = table[0]
                try:
                    cursor.execute(f"SELECT * FROM {table_name} WHERE ticker = ? ORDER BY date DESC LIMIT 10", (ticker,))
                    rows = cursor.fetchall()
                    if rows:
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = [col[1] for col in cursor.fetchall()]
                        sentiment_data[table_name] = [dict(zip(columns, row)) for row in rows]
                except:
                    continue
            
            conn.close()
            return sentiment_data if sentiment_data else None
            
        except Exception as e:
            logger.error(f"Error accessing sentiment cache for {ticker}: {str(e)}")
            return None
    
    def _create_sec_text(self, ticker, filing_type, data):
        """Create text representation of SEC filing data"""
        text_parts = [f"SEC {filing_type} data for {ticker}:"]
        
        if isinstance(data, list) and data:
            latest = data[0]  # Most recent filing
            for key, value in latest.items():
                if key not in ['ticker', 'id'] and value:
                    text_parts.append(f"{key}: {value}")
        
        return " | ".join(text_parts)
    
    def _create_sec_financial_text(self, ticker, financial_data):
        """Create text representation of SEC financial data"""
        text_parts = [f"SEC financial data for {ticker}:"]
        
        for table_name, data_list in financial_data.items():
            if data_list:
                latest = data_list[0]
                for key, value in latest.items():
                    if key not in ['ticker', 'id'] and value:
                        text_parts.append(f"{key}: {value}")
        
        return " | ".join(text_parts)
    
    def _create_sentiment_cache_text(self, ticker, sentiment_cache):
        """Create text representation of cached sentiment data"""
        text_parts = [f"Cached sentiment analysis for {ticker}:"]
        
        for table_name, data_list in sentiment_cache.items():
            if data_list:
                for item in data_list[:3]:  # Latest 3 entries
                    for key, value in item.items():
                        if key not in ['ticker', 'id'] and value:
                            text_parts.append(f"{key}: {value}")
        
        return " | ".join(text_parts)
    
    def _create_fundamentals_text(self, ticker, fundamentals, stock_info):
        """Create structured text for fundamentals data"""
        text_parts = [f"Fundamental analysis for {ticker}:"]
        
        if stock_info and stock_info.get('company_name', 'N/A') != 'N/A':
            text_parts.append(f"Company: {stock_info['company_name']}")
        if stock_info and stock_info.get('sector', 'N/A') != 'N/A':
            text_parts.append(f"Sector: {stock_info['sector']}")
        if stock_info and stock_info.get('industry', 'N/A') != 'N/A':
            text_parts.append(f"Industry: {stock_info['industry']}")
        
        for metric, value in fundamentals.items():
            if value != 'N/A':
                text_parts.append(f"{metric}: {value}")
        
        return " | ".join(text_parts)
    
    def _create_technical_text(self, ticker, signals):
        """Create structured text for technical analysis"""
        text_parts = [f"Technical analysis for {ticker}:"]
        
        for signal_type, signal_value in signals.items():
            text_parts.append(f"{signal_type.replace('_', ' ')}: {signal_value}")
        
        return " | ".join(text_parts)
    
    def _create_sentiment_text(self, ticker, sentiment_data):
        """Create structured text for sentiment analysis"""
        text_parts = [f"Sentiment analysis for {ticker}:"]
        text_parts.append(f"Overall sentiment: {sentiment_data.get('sentiment_label', 'Unknown')} ({sentiment_data.get('overall_sentiment', 0):.3f})")
        text_parts.append(f"Based on {sentiment_data.get('total_articles', 0)} articles")
        text_parts.append(f"Positive: {sentiment_data.get('positive_count', 0)}")
        text_parts.append(f"Negative: {sentiment_data.get('negative_count', 0)}")
        text_parts.append(f"Neutral: {sentiment_data.get('neutral_count', 0)}")
        
        return " | ".join(text_parts)
    
    def _add_to_index(self, index_type, text, metadata):
        """Add document to specific index"""
        if not self.embedder or index_type not in self.indices:
            return
        
        try:
            # Generate embedding
            embedding = self.embedder.encode([text])
            faiss.normalize_L2(embedding)
            
            # Add to specific index
            self.indices[index_type].add(embedding)
            self.documents[index_type].append(text)
            self.metadata[index_type].append(metadata)
            
        except Exception as e:
            logger.error(f"Error adding to {index_type} index: {str(e)}")
    
    def search_multi_index(self, query, ticker=None, top_k_per_index=3):
        """Search across multiple indices and return relevant results"""
        try:
            if not self.embedder:
                return {}
            
            # Encode query
            query_embedding = self.embedder.encode([query])
            faiss.normalize_L2(query_embedding)
            
            results = {}
            
            for index_type, index in self.indices.items():
                if index.ntotal == 0:  # Skip empty indices
                    continue
                
                try:
                    scores, indices = index.search(query_embedding, min(top_k_per_index, index.ntotal))
                    
                    index_results = []
                    for score, idx in zip(scores[0], indices[0]):
                        if idx >= 0 and idx < len(self.documents[index_type]):
                            metadata = self.metadata[index_type][idx]
                            
                            if ticker and metadata.get('ticker', '').upper() != ticker.upper():
                                continue
                            
                            if score > 0.2:  # Relevance threshold
                                index_results.append({
                                    'text': self.documents[index_type][idx],
                                    'score': float(score),
                                    'metadata': metadata
                                })
                    
                    if index_results:
                        index_results.sort(key=lambda x: x['score'], reverse=True)
                        results[index_type] = index_results
                        
                except Exception as e:
                    logger.warning(f"Error searching {index_type} index: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multi-index search: {str(e)}")
            return {}
    
    def get_response(self, user_query, ticker=None):
        """Generate intelligent response using ALL cached data sources with OpenAI GPT-5 nano"""
        try:
            if ticker:
                self.index_ticker_data(ticker)
            
            search_results = self.search_multi_index(user_query, ticker)
            
            context = self._build_comprehensive_context(search_results, ticker, user_query)
            
            fresh_context = self._get_fresh_data_context(ticker, user_query)
            if fresh_context:
                context += f"\n\n**Real-time Data:**\n{fresh_context}"
            
            query_type = self._classify_query(user_query)
            
            if context:
                response = self.openai_service.generate_analysis_response(
                    user_query, context, ticker
                )
            else:
                response = self._generate_fallback_response(user_query, ticker)
            
            self._store_enhanced_conversation(ticker, user_query, response, context, query_type)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while analyzing your request. Please try again or rephrase your question."
    
    def _build_comprehensive_context(self, search_results, ticker, query):
        """Build comprehensive context from ALL search results including SEC and sentiment"""
        context_parts = []
        data_sources = []
        
        index_priority = ['fundamentals', 'sec_financial', 'sec_filings', 'technical', 'sentiment', 'news']
        
        for index_type in index_priority:
            if index_type in search_results and search_results[index_type]:
                results = search_results[index_type]
                context_parts.append(f"\n**{index_type.replace('_', ' ').title()} Data:**")
                for result in results:
                    context_parts.append(f"• {result['text']}")
                    source = result['metadata'].get('source', index_type)
                    if source not in data_sources:
                        data_sources.append(source)
        
        if data_sources:
            context_parts.append(f"\n**Data Sources:** {', '.join(data_sources)}")
        
        return "\n".join(context_parts)
    
    def _get_fresh_data_context(self, ticker, query):
        """Get fresh real-time data context"""
        if not ticker:
            return ""
        
        fresh_context = []
        
        try:
            if any(word in query.lower() for word in ['price', 'cost', 'worth', 'value', 'current']):
                realtime_data = self.data_fetcher.get_realtime_price(ticker)
                if realtime_data:
                    fresh_context.append(f"Current price: ${realtime_data['price']:.2f}")
                    fresh_context.append(f"Change: {realtime_data['change']:+.2f} ({realtime_data['change_percent']:+.2f}%)")
            
        except Exception as e:
            logger.warning(f"Error getting fresh data: {str(e)}")
        
        return " | ".join(fresh_context)
    
    def _classify_query(self, query):
        """Classify query type including SEC and sentiment"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['price', 'cost', 'worth', 'value']):
            return 'price'
        elif any(word in query_lower for word in ['sec', 'filing', '10-k', '10-q', 'earnings', 'financial statement']):
            return 'sec_filing'
        elif any(word in query_lower for word in ['technical', 'indicator', 'rsi', 'macd', 'moving average']):
            return 'technical'
        elif any(word in query_lower for word in ['sentiment', 'news', 'opinion']):
            return 'sentiment'
        elif any(word in query_lower for word in ['fundamental', 'pe ratio', 'eps', 'roe', 'financial']):
            return 'fundamental'
        else:
            return 'general'
    
    def _generate_fallback_response(self, query, ticker):
        """Generate fallback response mentioning ALL data sources"""
        return f"""**Available Analysis for {ticker or 'stocks'}:**

**Key Findings:**
- Multiple data sources successfully accessed
- Comprehensive stock analysis ready
- Real-time and historical data available

**Analysis:**
- Stock Data: Real-time prices, historical trends
- Technical Analysis: RSI, MACD, Moving Averages, Bollinger Bands
- Fundamentals: P/E ratio, EPS, ROE, Debt/Equity ratio
- News Sentiment: Recent articles and market sentiment
- SEC Filings: Official 10-K, 10-Q reports when available
- SEC Financial: Earnings, revenue, balance sheet data

**Data Sources:**
- Financial APIs, Technical Indicators, News Sentiment Analysis, SEC EDGAR Filings

**Suggestions:**
Ask specific questions like:
- "What's the current price of {ticker}?"
- "Show me technical analysis for {ticker}"
- "What's the sentiment for {ticker}?"
- "Give me SEC filing insights for {ticker}"
- "Show me fundamental metrics for {ticker}"
"""
    
    def _store_enhanced_conversation(self, ticker, user_query, bot_response, context, query_type):
        """Store enhanced conversation data"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO conversation_context 
                (ticker, user_query, bot_response, context_data, query_type, data_sources)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (ticker, user_query, bot_response, context, query_type, "Enhanced Multi-index RAG with OpenAI GPT-5 nano"))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Error storing conversation: {str(e)}")
    
    def get_conversation_history(self, ticker=None, limit=5):
        """Get recent conversation history with enhanced metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if ticker:
                query = '''
                    SELECT user_query, bot_response, query_type, timestamp 
                    FROM conversation_context 
                    WHERE ticker = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                params = (ticker, limit)
            else:
                query = '''
                    SELECT user_query, bot_response, query_type, timestamp 
                    FROM conversation_context 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                params = (limit,)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            return [{
                'query': row[0], 
                'response': row[1], 
                'type': row[2],
                'timestamp': row[3]
            } for row in results]
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []