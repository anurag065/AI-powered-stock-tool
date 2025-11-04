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

# Import other modules
from ..data_module.data_fetcher import DataFetcher
from ..indicator_dashboard.technical_analysis import TechnicalAnalysis
from ..sentiment_module.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

class RAGChatbot:
    """
    Unified LLM Q&A Chatbot with RAG Architecture
    Integrates data from all modules: stock data, technical indicators, sentiment, and SEC filings
    """
    
    def __init__(self):
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "vector_store"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize vector store
        self._init_vector_store()
        
        # Initialize knowledge base
        self.db_path = self.cache_dir / "chatbot_knowledge.db"
        self._init_knowledge_base()
    
    def _init_vector_store(self):
        """Initialize sentence transformer and FAISS vector store"""
        try:
            # Use a lightweight sentence transformer model
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
            
            # Initialize FAISS index
            self.vector_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.documents = []  # Store document texts
            self.metadata = []   # Store document metadata
            
            # Try to load existing index
            self._load_vector_index()
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            self.embedder = None
            self.vector_index = None
    
    def _init_knowledge_base(self):
        """Initialize knowledge base for storing conversation context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                user_query TEXT,
                bot_response TEXT,
                context_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                data_type TEXT,
                content TEXT,
                embeddings BLOB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, data_type)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_vector_index(self):
        """Load existing vector index if available"""
        index_path = self.cache_dir / "faiss_index.bin"
        docs_path = self.cache_dir / "documents.pkl"
        meta_path = self.cache_dir / "metadata.pkl"
        
        if index_path.exists() and docs_path.exists() and meta_path.exists():
            try:
                self.vector_index = faiss.read_index(str(index_path))
                
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                with open(meta_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                logger.info(f"Loaded existing vector index with {len(self.documents)} documents")
            except Exception as e:
                logger.warning(f"Could not load existing vector index: {str(e)}")
    
    def _save_vector_index(self):
        """Save vector index to disk"""
        try:
            index_path = self.cache_dir / "faiss_index.bin"
            docs_path = self.cache_dir / "documents.pkl"
            meta_path = self.cache_dir / "metadata.pkl"
            
            faiss.write_index(self.vector_index, str(index_path))
            
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(meta_path, 'wb') as f:
                pickle.dump(self.metadata, f)
                
        except Exception as e:
            logger.error(f"Error saving vector index: {str(e)}")
    
    def index_ticker_data(self, ticker):
        """
        Index all available data for a ticker into the vector store
        """
        try:
            documents_to_add = []
            metadata_to_add = []
            
            # 1. Index stock fundamentals
            fundamentals = self.data_fetcher.get_fundamentals(ticker)
            stock_info = self.data_fetcher.get_stock_info(ticker)
            
            if fundamentals:
                fund_text = f"Stock fundamentals for {ticker}: "
                fund_text += f"Company: {stock_info.get('company_name', 'N/A')}, "
                fund_text += f"Sector: {stock_info.get('sector', 'N/A')}, "
                fund_text += f"P/E Ratio: {fundamentals.get('P/E Ratio', 'N/A')}, "
                fund_text += f"EPS: {fundamentals.get('EPS', 'N/A')}, "
                fund_text += f"ROE: {fundamentals.get('ROE', 'N/A')}, "
                fund_text += f"Debt to Equity: {fundamentals.get('Debt/Equity', 'N/A')}"
                
                documents_to_add.append(fund_text)
                metadata_to_add.append({
                    'ticker': ticker,
                    'type': 'fundamentals',
                    'timestamp': datetime.now().isoformat()
                })
            
            # 2. Index technical analysis
            signals = self.technical_analyzer.get_signal_analysis(ticker)
            if signals:
                tech_text = f"Technical analysis for {ticker}: "
                tech_text += f"Moving Average Signal: {signals.get('MA_Signal', 'N/A')}, "
                tech_text += f"RSI Signal: {signals.get('RSI_Signal', 'N/A')}, "
                tech_text += f"MACD Signal: {signals.get('MACD_Signal', 'N/A')}, "
                tech_text += f"Bollinger Bands Signal: {signals.get('BB_Signal', 'N/A')}"
                
                documents_to_add.append(tech_text)
                metadata_to_add.append({
                    'ticker': ticker,
                    'type': 'technical_analysis',
                    'timestamp': datetime.now().isoformat()
                })
            
            # 3. Index sentiment analysis
            sentiment_data = self.sentiment_analyzer.analyze_ticker_sentiment(ticker)
            if sentiment_data:
                sent_text = f"Sentiment analysis for {ticker}: "
                sent_text += f"Overall sentiment: {sentiment_data['sentiment_label']} ({sentiment_data['overall_sentiment']:.3f}), "
                sent_text += f"Based on {sentiment_data['total_articles']} articles, "
                sent_text += f"Positive: {sentiment_data['positive_count']}, "
                sent_text += f"Negative: {sentiment_data['negative_count']}, "
                sent_text += f"Neutral: {sentiment_data['neutral_count']}"
                
                documents_to_add.append(sent_text)
                metadata_to_add.append({
                    'ticker': ticker,
                    'type': 'sentiment',
                    'timestamp': datetime.now().isoformat()
                })
                
                # Also index recent headlines
                for headline in sentiment_data['headlines'][:5]:
                    headline_text = f"News for {ticker}: {headline['title']} - Sentiment: {headline['sentiment_label']} ({headline['sentiment']:.3f})"
                    documents_to_add.append(headline_text)
                    metadata_to_add.append({
                        'ticker': ticker,
                        'type': 'news',
                        'source': headline['source'],
                        'timestamp': headline['published'].isoformat() if headline['published'] else datetime.now().isoformat()
                    })
            
            # Add documents to vector store
            if documents_to_add and self.embedder:
                embeddings = self.embedder.encode(documents_to_add)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                
                # Add to FAISS index
                self.vector_index.add(embeddings)
                
                # Store documents and metadata
                self.documents.extend(documents_to_add)
                self.metadata.extend(metadata_to_add)
                
                # Save updated index
                self._save_vector_index()
                
                logger.info(f"Indexed {len(documents_to_add)} documents for {ticker}")
            
        except Exception as e:
            logger.error(f"Error indexing data for {ticker}: {str(e)}")
    
    def search_knowledge_base(self, query, ticker=None, top_k=5):
        """
        Search the knowledge base for relevant information
        """
        try:
            if not self.embedder or not self.documents:
                return []
            
            # Encode the query
            query_embedding = self.embedder.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.vector_index.search(query_embedding, min(top_k * 2, len(self.documents)))
            
            # Filter results and format
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    metadata = self.metadata[idx]
                    
                    # Filter by ticker if specified
                    if ticker and metadata.get('ticker', '').upper() != ticker.upper():
                        continue
                    
                    # Only include results with reasonable similarity
                    if score > 0.3:  # Threshold for relevance
                        results.append({
                            'text': self.documents[idx],
                            'score': float(score),
                            'metadata': metadata
                        })
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def get_response(self, user_query, ticker=None):
        """
        Generate response using RAG approach
        """
        try:
            # Ensure data is indexed for the ticker
            if ticker:
                self.index_ticker_data(ticker)
            
            # Search for relevant information
            relevant_docs = self.search_knowledge_base(user_query, ticker)
            
            # Build context from retrieved documents
            context = self._build_context(relevant_docs, ticker, user_query)
            
            # Generate response
            response = self._generate_response(user_query, context, ticker)
            
            # Store conversation for future reference
            self._store_conversation(ticker, user_query, response, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."
    
    def _build_context(self, relevant_docs, ticker, query):
        """
        Build context from retrieved documents and fresh data
        """
        context_parts = []
        
        # Add retrieved documents
        if relevant_docs:
            context_parts.append("Relevant information from knowledge base:")
            for doc in relevant_docs:
                context_parts.append(f"- {doc['text']}")
        
        # Add fresh data if needed
        if ticker:
            try:
                # Get current price
                realtime_data = self.data_fetcher.get_realtime_price(ticker)
                if realtime_data:
                    context_parts.append(f"\nCurrent price data for {ticker}:")
                    context_parts.append(f"- Current price: ${realtime_data['price']:.2f}")
                    context_parts.append(f"- Change: {realtime_data['change']:.2f} ({realtime_data['change_percent']:.2f}%)")
                
                # Add company info if query seems to ask for it
                if any(word in query.lower() for word in ['company', 'business', 'what is', 'about']):
                    stock_info = self.data_fetcher.get_stock_info(ticker)
                    if stock_info:
                        context_parts.append(f"\nCompany information for {ticker}:")
                        context_parts.append(f"- Company: {stock_info.get('company_name', 'N/A')}")
                        context_parts.append(f"- Sector: {stock_info.get('sector', 'N/A')}")
                        context_parts.append(f"- Industry: {stock_info.get('industry', 'N/A')}")
                
            except Exception as e:
                logger.warning(f"Error getting fresh data for context: {str(e)}")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query, context, ticker):
        """
        Generate response based on query and context
        This is a rule-based approach. In production, you'd use an LLM API
        """
        query_lower = query.lower()
        
        # Stock price queries
        if any(word in query_lower for word in ['price', 'cost', 'worth', 'value']):
            try:
                realtime_data = self.data_fetcher.get_realtime_price(ticker)
                if realtime_data:
                    return f"The current price of {ticker} is ${realtime_data['price']:.2f}, with a change of {realtime_data['change']:+.2f} ({realtime_data['change_percent']:+.2f}%) from the previous close."
            except:
                pass
        
        # Technical analysis queries
        if any(word in query_lower for word in ['technical', 'indicator', 'rsi', 'macd', 'moving average', 'signal']):
            try:
                signals = self.technical_analyzer.get_signal_analysis(ticker)
                if signals:
                    response = f"Technical analysis for {ticker}:\n"
                    for signal_type, signal_value in signals.items():
                        response += f"• {signal_type.replace('_', ' ')}: {signal_value}\n"
                    return response.strip()
            except:
                pass
        
        # Sentiment analysis queries
        if any(word in query_lower for word in ['sentiment', 'news', 'opinion', 'feeling', 'market sentiment']):
            try:
                sentiment_data = self.sentiment_analyzer.analyze_ticker_sentiment(ticker)
                if sentiment_data:
                    response = f"Sentiment analysis for {ticker}:\n"
                    response += f"• Overall sentiment: {sentiment_data['sentiment_label']} ({sentiment_data['overall_sentiment']:.3f})\n"
                    response += f"• Based on {sentiment_data['total_articles']} recent articles\n"
                    response += f"• Positive: {sentiment_data['positive_count']}, Negative: {sentiment_data['negative_count']}, Neutral: {sentiment_data['neutral_count']}\n"
                    
                    if sentiment_data['headlines']:
                        response += f"\nRecent headlines:\n"
                        for headline in sentiment_data['headlines'][:3]:
                            response += f"• {headline['title'][:100]}... (Sentiment: {headline['sentiment_label']})\n"
                    
                    return response.strip()
            except:
                pass
        
        # Fundamental analysis queries
        if any(word in query_lower for word in ['fundamental', 'pe ratio', 'eps', 'roe', 'debt', 'financial']):
            try:
                fundamentals = self.data_fetcher.get_fundamentals(ticker)
                if fundamentals:
                    response = f"Fundamental analysis for {ticker}:\n"
                    for key, value in fundamentals.items():
                        if value != 'N/A':
                            response += f"• {key}: {value}\n"
                    return response.strip()
            except:
                pass
        
        # Company information queries
        if any(word in query_lower for word in ['company', 'business', 'what is', 'about', 'sector', 'industry']):
            try:
                stock_info = self.data_fetcher.get_stock_info(ticker)
                if stock_info:
                    response = f"Information about {ticker}:\n"
                    if stock_info.get('company_name') != 'N/A':
                        response += f"• Company: {stock_info['company_name']}\n"
                    if stock_info.get('sector') != 'N/A':
                        response += f"• Sector: {stock_info['sector']}\n"
                    if stock_info.get('industry') != 'N/A':
                        response += f"• Industry: {stock_info['industry']}\n"
                    if stock_info.get('description') != 'N/A':
                        description = stock_info['description'][:300] + "..." if len(stock_info['description']) > 300 else stock_info['description']
                        response += f"• Description: {description}\n"
                    return response.strip()
            except:
                pass
        
        # If no specific handler, provide general response
        if context:
            return f"Based on the available data:\n\n{context}\n\nPlease let me know if you need more specific information about {ticker or 'the stock'} or have any other questions."
        else:
            return f"I can help you analyze {ticker or 'stocks'} using real-time data, technical indicators, sentiment analysis, and fundamental metrics. What specific information would you like to know?"
    
    def _store_conversation(self, ticker, user_query, bot_response, context):
        """Store conversation for future reference"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO conversation_context 
                (ticker, user_query, bot_response, context_data)
                VALUES (?, ?, ?, ?)
            ''', (ticker, user_query, bot_response, context))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Error storing conversation: {str(e)}")
    
    def get_conversation_history(self, ticker=None, limit=10):
        """Get recent conversation history"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if ticker:
                query = '''
                    SELECT user_query, bot_response, timestamp 
                    FROM conversation_context 
                    WHERE ticker = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                params = (ticker, limit)
            else:
                query = '''
                    SELECT user_query, bot_response, timestamp 
                    FROM conversation_context 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                params = (limit,)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            return [{'query': row[0], 'response': row[1], 'timestamp': row[2]} for row in results]
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []