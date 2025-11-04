import sqlite3
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Centralized database management for the stock analysis platform
    Handles cleanup and maintenance of all SQLite databases
    """
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.cache_dir = self.data_dir / "cache"
        self.vector_dir = self.data_dir / "vector_store"
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        
        self.databases = {
            'stock_data': self.cache_dir / "stock_data.db",
            'sentiment_data': self.cache_dir / "sentiment_data.db",
            'chatbot_knowledge': self.vector_dir / "chatbot_knowledge.db"
        }
    
    def cleanup_old_data(self, days_to_keep=2):
        """
        Clean up old data from all databases
        Keeps only the specified number of days of data
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        try:
            # Clean stock data
            self._cleanup_stock_data(cutoff_date)
            
            # Clean sentiment data
            self._cleanup_sentiment_data(cutoff_date)
            
            # Clean chatbot data
            self._cleanup_chatbot_data(cutoff_date)
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def _cleanup_stock_data(self, cutoff_date):
        """Clean up old stock data"""
        db_path = self.databases['stock_data']
        if not db_path.exists():
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Remove old price data
        cursor.execute('DELETE FROM stock_prices WHERE created_at < ?', (cutoff_date,))
        
        # Remove old fundamentals data
        cursor.execute('DELETE FROM fundamentals WHERE updated_at < ?', (cutoff_date,))
        
        conn.commit()
        conn.close()
    
    def _cleanup_sentiment_data(self, cutoff_date):
        """Clean up old sentiment data"""
        db_path = self.databases['sentiment_data']
        if not db_path.exists():
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Remove old news sentiment
        cursor.execute('DELETE FROM news_sentiment WHERE created_at < ?', (cutoff_date,))
        
        # Remove old sentiment summaries
        cursor.execute('DELETE FROM ticker_sentiment_summary WHERE updated_at < ?', (cutoff_date,))
        
        conn.commit()
        conn.close()
    
    def _cleanup_chatbot_data(self, cutoff_date):
        """Clean up old chatbot conversation data"""
        db_path = self.databases['chatbot_knowledge']
        if not db_path.exists():
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Keep conversation context for longer (1 week)
        conversation_cutoff = datetime.now() - timedelta(days=7)
        cursor.execute('DELETE FROM conversation_context WHERE timestamp < ?', (conversation_cutoff,))
        
        # Remove old knowledge cache
        cursor.execute('DELETE FROM knowledge_cache WHERE updated_at < ?', (cutoff_date,))
        
        conn.commit()
        conn.close()
    
    def get_database_stats(self):
        """Get statistics about database sizes and record counts"""
        stats = {}
        
        for db_name, db_path in self.databases.items():
            if not db_path.exists():
                stats[db_name] = {'size_mb': 0, 'tables': {}}
                continue
            
            try:
                # Get file size
                size_mb = db_path.stat().st_size / (1024 * 1024)
                
                # Get table record counts
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                table_stats = {}
                for (table_name,) in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    table_stats[table_name] = count
                
                conn.close()
                
                stats[db_name] = {
                    'size_mb': round(size_mb, 2),
                    'tables': table_stats
                }
                
            except Exception as e:
                logger.error(f"Error getting stats for {db_name}: {str(e)}")
                stats[db_name] = {'error': str(e)}
        
        return stats
    
    def vacuum_databases(self):
        """Vacuum all databases to reclaim space"""
        for db_name, db_path in self.databases.items():
            if not db_path.exists():
                continue
            
            try:
                conn = sqlite3.connect(db_path)
                conn.execute('VACUUM')
                conn.close()
                logger.info(f"Vacuumed database: {db_name}")
            except Exception as e:
                logger.error(f"Error vacuuming {db_name}: {str(e)}")
    
    def reset_all_data(self):
        """Reset all databases (delete all data)"""
        for db_name, db_path in self.databases.items():
            try:
                if db_path.exists():
                    db_path.unlink()
                    logger.info(f"Reset database: {db_name}")
            except Exception as e:
                logger.error(f"Error resetting {db_name}: {str(e)}")
    
    def export_data(self, ticker, output_dir):
        """Export all data for a specific ticker"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        export_data = {}
        
        try:
            # Export stock data
            stock_db = self.databases['stock_data']
            if stock_db.exists():
                conn = sqlite3.connect(stock_db)
                
                # Stock prices
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM stock_prices WHERE ticker = ?', (ticker,))
                export_data['stock_prices'] = cursor.fetchall()
                
                # Fundamentals
                cursor.execute('SELECT * FROM fundamentals WHERE ticker = ?', (ticker,))
                export_data['fundamentals'] = cursor.fetchall()
                
                conn.close()
            
            # Export sentiment data
            sentiment_db = self.databases['sentiment_data']
            if sentiment_db.exists():
                conn = sqlite3.connect(sentiment_db)
                
                # News sentiment
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM news_sentiment WHERE ticker = ?', (ticker,))
                export_data['news_sentiment'] = cursor.fetchall()
                
                # Sentiment summary
                cursor.execute('SELECT * FROM ticker_sentiment_summary WHERE ticker = ?', (ticker,))
                export_data['sentiment_summary'] = cursor.fetchall()
                
                conn.close()
            
            # Export chatbot data
            chatbot_db = self.databases['chatbot_knowledge']
            if chatbot_db.exists():
                conn = sqlite3.connect(chatbot_db)
                
                # Conversation context
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM conversation_context WHERE ticker = ?', (ticker,))
                export_data['conversations'] = cursor.fetchall()
                
                conn.close()
            
            # Save to JSON file
            import json
            output_file = output_path / f"{ticker}_data_export.json"
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported data for {ticker} to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting data for {ticker}: {str(e)}")
            return None