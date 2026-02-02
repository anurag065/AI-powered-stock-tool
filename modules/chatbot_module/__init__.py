"""
Chatbot Module for AI-Powered Stock Analysis Platform

Contains:
- EnhancedRAGChatbot: Fast, unified RAG chatbot with streaming support
- EnhancedOpenAIChatbot: Legacy chatbot (deprecated, use EnhancedRAGChatbot)
"""

from .enhanced_rag_chatbot import EnhancedRAGChatbot
from .rag_chatbot import EnhancedOpenAIChatbot

__all__ = ['EnhancedRAGChatbot', 'EnhancedOpenAIChatbot']
