import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """
    Technical Analysis module for creating indicator dashboard
    Implements SMA (50/200), RSI, MACD, and Bollinger Bands
    """
    
    def __init__(self):
        pass
    
    def calculate_indicators(self, ticker, period="3mo"):
        """
        Calculate all technical indicators for a given ticker
        Returns dictionary with indicator values and charts
        """
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"No data available for {ticker}")
                return None
            
            # Calculate indicators
            indicators = {}
            
            # Moving Averages
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
            
            # RSI
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(data['Close'])
            data['BB_Upper'] = bollinger.bollinger_hband()
            data['BB_Lower'] = bollinger.bollinger_lband()
            data['BB_Middle'] = bollinger.bollinger_mavg()
            
            # Store raw data
            indicators['data'] = data
            indicators['SMA_50'] = data['SMA_50'].iloc[-1] if not pd.isna(data['SMA_50'].iloc[-1]) else None
            indicators['SMA_200'] = data['SMA_200'].iloc[-1] if not pd.isna(data['SMA_200'].iloc[-1]) else None
            indicators['RSI'] = data['RSI'].iloc[-1] if not pd.isna(data['RSI'].iloc[-1]) else None
            
            # Create charts
            indicators['ma_chart'] = self._create_ma_chart(data, ticker)
            indicators['rsi_chart'] = self._create_rsi_chart(data, ticker)
            indicators['macd_chart'] = self._create_macd_chart(data, ticker)
            indicators['bollinger_chart'] = self._create_bollinger_chart(data, ticker)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {ticker}: {str(e)}")
            return None
    
    def _create_ma_chart(self, data, ticker):
        """Create moving averages chart"""
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        
        # SMA 50
        if not data['SMA_50'].isna().all():
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=1)
            ))
        
        # SMA 200
        if not data['SMA_200'].isna().all():
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_200'],
                mode='lines',
                name='SMA 200',
                line=dict(color='red', width=1)
            ))
        
        fig.update_layout(
            title=f'{ticker} - Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _create_rsi_chart(self, data, ticker):
        """Create RSI chart"""
        fig = go.Figure()
        
        # RSI line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ))
        
        # Overbought line (70)
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        
        # Oversold line (30)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        
        fig.update_layout(
            title=f'{ticker} - RSI (14)',
            xaxis_title='Date',
            yaxis_title='RSI',
            height=300,
            yaxis_range=[0, 100]
        )
        
        return fig
    
    def _create_macd_chart(self, data, ticker):
        """Create MACD chart"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('MACD Line & Signal', 'MACD Histogram'),
            vertical_spacing=0.1
        )
        
        # MACD line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Signal line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD_Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='red')
        ), row=1, col=1)
        
        # Histogram
        colors = ['green' if x >= 0 else 'red' for x in data['MACD_Histogram']]
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['MACD_Histogram'],
            name='Histogram',
            marker_color=colors
        ), row=2, col=1)
        
        fig.update_layout(
            title=f'{ticker} - MACD',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def _create_bollinger_chart(self, data, ticker):
        """Create Bollinger Bands chart"""
        fig = go.Figure()
        
        # Upper band
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Upper'],
            mode='lines',
            name='Upper Band',
            line=dict(color='gray', width=1),
            fill=None
        ))
        
        # Lower band
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Lower'],
            mode='lines',
            name='Lower Band',
            line=dict(color='gray', width=1),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ))
        
        # Middle band (SMA)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['BB_Middle'],
            mode='lines',
            name='Middle Band (SMA)',
            line=dict(color='orange', width=1)
        ))
        
        # Price
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f'{ticker} - Bollinger Bands',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def get_signal_analysis(self, ticker, period="3mo"):
        """
        Generate trading signals based on technical indicators
        """
        try:
            indicators = self.calculate_indicators(ticker, period)
            if not indicators:
                return None
            
            data = indicators['data']
            signals = {}
            
            # Moving Average Signal
            if indicators['SMA_50'] and indicators['SMA_200']:
                if indicators['SMA_50'] > indicators['SMA_200']:
                    signals['MA_Signal'] = 'Bullish (Golden Cross)'
                else:
                    signals['MA_Signal'] = 'Bearish (Death Cross)'
            else:
                signals['MA_Signal'] = 'Insufficient Data'
            
            # RSI Signal
            if indicators['RSI']:
                if indicators['RSI'] > 70:
                    signals['RSI_Signal'] = 'Overbought - Consider Selling'
                elif indicators['RSI'] < 30:
                    signals['RSI_Signal'] = 'Oversold - Consider Buying'
                else:
                    signals['RSI_Signal'] = 'Neutral'
            else:
                signals['RSI_Signal'] = 'Insufficient Data'
            
            # MACD Signal
            latest_macd = data['MACD'].iloc[-1]
            latest_signal = data['MACD_Signal'].iloc[-1]
            
            if not pd.isna(latest_macd) and not pd.isna(latest_signal):
                if latest_macd > latest_signal:
                    signals['MACD_Signal'] = 'Bullish Crossover'
                else:
                    signals['MACD_Signal'] = 'Bearish Crossover'
            else:
                signals['MACD_Signal'] = 'Insufficient Data'
            
            # Bollinger Bands Signal
            latest_price = data['Close'].iloc[-1]
            latest_upper = data['BB_Upper'].iloc[-1]
            latest_lower = data['BB_Lower'].iloc[-1]
            
            if not pd.isna(latest_upper) and not pd.isna(latest_lower):
                if latest_price > latest_upper:
                    signals['BB_Signal'] = 'Price Above Upper Band - Overbought'
                elif latest_price < latest_lower:
                    signals['BB_Signal'] = 'Price Below Lower Band - Oversold'
                else:
                    signals['BB_Signal'] = 'Price Within Bands - Normal'
            else:
                signals['BB_Signal'] = 'Insufficient Data'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signal analysis for {ticker}: {str(e)}")
            return None
    
    def get_support_resistance(self, ticker, period="6mo"):
        """
        Identify potential support and resistance levels
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                return None
            
            # Calculate local minima and maxima
            from scipy.signal import argrelextrema
            
            # Find local minima (support levels)
            support_indices = argrelextrema(data['Low'].values, np.less_equal, order=5)[0]
            support_levels = data['Low'].iloc[support_indices].tolist()
            
            # Find local maxima (resistance levels)  
            resistance_indices = argrelextrema(data['High'].values, np.greater_equal, order=5)[0]
            resistance_levels = data['High'].iloc[resistance_indices].tolist()
            
            # Get the most significant levels (recent and frequently tested)
            support_levels = sorted(support_levels)[-3:] if support_levels else []
            resistance_levels = sorted(resistance_levels)[-3:] if resistance_levels else []
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'current_price': data['Close'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance for {ticker}: {str(e)}")
            return None