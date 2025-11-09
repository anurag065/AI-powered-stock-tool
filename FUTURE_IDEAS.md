# Future Ideas & Improvements

This document tracks ideas for future enhancements to the AI-Powered Stock Analysis Platform.

---

## 📈 Option 3: Hybrid API Approach for Historical Price Charts

### Problem
- **Finnhub Free Tier**: Doesn't include historical candlestick data (403 error on `stock_candles`)
- **Yahoo Finance (yfinance)**: Unreliable, frequently blocked with 429 errors or JSON parsing failures
- **Result**: Price history charts don't work in the application

### Proposed Solution: Multi-API Hybrid Strategy

Use different free APIs for different data types based on their strengths:

| Data Type | API to Use | Reason |
|-----------|------------|--------|
| **Historical Price Charts** | **Twelve Data** | Free tier: 800 calls/day, includes OHLCV data |
| **Fundamentals** | **Finnhub** ✅ | Already working, excellent data quality |
| **Real-time Quotes** | **Finnhub** ✅ | Already working, accurate pricing |
| **Company Info** | **Finnhub** ✅ | Already working, good profiles |
| **News & Sentiment** | **Finnhub** | Has news endpoint (to be implemented) |
| **Technical Indicators** | Calculate locally | Use ta-lib or pandas_ta on fetched data |

---

## 🔧 Implementation Plan for Twelve Data Integration

### Step 1: Add Twelve Data Client

**New file**: `modules/utils/twelvedata_client.py`

```python
"""
Twelve Data API Client for Historical Price Data
Free tier: 800 API calls per day
"""

from twelvedata import TDClient
import pandas as pd
from datetime import datetime, timedelta
import logging

class TwelveDataClient:
    """Singleton client for Twelve Data API"""

    def __init__(self, api_key):
        self.client = TDClient(apikey=api_key)

    def get_time_series(self, symbol, interval='1day', outputsize=30):
        """
        Get historical OHLCV data

        Args:
            symbol: Stock ticker
            interval: 1min, 5min, 15min, 30min, 1h, 1day, 1week, 1month
            outputsize: Number of data points (max 5000)

        Returns:
            pandas.DataFrame with OHLCV data
        """
        ts = self.client.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=outputsize
        )

        df = ts.as_pandas()

        # Rename columns to match yfinance format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        return df
```

### Step 2: Update DataFetcher

**File**: `modules/data_module/data_fetcher.py`

Add Twelve Data for historical price data:

```python
from utils.twelvedata_client import TwelveDataClient

class DataFetcher:
    def __init__(self):
        # ... existing code ...

        # Initialize Twelve Data for historical charts
        try:
            self.twelvedata = TwelveDataClient(api_key=os.getenv('TWELVEDATA_API_KEY'))
            self.use_twelvedata = True
        except:
            self.twelvedata = None
            self.use_twelvedata = False

    def get_stock_data(self, ticker, period="1mo"):
        """Fetch historical price data using Twelve Data"""

        # Check cache first
        # ... existing cache logic ...

        # Try Twelve Data first for historical data
        if self.use_twelvedata:
            try:
                outputsize = self._period_to_outputsize(period)
                data = self.twelvedata.get_time_series(ticker, '1day', outputsize)

                if not data.empty:
                    logger.info(f"Retrieved {len(data)} data points from Twelve Data")
                    self._cache_stock_data(ticker, data)
                    return data
            except Exception as e:
                logger.warning(f"Twelve Data failed, falling back to yfinance: {e}")

        # Fallback to yfinance
        stock = self.ticker_cache.get_ticker(ticker)
        data = stock.history(period=period)
        return data
```

### Step 3: Update Environment Configuration

**File**: `.env.template`

```env
# Twelve Data API (for historical price charts)
# Get free API key at: https://twelvedata.com/register
TWELVEDATA_API_KEY=your_twelvedata_api_key_here

# Finnhub API (for fundamentals, quotes, company info)
FINNHUB_API_KEY=your_finnhub_api_key_here
```

### Step 4: Add to Requirements

**File**: `requirements.txt`

```
# Financial data APIs
finnhub-python==2.4.19
twelvedata==1.2.13
```

---

## 📊 Expected Benefits

### API Call Distribution (per ticker, all tabs loaded):

| Feature | API Used | Calls | Free Tier Limit |
|---------|----------|-------|-----------------|
| Historical price chart | Twelve Data | 1 | 800/day |
| Fundamentals (P/E, EPS, etc.) | Finnhub | 1 | 60/min |
| Real-time quote | Finnhub | 1 | 60/min |
| Company profile | Finnhub | 1 | 60/min |
| News headlines | Finnhub | 1 | 60/min |
| **Total per ticker** | Mixed | **5** | N/A |

### Daily Capacity:

- **Twelve Data limit**: 800 calls/day → 800 stocks with charts
- **Finnhub limit**: 60/min = 3,600/hour → effectively unlimited for our use case
- **Result**: Can analyze 800 different stocks per day with full features

### Advantages:

1. ✅ **All features work** (charts, fundamentals, quotes, news)
2. ✅ **100% free** (using free tiers of multiple APIs)
3. ✅ **Reliable** (official APIs, not scrapers)
4. ✅ **No single point of failure** (if one API down, others still work)
5. ✅ **Better rate limits** than single API
6. ✅ **Easy to swap out APIs** if one changes their free tier

### Disadvantages:

1. ⚠️ More API keys to manage (Finnhub + Twelve Data)
2. ⚠️ More complex codebase (multiple API clients)
3. ⚠️ Need to stay within free tier limits (800 charts/day)

---

## 🔄 Alternative APIs for Historical Data

If Twelve Data doesn't work out, here are alternatives:

### Polygon.io
- **Free tier**: 5 API calls per minute
- **Historical data**: ✅ Yes, 2 years
- **Pros**: Good documentation, stable API
- **Cons**: Lower rate limit than Twelve Data

### Alpha Vantage
- **Free tier**: 25 calls per day (too low!)
- **Historical data**: ✅ Yes
- **Pros**: Official API
- **Cons**: Rate limit too restrictive

### IEX Cloud
- **Free tier**: 50,000 calls per month
- **Historical data**: ✅ Yes
- **Pros**: Very generous free tier
- **Cons**: May require credit card for signup

### Financial Modeling Prep (FMP)
- **Free tier**: 250 calls per day
- **Historical data**: ✅ Yes
- **Pros**: Good fundamental data too
- **Cons**: Less generous than Twelve Data

---

## 🎯 Recommendation Priority

**When to implement:**
- ✅ **High priority** if you need price charts working soon
- ⏸️ **Medium priority** if fundamentals/quotes are enough for now
- ⏸️ **Low priority** if waiting to see if yfinance recovers

**Estimated implementation time:**
- Setting up Twelve Data client: 30 minutes
- Updating DataFetcher: 20 minutes
- Testing: 30 minutes
- **Total: ~1.5 hours**

---

## 📝 Notes

- This is **Option 3** from the Finnhub migration discussion
- Created on: 2025-11-09
- Status: **Not implemented** (future idea)
- Dependencies: Twelve Data API key (free signup at https://twelvedata.com)

---

## 🔗 Related Documentation

- Main migration progress: `FINNHUB_MIGRATION_PROGRESS.md`
- Fixes summary: `FIXES_SUMMARY.md`
- Environment template: `.env.template`

---

**Want to implement this?** Follow the steps above or ask for help!
