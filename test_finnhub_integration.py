"""
Test script for Hybrid API Integration (Finnhub + Twelve Data)
Run this to verify that the hybrid approach is working correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from modules.data_module.data_fetcher import DataFetcher
import logging

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_finnhub_integration():
    """Test Hybrid API Integration (Finnhub + Twelve Data)"""
    print("=" * 80)
    print("HYBRID API INTEGRATION TEST (Finnhub + Twelve Data)")
    print("=" * 80)

    # Initialize DataFetcher
    print("\n1. Initializing DataFetcher...")
    fetcher = DataFetcher()

    print("\nAPI Clients Status:")
    if fetcher.use_finnhub:
        print("  ✅ Finnhub client initialized (for fundamentals, quotes, company info, news)")
    else:
        print("  ❌ Finnhub client failed to initialize. Check your .env file.")
        print("     Make sure FINNHUB_API_KEY is set correctly.")

    if fetcher.use_twelvedata:
        print("  ✅ Twelve Data client initialized (for historical price charts)")
    else:
        print("  ⚠️  Twelve Data client not initialized. Will use yfinance fallback for charts.")
        print("     To enable Twelve Data: Set TWELVEDATA_API_KEY in .env file.")

    if not fetcher.use_finnhub and not fetcher.use_twelvedata:
        print("\n❌ Both API clients failed to initialize. Cannot proceed with tests.")
        print("   Please set up at least one API key in your .env file.")
        return

    # Test ticker
    ticker = "AAPL"
    print(f"\n2. Testing with ticker: {ticker}")

    # Test get_stock_data
    print(f"\n3. Testing get_stock_data() for {ticker}...")
    try:
        stock_data = fetcher.get_stock_data(ticker, period="1mo")
        if not stock_data.empty:
            print(f"✅ Stock data retrieved successfully!")
            print(f"   - Data points: {len(stock_data)}")
            print(f"   - Date range: {stock_data.index.min()} to {stock_data.index.max()}")
            print(f"   - Latest close: ${stock_data['Close'].iloc[-1]:.2f}")
        else:
            print("❌ Stock data is empty")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test get_fundamentals
    print(f"\n4. Testing get_fundamentals() for {ticker}...")
    try:
        fundamentals = fetcher.get_fundamentals(ticker)
        if fundamentals:
            print(f"✅ Fundamentals retrieved successfully!")
            for key, value in fundamentals.items():
                print(f"   - {key}: {value}")
        else:
            print("❌ Fundamentals are empty")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test get_realtime_price
    print(f"\n5. Testing get_realtime_price() for {ticker}...")
    try:
        price_data = fetcher.get_realtime_price(ticker)
        if price_data:
            print(f"✅ Realtime price retrieved successfully!")
            print(f"   - Current price: ${price_data['price']:.2f}")
            print(f"   - Change: ${price_data['change']:.2f} ({price_data['change_percent']:.2f}%)")
            print(f"   - Timestamp: {price_data['timestamp']}")
        else:
            print("❌ Price data is None")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test get_stock_info
    print(f"\n6. Testing get_stock_info() for {ticker}...")
    try:
        info = fetcher.get_stock_info(ticker)
        if info:
            print(f"✅ Stock info retrieved successfully!")
            print(f"   - Company: {info.get('company_name')}")
            print(f"   - Sector: {info.get('sector')}")
            print(f"   - Industry: {info.get('industry')}")
        else:
            print("❌ Stock info is empty")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test with historical data >1 year (should use yfinance)
    print(f"\n7. Testing with >1 year period (should use yfinance fallback)...")
    try:
        historical_data = fetcher.get_stock_data(ticker, period="2y")
        if not historical_data.empty:
            print(f"✅ Historical data (>1 year) retrieved successfully!")
            print(f"   - Data points: {len(historical_data)}")
            print(f"   - Date range: {historical_data.index.min()} to {historical_data.index.max()}")
        else:
            print("❌ Historical data is empty")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nNOTE: Check the logs above for any warnings or errors.")
    print("\nAPI Usage in Logs:")
    print("  - 'Twelve Data' = Using Twelve Data for historical price charts")
    print("  - 'Finnhub' = Using Finnhub for fundamentals/quotes/company info/news")
    print("  - 'yfinance' = Using yfinance as fallback")
    print("\nExpected Behavior:")
    print("  - Historical charts (1mo, 3mo, 6mo, 1y) → Twelve Data")
    print("  - Fundamentals, quotes, company info → Finnhub")
    print("  - News headlines → Finnhub")
    print("  - Fallback for all → yfinance (if APIs unavailable)")
    print("=" * 80)


if __name__ == "__main__":
    test_finnhub_integration()
