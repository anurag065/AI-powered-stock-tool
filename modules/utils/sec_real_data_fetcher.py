import requests
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import re

logger = logging.getLogger(__name__)

class SECFinancialDataFetcher:
    """
    Real-time SEC financial data fetcher using official SEC EDGAR API
    Fetches actual financial data from 10-K and 10-Q filings
    """
    
    def __init__(self):
        self.base_url = "https://data.sec.gov"
        self.headers = {
            'User-Agent': 'Stock Analysis Platform contact@yourcompany.com',
            'Accept': 'application/json',
            'Host': 'data.sec.gov'
        }
        
        # Cache setup
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "sec_real_data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "sec_financial_data.db"
        
        self._init_database()
        self.last_request_time = 0
        self.min_request_interval = 0.1  # SEC rate limit compliance
        
        # Common CIK mappings for major companies
        self.ticker_to_cik = self._load_ticker_mappings()
    
    def _load_ticker_mappings(self):
        """Load common ticker to CIK mappings"""
        return {
            'AAPL': '0000320193',
            'MSFT': '0000789019', 
            'GOOGL': '0001652044',
            'GOOG': '0001652044',
            'AMZN': '0001018724',
            'TSLA': '0001318605',
            'META': '0001326801',
            'NVDA': '0001045810',
            'NFLX': '0001065280',
            'AMD': '0000002488',
            'INTC': '0000050863',
            'CRM': '0001108524',
            'ORCL': '0000777476',
            'IBM': '0000051143',
            'ADBE': '0000796343',
            'PYPL': '0001633917',
            'DIS': '0001001039',
            'BA': '0000012927',
            'JPM': '0000019617',
            'JNJ': '0000200406'
        }
    
    def _rate_limit(self):
        """Enforce SEC API rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _init_database(self):
        """Initialize database for caching financial data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                cik TEXT,
                form_type TEXT,
                fiscal_year INTEGER,
                fiscal_period TEXT,
                filing_date TEXT,
                revenue REAL,
                net_income REAL,
                total_assets REAL,
                total_debt REAL,
                cash_equivalents REAL,
                shares_outstanding REAL,
                eps REAL,
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, form_type, fiscal_year, fiscal_period)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                fiscal_year INTEGER,
                segment_name TEXT,
                segment_revenue REAL,
                segment_percentage REAL,
                filing_date TEXT,
                UNIQUE(ticker, fiscal_year, segment_name)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_company_cik(self, ticker):
        """Get CIK for a ticker"""
        # First try our known mappings
        cik = self.ticker_to_cik.get(ticker.upper())
        if cik:
            return cik
        
        # Try to fetch from SEC company tickers endpoint
        try:
            self._rate_limit()
            url = f"{self.base_url}/files/company_tickers.json"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                companies = response.json()
                for company_data in companies.values():
                    if company_data.get('ticker', '').upper() == ticker.upper():
                        return str(company_data.get('cik_str', '')).zfill(10)
        except Exception as e:
            logger.warning(f"Could not fetch CIK from SEC API: {e}")
        
        return None
    
    def fetch_company_facts(self, ticker):
        """Fetch company financial facts from SEC EDGAR"""
        try:
            # Check cache first
            cached_data = self._get_cached_data(ticker)
            if cached_data:
                logger.info(f"Returning cached financial data for {ticker}")
                return cached_data
            
            # Get CIK
            cik = self.get_company_cik(ticker)
            if not cik:
                logger.warning(f"No CIK found for ticker {ticker}")
                return None
            
            self._rate_limit()
            
            # Fetch company facts
            facts_url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik}.json"
            response = requests.get(facts_url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"SEC API returned {response.status_code} for {ticker}")
                return None
            
            facts_data = response.json()
            
            # Parse financial data
            financial_data = self._parse_financial_facts(ticker, cik, facts_data)
            
            # Cache the data
            if financial_data:
                self._cache_financial_data(ticker, financial_data)
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error fetching SEC financial data for {ticker}: {str(e)}")
            return None
    
    def _parse_financial_facts(self, ticker, cik, facts_data):
        """Parse SEC facts data into structured financial information"""
        try:
            parsed_data = {
                'ticker': ticker,
                'cik': cik,
                'annual_data': [],
                'quarterly_data': [],
                'latest_annual': None,
                'latest_quarterly': None
            }
            
            # Get US GAAP facts
            us_gaap = facts_data.get('facts', {}).get('us-gaap', {})
            if not us_gaap:
                logger.warning(f"No US-GAAP data found for {ticker}")
                return None
            
            # Key financial metrics to extract
            financial_metrics = {
                'Revenues': 'revenues',
                'RevenueFromContractWithCustomerExcludingAssessedTax': 'revenues',
                'NetIncomeLoss': 'net_income',
                'Assets': 'total_assets',
                'AssetsCurrent': 'current_assets',
                'Liabilities': 'total_liabilities',
                'LiabilitiesCurrent': 'current_liabilities',
                'StockholdersEquity': 'stockholders_equity',
                'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents': 'cash_equivalents',
                'CashAndCashEquivalentsAtCarryingValue': 'cash_equivalents',
                'CommonStockSharesOutstanding': 'shares_outstanding',
                'EarningsPerShareBasic': 'eps_basic'
            }
            
            # Process each metric
            for sec_field, our_field in financial_metrics.items():
                if sec_field in us_gaap:
                    metric_data = us_gaap[sec_field]['units']
                    
                    # Process USD values
                    if 'USD' in metric_data:
                        for entry in metric_data['USD']:
                            fiscal_year = entry.get('fy')
                            fiscal_period = entry.get('fp', 'FY')
                            filed_date = entry.get('filed')
                            value = entry.get('val')
                            
                            if fiscal_year and value:
                                # Determine if annual or quarterly
                                if fiscal_period == 'FY':
                                    # Annual data
                                    annual_entry = self._find_or_create_period_entry(
                                        parsed_data['annual_data'], 
                                        fiscal_year, 
                                        'FY',
                                        filed_date
                                    )
                                    annual_entry[our_field] = value
                                else:
                                    # Quarterly data
                                    quarterly_entry = self._find_or_create_period_entry(
                                        parsed_data['quarterly_data'], 
                                        fiscal_year, 
                                        fiscal_period,
                                        filed_date
                                    )
                                    quarterly_entry[our_field] = value
                    
                    # Process shares values
                    elif 'shares' in metric_data:
                        for entry in metric_data['shares']:
                            fiscal_year = entry.get('fy')
                            fiscal_period = entry.get('fp', 'FY')
                            filed_date = entry.get('filed')
                            value = entry.get('val')
                            
                            if fiscal_year and value:
                                if fiscal_period == 'FY':
                                    annual_entry = self._find_or_create_period_entry(
                                        parsed_data['annual_data'], 
                                        fiscal_year, 
                                        'FY',
                                        filed_date
                                    )
                                    annual_entry[our_field] = value
                                else:
                                    quarterly_entry = self._find_or_create_period_entry(
                                        parsed_data['quarterly_data'], 
                                        fiscal_year, 
                                        fiscal_period,
                                        filed_date
                                    )
                                    quarterly_entry[our_field] = value
            
            # Sort by fiscal year (most recent first)
            parsed_data['annual_data'].sort(key=lambda x: x['fiscal_year'], reverse=True)
            parsed_data['quarterly_data'].sort(key=lambda x: (x['fiscal_year'], x['fiscal_period']), reverse=True)
            
            # Set latest data
            if parsed_data['annual_data']:
                parsed_data['latest_annual'] = parsed_data['annual_data'][0]
            if parsed_data['quarterly_data']:
                parsed_data['latest_quarterly'] = parsed_data['quarterly_data'][0]
            
            # Calculate additional metrics
            self._calculate_derived_metrics(parsed_data)
            
            logger.info(f"Successfully parsed SEC financial data for {ticker}")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing financial facts for {ticker}: {str(e)}")
            return None
    
    def _find_or_create_period_entry(self, data_list, fiscal_year, fiscal_period, filed_date):
        """Find existing period entry or create new one"""
        for entry in data_list:
            if entry['fiscal_year'] == fiscal_year and entry['fiscal_period'] == fiscal_period:
                return entry
        
        # Create new entry
        new_entry = {
            'fiscal_year': fiscal_year,
            'fiscal_period': fiscal_period,
            'filed_date': filed_date
        }
        data_list.append(new_entry)
        return new_entry
    
    def _calculate_derived_metrics(self, parsed_data):
        """Calculate derived financial metrics"""
        try:
            # Calculate for annual data
            for annual in parsed_data['annual_data']:
                # Revenue growth (if previous year exists)
                prev_year_data = None
                for data in parsed_data['annual_data']:
                    if data['fiscal_year'] == annual['fiscal_year'] - 1:
                        prev_year_data = data
                        break
                
                if prev_year_data and 'revenues' in annual and 'revenues' in prev_year_data:
                    if prev_year_data['revenues'] > 0:
                        growth = ((annual['revenues'] - prev_year_data['revenues']) / prev_year_data['revenues']) * 100
                        annual['revenue_growth'] = round(growth, 2)
                
                # Calculate EPS if not available
                if 'net_income' in annual and 'shares_outstanding' in annual and 'eps_basic' not in annual:
                    if annual['shares_outstanding'] > 0:
                        annual['eps_basic'] = annual['net_income'] / annual['shares_outstanding']
                
                # Calculate debt-to-equity
                if 'total_liabilities' in annual and 'stockholders_equity' in annual:
                    if annual['stockholders_equity'] > 0:
                        debt_to_equity = annual['total_liabilities'] / annual['stockholders_equity']
                        annual['debt_to_equity'] = round(debt_to_equity, 2)
                
                # Calculate current ratio
                if 'current_assets' in annual and 'current_liabilities' in annual:
                    if annual['current_liabilities'] > 0:
                        current_ratio = annual['current_assets'] / annual['current_liabilities']
                        annual['current_ratio'] = round(current_ratio, 2)
            
            # Similar calculations for quarterly data
            for quarterly in parsed_data['quarterly_data']:
                # Find same quarter previous year for YoY growth
                prev_year_quarterly = None
                for data in parsed_data['quarterly_data']:
                    if (data['fiscal_year'] == quarterly['fiscal_year'] - 1 and 
                        data['fiscal_period'] == quarterly['fiscal_period']):
                        prev_year_quarterly = data
                        break
                
                if (prev_year_quarterly and 'revenues' in quarterly and 
                    'revenues' in prev_year_quarterly and prev_year_quarterly['revenues'] > 0):
                    growth = ((quarterly['revenues'] - prev_year_quarterly['revenues']) / prev_year_quarterly['revenues']) * 100
                    quarterly['revenue_growth_yoy'] = round(growth, 2)
                
        except Exception as e:
            logger.error(f"Error calculating derived metrics: {str(e)}")
    
    def _get_cached_data(self, ticker):
        """Get cached financial data if recent"""
        conn = sqlite3.connect(self.db_path)
        
        # Check for recent cache (less than 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT raw_data FROM financial_data 
            WHERE ticker = ? AND created_at > ?
            ORDER BY fiscal_year DESC, created_at DESC
            LIMIT 1
        ''', (ticker, cutoff_time))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                return json.loads(result[0])
            except:
                return None
        
        return None
    
    def _cache_financial_data(self, ticker, data):
        """Cache financial data in database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Cache the complete data as JSON
            raw_data_json = json.dumps(data)
            
            # Cache latest annual data
            if data.get('latest_annual'):
                annual = data['latest_annual']
                conn.execute('''
                    INSERT OR REPLACE INTO financial_data 
                    (ticker, cik, form_type, fiscal_year, fiscal_period, filing_date,
                     revenue, net_income, total_assets, total_debt, cash_equivalents,
                     shares_outstanding, eps, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker, data['cik'], '10-K', annual['fiscal_year'], annual['fiscal_period'],
                    annual.get('filed_date'), annual.get('revenues'), annual.get('net_income'),
                    annual.get('total_assets'), annual.get('total_liabilities'), 
                    annual.get('cash_equivalents'), annual.get('shares_outstanding'),
                    annual.get('eps_basic'), raw_data_json
                ))
            
            # Cache latest quarterly data
            if data.get('latest_quarterly'):
                quarterly = data['latest_quarterly']
                conn.execute('''
                    INSERT OR REPLACE INTO financial_data 
                    (ticker, cik, form_type, fiscal_year, fiscal_period, filing_date,
                     revenue, net_income, total_assets, total_debt, cash_equivalents,
                     shares_outstanding, eps, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker, data['cik'], '10-Q', quarterly['fiscal_year'], quarterly['fiscal_period'],
                    quarterly.get('filed_date'), quarterly.get('revenues'), quarterly.get('net_income'),
                    quarterly.get('total_assets'), quarterly.get('total_liabilities'),
                    quarterly.get('cash_equivalents'), quarterly.get('shares_outstanding'),
                    quarterly.get('eps_basic'), raw_data_json
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error caching financial data for {ticker}: {str(e)}")
        finally:
            conn.close()
    
    def format_financial_value(self, value, format_type='currency'):
        """Format financial values for display"""
        if value is None:
            return 'N/A'
        
        try:
            if format_type == 'currency':
                if abs(value) >= 1e12:
                    return f"${value/1e12:.1f}T"
                elif abs(value) >= 1e9:
                    return f"${value/1e9:.1f}B"
                elif abs(value) >= 1e6:
                    return f"${value/1e6:.1f}M"
                elif abs(value) >= 1e3:
                    return f"${value/1e3:.1f}K"
                else:
                    return f"${value:,.0f}"
            elif format_type == 'percentage':
                return f"{value:.1f}%"
            elif format_type == 'ratio':
                return f"{value:.2f}"
            elif format_type == 'shares':
                if abs(value) >= 1e9:
                    return f"{value/1e9:.1f}B shares"
                elif abs(value) >= 1e6:
                    return f"{value/1e6:.1f}M shares"
                else:
                    return f"{value:,.0f} shares"
            else:
                return f"{value:,.0f}"
                
        except (ValueError, TypeError):
            return 'N/A'
    
    def get_financial_summary(self, ticker):
        """Get formatted financial summary for dashboard"""
        try:
            data = self.fetch_company_facts(ticker)
            
            if not data:
                return None
            
            summary = {
                'ticker': ticker,
                'has_data': True,
                'annual_summary': None,
                'quarterly_summary': None,
                'key_metrics': {}
            }
            
            # Format annual data
            if data.get('latest_annual'):
                annual = data['latest_annual']
                summary['annual_summary'] = {
                    'fiscal_year': annual['fiscal_year'],
                    'filed_date': annual.get('filed_date', 'N/A'),
                    'revenue': self.format_financial_value(annual.get('revenues')),
                    'revenue_growth': f"{annual.get('revenue_growth', 0):.1f}%" if annual.get('revenue_growth') else 'N/A',
                    'net_income': self.format_financial_value(annual.get('net_income')),
                    'total_assets': self.format_financial_value(annual.get('total_assets')),
                    'cash': self.format_financial_value(annual.get('cash_equivalents')),
                    'eps': f"${annual.get('eps_basic', 0):.2f}" if annual.get('eps_basic') else 'N/A',
                    'debt_to_equity': f"{annual.get('debt_to_equity', 0):.2f}" if annual.get('debt_to_equity') else 'N/A'
                }
            
            # Format quarterly data
            if data.get('latest_quarterly'):
                quarterly = data['latest_quarterly']
                summary['quarterly_summary'] = {
                    'fiscal_year': quarterly['fiscal_year'],
                    'fiscal_period': quarterly['fiscal_period'],
                    'filed_date': quarterly.get('filed_date', 'N/A'),
                    'revenue': self.format_financial_value(quarterly.get('revenues')),
                    'revenue_growth_yoy': f"{quarterly.get('revenue_growth_yoy', 0):.1f}%" if quarterly.get('revenue_growth_yoy') else 'N/A',
                    'net_income': self.format_financial_value(quarterly.get('net_income')),
                    'eps': f"${quarterly.get('eps_basic', 0):.2f}" if quarterly.get('eps_basic') else 'N/A'
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating financial summary for {ticker}: {str(e)}")
            return {'ticker': ticker, 'has_data': False, 'error': str(e)}

# Integration class for the main app
class RealSECIntegration:
    """Integration class for real SEC financial data"""
    
    def __init__(self):
        self.fetcher = SECFinancialDataFetcher()
    
    def get_real_sec_data(self, ticker):
        """Get real SEC financial data for dashboard"""
        return self.fetcher.get_financial_summary(ticker)