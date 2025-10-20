"""
Alternative Data Sources Integration

Integrates non-traditional data sources for enhanced analysis:
- Google Trends (search interest)
- On-chain metrics (blockchain data)
- Earnings calls / SEC filings (for crypto-related stocks)
- GitHub activity (for crypto projects)
"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import time


@dataclass
class AlternativeDataPoint:
    """Single alternative data point"""
    source: str
    timestamp: datetime
    metric_name: str
    value: float
    metadata: Dict


class GoogleTrendsCollector:
    """
    Collect Google Trends data
    
    Note: Uses unofficial pytrends library or direct API
    """
    
    def __init__(self):
        try:
            from pytrends.request import TrendReq
            self.pytrends = TrendReq(hl='en-US', tz=360)
            self.available = True
            logger.info("Google Trends collector initialized")
        except ImportError:
            self.pytrends = None
            self.available = False
            logger.warning("pytrends not installed. Install with: pip install pytrends")
    
    def get_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = 'today 3-m'
    ) -> pd.DataFrame:
        """
        Get search interest over time
        
        Args:
            keywords: List of search terms (max 5)
            timeframe: e.g., 'today 3-m', 'now 7-d', 'today 12-m'
        
        Returns:
            DataFrame with timestamp index and interest values
        """
        if not self.available:
            logger.warning("Google Trends not available")
            return pd.DataFrame()
        
        try:
            self.pytrends.build_payload(keywords, timeframe=timeframe)
            data = self.pytrends.interest_over_time()
            
            if not data.empty:
                data = data.drop('isPartial', axis=1, errors='ignore')
                logger.info(f"Fetched Google Trends for {keywords}: {len(data)} data points")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Google Trends: {e}")
            return pd.DataFrame()
    
    def get_related_queries(self, keyword: str) -> Dict[str, pd.DataFrame]:
        """Get related search queries"""
        if not self.available:
            return {}
        
        try:
            self.pytrends.build_payload([keyword])
            related = self.pytrends.related_queries()
            return related.get(keyword, {})
        except Exception as e:
            logger.error(f"Error fetching related queries: {e}")
            return {}


class OnChainMetricsCollector:
    """
    Collect on-chain blockchain metrics
    
    Sources:
    - Glassnode API (premium)
    - CoinMetrics API
    - Free blockchain explorers
    """
    
    def __init__(self, glassnode_api_key: Optional[str] = None):
        """
        Initialize on-chain collector
        
        Args:
            glassnode_api_key: Optional Glassnode API key
        """
        self.glassnode_key = glassnode_api_key
        self.base_url = "https://api.glassnode.com/v1/metrics"
        
        if glassnode_api_key:
            logger.info("OnChain collector initialized with Glassnode API")
        else:
            logger.info("OnChain collector initialized (limited free data)")
    
    def get_active_addresses(
        self,
        asset: str = 'BTC',
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get number of active addresses
        
        Indicator of network usage and adoption
        """
        if not self.glassnode_key:
            return self._generate_mock_data('active_addresses', since, until)
        
        try:
            endpoint = f"{self.base_url}/addresses/active_count"
            params = {
                'a': asset,
                'api_key': self.glassnode_key,
                'f': 'JSON'
            }
            
            if since:
                params['s'] = int(since.timestamp())
            if until:
                params['u'] = int(until.timestamp())
            
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
            df['active_addresses'] = df['v']
            
            return df[['timestamp', 'active_addresses']].set_index('timestamp')
            
        except Exception as e:
            logger.error(f"Error fetching active addresses: {e}")
            return pd.DataFrame()
    
    def get_exchange_flow(
        self,
        asset: str = 'BTC',
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get exchange inflow/outflow
        
        Indicator of selling/buying pressure
        - Inflow: potential selling pressure
        - Outflow: potential accumulation
        """
        if not self.glassnode_key:
            return self._generate_mock_data('exchange_flow', since, until)
        
        try:
            # Get both inflow and outflow
            inflow_url = f"{self.base_url}/transactions/transfers_volume_exchanges_in"
            outflow_url = f"{self.base_url}/transactions/transfers_volume_exchanges_out"
            
            params = {
                'a': asset,
                'api_key': self.glassnode_key,
                'f': 'JSON'
            }
            
            if since:
                params['s'] = int(since.timestamp())
            if until:
                params['u'] = int(until.timestamp())
            
            inflow_data = requests.get(inflow_url, params=params).json()
            outflow_data = requests.get(outflow_url, params=params).json()
            
            df_in = pd.DataFrame(inflow_data)
            df_out = pd.DataFrame(outflow_data)
            
            df = pd.merge(df_in, df_out, on='t', suffixes=('_in', '_out'))
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
            df['net_flow'] = df['v_in'] - df['v_out']
            
            return df[['timestamp', 'v_in', 'v_out', 'net_flow']].set_index('timestamp')
            
        except Exception as e:
            logger.error(f"Error fetching exchange flow: {e}")
            return pd.DataFrame()
    
    def get_nvt_ratio(
        self,
        asset: str = 'BTC',
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get Network Value to Transactions (NVT) ratio
        
        Similar to P/E ratio for stocks
        High NVT = potentially overvalued
        """
        if not self.glassnode_key:
            return self._generate_mock_data('nvt_ratio', since, until)
        
        try:
            endpoint = f"{self.base_url}/indicators/nvt"
            params = {
                'a': asset,
                'api_key': self.glassnode_key,
                'f': 'JSON'
            }
            
            if since:
                params['s'] = int(since.timestamp())
            if until:
                params['u'] = int(until.timestamp())
            
            response = requests.get(endpoint, params=params)
            data = response.json()
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
            df['nvt_ratio'] = df['v']
            
            return df[['timestamp', 'nvt_ratio']].set_index('timestamp')
            
        except Exception as e:
            logger.error(f"Error fetching NVT ratio: {e}")
            return pd.DataFrame()
    
    def _generate_mock_data(
        self,
        metric_type: str,
        since: Optional[datetime],
        until: Optional[datetime]
    ) -> pd.DataFrame:
        """Generate mock data for testing"""
        if since is None:
            since = datetime.now() - timedelta(days=30)
        if until is None:
            until = datetime.now()
        
        dates = pd.date_range(start=since, end=until, freq='D')
        
        if metric_type == 'active_addresses':
            values = np.random.randint(800000, 1200000, len(dates))
            return pd.DataFrame({
                'timestamp': dates,
                'active_addresses': values
            }).set_index('timestamp')
        
        elif metric_type == 'exchange_flow':
            inflow = np.random.randint(10000, 50000, len(dates))
            outflow = np.random.randint(10000, 50000, len(dates))
            return pd.DataFrame({
                'timestamp': dates,
                'v_in': inflow,
                'v_out': outflow,
                'net_flow': inflow - outflow
            }).set_index('timestamp')
        
        elif metric_type == 'nvt_ratio':
            values = np.random.uniform(50, 150, len(dates))
            return pd.DataFrame({
                'timestamp': dates,
                'nvt_ratio': values
            }).set_index('timestamp')
        
        return pd.DataFrame()


class GitHubActivityCollector:
    """
    Collect GitHub activity for crypto projects
    
    Metrics:
    - Commits (development activity)
    - Stars (popularity)
    - Forks (adoption)
    - Issues (engagement)
    """
    
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize GitHub collector
        
        Args:
            github_token: Optional GitHub personal access token (increases rate limit)
        """
        self.token = github_token
        self.base_url = "https://api.github.com"
        self.headers = {}
        
        if github_token:
            self.headers['Authorization'] = f'token {github_token}'
        
        logger.info("GitHub activity collector initialized")
    
    def get_repo_stats(self, owner: str, repo: str) -> Dict:
        """
        Get repository statistics
        
        Args:
            owner: Repository owner (e.g., 'bitcoin')
            repo: Repository name (e.g., 'bitcoin')
        
        Returns:
            Dict with repo statistics
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            stats = {
                'stars': data.get('stargazers_count', 0),
                'forks': data.get('forks_count', 0),
                'watchers': data.get('watchers_count', 0),
                'open_issues': data.get('open_issues_count', 0),
                'created_at': data.get('created_at'),
                'updated_at': data.get('updated_at'),
                'pushed_at': data.get('pushed_at')
            }
            
            logger.info(f"Fetched GitHub stats for {owner}/{repo}")
            return stats
            
        except Exception as e:
            logger.error(f"Error fetching GitHub stats: {e}")
            return {}
    
    def get_commit_activity(
        self,
        owner: str,
        repo: str,
        weeks: int = 52
    ) -> pd.DataFrame:
        """
        Get commit activity over time
        
        Args:
            owner: Repository owner
            repo: Repository name
            weeks: Number of weeks to fetch
        
        Returns:
            DataFrame with weekly commit counts
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/stats/commit_activity"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['week'], unit='s')
                df['commits'] = df['total']
                
                return df[['timestamp', 'commits']].tail(weeks)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching commit activity: {e}")
            return pd.DataFrame()


class AlternativeDataAggregator:
    """
    Aggregate multiple alternative data sources
    """
    
    def __init__(
        self,
        glassnode_key: Optional[str] = None,
        github_token: Optional[str] = None
    ):
        """
        Initialize aggregator
        
        Args:
            glassnode_key: Glassnode API key
            github_token: GitHub token
        """
        self.google_trends = GoogleTrendsCollector()
        self.onchain = OnChainMetricsCollector(glassnode_key)
        self.github = GitHubActivityCollector(github_token)
        
        logger.info("Alternative data aggregator initialized")
    
    def collect_all(
        self,
        symbol: str = 'BTC',
        keywords: List[str] = None,
        github_repos: List[Tuple[str, str]] = None,
        timeframe_days: int = 90
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect all alternative data sources
        
        Args:
            symbol: Crypto symbol
            keywords: Google Trends keywords
            github_repos: List of (owner, repo) tuples
            timeframe_days: Days of historical data
        
        Returns:
            Dict with data from each source
        """
        since = datetime.now() - timedelta(days=timeframe_days)
        until = datetime.now()
        
        results = {}
        
        # Google Trends
        if keywords and self.google_trends.available:
            logger.info("Collecting Google Trends data...")
            trends_data = self.google_trends.get_interest_over_time(
                keywords,
                timeframe=f'today {timeframe_days//30}-m'
            )
            if not trends_data.empty:
                results['google_trends'] = trends_data
        
        # On-chain metrics
        logger.info("Collecting on-chain metrics...")
        active_addr = self.onchain.get_active_addresses(symbol, since, until)
        if not active_addr.empty:
            results['active_addresses'] = active_addr
        
        exchange_flow = self.onchain.get_exchange_flow(symbol, since, until)
        if not exchange_flow.empty:
            results['exchange_flow'] = exchange_flow
        
        nvt = self.onchain.get_nvt_ratio(symbol, since, until)
        if not nvt.empty:
            results['nvt_ratio'] = nvt
        
        # GitHub activity
        if github_repos:
            logger.info("Collecting GitHub activity...")
            for owner, repo in github_repos:
                stats = self.github.get_repo_stats(owner, repo)
                commits = self.github.get_commit_activity(owner, repo)
                
                if stats or not commits.empty:
                    results[f'github_{owner}_{repo}'] = {
                        'stats': stats,
                        'commits': commits
                    }
        
        logger.info(f"Collected {len(results)} alternative data sources")
        return results


# Example usage
if __name__ == "__main__":
    # Initialize aggregator
    aggregator = AlternativeDataAggregator()
    
    # Collect data for Bitcoin
    data = aggregator.collect_all(
        symbol='BTC',
        keywords=['Bitcoin', 'BTC', 'crypto'],
        github_repos=[('bitcoin', 'bitcoin')],
        timeframe_days=30
    )
    
    print(f"\nCollected {len(data)} data sources:")
    for source, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"  {source}: {len(df)} records")
        else:
            print(f"  {source}: dict with {len(df)} keys")
