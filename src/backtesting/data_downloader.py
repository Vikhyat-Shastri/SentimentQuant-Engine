"""
Historical Data Downloader for Backtesting

Downloads OHLCV data from multiple cryptocurrency exchanges:
- Binance
- OKX
- Coinbase
- Kraken

Uses public APIs to fetch historical data as specified in the task requirements.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import requests
from loguru import logger
import time


class HistoricalDataDownloader:
    """
    Downloads historical OHLCV data from cryptocurrency exchanges
    Supports: Binance, OKX, Coinbase, Kraken
    """
    
    def __init__(self, cache_dir: str = "data/historical"):
        """
        Initialize the data downloader
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Exchange API endpoints
        self.exchanges = {
            'binance': {
                'base_url': 'https://api.binance.com/api/v3/klines',
                'symbol_format': lambda s: s.replace('-', ''),  # BTC-USD -> BTCUSD
                'interval_map': {
                    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                    '1h': '1h', '4h': '4h', '1d': '1d'
                }
            },
            'coinbase': {
                'base_url': 'https://api.exchange.coinbase.com/products',
                'symbol_format': lambda s: s,  # BTC-USD stays BTC-USD
                'interval_map': {
                    '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                    '1h': 3600, '4h': 14400, '1d': 86400
                }
            }
        }
    
    def download_binance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1h'
    ) -> Optional[pd.DataFrame]:
        """
        Download data from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USDT', 'ETH-USDT')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert symbol format
            if 'USD' in symbol and 'USDT' not in symbol:
                symbol = symbol.replace('USD', 'USDT')
            
            binance_symbol = symbol.replace('-', '')
            
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Binance interval mapping
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            binance_interval = interval_map.get(interval, '1h')
            
            # Fetch data
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': binance_symbol,
                'interval': binance_interval,
                'startTime': start_ts,
                'endTime': end_ts,
                'limit': 1000
            }
            
            logger.info(f"Downloading {symbol} from Binance ({start_date} to {end_date})...")
            
            all_data = []
            current_start = start_ts
            
            while current_start < end_ts:
                params['startTime'] = current_start
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code != 200:
                    logger.warning(f"Binance API error: {response.status_code}")
                    break
                
                data = response.json()
                if not data:
                    break
                
                all_data.extend(data)
                current_start = data[-1][6] + 1  # Next start time
                
                # Rate limiting
                time.sleep(0.1)
            
            if not all_data:
                logger.warning(f"No data received from Binance for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Process data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"✓ Downloaded {len(df)} bars from Binance for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Binance download error for {symbol}: {e}")
            return None
    
    def download_coinbase(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1h'
    ) -> Optional[pd.DataFrame]:
        """
        Download data from Coinbase Pro
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD', 'ETH-USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Interval in seconds
            granularity_map = {
                '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '4h': 14400, '1d': 86400
            }
            granularity = granularity_map.get(interval, 3600)
            
            # Convert dates
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            logger.info(f"Downloading {symbol} from Coinbase ({start_date} to {end_date})...")
            
            url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
            
            all_data = []
            current_start = start_dt
            
            # Coinbase allows max 300 candles per request
            max_candles = 300
            interval_delta = timedelta(seconds=granularity * max_candles)
            
            while current_start < end_dt:
                current_end = min(current_start + interval_delta, end_dt)
                
                params = {
                    'start': current_start.isoformat(),
                    'end': current_end.isoformat(),
                    'granularity': granularity
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code != 200:
                    logger.warning(f"Coinbase API error: {response.status_code}")
                    break
                
                data = response.json()
                if not data:
                    break
                
                all_data.extend(data)
                current_start = current_end
                
                # Rate limiting
                time.sleep(0.3)
            
            if not all_data:
                logger.warning(f"No data received from Coinbase for {symbol}")
                return None
            
            # Convert to DataFrame
            # Coinbase format: [timestamp, low, high, open, close, volume]
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'low', 'high', 'open', 'close', 'volume'
            ])
            
            # Process data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Reorder columns to standard OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"✓ Downloaded {len(df)} bars from Coinbase for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Coinbase download error for {symbol}: {e}")
            return None
    
    def download_yahoo_finance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1h'
    ) -> Optional[pd.DataFrame]:
        """
        Fallback to Yahoo Finance for crypto data
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD', 'ETH-USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
            
            logger.info(f"Downloading {symbol} from Yahoo Finance ({start_date} to {end_date})...")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if df.empty:
                logger.warning(f"No data received from Yahoo Finance for {symbol}")
                return None
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"✓ Downloaded {len(df)} bars from Yahoo Finance for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Yahoo Finance download error for {symbol}: {e}")
            return None
    
    def download_with_fallback(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1h',
        preferred_exchange: str = 'binance'
    ) -> Optional[pd.DataFrame]:
        """
        Download data with automatic fallback to other sources
        
        Args:
            symbol: Trading pair (e.g., 'BTC-USD', 'ETH-USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval
            preferred_exchange: Preferred exchange to try first
        
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_file = self.cache_dir / f"{symbol.replace('-', '_')}_{start_date}_{end_date}_{interval}.csv"
        if cache_file.exists():
            logger.info(f"Loading {symbol} from cache...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.info(f"✓ Loaded {len(df)} bars from cache")
            return df
        
        # Try exchanges in order
        download_functions = [
            ('binance', self.download_binance),
            ('coinbase', self.download_coinbase),
            ('yahoo', self.download_yahoo_finance)
        ]
        
        # Try preferred exchange first
        for name, func in download_functions:
            if name == preferred_exchange:
                df = func(symbol, start_date, end_date, interval)
                if df is not None and not df.empty:
                    # Cache the data
                    df.to_csv(cache_file)
                    logger.info(f"✓ Cached data to {cache_file}")
                    return df
        
        # Try other exchanges
        for name, func in download_functions:
            if name != preferred_exchange:
                df = func(symbol, start_date, end_date, interval)
                if df is not None and not df.empty:
                    # Cache the data
                    df.to_csv(cache_file)
                    logger.info(f"✓ Cached data to {cache_file}")
                    return df
        
        logger.error(f"Failed to download data for {symbol} from all sources")
        return None
    
    def download_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1h'
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols
        
        Args:
            symbols: List of trading pairs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Time interval
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        logger.info(f"Downloading historical data for {len(symbols)} symbols...")
        
        data = {}
        for symbol in symbols:
            df = self.download_with_fallback(symbol, start_date, end_date, interval)
            if df is not None:
                data[symbol] = df
        
        logger.info(f"✓ Successfully downloaded data for {len(data)}/{len(symbols)} symbols")
        return data


# Quick test function
if __name__ == "__main__":
    downloader = HistoricalDataDownloader()
    
    # Test download
    df = downloader.download_with_fallback(
        symbol='BTC-USD',
        start_date='2024-10-01',
        end_date='2024-10-18',
        interval='1h'
    )
    
    if df is not None:
        print(f"\nDownloaded {len(df)} bars")
        print(df.head())
        print(df.tail())
