"""
Market data ingestion from cryptocurrency exchanges.
Provides real-time price, volume, and order book data.
"""

import time
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from src.utils import (
    thread_manager,
    DataPacket,
    DataType,
    get_timestamp,
    handle_errors_gracefully
)


@dataclass
class MarketData:
    """Container for market data."""
    symbol: str
    price: float
    volume_24h: float
    price_change_24h: float
    timestamp: float
    high_24h: float
    low_24h: float
    bid: float
    ask: float


class MarketDataFeed:
    """
    Real-time market data feed for cryptocurrencies.
    Uses yfinance and simulated data for testing.
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        update_interval: int = 5
    ):
        """
        Initialize market data feed.
        
        Args:
            symbols: List of trading symbols to monitor
            update_interval: Seconds between updates
        """
        self.symbols = symbols or [
            'BTC-USD',
            'ETH-USD',
            'BNB-USD',
            'SOL-USD',
            'ADA-USD'
        ]
        
        self.update_interval = update_interval
        self.running = False
        self.last_prices = {}
        
        logger.info(f"MarketDataFeed initialized for symbols: {self.symbols}")
    
    def start(self, stop_event: threading.Event) -> None:
        """
        Start collecting market data.
        
        Args:
            stop_event: Threading event to signal stop
        """
        self.running = True
        logger.info("Starting market data feed...")
        
        # Use real market data from yfinance
        self._start_real_feed(stop_event)
    
    @handle_errors_gracefully()
    def _fetch_real_data(self, symbol: str) -> Optional[MarketData]:
        """
        Fetch real market data using yfinance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            
        Returns:
            MarketData object or None
        """
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            
            # Get fast info for current data
            try:
                # Try to get fast info first (faster but less data)
                fast_info = ticker.fast_info
                current_price = fast_info.get('last_price', 0)
                
                if current_price == 0:
                    # Fallback to full info
                    info = ticker.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                else:
                    # Get additional data from history for 24h stats
                    hist = ticker.history(period='1d', interval='1m')
                    if not hist.empty:
                        high_24h = hist['High'].max()
                        low_24h = hist['Low'].min()
                        volume_24h = hist['Volume'].sum()
                        
                        # Calculate 24h price change
                        if len(hist) > 0:
                            first_price = hist['Close'].iloc[0]
                            price_change_24h = ((current_price - first_price) / first_price) * 100 if first_price > 0 else 0
                        else:
                            price_change_24h = 0
                    else:
                        high_24h = current_price
                        low_24h = current_price
                        volume_24h = 0
                        price_change_24h = 0
                        
                    data = MarketData(
                        symbol=symbol,
                        price=current_price,
                        volume_24h=volume_24h,
                        price_change_24h=price_change_24h,
                        timestamp=get_timestamp(),
                        high_24h=high_24h,
                        low_24h=low_24h,
                        bid=current_price * 0.999,  # Approximate bid/ask
                        ask=current_price * 1.001
                    )
                    return data
                    
            except Exception as e:
                logger.debug(f"Fast info not available for {symbol}, using full info: {e}")
                # Fallback to full info method
                info = ticker.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            
            if current_price == 0:
                logger.warning(f"No price data available for {symbol}")
                return None
            
            data = MarketData(
                symbol=symbol,
                price=current_price,
                volume_24h=info.get('volume', 0),
                price_change_24h=info.get('regularMarketChangePercent', 0),
                timestamp=get_timestamp(),
                high_24h=info.get('dayHigh', current_price),
                low_24h=info.get('dayLow', current_price),
                bid=info.get('bid', current_price * 0.999),
                ask=info.get('ask', current_price * 1.001)
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _start_real_feed(self, stop_event: threading.Event) -> None:
        """
        Start real market data feed using yfinance.
        
        Args:
            stop_event: Threading event to signal stop
        """
        logger.info("Running market data feed in LIVE mode")
        logger.success("✅ Fetching real-time prices from Yahoo Finance")
        
        while self.running and not stop_event.is_set():
            try:
                for symbol in self.symbols:
                    # Fetch real market data
                    data = self._fetch_real_data(symbol)
                    
                    if data is None:
                        logger.warning(f"Could not fetch data for {symbol}, skipping...")
                        continue
                    
                    # Create data packet
                    packet = DataPacket(
                        data_type=DataType.MARKET,
                        timestamp=data.timestamp,
                        source="yfinance",
                        data=data,
                        metadata={
                            'symbol': symbol,
                            'price': data.price,
                            'volume': data.volume_24h,
                            'change_24h': data.price_change_24h
                        }
                    )
                    
                    # Put in processing queue
                    thread_manager.put_data('raw_data', packet, block=False)
                    
                    logger.info(f"📊 {symbol}: ${data.price:,.2f} ({data.price_change_24h:+.2f}%) | Vol: ${data.volume_24h:,.0f}")
                
                # Sleep until next update
                logger.debug(f"Sleeping for {self.update_interval} seconds before next update...")
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in real market data feed: {e}")
                time.sleep(5)
        
        logger.info("Market data feed stopped")
    
    def _simulate_feed(self, stop_event: threading.Event) -> None:
        """
        Simulate market data feed for testing.
        
        Args:
            stop_event: Threading event to signal stop
        """
        logger.info("Running market data feed in simulation mode")
        logger.warning("Using simulated prices - configure exchange APIs for live data")
        
        # Initialize simulated prices
        base_prices = {
            'BTC-USD': 65000.0,
            'ETH-USD': 3500.0,
            'BNB-USD': 580.0,
            'SOL-USD': 140.0,
            'ADA-USD': 0.45
        }
        
        import random
        
        while self.running and not stop_event.is_set():
            try:
                for symbol in self.symbols:
                    # Simulate price movement
                    if symbol not in self.last_prices:
                        self.last_prices[symbol] = base_prices.get(symbol, 100.0)
                    
                    # Random price change (-1% to +1%)
                    price_change = random.uniform(-0.01, 0.01)
                    new_price = self.last_prices[symbol] * (1 + price_change)
                    self.last_prices[symbol] = new_price
                    
                    # Create market data
                    data = MarketData(
                        symbol=symbol,
                        price=new_price,
                        volume_24h=random.uniform(1e9, 1e10),
                        price_change_24h=random.uniform(-5, 5),
                        timestamp=get_timestamp(),
                        high_24h=new_price * 1.02,
                        low_24h=new_price * 0.98,
                        bid=new_price * 0.999,
                        ask=new_price * 1.001
                    )
                    
                    # Create data packet
                    packet = DataPacket(
                        data_type=DataType.MARKET,
                        timestamp=data.timestamp,
                        source="market_data_feed",
                        data=data,
                        metadata={
                            'symbol': symbol,
                            'price': data.price,
                            'volume': data.volume_24h
                        }
                    )
                    
                    # Put in processing queue
                    thread_manager.put_data('raw_data', packet, block=False)
                    
                    logger.debug(f"Market update {symbol}: ${data.price:.2f} ({data.price_change_24h:+.2f}%)")
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in market data feed: {e}")
                time.sleep(5)
        
        logger.info("Market data feed stopped")
    
    def stop(self) -> None:
        """Stop the market data feed."""
        self.running = False
        logger.info("Stopping market data feed...")


def start_market_feed(stop_event: threading.Event) -> None:
    """
    Thread worker function for market data feed.
    
    Args:
        stop_event: Threading event to signal stop
    """
    feed = MarketDataFeed(update_interval=5)
    feed.start(stop_event)


if __name__ == "__main__":
    # Test the market data feed
    import signal
    
    stop_event = threading.Event()
    
    def signal_handler(sig, frame):
        logger.info("Received stop signal")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting market data feed test...")
    logger.info("Press Ctrl+C to stop")
    
    # Start in main thread for testing
    feed = MarketDataFeed(update_interval=2)
    feed.start(stop_event)
