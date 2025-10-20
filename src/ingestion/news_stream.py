"""
News data ingestion using NewsAPI and other financial news sources.
Monitors cryptocurrency and financial news for sentiment analysis.
"""

import time
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

# NewsAPI support
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    logger.warning("newsapi-python not installed. Install with: pip install newsapi-python")

from src.utils import (
    thread_manager,
    DataPacket,
    DataType,
    get_timestamp,
    handle_errors_gracefully,
    config_manager
)


@dataclass
class NewsArticle:
    """Container for news article data."""
    title: str
    description: str
    content: str
    source: str
    author: str
    timestamp: float
    url: str


class NewsStream:
    """
    News data stream for crypto sentiment analysis.
    Supports both simulation mode and live NewsAPI integration.
    """
    
    def __init__(
        self,
        keywords: List[str] = None,
        sources: List[str] = None,
        max_articles_per_hour: int = 100,
        mode: str = "simulation"
    ):
        """
        Initialize News stream.
        
        Args:
            keywords: List of keywords to search for
            sources: News sources to monitor (optional)
            max_articles_per_hour: Rate limit for processing
            mode: "simulation" or "live" (requires NewsAPI key)
        """
        self.keywords = keywords or [
            'bitcoin', 'BTC',
            'ethereum', 'ETH',
            'cryptocurrency', 'crypto',
            'blockchain', 'altcoin'
        ]
        
        self.sources = sources  # e.g., ['bloomberg', 'reuters', 'cnbc']
        self.max_articles_per_hour = max_articles_per_hour
        self.mode = mode
        self.running = False
        self.news_client: Optional[NewsApiClient] = None
        
        # Try to load NewsAPI credentials
        self.api_key = None
        if mode == "live":
            try:
                api_config = config_manager.get_section('api_keys', 'news')
                self.api_key = api_config.get('newsapi', {}).get('api_key')
                if not self.api_key or self.api_key == "YOUR_NEWSAPI_KEY":
                    logger.warning("NewsAPI key not found in config. Falling back to simulation mode.")
                    self.mode = "simulation"
            except Exception as e:
                logger.warning(f"Could not load NewsAPI credentials: {e}. Using simulation mode.")
                self.mode = "simulation"
        
        logger.info(f"NewsStream initialized in {self.mode} mode with keywords: {self.keywords}")
    
    def start(self, stop_event: threading.Event) -> None:
        """
        Start collecting news articles.
        
        Args:
            stop_event: Threading event to signal stop
        """
        self.running = True
        
        if self.mode == "live" and NEWSAPI_AVAILABLE and self.api_key:
            logger.info("Starting News stream in LIVE mode (NewsAPI)...")
            self._start_live_stream(stop_event)
        else:
            logger.info("Starting News stream in SIMULATION mode...")
            logger.info("To use live mode: Set mode='live' and add NewsAPI key to config/api_keys.yaml")
            self._simulate_stream(stop_event)
    
    def _start_live_stream(self, stop_event: threading.Event) -> None:
        """
        Start live NewsAPI stream.
        
        Args:
            stop_event: Threading event to signal stop
        """
        try:
            # Initialize NewsAPI client
            self.news_client = NewsApiClient(api_key=self.api_key)
            logger.success("✅ NewsAPI client initialized")
            
            # Track processed article URLs to avoid duplicates
            processed_urls = set()
            
            while self.running and not stop_event.is_set():
                try:
                    # Fetch news for each keyword
                    for keyword in self.keywords:
                        if not self.running or stop_event.is_set():
                            break
                        
                        # Get recent articles
                        # Note: Free tier doesn't support 'from_param', so we get latest articles
                        
                        # Call NewsAPI (free tier)
                        response = self.news_client.get_everything(
                            q=keyword,
                            language='en',
                            sort_by='publishedAt',
                            page_size=10  # Free tier limit
                        )
                        
                        if response['status'] == 'ok':
                            articles = response.get('articles', [])
                            
                            for article_data in articles:
                                # Skip if already processed
                                url = article_data.get('url', '')
                                if url in processed_urls:
                                    continue
                                
                                processed_urls.add(url)
                                
                                # Create NewsArticle object
                                article = NewsArticle(
                                    title=article_data.get('title', ''),
                                    description=article_data.get('description', ''),
                                    content=article_data.get('content', ''),
                                    source=article_data.get('source', {}).get('name', 'Unknown'),
                                    author=article_data.get('author', 'Unknown'),
                                    timestamp=self._parse_timestamp(article_data.get('publishedAt')),
                                    url=url
                                )
                                
                                # Create data packet
                                packet = DataPacket(
                                    data_type=DataType.NEWS,
                                    timestamp=article.timestamp,
                                    source="newsapi",
                                    data=article,
                                    metadata={
                                        'source': article.source,
                                        'author': article.author,
                                        'keyword': keyword
                                    }
                                )
                                
                                # Put in processing queue
                                thread_manager.put_data('raw_data', packet, block=False)
                                
                                logger.debug(f"Fetched article: {article.title[:50]}...")
                            
                            logger.info(f"Fetched {len(articles)} articles for keyword: {keyword}")
                    
                    # Clean up old URLs from processed set (keep last 1000)
                    if len(processed_urls) > 1000:
                        processed_urls = set(list(processed_urls)[-1000:])
                    
                    # Sleep to respect rate limits (Free tier: 100 requests/day)
                    # With ~10 keywords and hourly polling, that's ~240 requests/day
                    sleep_time = 3600 / max(1, self.max_articles_per_hour / len(self.keywords))
                    
                    logger.debug(f"Sleeping for {sleep_time:.1f} seconds before next poll...")
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error fetching news: {e}")
                    time.sleep(60)  # Wait before retrying
            
            logger.info("News stream stopped")
            
        except Exception as e:
            logger.exception(f"Error in live news stream: {e}")
            logger.warning("Falling back to simulation mode")
            self._simulate_stream(stop_event)
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """
        Parse ISO timestamp string to Unix timestamp.
        
        Args:
            timestamp_str: ISO format timestamp
            
        Returns:
            Unix timestamp
        """
        try:
            if timestamp_str:
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                return dt.timestamp()
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
        
        return time.time()
    
    def _simulate_stream(self, stop_event: threading.Event) -> None:
        """
        Simulate news article collection for testing.
        
        Args:
            stop_event: Threading event to signal stop
        """
        logger.info("Running News stream in SIMULATION mode")
        logger.info("To use live news data, set mode='live' and configure NewsAPI credentials")
        
        # Sample news articles for simulation
        sample_articles = [
            (
                "Bitcoin Price Surges to New All-Time High",
                "Bitcoin (BTC) has reached a new all-time high today, breaking through the $100,000 barrier for the first time in history.",
                "CryptoNews", 0.9
            ),
            (
                "Ethereum 2.0 Upgrade Completes Successfully",
                "The Ethereum network has successfully completed its transition to proof-of-stake, marking a major milestone.",
                "CoinDesk", 0.8
            ),
            (
                "Regulatory Concerns Weigh on Crypto Markets",
                "Cryptocurrency markets face selling pressure as regulators announce stricter compliance requirements.",
                "Bloomberg", -0.6
            ),
            (
                "Major Institution Adopts Bitcoin as Treasury Reserve",
                "A Fortune 500 company announces significant Bitcoin purchase for its corporate treasury.",
                "Reuters", 0.7
            ),
            (
                "DeFi Protocol Suffers Major Security Breach",
                "A popular decentralized finance protocol reports a critical vulnerability leading to millions in losses.",
                "CryptoDaily", -0.8
            ),
            (
                "Central Bank Explores Digital Currency Initiative",
                "The Federal Reserve announces plans to study central bank digital currency implementation.",
                "Financial Times", 0.5
            ),
            (
                "Crypto Market Volatility Reaches Record Levels",
                "Trading volumes surge as cryptocurrency markets experience extreme price swings.",
                "CNBC", -0.4
            ),
            (
                "Institutional Adoption of Crypto Accelerates",
                "Survey shows 60% of institutional investors now hold digital assets in their portfolios.",
                "Forbes", 0.8
            ),
        ]
        
        article_index = 0
        
        while self.running and not stop_event.is_set():
            try:
                # Generate simulated article
                title, description, source, _ = sample_articles[article_index % len(sample_articles)]
                
                article = NewsArticle(
                    title=title,
                    description=description,
                    content=description + " " + title,  # Combine for more text
                    source=source,
                    author="News Simulation",
                    timestamp=get_timestamp(),
                    url=f"https://news.example.com/article/{article_index}"
                )
                
                # Create data packet
                packet = DataPacket(
                    data_type=DataType.NEWS,
                    timestamp=article.timestamp,
                    source="news_simulation",
                    data=article,
                    metadata={
                        'source': article.source,
                        'author': article.author
                    }
                )
                
                # Put in processing queue
                thread_manager.put_data('raw_data', packet, block=False)
                
                logger.debug(f"Simulated news article: {title[:50]}...")
                
                article_index += 1
                
                # Sleep to respect rate limit
                sleep_time = 3600.0 / self.max_articles_per_hour
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in news simulation: {e}")
                time.sleep(5)  # Wait before retrying
        
        logger.info("News stream stopped")
    
    def stop(self) -> None:
        """Stop the news stream."""
        self.running = False
        logger.info("Stopping news stream...")


def start_news_stream(stop_event: threading.Event, mode: str = "simulation") -> None:
    """
    Thread worker function for News stream.
    
    Args:
        stop_event: Threading event to signal stop
        mode: "simulation" or "live"
    """
    stream = NewsStream(max_articles_per_hour=50, mode=mode)
    stream.start(stop_event)


if __name__ == "__main__":
    # Test the News stream
    import signal
    import argparse
    
    parser = argparse.ArgumentParser(description='Test News Stream')
    parser.add_argument('--mode', choices=['simulation', 'live'], default='simulation',
                       help='Stream mode: simulation or live (requires NewsAPI key)')
    args = parser.parse_args()
    
    stop_event = threading.Event()
    
    def signal_handler(sig, frame):
        logger.info("Received stop signal")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info(f"Starting News stream test in {args.mode} mode...")
    logger.info("Press Ctrl+C to stop")
    
    # Start in main thread for testing
    stream = NewsStream(max_articles_per_hour=10, mode=args.mode)
    stream.start(stop_event)
