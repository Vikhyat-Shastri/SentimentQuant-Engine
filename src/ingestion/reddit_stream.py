"""
Reddit data ingestion using PRAW.
Monitors cryptocurrency subreddits for sentiment analysis.
"""

import time
import threading
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger

from src.utils import (
    thread_manager,
    DataPacket,
    DataType,
    get_timestamp,
    config_manager,
    handle_errors_gracefully,
    retry_with_backoff
)


@dataclass
class RedditPost:
    """Container for Reddit post/comment data."""
    text: str
    author: str
    timestamp: float
    karma: int
    awards: int
    subreddit: str
    post_type: str  # 'submission' or 'comment'
    url: str


class RedditStream:
    """
    Reddit data stream for crypto sentiment analysis.
    Supports both simulation mode and live PRAW streaming.
    """
    
    def __init__(
        self,
        subreddits: List[str] = None,
        max_posts_per_minute: int = 100,
        mode: str = "simulation"
    ):
        """
        Initialize Reddit stream.
        
        Args:
            subreddits: List of subreddits to monitor
            max_posts_per_minute: Rate limit for processing
            mode: "simulation" or "live" (requires Reddit API credentials)
        """
        self.subreddits = subreddits or [
            'cryptocurrency',
            'Bitcoin',
            'ethereum',
            'CryptoMarkets',
            'CryptoCurrency',
            'altcoin'
        ]
        
        self.max_posts_per_minute = max_posts_per_minute
        self.mode = mode
        self.running = False
        self.reddit_client = None
        
        logger.info(f"RedditStream initialized in {mode} mode for subreddits: {self.subreddits}")
    
    def _init_reddit_client(self) -> bool:
        """
        Initialize PRAW Reddit client.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get Reddit credentials from config
            client_id = config_manager.get('api_keys', 'reddit.client_id', '')
            client_secret = config_manager.get('api_keys', 'reddit.client_secret', '')
            user_agent = config_manager.get('api_keys', 'reddit.user_agent', 'sentiment_engine/1.0')
            
            if not client_id or not client_secret or client_id == "YOUR_CLIENT_ID":
                logger.warning("Reddit API credentials not configured")
                logger.warning("Running in simulation mode - configure config/api_keys.yaml for live data")
                return False
            
            import praw
            
            self.reddit_client = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            # Test connection
            self.reddit_client.user.me()
            
            logger.success("Reddit client initialized successfully")
            return True
            
        except ImportError:
            logger.error("PRAW not installed. Install with: pip install praw")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            return False
    
    def start(self, stop_event: threading.Event) -> None:
        """
        Start monitoring Reddit.
        
        Args:
            stop_event: Threading event to signal stop
        """
        self.running = True
        
        if self.mode == "live":
            logger.info("Starting Reddit stream in LIVE mode (PRAW)...")
            # Try to initialize real client, fall back to simulation
            if self._init_reddit_client():
                self._stream_reddit(stop_event)
            else:
                logger.warning("Failed to initialize PRAW. Falling back to simulation mode.")
                self._simulate_stream(stop_event)
        else:
            logger.info("Starting Reddit stream in SIMULATION mode...")
            logger.info("To use live Reddit data, set mode='live' and configure Reddit API credentials")
            self._simulate_stream(stop_event)
    
    def _stream_reddit(self, stop_event: threading.Event) -> None:
        """
        Stream real Reddit data using PRAW.
        
        Args:
            stop_event: Threading event to signal stop
        """
        if not self.reddit_client:
            logger.error("Reddit client not initialized")
            return
        
        logger.info("Streaming live Reddit data...")
        
        try:
            # Create subreddit stream
            subreddit_str = '+'.join(self.subreddits)
            subreddit = self.reddit_client.subreddit(subreddit_str)
            
            # Stream comments and submissions
            for item in subreddit.stream.comments(skip_existing=True):
                if not self.running or stop_event.is_set():
                    break
                
                try:
                    # Process comment
                    self._process_reddit_item(item, 'comment')
                    
                    # Rate limiting
                    time.sleep(60.0 / self.max_posts_per_minute)
                    
                except Exception as e:
                    logger.error(f"Error processing Reddit item: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in Reddit stream: {e}")
        finally:
            logger.info("Reddit stream stopped")
    
    def _process_reddit_item(self, item, item_type: str) -> None:
        """
        Process a Reddit post/comment.
        
        Args:
            item: PRAW submission or comment object
            item_type: 'submission' or 'comment'
        """
        try:
            # Extract text
            if item_type == 'submission':
                text = f"{item.title} {item.selftext}"
            else:
                text = item.body
            
            # Skip deleted/removed content
            if text in ['[deleted]', '[removed]', '']:
                return
            
            # Create RedditPost object
            post = RedditPost(
                text=text,
                author=str(item.author) if item.author else '[deleted]',
                timestamp=get_timestamp(),
                karma=item.score,
                awards=item.total_awards_received if hasattr(item, 'total_awards_received') else 0,
                subreddit=str(item.subreddit),
                post_type=item_type,
                url=f"https://reddit.com{item.permalink}"
            )
            
            # Create data packet
            packet = DataPacket(
                data_type=DataType.REDDIT,
                timestamp=post.timestamp,
                source="reddit_stream",
                data=post,
                metadata={
                    'karma': post.karma,
                    'awards': post.awards,
                    'subreddit': post.subreddit
                }
            )
            
            # Put in processing queue
            thread_manager.put_data('raw_data', packet, block=False)
            
            logger.debug(f"Collected Reddit {item_type} from r/{post.subreddit}: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing Reddit item: {e}")
    
    def _simulate_stream(self, stop_event: threading.Event) -> None:
        """
        Simulate Reddit data collection for testing.
        
        Args:
            stop_event: Threading event to signal stop
        """
        logger.info("Running Reddit stream in simulation mode")
        logger.warning("Configure Reddit API keys for live data")
        
        # Sample posts for simulation
        sample_posts = [
            ("Just bought more Bitcoin. This is the way! 💎🙌", "cryptocurrency", 150, 2),
            ("Why is everything dumping? Should I panic sell?", "CryptoMarkets", 45, 0),
            ("ETH 2.0 is revolutionary. Long term bullish.", "ethereum", 200, 5),
            ("Unpopular opinion: Most altcoins will go to zero", "cryptocurrency", 80, 1),
            ("My portfolio is up 300% this year! 🚀", "Bitcoin", 500, 10),
            ("Bear market confirmed. Time to short everything.", "CryptoMarkets", 30, 0),
            ("HODL strategy has never failed me", "Bitcoin", 120, 3),
            ("This correction is healthy for the market", "cryptocurrency", 95, 1),
        ]
        
        post_index = 0
        
        while self.running and not stop_event.is_set():
            try:
                # Generate simulated post
                text, subreddit, karma, awards = sample_posts[post_index % len(sample_posts)]
                
                post = RedditPost(
                    text=text,
                    author=f"user_{post_index % 10}",
                    timestamp=get_timestamp(),
                    karma=karma,
                    awards=awards,
                    subreddit=subreddit,
                    post_type='comment',
                    url=f"https://reddit.com/r/{subreddit}/comments/abc123"
                )
                
                # Create data packet
                packet = DataPacket(
                    data_type=DataType.REDDIT,
                    timestamp=post.timestamp,
                    source="reddit_stream",
                    data=post,
                    metadata={
                        'karma': post.karma,
                        'awards': post.awards,
                        'subreddit': post.subreddit
                    }
                )
                
                # Put in processing queue
                thread_manager.put_data('raw_data', packet, block=False)
                
                logger.debug(f"Simulated Reddit post from r/{subreddit}: {text[:50]}...")
                
                post_index += 1
                
                # Sleep to respect rate limit
                sleep_time = 60.0 / self.max_posts_per_minute
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in Reddit simulation: {e}")
                time.sleep(5)
        
        logger.info("Reddit stream stopped")
    
    def stop(self) -> None:
        """Stop the Reddit stream."""
        self.running = False
        logger.info("Stopping Reddit stream...")


def start_reddit_stream(stop_event: threading.Event) -> None:
    """
    Thread worker function for Reddit stream.
    
    Args:
        stop_event: Threading event to signal stop
    """
    stream = RedditStream(max_posts_per_minute=60)
    stream.start(stop_event)


if __name__ == "__main__":
    # Test the Reddit stream
    import signal
    
    stop_event = threading.Event()
    
    def signal_handler(sig, frame):
        logger.info("Received stop signal")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting Reddit stream test...")
    logger.info("Press Ctrl+C to stop")
    
    # Start in main thread for testing
    stream = RedditStream(max_posts_per_minute=10)
    stream.start(stop_event)
