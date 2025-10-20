"""
Twitter data ingestion supporting both simulation and live Twitter API v2.
Uses tweepy for real-time filtered streams when API credentials are available.
"""

import time
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

# Twitter API v2 support
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    logger.warning("tweepy not installed. Twitter API features disabled. Install with: pip install tweepy")

from src.utils import (
    thread_manager,
    DataPacket,
    DataType,
    get_timestamp,
    handle_errors_gracefully,
    config_manager
)


@dataclass
class Tweet:
    """Container for tweet data."""
    text: str
    username: str
    timestamp: float
    followers: int
    retweets: int
    likes: int
    verified: bool
    url: str


class TwitterStreamListener(tweepy.StreamingClient):
    """
    Custom Twitter API v2 streaming client.
    Handles real-time tweet data from Twitter filtered stream.
    """
    
    def __init__(self, bearer_token: str, max_tweets_per_minute: int = 100, **kwargs):
        """
        Initialize streaming client.
        
        Args:
            bearer_token: Twitter API v2 Bearer Token
            max_tweets_per_minute: Rate limit for processing
        """
        super().__init__(bearer_token, **kwargs)
        self.max_tweets_per_minute = max_tweets_per_minute
        self.tweet_count = 0
        self.start_time = time.time()
        logger.info("TwitterStreamListener initialized")
    
    def on_tweet(self, tweet):
        """
        Called when a new tweet is received from the stream.
        
        Args:
            tweet: tweepy.Tweet object
        """
        try:
            # Extract tweet data
            text = tweet.text
            tweet_id = tweet.id
            author_id = tweet.author_id if hasattr(tweet, 'author_id') else "unknown"
            created_at = tweet.created_at if hasattr(tweet, 'created_at') else datetime.now()
            
            # Get public metrics if available
            public_metrics = tweet.public_metrics if hasattr(tweet, 'public_metrics') else {}
            retweets = public_metrics.get('retweet_count', 0)
            likes = public_metrics.get('like_count', 0)
            replies = public_metrics.get('reply_count', 0)
            
            # Create Tweet object
            tweet_obj = Tweet(
                text=text,
                username=f"user_{author_id}",  # Username would need expansion to get actual username
                timestamp=created_at.timestamp() if isinstance(created_at, datetime) else time.time(),
                followers=0,  # Would need user expansion to get follower count
                retweets=retweets,
                likes=likes,
                verified=False,  # Would need user expansion
                url=f"https://twitter.com/user/status/{tweet_id}"
            )
            
            # Create data packet
            packet = DataPacket(
                data_type=DataType.TWITTER,
                timestamp=tweet_obj.timestamp,
                source="twitter_api_v2",
                data=tweet_obj,
                metadata={
                    'tweet_id': tweet_id,
                    'author_id': author_id,
                    'engagement': retweets + likes + replies,
                    'retweets': retweets,
                    'likes': likes,
                    'replies': replies
                }
            )
            
            # Put in processing queue
            thread_manager.put_data('raw_data', packet, block=False)
            
            self.tweet_count += 1
            
            # Log rate
            elapsed = time.time() - self.start_time
            if elapsed > 60:
                rate = self.tweet_count / (elapsed / 60)
                logger.info(f"Twitter stream rate: {rate:.1f} tweets/min")
                self.tweet_count = 0
                self.start_time = time.time()
            
            logger.debug(f"Received tweet {tweet_id}: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
    
    def on_errors(self, errors):
        """
        Called when an error occurs in the stream.
        
        Args:
            errors: Error information
        """
        logger.error(f"Twitter stream error: {errors}")
        return True  # Continue streaming
    
    def on_connection_error(self):
        """Called when a connection error occurs."""
        logger.error("Twitter stream connection error")
        return True  # Attempt to reconnect


class TwitterStream:
    """
    Twitter data stream for crypto sentiment analysis.
    Supports both simulation mode and live Twitter API v2 streaming.
    """
    
    def __init__(
        self,
        keywords: List[str] = None,
        max_tweets_per_minute: int = 100,
        languages: List[str] = None,
        mode: str = "simulation"
    ):
        """
        Initialize Twitter stream.
        
        Args:
            keywords: List of keywords to search for
            max_tweets_per_minute: Rate limit for processing
            languages: Languages to filter (e.g., ['en'])
            mode: "simulation" or "live" (requires Twitter API credentials)
        """
        self.keywords = keywords or [
            'bitcoin', 'BTC', '$BTC',
            'ethereum', 'ETH', '$ETH',
            'crypto', 'cryptocurrency',
            'altcoin', 'bullish', 'bearish'
        ]
        
        self.max_tweets_per_minute = max_tweets_per_minute
        self.languages = languages or ['en']
        self.mode = mode
        self.running = False
        self.stream_client: Optional[TwitterStreamListener] = None
        
        # Try to load Twitter API credentials
        self.bearer_token = None
        if mode == "live":
            try:
                api_config = config_manager.get_section('api_keys', 'twitter')
                self.bearer_token = api_config.get('bearer_token')
                if not self.bearer_token:
                    logger.warning("Twitter bearer_token not found in config. Falling back to simulation mode.")
                    self.mode = "simulation"
            except Exception as e:
                logger.warning(f"Could not load Twitter API credentials: {e}. Using simulation mode.")
                self.mode = "simulation"
        
        logger.info(f"TwitterStream initialized in {self.mode} mode with keywords: {self.keywords}")
    
    def start(self, stop_event: threading.Event) -> None:
        """
        Start collecting tweets in a loop.
        
        Args:
            stop_event: Threading event to signal stop
        """
        self.running = True
        
        if self.mode == "live" and TWEEPY_AVAILABLE and self.bearer_token:
            logger.info("Starting Twitter stream in LIVE mode (Twitter API v2)...")
            self._start_live_stream(stop_event)
        else:
            logger.info("Starting Twitter stream in SIMULATION mode...")
            logger.warning("To use live mode: Set mode='live' and add Twitter API bearer_token to config/api_keys.yaml")
            self._simulate_stream(stop_event)
    
    def _start_live_stream(self, stop_event: threading.Event) -> None:
        """
        Start live Twitter API v2 filtered stream.
        
        Args:
            stop_event: Threading event to signal stop
        """
        try:
            # Create streaming client
            self.stream_client = TwitterStreamListener(
                bearer_token=self.bearer_token,
                max_tweets_per_minute=self.max_tweets_per_minute
            )
            
            # Delete existing rules (if any)
            try:
                existing_rules = self.stream_client.get_rules()
                if existing_rules.data:
                    rule_ids = [rule.id for rule in existing_rules.data]
                    self.stream_client.delete_rules(rule_ids)
                    logger.info(f"Deleted {len(rule_ids)} existing stream rules")
            except Exception as e:
                logger.warning(f"Could not delete existing rules: {e}")
            
            # Add new rules for keywords
            rules = []
            for keyword in self.keywords:
                # Twitter API v2 rule format
                rule_value = f"{keyword} lang:en -is:retweet"
                rules.append(tweepy.StreamRule(rule_value))
            
            if rules:
                self.stream_client.add_rules(rules)
                logger.info(f"Added {len(rules)} stream rules for keywords")
            
            # Start streaming with expansions and fields
            logger.success("✅ Twitter API v2 live stream starting...")
            logger.info(f"Filtering for keywords: {', '.join(self.keywords)}")
            
            # Run stream in separate thread
            stream_thread = threading.Thread(
                target=self._run_stream,
                args=(stop_event,),
                daemon=True,
                name="TwitterAPIStream"
            )
            stream_thread.start()
            
            # Wait for stop event
            while self.running and not stop_event.is_set():
                time.sleep(1)
            
            # Disconnect stream
            if self.stream_client:
                self.stream_client.disconnect()
                logger.info("Twitter API stream disconnected")
            
        except Exception as e:
            logger.exception(f"Error in live Twitter stream: {e}")
            logger.warning("Falling back to simulation mode")
            self._simulate_stream(stop_event)
    
    def _run_stream(self, stop_event: threading.Event) -> None:
        """
        Run the Twitter stream (called in separate thread).
        
        Args:
            stop_event: Threading event to signal stop
        """
        try:
            self.stream_client.filter(
                tweet_fields=['created_at', 'public_metrics', 'author_id'],
                expansions=['author_id'],
                user_fields=['username', 'verified', 'public_metrics'],
                threaded=False  # Run in current thread
            )
        except Exception as e:
            logger.error(f"Stream error: {e}")
            if not stop_event.is_set():
                logger.info("Attempting to reconnect in 10 seconds...")
                time.sleep(10)
                if self.running and not stop_event.is_set():
                    self._run_stream(stop_event)

    
    def _simulate_stream(self, stop_event: threading.Event) -> None:
        """
        Simulate tweet collection for testing.
        Replace this with actual scraping in production.
        
        Args:
            stop_event: Threading event to signal stop
        """
        logger.info("Running Twitter stream in SIMULATION mode")
        logger.info("To use live Twitter data, set mode='live' and configure Twitter API credentials")
        
        # Sample tweets for simulation
        sample_tweets = [
            ("Bitcoin is going to the moon! 🚀 $BTC", "crypto_bull", 10000, True, 0.8),
            ("Bearish on crypto right now. Expecting a dump. $BTC", "trader_joe", 5000, False, -0.6),
            ("Just bought more ETH! HODL! 💎", "eth_maxi", 15000, False, 0.9),
            ("This market is scary. Time to sell? $BTC $ETH", "worried_investor", 2000, False, -0.4),
            ("Bitcoin fundamentals are stronger than ever!", "btc_believer", 50000, True, 0.7),
            ("Massive sell-off incoming! Get out now! 📉", "doom_sayer", 8000, False, -0.9),
            ("Accumulation phase. Smart money is buying.", "whale_watcher", 30000, True, 0.6),
            ("FOMO is real! Everyone is buying now 🚀", "fomo_trader", 3000, False, 0.5),
        ]
        
        tweet_index = 0
        
        while self.running and not stop_event.is_set():
            try:
                # Generate simulated tweet
                text, username, followers, verified, _ = sample_tweets[tweet_index % len(sample_tweets)]
                
                tweet = Tweet(
                    text=text,
                    username=username,
                    timestamp=get_timestamp(),
                    followers=followers,
                    retweets=10,
                    likes=50,
                    verified=verified,
                    url=f"https://twitter.com/{username}/status/123456"
                )
                
                # Create data packet
                packet = DataPacket(
                    data_type=DataType.TWITTER,
                    timestamp=tweet.timestamp,
                    source="twitter_simulation",
                    data=tweet,
                    metadata={
                        'followers': tweet.followers,
                        'verified': tweet.verified,
                        'engagement': tweet.retweets + tweet.likes
                    }
                )
                
                # Put in processing queue
                thread_manager.put_data('raw_data', packet, block=False)
                
                logger.debug(f"Simulated tweet from @{username}: {text[:50]}...")
                
                tweet_index += 1
                
                # Sleep to respect rate limit
                sleep_time = 60.0 / self.max_tweets_per_minute
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in Twitter simulation: {e}")
                time.sleep(5)  # Wait before retrying
        
        logger.info("Twitter stream stopped")
    
    @handle_errors_gracefully(default_return=None)
    def search_tweets(self, query: str, max_results: int = 100) -> List[Tweet]:
        """
        Search for tweets matching query using Twitter API v2.
        
        Args:
            query: Search query
            max_results: Maximum number of tweets to return (10-100 for free tier)
            
        Returns:
            List of Tweet objects
        """
        if self.mode != "live" or not TWEEPY_AVAILABLE or not self.bearer_token:
            logger.warning("Search tweets requires live mode with Twitter API credentials")
            return []
        
        try:
            # Create API client
            client = tweepy.Client(bearer_token=self.bearer_token)
            
            # Search recent tweets
            response = client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),  # API limit
                tweet_fields=['created_at', 'public_metrics', 'author_id'],
                expansions=['author_id'],
                user_fields=['username', 'verified', 'public_metrics']
            )
            
            tweets = []
            if response.data:
                # Build user lookup dict
                users = {user.id: user for user in response.includes.get('users', [])} if response.includes else {}
                
                for tweet in response.data:
                    author_id = tweet.author_id
                    user = users.get(author_id)
                    
                    tweet_obj = Tweet(
                        text=tweet.text,
                        username=user.username if user else f"user_{author_id}",
                        timestamp=tweet.created_at.timestamp(),
                        followers=user.public_metrics.get('followers_count', 0) if user and hasattr(user, 'public_metrics') else 0,
                        retweets=tweet.public_metrics.get('retweet_count', 0) if hasattr(tweet, 'public_metrics') else 0,
                        likes=tweet.public_metrics.get('like_count', 0) if hasattr(tweet, 'public_metrics') else 0,
                        verified=user.verified if user and hasattr(user, 'verified') else False,
                        url=f"https://twitter.com/{user.username}/status/{tweet.id}" if user else f"https://twitter.com/i/status/{tweet.id}"
                    )
                    tweets.append(tweet_obj)
            
            logger.info(f"Found {len(tweets)} tweets for query: {query}")
            return tweets
            
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []
    
    def stop(self) -> None:
        """Stop the Twitter stream."""
        self.running = False
        
        if self.stream_client:
            try:
                self.stream_client.disconnect()
                logger.info("Disconnected Twitter API stream")
            except Exception as e:
                logger.error(f"Error disconnecting stream: {e}")
        
        logger.info("Twitter stream stopped")


def start_twitter_stream(stop_event: threading.Event, mode: str = "simulation") -> None:
    """
    Thread worker function for Twitter stream.
    
    Args:
        stop_event: Threading event to signal stop
        mode: "simulation" or "live"
    """
    stream = TwitterStream(max_tweets_per_minute=60, mode=mode)
    stream.start(stop_event)


if __name__ == "__main__":
    # Test the Twitter stream
    import signal
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Twitter Stream')
    parser.add_argument('--mode', choices=['simulation', 'live'], default='simulation',
                       help='Stream mode: simulation or live (requires API credentials)')
    args = parser.parse_args()
    
    stop_event = threading.Event()
    
    def signal_handler(sig, frame):
        logger.info("Received stop signal")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info(f"Starting Twitter stream test in {args.mode} mode...")
    logger.info("Press Ctrl+C to stop")
    
    # Start in main thread for testing
    stream = TwitterStream(max_tweets_per_minute=10, mode=args.mode)
    stream.start(stop_event)

