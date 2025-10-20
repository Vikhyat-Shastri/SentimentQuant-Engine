"""Data ingestion package for sentiment analysis."""

from .twitter_stream import TwitterStream, start_twitter_stream
from .reddit_stream import RedditStream, start_reddit_stream
from .market_data import MarketDataFeed, start_market_feed

__all__ = [
    'TwitterStream',
    'start_twitter_stream',
    'RedditStream',
    'start_reddit_stream',
    'MarketDataFeed',
    'start_market_feed'
]
