"""
Sentiment aggregator that combines sentiment from multiple sources.
Applies source weights, time decay, and influence scoring.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from src.utils import (
    get_timestamp,
    exponential_decay,
    calculate_influence_score,
    weighted_average,
    normalize_score,
    config_manager
)


@dataclass
class SentimentDataPoint:
    """
    Individual sentiment data point from a source.
    
    Attributes:
        text: Original text
        sentiment_score: Compound sentiment [-1 to 1]
        timestamp: Unix timestamp
        source: Data source (twitter, reddit, news)
        asset: Asset ticker (BTC, ETH, etc.)
        metadata: Additional data (followers, karma, etc.)
    """
    text: str
    sentiment_score: float
    timestamp: float
    source: str
    asset: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class AggregatedSentiment:
    """
    Aggregated sentiment for an asset.
    
    Attributes:
        asset: Asset ticker
        overall_score: Overall sentiment [-1 to 1]
        fear_greed_score: Fear/Greed index [0-100]
        volume: Number of mentions
        momentum: Rate of change
        confidence: Confidence in the score [0-1]
        timestamp: When aggregated
        breakdown: Sentiment by source
    """
    asset: str
    overall_score: float
    fear_greed_score: float
    volume: int
    momentum: float
    confidence: float
    timestamp: float
    breakdown: Dict[str, float] = field(default_factory=dict)


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources with weighting and time decay.
    """
    
    def __init__(self, time_window_seconds: int = 3600):
        """
        Initialize sentiment aggregator.
        
        Args:
            time_window_seconds: How far back to consider data (default: 1 hour)
        """
        self.time_window = time_window_seconds
        
        # Storage for sentiment data points
        self.data_points: Dict[str, List[SentimentDataPoint]] = defaultdict(list)
        
        # Historical aggregated scores for momentum calculation
        self.historical_scores: Dict[str, List[tuple]] = defaultdict(list)
        
        # Load configuration
        self.source_weights = config_manager.get(
            'sentiment_config',
            'sentiment.source_weights',
            {
                'news_verified': 1.5,
                'twitter_verified': 1.2,
                'twitter_regular': 1.0,
                'reddit_high_karma': 1.1,
                'reddit_regular': 1.0
            }
        )
        
        self.time_decay_halflife = config_manager.get(
            'sentiment_config',
            'sentiment.aggregation.time_decay_halflife_hours',
            4
        ) * 3600  # Convert to seconds
        
        logger.info(f"SentimentAggregator initialized (window: {time_window_seconds}s)")
    
    def add_data_point(self, data_point: SentimentDataPoint) -> None:
        """
        Add a sentiment data point.
        
        Args:
            data_point: SentimentDataPoint to add
        """
        asset = data_point.asset or 'OVERALL'
        self.data_points[asset].append(data_point)
        
        # Clean old data points
        self._cleanup_old_data(asset)
    
    def _cleanup_old_data(self, asset: str) -> None:
        """
        Remove data points older than time window.
        
        Args:
            asset: Asset ticker
        """
        current_time = get_timestamp()
        cutoff_time = current_time - self.time_window
        
        self.data_points[asset] = [
            dp for dp in self.data_points[asset]
            if dp.timestamp >= cutoff_time
        ]
    
    def _get_source_weight(self, data_point: SentimentDataPoint) -> float:
        """
        Get weight for a data point based on source credibility.
        
        Args:
            data_point: SentimentDataPoint
            
        Returns:
            Weight multiplier
        """
        source = data_point.source
        metadata = data_point.metadata
        
        # Base weight from source type
        if source == 'twitter':
            if metadata.get('verified', False):
                base_weight = self.source_weights.get('twitter_verified', 1.2)
            else:
                base_weight = self.source_weights.get('twitter_regular', 1.0)
        elif source == 'reddit':
            karma = metadata.get('karma', 0)
            if karma > 1000:
                base_weight = self.source_weights.get('reddit_high_karma', 1.1)
            else:
                base_weight = self.source_weights.get('reddit_regular', 1.0)
        elif source == 'news':
            base_weight = self.source_weights.get('news_verified', 1.5)
        else:
            base_weight = 1.0
        
        # Apply influence scoring
        followers = metadata.get('followers', 0)
        karma = metadata.get('karma', 0)
        verified = metadata.get('verified', False)
        
        influence_multiplier = calculate_influence_score(
            followers=followers,
            karma=karma,
            verified=verified
        )
        
        return base_weight * influence_multiplier
    
    def _calculate_time_weight(self, data_point: SentimentDataPoint) -> float:
        """
        Calculate time decay weight for a data point.
        
        Args:
            data_point: SentimentDataPoint
            
        Returns:
            Time decay multiplier [0-1]
        """
        current_time = get_timestamp()
        age_seconds = current_time - data_point.timestamp
        
        return exponential_decay(
            value=1.0,
            age_seconds=age_seconds,
            half_life_seconds=self.time_decay_halflife
        )
    
    def aggregate(self, asset: Optional[str] = None) -> AggregatedSentiment:
        """
        Aggregate sentiment for an asset.
        
        Args:
            asset: Asset ticker (None for overall market)
            
        Returns:
            AggregatedSentiment object
        """
        asset = asset or 'OVERALL'
        
        # Get data points
        data_points = self.data_points.get(asset, [])
        
        if not data_points:
            return AggregatedSentiment(
                asset=asset,
                overall_score=0.0,
                fear_greed_score=50.0,
                volume=0,
                momentum=0.0,
                confidence=0.0,
                timestamp=get_timestamp(),
                breakdown={}
            )
        
        # Calculate weighted sentiment
        weighted_scores = []
        weights = []
        source_scores = defaultdict(list)
        
        for dp in data_points:
            # Get source weight
            source_weight = self._get_source_weight(dp)
            
            # Get time weight
            time_weight = self._calculate_time_weight(dp)
            
            # Combined weight
            total_weight = source_weight * time_weight
            
            # Add to lists
            weighted_scores.append(dp.sentiment_score * total_weight)
            weights.append(total_weight)
            
            # Track by source
            source_scores[dp.source].append(dp.sentiment_score)
        
        # Calculate overall score
        if sum(weights) > 0:
            overall_score = sum(weighted_scores) / sum(weights)
        else:
            overall_score = 0.0
        
        # Normalize to [-1, 1]
        overall_score = max(-1.0, min(1.0, overall_score))
        
        # Convert to Fear/Greed scale [0-100]
        fear_greed_score = normalize_score(
            overall_score,
            min_val=-1.0,
            max_val=1.0,
            target_min=0.0,
            target_max=100.0
        )
        
        # Calculate breakdown by source
        breakdown = {}
        for source, scores in source_scores.items():
            if scores:
                breakdown[source] = sum(scores) / len(scores)
        
        # Calculate confidence (based on volume and agreement)
        volume = len(data_points)
        confidence = min(1.0, volume / 50.0)  # Max confidence at 50+ data points
        
        # Calculate momentum (compare to previous aggregation)
        momentum = self._calculate_momentum(asset, overall_score)
        
        # Store for momentum calculation
        current_time = get_timestamp()
        self.historical_scores[asset].append((current_time, overall_score))
        
        # Keep only recent history (last hour)
        cutoff = current_time - 3600
        self.historical_scores[asset] = [
            (t, s) for t, s in self.historical_scores[asset]
            if t >= cutoff
        ]
        
        return AggregatedSentiment(
            asset=asset,
            overall_score=overall_score,
            fear_greed_score=fear_greed_score,
            volume=volume,
            momentum=momentum,
            confidence=confidence,
            timestamp=current_time,
            breakdown=breakdown
        )
    
    def _calculate_momentum(self, asset: str, current_score: float) -> float:
        """
        Calculate sentiment momentum (rate of change).
        
        Args:
            asset: Asset ticker
            current_score: Current sentiment score
            
        Returns:
            Momentum score
        """
        history = self.historical_scores.get(asset, [])
        
        if len(history) < 2:
            return 0.0
        
        # Get score from 15 minutes ago
        current_time = get_timestamp()
        lookback_time = current_time - 900  # 15 minutes
        
        # Find closest historical score
        past_score = None
        for timestamp, score in history:
            if timestamp <= lookback_time:
                past_score = score
        
        if past_score is None:
            return 0.0
        
        # Calculate momentum
        momentum = current_score - past_score
        
        return momentum
    
    def get_all_assets(self) -> List[str]:
        """
        Get list of all tracked assets.
        
        Returns:
            List of asset tickers
        """
        return list(self.data_points.keys())
    
    def aggregate_all(self) -> Dict[str, AggregatedSentiment]:
        """
        Aggregate sentiment for all tracked assets.
        
        Returns:
            Dictionary of asset -> AggregatedSentiment
        """
        results = {}
        for asset in self.get_all_assets():
            results[asset] = self.aggregate(asset)
        return results
    
    def get_market_sentiment(self) -> AggregatedSentiment:
        """
        Get overall market sentiment (all assets combined).
        
        Returns:
            AggregatedSentiment for entire market
        """
        return self.aggregate('OVERALL')


if __name__ == "__main__":
    # Test the aggregator
    print("=" * 70)
    print("Testing Sentiment Aggregator")
    print("=" * 70)
    
    aggregator = SentimentAggregator(time_window_seconds=3600)
    
    # Add some test data points
    current_time = get_timestamp()
    
    test_points = [
        SentimentDataPoint(
            text="Bitcoin to the moon!",
            sentiment_score=0.8,
            timestamp=current_time,
            source="twitter",
            asset="BTC",
            metadata={'followers': 10000, 'verified': False}
        ),
        SentimentDataPoint(
            text="Bearish on crypto",
            sentiment_score=-0.6,
            timestamp=current_time - 600,  # 10 min ago
            source="twitter",
            asset="BTC",
            metadata={'followers': 5000, 'verified': False}
        ),
        SentimentDataPoint(
            text="Bitcoin fundamentals strong",
            sentiment_score=0.5,
            timestamp=current_time - 1800,  # 30 min ago
            source="news",
            asset="BTC",
            metadata={}
        ),
        SentimentDataPoint(
            text="HODL!",
            sentiment_score=0.7,
            timestamp=current_time - 300,  # 5 min ago
            source="reddit",
            asset="BTC",
            metadata={'karma': 1500}
        ),
    ]
    
    # Add data points
    for point in test_points:
        aggregator.add_data_point(point)
    
    # Aggregate
    result = aggregator.aggregate('BTC')
    
    print(f"\nAggregated Sentiment for BTC:")
    print(f"  Overall Score: {result.overall_score:+.3f}")
    print(f"  Fear/Greed Index: {result.fear_greed_score:.1f}/100")
    print(f"  Volume: {result.volume} mentions")
    print(f"  Momentum: {result.momentum:+.3f}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"\nBreakdown by source:")
    for source, score in result.breakdown.items():
        print(f"  {source}: {score:+.3f}")
