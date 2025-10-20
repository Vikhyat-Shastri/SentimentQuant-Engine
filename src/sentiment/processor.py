"""
Sentiment Processor - Orchestrates the complete sentiment analysis pipeline.

This module coordinates:
1. Text preprocessing (cleaning, entity extraction)
2. Sentiment analysis (VADER + Fear/Greed ensemble)
3. Sentiment aggregation (multi-source, time-weighted)
4. Fear & Greed Index calculation

Processes data from raw_data queue → sentiment_queue
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

# Add project root to path for imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.processing.text_preprocessor import TextPreprocessor
from src.sentiment.sentiment_analyzer import EnsembleSentimentAnalyzer
from src.sentiment.aggregator import SentimentAggregator, SentimentDataPoint
from src.sentiment.fear_greed_index import FearGreedIndexCalculator
from src.utils import get_timestamp


@dataclass
class ProcessedSentiment:
    """
    Complete sentiment processing result.
    
    Attributes:
        source: Data source (twitter, reddit, news, market)
        asset: Asset ticker (BTC, ETH, etc.)
        original_text: Original text
        cleaned_text: Preprocessed text
        sentiment_score: Compound sentiment [-1 to 1]
        confidence: Confidence in sentiment [0-1]
        entities: Extracted entities (tickers, hashtags, mentions)
        timestamp: Processing timestamp
        metadata: Additional metadata
    """
    source: str
    asset: str
    original_text: str
    cleaned_text: str
    sentiment_score: float
    confidence: float
    entities: Dict[str, List[str]]
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class AggregatedResult:
    """
    Complete aggregated sentiment result.
    
    Attributes:
        asset: Asset ticker
        overall_sentiment: Overall sentiment score [-1 to 1]
        fear_greed_index: Fear/Greed index [0-100]
        fear_greed_label: Text label (Extreme Fear, Fear, etc.)
        volume: Number of mentions
        momentum: Sentiment momentum
        confidence: Overall confidence
        source_breakdown: Sentiment by source
        timestamp: Result timestamp
    """
    asset: str
    overall_sentiment: float
    fear_greed_index: float
    fear_greed_label: str
    volume: int
    momentum: float
    confidence: float
    source_breakdown: Dict[str, float]
    timestamp: float


class SentimentProcessor:
    """
    Orchestrates the complete sentiment analysis pipeline.
    
    Pipeline:
        Raw Data → Preprocessing → Sentiment Analysis → Aggregation → F&G Index
    """
    
    def __init__(self):
        """Initialize sentiment processor with all components."""
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = EnsembleSentimentAnalyzer()
        self.aggregator = SentimentAggregator()
        self.fgi_calculator = FearGreedIndexCalculator()
        
        # Statistics
        self.processed_count = 0
        self.error_count = 0
        self.start_time = get_timestamp()
        
        logger.info("SentimentProcessor initialized with full pipeline")
    
    def process_single(
        self,
        text: str,
        source: str,
        asset: str = "BTC",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ProcessedSentiment]:
        """
        Process a single text through the sentiment pipeline.
        
        Args:
            text: Raw text to process
            source: Data source (twitter, reddit, news)
            asset: Asset ticker (BTC, ETH, etc.)
            metadata: Additional metadata (followers, karma, etc.)
            
        Returns:
            ProcessedSentiment object or None if processing fails
        """
        try:
            timestamp = get_timestamp()
            
            # Step 1: Preprocess text
            logger.debug(f"Preprocessing text: {text[:50]}...")
            processed = self.preprocessor.preprocess(text)
            logger.debug(f"Preprocessor returned type: {type(processed)}")
            
            # Check if preprocessing returned valid result
            if not processed:
                logger.warning(f"Preprocessing returned None for: {text[:50]}...")
                return None
            
            # Check type
            if not hasattr(processed, 'cleaned'):
                logger.error(f"Preprocessor returned unexpected type: {type(processed)}, value: {str(processed)[:100]}")
                return None
            
            # Validate the processed text
            if not self.preprocessor.is_valid_text(processed):
                logger.warning(f"Invalid text after preprocessing: {text[:50]}...")
                return None
            
            # Step 2: Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze(processed.cleaned)
            
            # Step 3: Create processed result
            result = ProcessedSentiment(
                source=source,
                asset=asset,
                original_text=text,
                cleaned_text=processed.cleaned,
                sentiment_score=sentiment.compound,
                confidence=sentiment.confidence,
                entities=processed.entities,
                timestamp=timestamp,
                metadata=metadata or {}
            )
            
            self.processed_count += 1
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing text: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> List[ProcessedSentiment]:
        """
        Process a batch of items through sentiment pipeline.
        
        Args:
            items: List of dicts with keys: text, source, asset, metadata
            
        Returns:
            List of ProcessedSentiment objects
        """
        results = []
        
        for item in items:
            result = self.process_single(
                text=item.get('text', ''),
                source=item.get('source', 'unknown'),
                asset=item.get('asset', 'BTC'),
                metadata=item.get('metadata', {})
            )
            
            if result:
                results.append(result)
        
        logger.info(f"Processed batch: {len(results)}/{len(items)} successful")
        
        return results
    
    def add_to_aggregator(self, processed: ProcessedSentiment) -> None:
        """
        Add processed sentiment to aggregator.
        
        Args:
            processed: ProcessedSentiment object
        """
        # Calculate influence score from metadata
        from src.utils import calculate_influence_score
        
        influence = calculate_influence_score(
            followers=processed.metadata.get('followers', 0),
            karma=processed.metadata.get('karma', 0),
            verified=processed.metadata.get('verified', False)
        )
        
        # Create sentiment data point
        data_point = SentimentDataPoint(
            text=processed.original_text,
            sentiment_score=processed.sentiment_score,
            timestamp=processed.timestamp,
            source=processed.source,
            asset=processed.asset,
            metadata={
                **processed.metadata,
                'confidence': processed.confidence,
                'influence': influence,
                'entities': processed.entities
            }
        )
        
        # Add to aggregator
        self.aggregator.add_data_point(data_point)
        
        logger.debug(f"Added to aggregator: {processed.asset} ({processed.source}) = {processed.sentiment_score:.2f}")
    
    def get_aggregated_sentiment(
        self,
        asset: str = "BTC",
        include_fgi: bool = True,
        current_price: Optional[float] = None,
        current_volume: Optional[float] = None,
        social_mentions: Optional[int] = None
    ) -> Optional[AggregatedResult]:
        """
        Get aggregated sentiment for an asset.
        
        Args:
            asset: Asset ticker
            include_fgi: Whether to calculate Fear & Greed Index
            current_price: Current asset price (for FGI calculation)
            current_volume: Current trading volume (for FGI calculation)
            social_mentions: Number of social mentions (for FGI calculation)
            
        Returns:
            AggregatedResult object or None if no data
        """
        try:
            # Get aggregated sentiment
            agg = self.aggregator.aggregate(asset=asset)
            
            if agg.volume == 0:
                logger.warning(f"No data points for asset: {asset}")
                return None
            
            # Calculate Fear & Greed Index if requested
            fgi_index = 50.0
            fgi_label = "Neutral"
            
            if include_fgi and current_price and current_volume and social_mentions:
                fgi = self.fgi_calculator.calculate(
                    sentiment_score=agg.overall_score,
                    current_volume=current_volume,
                    current_price=current_price,
                    social_mentions=social_mentions,
                    btc_dominance=55.0  # Default BTC dominance
                )
                fgi_index = fgi.index
                fgi_label = fgi.label
            
            # Create result
            result = AggregatedResult(
                asset=asset,
                overall_sentiment=agg.overall_score,
                fear_greed_index=fgi_index,
                fear_greed_label=fgi_label,
                volume=agg.volume,
                momentum=agg.momentum,
                confidence=agg.confidence,
                source_breakdown=agg.breakdown,
                timestamp=agg.timestamp
            )
            
            logger.info(
                f"{asset}: Sentiment={result.overall_sentiment:.3f}, "
                f"F&G={result.fear_greed_index:.1f} ({result.fear_greed_label}), "
                f"Volume={result.volume}, Confidence={result.confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting aggregated sentiment: {e}", exc_info=True)
            return None
    
    def process_and_aggregate(
        self,
        text: str,
        source: str,
        asset: str = "BTC",
        metadata: Optional[Dict[str, Any]] = None,
        get_aggregated: bool = True,
        **fgi_kwargs
    ) -> Dict[str, Any]:
        """
        Complete pipeline: process text and optionally get aggregated result.
        
        Args:
            text: Raw text to process
            source: Data source
            asset: Asset ticker
            metadata: Additional metadata
            get_aggregated: Whether to return aggregated sentiment
            **fgi_kwargs: Arguments for Fear & Greed Index calculation
            
        Returns:
            Dict with processed sentiment and optional aggregated result
        """
        # Process single text
        processed = self.process_single(text, source, asset, metadata)
        
        if not processed:
            return {'success': False, 'error': 'Processing failed'}
        
        # Add to aggregator
        self.add_to_aggregator(processed)
        
        result = {
            'success': True,
            'processed': processed
        }
        
        # Get aggregated sentiment if requested
        if get_aggregated:
            aggregated = self.get_aggregated_sentiment(asset=asset, **fgi_kwargs)
            result['aggregated'] = aggregated
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dict with statistics
        """
        elapsed = get_timestamp() - self.start_time
        rate = self.processed_count / elapsed if elapsed > 0 else 0
        
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'success_rate': self.processed_count / (self.processed_count + self.error_count) if (self.processed_count + self.error_count) > 0 else 0,
            'processing_rate': rate,  # items per second
            'elapsed_time': elapsed,
            'aggregator_stats': {
                asset: len(points) 
                for asset, points in self.aggregator.data_points.items()
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.processed_count = 0
        self.error_count = 0
        self.start_time = get_timestamp()
        logger.info("Statistics reset")


if __name__ == "__main__":
    # Test the sentiment processor
    print("=" * 70)
    print("Testing Sentiment Processor")
    print("=" * 70)
    
    processor = SentimentProcessor()
    
    # Test cases
    test_data = [
        {
            'text': "Bitcoin is breaking out! 🚀 This could be the start of a massive bull run!",
            'source': 'twitter',
            'asset': 'BTC',
            'metadata': {'followers': 5000, 'verified': True}
        },
        {
            'text': "Market crash incoming! Everyone is panic selling! Get out now!",
            'source': 'twitter',
            'asset': 'BTC',
            'metadata': {'followers': 1000, 'verified': False}
        },
        {
            'text': "Major exchange announces Bitcoin ETF approval. Institutional adoption accelerating.",
            'source': 'news',
            'asset': 'BTC',
            'metadata': {'source_credibility': 'high'}
        },
        {
            'text': "Detailed technical analysis shows BTC holding support at $45k. Bullish divergence forming.",
            'source': 'reddit',
            'asset': 'BTC',
            'metadata': {'karma': 5000, 'awards': 3}
        }
    ]
    
    print("\nProcessing individual texts...")
    for i, item in enumerate(test_data, 1):
        print(f"\n--- Item {i} ---")
        # Encode text safely for Windows terminal
        safe_text = item['text'][:60].encode('ascii', 'ignore').decode('ascii')
        print(f"Text: {safe_text}...")
        
        result = processor.process_and_aggregate(
            text=item['text'],
            source=item['source'],
            asset=item['asset'],
            metadata=item['metadata'],
            get_aggregated=True,
            include_fgi=True,
            current_price=45000,
            current_volume=1000000,
            social_mentions=100
        )
        
        if result['success']:
            proc = result['processed']
            print(f"Sentiment: {proc.sentiment_score:.3f} (confidence: {proc.confidence:.2f})")
            print(f"Entities: Tickers={proc.entities['tickers']}, Hashtags={proc.entities['hashtags']}")
            
            if 'aggregated' in result and result['aggregated']:
                agg = result['aggregated']
                print(f"Aggregated: {agg.overall_sentiment:.3f}, F&G: {agg.fear_greed_index:.1f} ({agg.fear_greed_label})")
                print(f"Volume: {agg.volume}, Confidence: {agg.confidence:.2f}")
    
    # Show statistics
    print("\n" + "=" * 70)
    print("Processing Statistics")
    print("=" * 70)
    stats = processor.get_statistics()
    print(f"Processed: {stats['processed_count']}")
    print(f"Errors: {stats['error_count']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Processing Rate: {stats['processing_rate']:.1f} items/sec")
    print(f"Aggregator Data Points: {stats['aggregator_stats']}")
