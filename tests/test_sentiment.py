"""
Test sentiment analysis system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from loguru import logger

from src.sentiment.sentiment_analyzer import (
    VADERSentimentAnalyzer,
    FearGreedClassifier,
    EnsembleSentimentAnalyzer
)
from src.sentiment.aggregator import SentimentAggregator, SentimentDataPoint
from src.sentiment.fear_greed_index import FearGreedIndexCalculator


def test_vader_analyzer():
    """Test VADER sentiment analyzer."""
    print("\n" + "=" * 70)
    print("Testing VADER Sentiment Analyzer")
    print("=" * 70)
    
    analyzer = VADERSentimentAnalyzer()
    
    test_texts = [
        "Bitcoin is crashing! I'm panicking and selling everything!",
        "BTC to the moon! 🚀 This rally is amazing!",
        "Market is flat today, nothing much happening.",
        "Ethereum looks stable, holding support well."
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"Result: pos={result.positive:.2f}, neg={result.negative:.2f}, "
              f"compound={result.compound:.2f}, confidence={result.confidence:.2f}")


def test_fear_greed_classifier():
    """Test Fear & Greed classifier."""
    print("\n" + "=" * 70)
    print("Testing Fear & Greed Classifier")
    print("=" * 70)
    
    classifier = FearGreedClassifier()
    
    test_texts = [
        "Market crash incoming! Everyone is dumping! PANIC SELL!",
        "FOMO is real! To the moon! Buy now before it pumps! 🚀🚀🚀",
        "Bitcoin looks steady at $45k support level.",
        "Scary market conditions. Feeling bearish. Might sell."
    ]
    
    for text in test_texts:
        result = classifier.analyze(text)
        print(f"\nText: {text}")
        print(f"Fear/Greed: {result.compound:.2f} "
              f"(pos={result.positive:.2f}, neg={result.negative:.2f})")


def test_ensemble_analyzer():
    """Test ensemble sentiment analyzer."""
    print("\n" + "=" * 70)
    print("Testing Ensemble Sentiment Analyzer")
    print("=" * 70)
    
    analyzer = EnsembleSentimentAnalyzer()
    
    test_texts = [
        "Bitcoin is crashing hard! 📉 Everyone is panic selling!",
        "BTC to the moon! 🚀 This is the beginning of the bull run!",
        "Stable market today. BTC holding $45k nicely.",
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"Ensemble Score: compound={result.compound:.2f}, confidence={result.confidence:.2f}")


def test_sentiment_aggregator():
    """Test sentiment aggregator."""
    print("\n" + "=" * 70)
    print("Testing Sentiment Aggregator")
    print("=" * 70)
    
    aggregator = SentimentAggregator()
    
    # Simulate data from multiple sources
    from src.utils import get_timestamp
    from src.sentiment.aggregator import SentimentDataPoint
    
    current_time = get_timestamp()
    
    # Add some tweets (recent, high frequency)
    for i in range(5):
        dp = SentimentDataPoint(
            text=f"BTC is pumping! Tweet {i} 🚀",
            source="twitter",
            sentiment_score=0.7,
            timestamp=current_time - i * 60,  # 1 minute apart
            asset="BTC",
            metadata={'followers': 1000}
        )
        aggregator.add_data_point(dp)
    
    # Add some reddit posts (less frequent, more detailed)
    for i in range(3):
        dp = SentimentDataPoint(
            text=f"Detailed analysis showing bullish trend. Post {i}",
            source="reddit",
            sentiment_score=0.5,
            timestamp=current_time - i * 300,  # 5 minutes apart
            asset="BTC",
            metadata={'karma': 500}
        )
        aggregator.add_data_point(dp)
    
    # Add some news (highest weight)
    dp = SentimentDataPoint(
        text="Major institution announces Bitcoin investment",
        source="news",
        sentiment_score=0.8,
        timestamp=current_time - 600,  # 10 minutes ago
        asset="BTC"
    )
    aggregator.add_data_point(dp)
    
    # Add some older fearful tweets
    for i in range(3):
        dp = SentimentDataPoint(
            text=f"BTC crashing! Old tweet {i} 📉",
            source="twitter",
            sentiment_score=-0.6,
            timestamp=current_time - 4 * 3600 - i * 60,  # 4+ hours ago (should decay)
            asset="BTC",
            metadata={'followers': 500}
        )
        aggregator.add_data_point(dp)
    
    # Aggregate
    result = aggregator.aggregate(asset="BTC")
    
    print(f"\nAggregated Results:")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Fear/Greed Score: {result.fear_greed_score:.1f}/100")
    print(f"Volume: {result.volume} data points")
    print(f"Momentum: {result.momentum:.3f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nSource Breakdown:")
    for source, score in result.breakdown.items():
        print(f"  {source}: {score:.3f}")


def test_fear_greed_index():
    """Test Fear & Greed Index calculator."""
    print("\n" + "=" * 70)
    print("Testing Fear & Greed Index Calculator")
    print("=" * 70)
    
    calculator = FearGreedIndexCalculator()
    
    # Simulate some data collection first
    test_data = [
        # Initial neutral data
        (0.0, 1000000, 45000, 100, 55),
        (0.1, 1100000, 46000, 110, 54),
        (0.0, 1000000, 45500, 105, 55),
        # Extreme fear scenario
        (-0.7, 500000, 42000, 50, 65),
    ]
    
    for i, (sentiment, volume, price, mentions, dominance) in enumerate(test_data):
        result = calculator.calculate(
            sentiment_score=sentiment,
            current_volume=volume,
            current_price=price,
            social_mentions=mentions,
            btc_dominance=dominance
        )
        
        print(f"\n--- Data Point {i+1} ---")
        print(f"Index: {result.index:.1f}/100 - {result.label}")
        print(f"Components: sentiment={result.components['sentiment']:.1f}, "
              f"volume={result.components['volume_momentum']:.1f}, "
              f"volatility={result.components['volatility']:.1f}")


def test_full_pipeline():
    """Test full sentiment analysis pipeline."""
    print("\n" + "=" * 70)
    print("Testing Full Sentiment Analysis Pipeline")
    print("=" * 70)
    
    # Initialize all components
    analyzer = EnsembleSentimentAnalyzer()
    aggregator = SentimentAggregator()
    fgi_calculator = FearGreedIndexCalculator()
    
    from src.utils import get_timestamp
    
    # Simulate incoming data stream
    sample_data = [
        {
            'text': "Bitcoin breaking out! 🚀 This could be the start of the bull run!",
            'source': 'twitter',
            'influence': 1.5,
            'price': 46000,
            'volume': 1200000,
            'mentions': 150
        },
        {
            'text': "Major exchange announces support for Bitcoin ETF",
            'source': 'news',
            'influence': 2.0,
            'price': 46500,
            'volume': 1300000,
            'mentions': 200
        },
        {
            'text': "Market looks unstable. Might be a good time to take profits.",
            'source': 'reddit',
            'influence': 1.3,
            'price': 46200,
            'volume': 1250000,
            'mentions': 180
        },
        {
            'text': "FOMO is kicking in! Everyone wants to buy now! 💰💰💰",
            'source': 'twitter',
            'influence': 1.2,
            'price': 47000,
            'volume': 1400000,
            'mentions': 220
        },
    ]
    
    current_time = get_timestamp()
    
    print("\nProcessing data stream...")
    
    for i, data in enumerate(sample_data):
        print(f"\n--- Item {i+1} ---")
        print(f"Text: {data['text'][:60]}...")
        
        # Step 1: Analyze sentiment
        sentiment = analyzer.analyze(data['text'])
        print(f"Sentiment: {sentiment.compound:.2f} (confidence: {sentiment.confidence:.2f})")
        
        # Step 2: Add to aggregator
        dp = SentimentDataPoint(
            text=data['text'],
            source=data['source'],
            sentiment_score=sentiment.compound,
            timestamp=current_time + i * 60,  # 1 minute apart
            asset="BTC"
        )
        aggregator.add_data_point(dp)
        
        # Step 3: Get aggregated sentiment
        agg_result = aggregator.aggregate(asset="BTC")
        print(f"Aggregated Score: {agg_result.overall_score:.3f} (volume: {agg_result.volume})")
        
        # Step 4: Calculate Fear & Greed Index
        fgi_result = fgi_calculator.calculate(
            sentiment_score=agg_result.overall_score,
            current_volume=data['volume'],
            current_price=data['price'],
            social_mentions=data['mentions'],
            btc_dominance=55.0
        )
        print(f"Fear & Greed Index: {fgi_result.index:.1f}/100 - {fgi_result.label}")
    
    print("\n" + "=" * 70)
    print("Pipeline Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    logger.info("Starting sentiment analysis tests...")
    
    try:
        # Run individual component tests
        test_vader_analyzer()
        time.sleep(1)
        
        test_fear_greed_classifier()
        time.sleep(1)
        
        test_ensemble_analyzer()
        time.sleep(1)
        
        test_sentiment_aggregator()
        time.sleep(1)
        
        test_fear_greed_index()
        time.sleep(1)
        
        # Run full pipeline test
        test_full_pipeline()
        
        print("\n" + "=" * 70)
        print("✅ ALL SENTIMENT TESTS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
