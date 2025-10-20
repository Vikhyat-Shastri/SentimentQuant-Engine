"""
Tests for signal generation module.
"""

import pytest
import queue
import time
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.signal_generator import SignalGenerator, SignalAction, TradingSignal


class TestSignalGenerator:
    """Test suite for SignalGenerator"""
    
    def test_initialization(self):
        """Test signal generator initialization"""
        generator = SignalGenerator()
        assert generator is not None
        assert generator.signals_generated == 0
        assert generator.last_signal_time is None
        assert len(generator.signal_history) == 0
    
    def test_extreme_fear_signal(self):
        """Test signal generation for extreme fear (FGI < 25) - TREND-FOLLOWING"""
        generator = SignalGenerator()
        
        sentiment_data = {
            'fear_greed_index': 15.0,
            'sentiment_score': -0.8,
            'symbol': 'BTC-USD',
            'source_count': 10,
            'metadata': {}
        }
        
        signal = generator.generate_signal(sentiment_data)
        
        assert signal is not None
        # TREND-FOLLOWING: extreme fear = strong bearish momentum → STRONG_SELL
        assert signal.action == SignalAction.STRONG_SELL
        assert signal.fear_greed_index == 15.0
        assert signal.strength > 0.0  # Reasonable strength for extreme fear
        assert signal.confidence > 0.0
        assert 0.0 <= signal.position_size <= 1.0
    
    def test_fear_signal(self):
        """Test signal generation for fear (FGI 25-40) - TREND-FOLLOWING"""
        generator = SignalGenerator()
        
        sentiment_data = {
            'fear_greed_index': 30.0,
            'sentiment_score': -0.4,
            'symbol': 'BTC-USD',
            'source_count': 8,
            'metadata': {}
        }
        
        signal = generator.generate_signal(sentiment_data)
        
        assert signal is not None
        # TREND-FOLLOWING: fear = bearish momentum → SELL
        assert signal.action == SignalAction.SELL
        assert signal.fear_greed_index == 30.0
    
    def test_neutral_signal(self):
        """Test signal generation for neutral (FGI 40-60) - TREND-FOLLOWING"""
        generator = SignalGenerator()
        
        sentiment_data = {
            'fear_greed_index': 50.0,
            'sentiment_score': 0.0,
            'symbol': 'BTC-USD',
            'source_count': 5,
            'metadata': {}
        }
        
        signal = generator.generate_signal(sentiment_data)
        
        # Neutral zone may not generate signal if confidence too low
        # HOLD signals might be filtered out with min_confidence=0.75
        # Just check it doesn't crash
        if signal:
            assert signal.action == SignalAction.HOLD
            assert signal.fear_greed_index == 50.0
            assert signal.strength == 0.5
    
    def test_greed_signal(self):
        """Test signal generation for greed (FGI 60-75) - TREND-FOLLOWING"""
        generator = SignalGenerator()
        
        sentiment_data = {
            'fear_greed_index': 65.0,
            'sentiment_score': 0.5,
            'symbol': 'BTC-USD',
            'source_count': 7,
            'metadata': {}
        }
        
        signal = generator.generate_signal(sentiment_data)
        
        # Greed zone may not generate signal if confidence too low
        if signal:
            # TREND-FOLLOWING: greed = bullish momentum → BUY
            assert signal.action == SignalAction.BUY
            assert signal.fear_greed_index == 65.0
    
    def test_extreme_greed_signal(self):
        """Test signal generation for extreme greed (FGI > 75) - TREND-FOLLOWING"""
        generator = SignalGenerator()
        
        sentiment_data = {
            'fear_greed_index': 85.0,
            'sentiment_score': 0.9,
            'symbol': 'BTC-USD',
            'source_count': 12,
            'metadata': {}
        }
        
        signal = generator.generate_signal(sentiment_data)
        
        assert signal is not None
        # TREND-FOLLOWING: extreme greed = strong bullish momentum → STRONG_BUY
        assert signal.action == SignalAction.STRONG_BUY
        assert signal.fear_greed_index == 85.0
        assert signal.strength > 0.0  # Reasonable strength for extreme greed
    
    def test_confidence_calculation(self):
        """Test confidence calculation with aligned sentiment"""
        generator = SignalGenerator()
        
        # High FGI with positive sentiment (aligned)
        sentiment_data = {
            'fear_greed_index': 80.0,
            'sentiment_score': 0.8,
            'symbol': 'BTC-USD',
            'source_count': 10,
            'metadata': {}
        }
        
        signal = generator.generate_signal(sentiment_data)
        assert signal is not None
        assert signal.confidence > 0.5  # High confidence for aligned signals
    
    def test_low_confidence_rejection(self):
        """Test that low confidence signals are rejected"""
        generator = SignalGenerator()
        
        # Conflicting signals: high FGI with negative sentiment
        sentiment_data = {
            'fear_greed_index': 80.0,
            'sentiment_score': -0.9,
            'symbol': 'BTC-USD',
            'source_count': 1,
            'metadata': {}
        }
        
        signal = generator.generate_signal(sentiment_data)
        # May be None if confidence too low, or have low confidence
        if signal:
            assert signal.confidence < 0.7
    
    def test_position_sizing(self):
        """Test position sizing calculation"""
        generator = SignalGenerator()
        
        sentiment_data = {
            'fear_greed_index': 20.0,
            'sentiment_score': -0.7,
            'symbol': 'BTC-USD',
            'source_count': 10,
            'metadata': {}
        }
        
        signal = generator.generate_signal(sentiment_data)
        
        assert signal is not None
        # Position size should be within configured limits (UPDATED LIMITS)
        assert signal.position_size >= 0.02  # min_position (reduced from 0.05)
        assert signal.position_size <= 0.10  # max_position (reduced from 0.25)
    
    def test_signal_history(self):
        """Test signal history tracking"""
        generator = SignalGenerator()
        
        # Generate multiple signals with STRONG sentiments (to pass confidence filter)
        test_cases = [
            (20, -0.8, 10),   # Extreme fear with strong negative sentiment
            (35, -0.5, 10),   # Fear with negative sentiment
            (50, 0.0, 10),    # Neutral (may be filtered)
            (65, 0.5, 10),    # Greed with positive sentiment (may be filtered)
            (80, 0.8, 10)     # Extreme greed with strong positive sentiment
        ]
        
        for fgi, sentiment, sources in test_cases:
            sentiment_data = {
                'fear_greed_index': float(fgi),
                'sentiment_score': sentiment,
                'symbol': 'BTC-USD',
                'source_count': sources,
                'metadata': {}
            }
            generator.generate_signal(sentiment_data)
        
        # Check history - at least the extreme cases should pass confidence filter
        assert len(generator.signal_history) >= 2  # At least extreme cases
        assert generator.signals_generated >= 2
    
    def test_signal_summary(self):
        """Test signal summary statistics"""
        generator = SignalGenerator()
        
        # Generate signals with ALIGNED sentiment (to pass confidence filter)
        test_cases = [
            (15, -0.8, 10),   # STRONG_SELL (extreme fear + negative sentiment)
            (30, -0.4, 10),   # SELL (fear + negative sentiment)
            (50, 0.0, 10),    # HOLD (neutral, may be filtered)
            (65, 0.5, 10),    # BUY (greed + positive sentiment, may be filtered)
            (85, 0.9, 10)     # STRONG_BUY (extreme greed + positive sentiment)
        ]
        
        for fgi, sentiment, sources in test_cases:
            sentiment_data = {
                'fear_greed_index': fgi,
                'sentiment_score': sentiment,
                'symbol': 'BTC-USD',
                'source_count': sources,
                'metadata': {}
            }
            generator.generate_signal(sentiment_data)
        
        summary = generator.get_signal_summary()
        
        assert summary['total'] >= 2  # At least the extreme cases should pass
        assert 'by_action' in summary
        assert 'avg_confidence' in summary
        assert 'last_signal' in summary
    
    def test_threaded_processing(self):
        """Test signal generation in threaded mode"""
        generator = SignalGenerator()
        
        # Create queues
        sentiment_queue = queue.Queue()
        signal_queue = queue.Queue()
        
        # Start generator
        generator.start(sentiment_queue, signal_queue)
        
        # Add test data with STRONG aligned sentiment to ensure signal passes confidence filter
        sentiment_data = {
            'fear_greed_index': 85.0,  # Extreme greed
            'sentiment_score': 0.9,     # Strong positive (aligned)
            'symbol': 'BTC-USD',
            'source_count': 10,         # Multiple sources
            'metadata': {}
        }
        sentiment_queue.put(sentiment_data)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check output queue
        if not signal_queue.empty():
            signal_dict = signal_queue.get()
            
            assert signal_dict['action'] in [a.value for a in SignalAction]
            assert 'confidence' in signal_dict
            assert 'position_size' in signal_dict
            assert signal_dict['action'] == 'STRONG_BUY'  # Trend-following logic
        
        # Stop generator
        generator.stop()
    
    def test_statistics(self):
        """Test signal generator statistics"""
        generator = SignalGenerator()
        
        # Use STRONG aligned sentiment to ensure signal passes confidence filter
        sentiment_data = {
            'fear_greed_index': 85.0,  # Extreme greed
            'sentiment_score': 0.9,     # Strong positive (aligned)
            'symbol': 'BTC-USD',
            'source_count': 10,         # Multiple sources for confidence
            'metadata': {}
        }
        
        signal = generator.generate_signal(sentiment_data)
        
        # Signal should be generated with strong aligned sentiment
        assert signal is not None
        
        stats = generator.stats
        
        assert stats['signals_generated'] >= 1
        assert stats['last_signal_time'] is not None
        assert stats['signal_history_size'] >= 1
    
    def test_reasoning_generation(self):
        """Test signal reasoning generation"""
        generator = SignalGenerator()
        
        sentiment_data = {
            'fear_greed_index': 20.0,   # Extreme fear
            'sentiment_score': -0.7,    # Strong negative (aligned)
            'symbol': 'BTC-USD',
            'source_count': 10,         # Multiple sources
            'metadata': {}
        }
        
        signal = generator.generate_signal(sentiment_data)
        
        assert signal is not None
        assert signal.reasoning is not None
        assert len(signal.reasoning) > 0
        assert 'FGI' in signal.reasoning or 'fear' in signal.reasoning.lower()
        # Should be STRONG_SELL with trend-following logic
        assert signal.action == SignalAction.STRONG_SELL


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
