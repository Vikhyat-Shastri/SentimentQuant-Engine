"""
Behavioral Bias Detection

Identifies psychological biases in market sentiment:
- Herding behavior (following the crowd)
- FOMO (Fear of Missing Out)
- FUD (Fear, Uncertainty, Doubt)
- Confirmation bias
- Recency bias
- Loss aversion patterns
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class BiasSignal:
    """Detected behavioral bias"""
    bias_type: str
    timestamp: datetime
    strength: float  # 0-1
    description: str
    indicators: Dict[str, float]


@dataclass
class BiasAnalysisResult:
    """Complete bias analysis"""
    symbol: str
    period_start: datetime
    period_end: datetime
    detected_biases: List[BiasSignal]
    herding_score: float
    fomo_score: float
    fud_score: float
    confirmation_bias_score: float
    overall_bias_level: float


class BehavioralBiasDetector:
    """
    Detects behavioral biases in market sentiment
    
    Features:
    - Herding behavior detection (momentum + volume + sentiment alignment)
    - FOMO detection (rapid price increase + positive sentiment surge)
    - FUD detection (negative sentiment spike + volatility)
    - Confirmation bias (sentiment vs reality divergence)
    """
    
    def __init__(
        self,
        herding_threshold: float = 0.7,
        fomo_threshold: float = 0.75,
        fud_threshold: float = 0.75
    ):
        """
        Initialize bias detector
        
        Args:
            herding_threshold: Threshold for herding detection (0-1)
            fomo_threshold: Threshold for FOMO detection (0-1)
            fud_threshold: Threshold for FUD detection (0-1)
        """
        self.herding_threshold = herding_threshold
        self.fomo_threshold = fomo_threshold
        self.fud_threshold = fud_threshold
        
        logger.info("BehavioralBiasDetector initialized")
    
    def analyze(
        self,
        price_data: pd.DataFrame,
        sentiment_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> BiasAnalysisResult:
        """
        Analyze behavioral biases
        
        Args:
            price_data: DataFrame with 'timestamp' and 'close' columns
            sentiment_data: DataFrame with 'timestamp' and 'sentiment' columns
            volume_data: Optional DataFrame with 'timestamp' and 'volume' columns
        
        Returns:
            BiasAnalysisResult with detected biases
        """
        # Merge data
        data = self._merge_data(price_data, sentiment_data, volume_data)
        
        if len(data) < 10:
            logger.warning("Insufficient data for bias analysis")
            return self._empty_result(price_data)
        
        # Detect each type of bias
        detected_biases = []
        
        # 1. Herding behavior
        herding_signals = self._detect_herding(data)
        detected_biases.extend(herding_signals)
        herding_score = np.mean([s.strength for s in herding_signals]) if herding_signals else 0.0
        
        # 2. FOMO
        fomo_signals = self._detect_fomo(data)
        detected_biases.extend(fomo_signals)
        fomo_score = np.mean([s.strength for s in fomo_signals]) if fomo_signals else 0.0
        
        # 3. FUD
        fud_signals = self._detect_fud(data)
        detected_biases.extend(fud_signals)
        fud_score = np.mean([s.strength for s in fud_signals]) if fud_signals else 0.0
        
        # 4. Confirmation bias
        conf_signals = self._detect_confirmation_bias(data)
        detected_biases.extend(conf_signals)
        conf_score = np.mean([s.strength for s in conf_signals]) if conf_signals else 0.0
        
        # Overall bias level
        overall_bias = np.mean([herding_score, fomo_score, fud_score, conf_score])
        
        # Sort by timestamp
        detected_biases.sort(key=lambda x: x.timestamp)
        
        return BiasAnalysisResult(
            symbol=price_data.get('symbol', ['UNKNOWN'])[0] if 'symbol' in price_data else 'UNKNOWN',
            period_start=data['timestamp'].min(),
            period_end=data['timestamp'].max(),
            detected_biases=detected_biases,
            herding_score=herding_score,
            fomo_score=fomo_score,
            fud_score=fud_score,
            confirmation_bias_score=conf_score,
            overall_bias_level=overall_bias
        )
    
    def _merge_data(
        self,
        price_data: pd.DataFrame,
        sentiment_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge price, sentiment, and volume data"""
        # Ensure timestamp columns
        if 'timestamp' not in price_data.columns:
            price_data = price_data.copy()
            price_data['timestamp'] = price_data.index
        
        if 'timestamp' not in sentiment_data.columns:
            sentiment_data = sentiment_data.copy()
            sentiment_data['timestamp'] = sentiment_data.index
        
        # Merge on timestamp
        data = pd.merge(
            price_data[['timestamp', 'close']],
            sentiment_data[['timestamp', 'sentiment']],
            on='timestamp',
            how='inner'
        )
        
        if volume_data is not None and 'volume' in volume_data.columns:
            if 'timestamp' not in volume_data.columns:
                volume_data = volume_data.copy()
                volume_data['timestamp'] = volume_data.index
            
            data = pd.merge(
                data,
                volume_data[['timestamp', 'volume']],
                on='timestamp',
                how='left'
            )
        else:
            data['volume'] = 0
        
        return data.sort_values('timestamp').reset_index(drop=True)
    
    def _detect_herding(self, data: pd.DataFrame) -> List[BiasSignal]:
        """
        Detect herding behavior
        
        Herding occurs when:
        - Price momentum is strong (many moving in same direction)
        - Volume is high (many participants)
        - Sentiment is aligned (everyone agrees)
        """
        signals = []
        
        # Calculate indicators
        data['returns'] = data['close'].pct_change()
        data['momentum'] = data['returns'].rolling(window=5).sum()
        
        # Volume ratio
        if 'volume' in data.columns and data['volume'].sum() > 0:
            avg_volume = data['volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['volume'] / avg_volume.replace(0, 1)
        else:
            data['volume_ratio'] = 1.0
        
        # Sentiment alignment (low variance = high agreement)
        data['sentiment_std'] = data['sentiment'].rolling(window=10).std()
        
        # Detect herding moments
        for i in range(10, len(data)):
            momentum = abs(data.loc[i, 'momentum'])
            volume_ratio = data.loc[i, 'volume_ratio']
            sentiment_std = data.loc[i, 'sentiment_std']
            
            # High momentum + high volume + low sentiment variance = herding
            if momentum > 0.05 and volume_ratio > 1.5 and sentiment_std < 0.2:
                strength = min((momentum * volume_ratio) / (sentiment_std + 0.1), 1.0)
                
                if strength > self.herding_threshold:
                    signals.append(BiasSignal(
                        bias_type='HERDING',
                        timestamp=data.loc[i, 'timestamp'],
                        strength=strength,
                        description='Strong herding behavior detected - crowd following same direction',
                        indicators={
                            'momentum': momentum,
                            'volume_ratio': volume_ratio,
                            'sentiment_variance': sentiment_std
                        }
                    ))
        
        return signals
    
    def _detect_fomo(self, data: pd.DataFrame) -> List[BiasSignal]:
        """
        Detect FOMO (Fear of Missing Out)
        
        FOMO occurs when:
        - Rapid price increase
        - Sudden surge in positive sentiment
        - Accelerating momentum
        """
        signals = []
        
        # Calculate indicators
        data['returns'] = data['close'].pct_change()
        data['returns_5d'] = data['returns'].rolling(window=5).sum()
        data['sentiment_change'] = data['sentiment'].diff()
        data['acceleration'] = data['returns'].diff()
        
        for i in range(5, len(data)):
            returns_5d = data.loc[i, 'returns_5d']
            sentiment = data.loc[i, 'sentiment']
            sentiment_change = data.loc[i, 'sentiment_change']
            acceleration = data.loc[i, 'acceleration']
            
            # Rapid price increase + positive sentiment surge + acceleration
            if (returns_5d > 0.1 and  # 10% gain in 5 periods
                sentiment > 0.5 and  # Strong positive sentiment
                sentiment_change > 0.3 and  # Sudden sentiment increase
                acceleration > 0):  # Accelerating
                
                strength = min(
                    (returns_5d * sentiment * sentiment_change * 2),
                    1.0
                )
                
                if strength > self.fomo_threshold:
                    signals.append(BiasSignal(
                        bias_type='FOMO',
                        timestamp=data.loc[i, 'timestamp'],
                        strength=strength,
                        description='FOMO detected - rapid price rise with sentiment surge',
                        indicators={
                            'returns_5d': returns_5d,
                            'sentiment': sentiment,
                            'sentiment_change': sentiment_change,
                            'acceleration': acceleration
                        }
                    ))
        
        return signals
    
    def _detect_fud(self, data: pd.DataFrame) -> List[BiasSignal]:
        """
        Detect FUD (Fear, Uncertainty, Doubt)
        
        FUD occurs when:
        - Sudden negative sentiment spike
        - Increased volatility
        - Negative price action
        """
        signals = []
        
        # Calculate indicators
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=10).std()
        data['sentiment_change'] = data['sentiment'].diff()
        
        for i in range(10, len(data)):
            sentiment = data.loc[i, 'sentiment']
            sentiment_change = data.loc[i, 'sentiment_change']
            volatility = data.loc[i, 'volatility']
            returns = data.loc[i, 'returns']
            
            # Negative sentiment spike + high volatility + price drop
            if (sentiment < -0.5 and  # Strong negative sentiment
                sentiment_change < -0.3 and  # Sudden sentiment drop
                volatility > data['volatility'].median() * 1.5 and  # High volatility
                returns < 0):  # Price drop
                
                strength = min(
                    (abs(sentiment) * abs(sentiment_change) * volatility * 10),
                    1.0
                )
                
                if strength > self.fud_threshold:
                    signals.append(BiasSignal(
                        bias_type='FUD',
                        timestamp=data.loc[i, 'timestamp'],
                        strength=strength,
                        description='FUD detected - negative sentiment spike with volatility',
                        indicators={
                            'sentiment': sentiment,
                            'sentiment_change': sentiment_change,
                            'volatility': volatility,
                            'returns': returns
                        }
                    ))
        
        return signals
    
    def _detect_confirmation_bias(self, data: pd.DataFrame) -> List[BiasSignal]:
        """
        Detect confirmation bias
        
        Confirmation bias occurs when:
        - Sentiment remains positive despite price drops (or vice versa)
        - Sentiment-reality divergence
        """
        signals = []
        
        # Calculate indicators
        data['returns'] = data['close'].pct_change()
        data['returns_10d'] = data['returns'].rolling(window=10).sum()
        data['sentiment_10d'] = data['sentiment'].rolling(window=10).mean()
        
        for i in range(10, len(data)):
            returns_10d = data.loc[i, 'returns_10d']
            sentiment_10d = data.loc[i, 'sentiment_10d']
            
            # Sentiment-price divergence
            # Case 1: Price dropping but sentiment still positive
            if returns_10d < -0.05 and sentiment_10d > 0.3:
                strength = min(abs(returns_10d) * sentiment_10d * 5, 1.0)
                
                signals.append(BiasSignal(
                    bias_type='CONFIRMATION_BIAS',
                    timestamp=data.loc[i, 'timestamp'],
                    strength=strength,
                    description='Confirmation bias - positive sentiment despite price drop',
                    indicators={
                        'returns_10d': returns_10d,
                        'sentiment_10d': sentiment_10d,
                        'divergence': sentiment_10d - (returns_10d * 10)
                    }
                ))
            
            # Case 2: Price rising but sentiment still negative
            elif returns_10d > 0.05 and sentiment_10d < -0.3:
                strength = min(returns_10d * abs(sentiment_10d) * 5, 1.0)
                
                signals.append(BiasSignal(
                    bias_type='CONFIRMATION_BIAS',
                    timestamp=data.loc[i, 'timestamp'],
                    strength=strength,
                    description='Confirmation bias - negative sentiment despite price rise',
                    indicators={
                        'returns_10d': returns_10d,
                        'sentiment_10d': sentiment_10d,
                        'divergence': sentiment_10d - (returns_10d * 10)
                    }
                ))
        
        return signals
    
    def _empty_result(self, price_data: pd.DataFrame) -> BiasAnalysisResult:
        """Return empty result when insufficient data"""
        return BiasAnalysisResult(
            symbol='UNKNOWN',
            period_start=datetime.now(),
            period_end=datetime.now(),
            detected_biases=[],
            herding_score=0.0,
            fomo_score=0.0,
            fud_score=0.0,
            confirmation_bias_score=0.0,
            overall_bias_level=0.0
        )


# Test function
if __name__ == "__main__":
    # Generate test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    
    # Simulate price data with FOMO event
    price_data = pd.DataFrame({
        'timestamp': dates,
        'close': np.concatenate([
            np.random.randn(70).cumsum() + 100,  # Normal
            np.random.randn(30).cumsum() + 5 + 100  # Rapid increase (FOMO)
        ])
    })
    
    # Simulate sentiment data
    sentiment_data = pd.DataFrame({
        'timestamp': dates,
        'sentiment': np.concatenate([
            np.random.randn(70) * 0.2,  # Normal
            np.random.randn(30) * 0.1 + 0.6  # Positive surge (FOMO)
        ])
    })
    
    # Run analysis
    detector = BehavioralBiasDetector()
    result = detector.analyze(price_data, sentiment_data)
    
    print(f"\nBehavioral Bias Analysis:")
    print(f"Period: {result.period_start} to {result.period_end}")
    print(f"Overall bias level: {result.overall_bias_level:.2f}")
    print(f"\nBias Scores:")
    print(f"  Herding: {result.herding_score:.2f}")
    print(f"  FOMO: {result.fomo_score:.2f}")
    print(f"  FUD: {result.fud_score:.2f}")
    print(f"  Confirmation: {result.confirmation_bias_score:.2f}")
    print(f"\nDetected {len(result.detected_biases)} bias signals")
    
    for signal in result.detected_biases[:5]:  # Show first 5
        print(f"\n{signal.bias_type} at {signal.timestamp}")
        print(f"  Strength: {signal.strength:.2f}")
        print(f"  {signal.description}")
