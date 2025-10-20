"""
Fear & Greed Index calculator for cryptocurrency markets.
Combines multiple market indicators into a single index.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

from src.utils import normalize_score, weighted_average, config_manager


@dataclass
class FearGreedIndex:
    """
    Fear & Greed Index result.
    
    Attributes:
        index: Overall index [0-100]
        label: Text label (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)
        components: Breakdown by component
        timestamp: When calculated
    """
    index: float
    label: str
    components: Dict[str, float]
    timestamp: float
    
    @property
    def is_extreme_fear(self) -> bool:
        """Check if extreme fear."""
        return self.index < 25
    
    @property
    def is_fear(self) -> bool:
        """Check if fear."""
        return 25 <= self.index < 45
    
    @property
    def is_neutral(self) -> bool:
        """Check if neutral."""
        return 45 <= self.index <= 55
    
    @property
    def is_greed(self) -> bool:
        """Check if greed."""
        return 55 < self.index <= 75
    
    @property
    def is_extreme_greed(self) -> bool:
        """Check if extreme greed."""
        return self.index > 75


class FearGreedIndexCalculator:
    """
    Calculate Fear & Greed Index from multiple market indicators.
    
    Index components:
    1. Sentiment Score (25%): Social media and news sentiment
    2. Volume/Momentum (25%): Trading volume changes
    3. Volatility (15%): Price volatility
    4. Social Volume (15%): Mentions and social activity
    5. Dominance (10%): BTC dominance changes
    6. Trends (10%): Google Trends data (optional)
    """
    
    def __init__(self):
        """Initialize Fear & Greed calculator."""
        # Load component weights from config
        config_components = config_manager.get_section(
            'signal_config',
            'signals'
        ).get('fear_greed_index', {}).get('components', {})
        
        self.weights = {
            'sentiment_score': config_components.get('sentiment_score', {}).get('weight', 0.25),
            'volume_momentum': config_components.get('volume_momentum', {}).get('weight', 0.25),
            'volatility': config_components.get('volatility', {}).get('weight', 0.15),
            'social_volume': config_components.get('social_volume', {}).get('weight', 0.15),
            'dominance': config_components.get('dominance', {}).get('weight', 0.10),
            'trends': config_components.get('trends', {}).get('weight', 0.10),
        }
        
        # Historical data for calculations
        self.volume_history: List[float] = []
        self.price_history: List[float] = []
        self.social_volume_history: List[float] = []
        
        logger.info(f"Fear & Greed Index calculator initialized with weights: {self.weights}")
    
    def calculate(
        self,
        sentiment_score: float,
        current_volume: float,
        current_price: float,
        social_mentions: int,
        btc_dominance: Optional[float] = None,
        trends_score: Optional[float] = None
    ) -> FearGreedIndex:
        """
        Calculate Fear & Greed Index.
        
        Args:
            sentiment_score: Aggregated sentiment score [-1 to 1]
            current_volume: Current 24h trading volume
            current_price: Current price
            social_mentions: Number of social media mentions
            btc_dominance: Bitcoin dominance % (optional)
            trends_score: Google Trends score (optional)
            
        Returns:
            FearGreedIndex object
        """
        from src.utils import get_timestamp
        
        components = {}
        
        # 1. Sentiment Score Component (25%)
        sentiment_component = self._calculate_sentiment_component(sentiment_score)
        components['sentiment'] = sentiment_component
        
        # 2. Volume/Momentum Component (25%)
        volume_component = self._calculate_volume_component(current_volume)
        components['volume_momentum'] = volume_component
        
        # 3. Volatility Component (15%)
        volatility_component = self._calculate_volatility_component(current_price)
        components['volatility'] = volatility_component
        
        # 4. Social Volume Component (15%)
        social_component = self._calculate_social_volume_component(social_mentions)
        components['social_volume'] = social_component
        
        # 5. Dominance Component (10%)
        if btc_dominance is not None:
            dominance_component = self._calculate_dominance_component(btc_dominance)
        else:
            dominance_component = 50.0  # Neutral if not available
        components['dominance'] = dominance_component
        
        # 6. Trends Component (10%)
        if trends_score is not None:
            trends_component = trends_score
        else:
            trends_component = 50.0  # Neutral if not available
        components['trends'] = trends_component
        
        # Calculate weighted index
        values = []
        weights = []
        
        for component_name, component_value in components.items():
            weight = self.weights.get(component_name, 0.0)
            if weight > 0:
                values.append(component_value)
                weights.append(weight)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            index = weighted_average(values, normalized_weights)
        else:
            index = 50.0
        
        # Clamp to [0, 100]
        index = max(0.0, min(100.0, index))
        
        # Determine label
        label = self._get_label(index)
        
        return FearGreedIndex(
            index=index,
            label=label,
            components=components,
            timestamp=get_timestamp()
        )
    
    def _calculate_sentiment_component(self, sentiment_score: float) -> float:
        """
        Convert sentiment score to index component [0-100].
        
        Args:
            sentiment_score: Sentiment score [-1 to 1]
            
        Returns:
            Component score [0-100]
        """
        return normalize_score(
            sentiment_score,
            min_val=-1.0,
            max_val=1.0,
            target_min=0.0,
            target_max=100.0
        )
    
    def _calculate_volume_component(self, current_volume: float) -> float:
        """
        Calculate volume momentum component.
        
        High volume = more conviction = higher index
        
        Args:
            current_volume: Current trading volume
            
        Returns:
            Component score [0-100]
        """
        # Store volume history
        self.volume_history.append(current_volume)
        
        # Keep last 30 data points
        if len(self.volume_history) > 30:
            self.volume_history = self.volume_history[-30:]
        
        if len(self.volume_history) < 2:
            return 50.0  # Neutral if not enough data
        
        # Calculate average volume
        avg_volume = sum(self.volume_history[:-1]) / len(self.volume_history[:-1])
        
        if avg_volume == 0:
            return 50.0
        
        # Calculate volume ratio
        volume_ratio = current_volume / avg_volume
        
        # Higher volume = more greed (assuming upward price movement)
        # Lower volume = more fear
        # Scale: 0.5x volume = 0, 1x = 50, 2x+ = 100
        if volume_ratio < 0.5:
            score = 0.0
        elif volume_ratio > 2.0:
            score = 100.0
        else:
            score = (volume_ratio - 0.5) / 1.5 * 100.0
        
        return score
    
    def _calculate_volatility_component(self, current_price: float) -> float:
        """
        Calculate volatility component.
        
        High volatility = uncertainty = lower index
        
        Args:
            current_price: Current price
            
        Returns:
            Component score [0-100]
        """
        # Store price history
        self.price_history.append(current_price)
        
        # Keep last 30 data points
        if len(self.price_history) > 30:
            self.price_history = self.price_history[-30:]
        
        if len(self.price_history) < 5:
            return 50.0  # Neutral if not enough data
        
        # Calculate standard deviation (volatility)
        import numpy as np
        prices = np.array(self.price_history)
        volatility = np.std(prices) / np.mean(prices)  # Coefficient of variation
        
        # Higher volatility = more fear
        # Scale: 0% = 100 (greed), 5%+ = 0 (fear)
        if volatility >= 0.05:
            score = 0.0
        else:
            score = (1 - volatility / 0.05) * 100.0
        
        return score
    
    def _calculate_social_volume_component(self, social_mentions: int) -> float:
        """
        Calculate social volume component.
        
        High social activity = more interest = higher index
        
        Args:
            social_mentions: Number of social mentions
            
        Returns:
            Component score [0-100]
        """
        # Store social volume history
        self.social_volume_history.append(float(social_mentions))
        
        # Keep last 30 data points
        if len(self.social_volume_history) > 30:
            self.social_volume_history = self.social_volume_history[-30:]
        
        if len(self.social_volume_history) < 2:
            return 50.0
        
        # Calculate average social volume
        avg_volume = sum(self.social_volume_history[:-1]) / len(self.social_volume_history[:-1])
        
        if avg_volume == 0:
            return 50.0
        
        # Calculate ratio
        ratio = social_mentions / avg_volume
        
        # Higher social volume = more FOMO/greed
        # Scale: 0.5x = 0, 1x = 50, 2x+ = 100
        if ratio < 0.5:
            score = 0.0
        elif ratio > 2.0:
            score = 100.0
        else:
            score = (ratio - 0.5) / 1.5 * 100.0
        
        return score
    
    def _calculate_dominance_component(self, btc_dominance: float) -> float:
        """
        Calculate BTC dominance component.
        
        Rising dominance = flight to safety = fear
        Falling dominance = altcoin season = greed
        
        Args:
            btc_dominance: Bitcoin dominance %
            
        Returns:
            Component score [0-100]
        """
        # BTC dominance typically ranges from 40% to 70%
        # 70%+ = extreme fear (flight to safety)
        # 40% = extreme greed (altcoin mania)
        
        if btc_dominance >= 70:
            score = 0.0
        elif btc_dominance <= 40:
            score = 100.0
        else:
            # Inverse: higher dominance = lower score
            score = (70 - btc_dominance) / 30 * 100.0
        
        return score
    
    def _get_label(self, index: float) -> str:
        """
        Get text label for index value.
        
        Args:
            index: Index value [0-100]
            
        Returns:
            Text label
        """
        if index < 25:
            return "Extreme Fear"
        elif index < 45:
            return "Fear"
        elif index <= 55:
            return "Neutral"
        elif index <= 75:
            return "Greed"
        else:
            return "Extreme Greed"


if __name__ == "__main__":
    # Test the Fear & Greed calculator
    print("=" * 70)
    print("Testing Fear & Greed Index Calculator")
    print("=" * 70)
    
    calculator = FearGreedIndexCalculator()
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Extreme Fear Scenario',
            'sentiment': -0.7,
            'volume': 500000,
            'price': 30000,
            'mentions': 50,
            'dominance': 65
        },
        {
            'name': 'Extreme Greed Scenario',
            'sentiment': 0.8,
            'volume': 2000000,
            'price': 60000,
            'mentions': 200,
            'dominance': 45
        },
        {
            'name': 'Neutral Scenario',
            'sentiment': 0.0,
            'volume': 1000000,
            'price': 45000,
            'mentions': 100,
            'dominance': 55
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 70)
        
        result = calculator.calculate(
            sentiment_score=scenario['sentiment'],
            current_volume=scenario['volume'],
            current_price=scenario['price'],
            social_mentions=scenario['mentions'],
            btc_dominance=scenario['dominance']
        )
        
        print(f"Index: {result.index:.1f}/100 - {result.label}")
        print(f"\nComponent Breakdown:")
        for component, value in result.components.items():
            print(f"  {component}: {value:.1f}")
