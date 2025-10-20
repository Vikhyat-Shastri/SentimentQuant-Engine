"""
VADER-based sentiment analyzer for social media text.
Fast, lexicon-based approach optimized for short texts.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger


@dataclass
class SentimentScore:
    """
    Container for sentiment analysis results.
    
    Attributes:
        positive: Positive sentiment score [0-1]
        negative: Negative sentiment score [0-1]
        neutral: Neutral sentiment score [0-1]
        compound: Overall sentiment score [-1 to 1]
        confidence: Confidence in the sentiment [0-1]
    """
    positive: float
    negative: float
    neutral: float
    compound: float
    confidence: float
    
    @property
    def is_positive(self) -> bool:
        """Check if sentiment is positive."""
        return self.compound > 0.05
    
    @property
    def is_negative(self) -> bool:
        """Check if sentiment is negative."""
        return self.compound < -0.05
    
    @property
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral."""
        return -0.05 <= self.compound <= 0.05
    
    @property
    def sentiment_label(self) -> str:
        """Get sentiment label."""
        if self.is_positive:
            return "positive"
        elif self.is_negative:
            return "negative"
        else:
            return "neutral"


class VADERSentimentAnalyzer:
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) analyzer.
    Optimized for social media text with emojis, slang, and informal language.
    """
    
    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER sentiment analyzer initialized")
    
    def analyze(self, text: str) -> SentimentScore:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentScore object with detailed scores
        """
        if not text or len(text.strip()) == 0:
            return SentimentScore(
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                compound=0.0,
                confidence=0.0
            )
        
        try:
            # Get VADER scores
            scores = self.analyzer.polarity_scores(text)
            
            # Calculate confidence (higher when sentiment is clear)
            confidence = max(scores['pos'], scores['neg'], scores['neu'])
            
            return SentimentScore(
                positive=scores['pos'],
                negative=scores['neg'],
                neutral=scores['neu'],
                compound=scores['compound'],
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return SentimentScore(
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                compound=0.0,
                confidence=0.0
            )
    
    def batch_analyze(self, texts: list[str]) -> list[SentimentScore]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentScore objects
        """
        return [self.analyze(text) for text in texts]


class FearGreedClassifier:
    """
    Custom classifier for fear and greed indicators in financial text.
    Uses keyword-based approach to detect extreme emotions.
    """
    
    def __init__(self):
        """Initialize fear/greed classifier."""
        # Fear keywords (weighted by intensity)
        self.fear_keywords = {
            'crash': 2.0,
            'dump': 1.5,
            'panic': 2.0,
            'sell': 1.0,
            'fear': 1.5,
            'blood': 1.8,
            'bottom': 1.2,
            'bear': 1.3,
            'collapse': 2.0,
            'drop': 1.0,
            'fall': 1.0,
            'scary': 1.5,
            'worried': 1.3,
            'uncertain': 1.2,
            'risk': 1.0,
            'loss': 1.3
        }
        
        # Greed keywords (weighted by intensity)
        self.greed_keywords = {
            'moon': 2.0,
            'pump': 1.5,
            'fomo': 2.0,
            'bullish': 1.3,
            'buy': 1.0,
            'greed': 1.5,
            'ath': 1.8,  # All-time high
            '100x': 2.0,
            'lambo': 1.8,
            'bull': 1.3,
            'surge': 1.2,
            'rally': 1.2,
            'rocket': 1.5,
            'hodl': 1.2,
            'diamond': 1.5,  # diamond hands
            'accumulate': 1.0
        }
        
        logger.info("Fear/Greed classifier initialized")
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze fear and greed indicators in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with fear_score, greed_score, and emotion
        """
        if not text:
            return {'fear_score': 0.0, 'greed_score': 0.0, 'emotion': 'neutral'}
        
        text_lower = text.lower()
        
        # Calculate fear score
        fear_score = 0.0
        for keyword, weight in self.fear_keywords.items():
            if keyword in text_lower:
                fear_score += weight
        
        # Calculate greed score
        greed_score = 0.0
        for keyword, weight in self.greed_keywords.items():
            if keyword in text_lower:
                greed_score += weight
        
        # Normalize scores (cap at 10)
        fear_score = min(fear_score, 10.0) / 10.0
        greed_score = min(greed_score, 10.0) / 10.0
        
        # Calculate net sentiment (-1 to +1)
        net_sentiment = greed_score - fear_score
        
        # Convert to sentiment score format
        # positive = greed, negative = fear, neutral = low scores
        positive = greed_score
        negative = fear_score
        neutral = 1.0 - max(fear_score, greed_score)
        compound = net_sentiment  # Maps to -1 (fear) to +1 (greed)
        confidence = abs(net_sentiment)  # Higher difference = more confident
        
        return SentimentScore(
            positive=positive,
            negative=negative,
            neutral=neutral,
            compound=compound,
            confidence=confidence
        )


class EnsembleSentimentAnalyzer:
    """
    Combines multiple sentiment analyzers for more accurate results.
    Uses VADER + Fear/Greed classifier.
    """
    
    def __init__(self, vader_weight: float = 0.7, fear_greed_weight: float = 0.3):
        """
        Initialize ensemble analyzer.
        
        Args:
            vader_weight: Weight for VADER scores (0-1)
            fear_greed_weight: Weight for fear/greed scores (0-1)
        """
        self.vader = VADERSentimentAnalyzer()
        self.fear_greed = FearGreedClassifier()
        self.vader_weight = vader_weight
        self.fear_greed_weight = fear_greed_weight
        
        logger.info(f"Ensemble analyzer initialized (VADER: {vader_weight}, Fear/Greed: {fear_greed_weight})")
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text with ensemble of methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with combined sentiment scores
        """
        # Get VADER sentiment
        vader_result = self.vader.analyze(text)
        
        # Get fear/greed sentiment
        fg_result = self.fear_greed.analyze(text)
        
        # Combine scores (weighted average)
        combined_compound = (
            vader_result.compound * self.vader_weight +
            fg_result.compound * self.fear_greed_weight
        )
        
        # Combine positive/negative/neutral
        combined_positive = (
            vader_result.positive * self.vader_weight +
            fg_result.positive * self.fear_greed_weight
        )
        combined_negative = (
            vader_result.negative * self.vader_weight +
            fg_result.negative * self.fear_greed_weight
        )
        combined_neutral = (
            vader_result.neutral * self.vader_weight +
            fg_result.neutral * self.fear_greed_weight
        )
        
        # Combine confidence
        combined_confidence = (
            vader_result.confidence * self.vader_weight +
            fg_result.confidence * self.fear_greed_weight
        )
        
        return SentimentScore(
            positive=combined_positive,
            negative=combined_negative,
            neutral=combined_neutral,
            compound=combined_compound,
            confidence=combined_confidence
        )


if __name__ == "__main__":
    # Test the sentiment analyzers
    print("=" * 70)
    print("Testing Sentiment Analyzers")
    print("=" * 70)
    
    # Test texts
    test_texts = [
        "Bitcoin is going to the moon! 🚀 Buy now!",
        "This is a terrible crash. Sell everything!",
        "The market looks stable today.",
        "FOMO is real! Everyone is buying! 100x incoming!",
        "Panic selling everywhere. Blood in the streets.",
    ]
    
    # VADER only
    print("\n1. VADER Sentiment Analyzer:")
    print("-" * 70)
    vader = VADERSentimentAnalyzer()
    for text in test_texts:
        result = vader.analyze(text)
        print(f"Text: {text}")
        print(f"  Compound: {result.compound:+.3f} | Label: {result.sentiment_label} | Confidence: {result.confidence:.2f}")
        print()
    
    # Fear/Greed Classifier
    print("\n2. Fear/Greed Classifier:")
    print("-" * 70)
    fg = FearGreedClassifier()
    for text in test_texts:
        result = fg.analyze(text)
        print(f"Text: {text}")
        print(f"  Fear: {result['fear_score']:.2f} | Greed: {result['greed_score']:.2f} | Emotion: {result['emotion']}")
        print()
    
    # Ensemble
    print("\n3. Ensemble Sentiment Analyzer:")
    print("-" * 70)
    ensemble = EnsembleSentimentAnalyzer()
    for text in test_texts:
        result = ensemble.analyze(text)
        print(f"Text: {text}")
        print(f"  VADER: {result['vader']['compound']:+.3f}")
        print(f"  Fear/Greed: {result['fear_greed']['net_sentiment']:+.3f}")
        print(f"  Combined: {result['combined']['compound']:+.3f} ({result['combined']['sentiment']})")
        print(f"  Confidence: {result['combined']['confidence']:.2f}")
        print()
