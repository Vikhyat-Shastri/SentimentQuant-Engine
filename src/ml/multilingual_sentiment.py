"""
Multi-Language Sentiment Analysis

Supports sentiment analysis in multiple languages crucial for crypto markets:
- English (primary)
- Chinese (Simplified & Traditional) - Major crypto market
- Japanese - Active crypto trading
- Korean - Major crypto exchanges
- Spanish - Growing market

Uses language detection and translation where needed.
"""
import re
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    logger.warning("langdetect not installed. Multi-language support limited.")
    LANGDETECT_AVAILABLE = False

try:
    from googletrans import Translator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    logger.warning("googletrans not installed. Translation disabled.")
    TRANSLATOR_AVAILABLE = False


@dataclass
class MultiLingualSentiment:
    """Result of multi-lingual sentiment analysis"""
    text: str
    detected_language: str
    translated_text: Optional[str]
    sentiment_score: float
    confidence: float
    language_confidence: float


class MultiLingualSentimentAnalyzer:
    """
    Analyzes sentiment across multiple languages
    
    Features:
    - Automatic language detection
    - Optional translation to English for analysis
    - Native sentiment analysis for supported languages
    - Language-specific preprocessing
    """
    
    def __init__(self, translate_threshold: float = 0.8):
        """
        Initialize multi-lingual analyzer
        
        Args:
            translate_threshold: Confidence threshold for translation (0-1)
        """
        self.translate_threshold = translate_threshold
        self.translator = Translator() if TRANSLATOR_AVAILABLE else None
        
        # Supported languages
        self.supported_languages = {
            'en': 'English',
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)',
            'ja': 'Japanese',
            'ko': 'Korean',
            'es': 'Spanish',
            'pt': 'Portuguese',
            'ru': 'Russian'
        }
        
        # Language-specific sentiment lexicons (basic implementation)
        self.sentiment_keywords = {
            'zh-cn': {
                'positive': ['好', '棒', '涨', '牛', '看多', '突破', '强势', '利好'],
                'negative': ['差', '跌', '熊', '看空', '崩', '暴跌', '利空', '套牢']
            },
            'ja': {
                'positive': ['良い', '上がる', '強気', '突破'],
                'negative': ['悪い', '下がる', '弱気', '暴落']
            },
            'ko': {
                'positive': ['좋다', '상승', '강세', '돌파'],
                'negative': ['나쁘다', '하락', '약세', '폭락']
            }
        }
        
        logger.info(f"MultiLingualSentimentAnalyzer initialized")
        logger.info(f"Supported languages: {', '.join(self.supported_languages.values())}")
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of text
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (language_code, confidence)
        """
        if not LANGDETECT_AVAILABLE:
            return 'en', 1.0  # Default to English
        
        try:
            # Clean text for better detection
            clean_text = re.sub(r'http\S+|@\w+|#\w+', '', text)
            clean_text = clean_text.strip()
            
            if len(clean_text) < 3:
                return 'en', 0.5
            
            lang = detect(clean_text)
            
            # langdetect returns 2-letter codes, map to our format
            if lang == 'zh':
                # Try to distinguish simplified vs traditional
                if self._contains_simplified_chinese(text):
                    lang = 'zh-cn'
                else:
                    lang = 'zh-tw'
            
            confidence = 0.9  # langdetect doesn't provide confidence
            
            return lang, confidence
            
        except LangDetectException:
            return 'en', 0.3  # Low confidence default
    
    def _contains_simplified_chinese(self, text: str) -> bool:
        """Check if text contains simplified Chinese characters"""
        # Simplified-specific characters (common ones)
        simplified_chars = set('国会来个问题')
        # Traditional-specific characters
        traditional_chars = set('國會來個問題')
        
        has_simplified = any(c in text for c in simplified_chars)
        has_traditional = any(c in text for c in traditional_chars)
        
        if has_simplified and not has_traditional:
            return True
        elif has_traditional and not has_simplified:
            return False
        else:
            return True  # Default to simplified
    
    def translate_to_english(self, text: str, source_lang: str) -> Optional[str]:
        """
        Translate text to English
        
        Args:
            text: Text to translate
            source_lang: Source language code
        
        Returns:
            Translated text or None if translation fails
        """
        if not self.translator:
            logger.warning("Translation not available (googletrans not installed)")
            return None
        
        if source_lang == 'en':
            return text
        
        try:
            # Translate to English
            result = self.translator.translate(text, src=source_lang, dest='en')
            translated = result.text
            
            logger.debug(f"Translated from {source_lang}: {text[:50]}... -> {translated[:50]}...")
            return translated
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None
    
    def analyze_native(self, text: str, language: str) -> Tuple[float, float]:
        """
        Analyze sentiment in native language using keyword matching
        
        Args:
            text: Text in native language
            language: Language code
        
        Returns:
            Tuple of (sentiment_score, confidence)
        """
        if language not in self.sentiment_keywords:
            return 0.0, 0.0  # Unsupported language
        
        keywords = self.sentiment_keywords[language]
        positive_words = keywords['positive']
        negative_words = keywords['negative']
        
        # Count keyword matches
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        total_count = positive_count + negative_count
        
        if total_count == 0:
            return 0.0, 0.0  # No keywords found
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / total_count
        
        # Confidence based on number of matches
        confidence = min(total_count / 5.0, 1.0)  # Max confidence at 5+ keywords
        
        return sentiment_score, confidence
    
    def analyze(
        self,
        text: str,
        vader_analyzer=None,
        ml_model=None
    ) -> MultiLingualSentiment:
        """
        Analyze sentiment with automatic language detection
        
        Args:
            text: Input text
            vader_analyzer: Optional VADER analyzer for English
            ml_model: Optional ML model for English
        
        Returns:
            MultiLingualSentiment result
        """
        # Detect language
        lang, lang_confidence = self.detect_language(text)
        
        logger.debug(f"Detected language: {lang} (confidence: {lang_confidence:.2f})")
        
        # Strategy 1: Native language analysis (for supported languages)
        if lang in self.sentiment_keywords:
            native_score, native_conf = self.analyze_native(text, lang)
            
            if native_conf > 0.3:  # Decent confidence
                return MultiLingualSentiment(
                    text=text,
                    detected_language=lang,
                    translated_text=None,
                    sentiment_score=native_score,
                    confidence=native_conf,
                    language_confidence=lang_confidence
                )
        
        # Strategy 2: Translate to English and analyze
        if lang != 'en' and lang_confidence > self.translate_threshold:
            translated = self.translate_to_english(text, lang)
            
            if translated:
                # Use English analyzers on translated text
                if ml_model:
                    # Use ML model (most accurate)
                    result = ml_model.analyze(translated)
                    sentiment_score = result.score if result.label == 'positive' else -result.score
                    confidence = result.score
                elif vader_analyzer:
                    # Use VADER
                    result = vader_analyzer.analyze(translated)
                    sentiment_score = result.compound
                    confidence = abs(sentiment_score)
                else:
                    sentiment_score = 0.0
                    confidence = 0.0
                
                return MultiLingualSentiment(
                    text=text,
                    detected_language=lang,
                    translated_text=translated,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    language_confidence=lang_confidence
                )
        
        # Strategy 3: English analysis (default)
        if vader_analyzer:
            result = vader_analyzer.analyze(text)
            sentiment_score = result.compound
            confidence = abs(sentiment_score)
        else:
            sentiment_score = 0.0
            confidence = 0.0
        
        return MultiLingualSentiment(
            text=text,
            detected_language=lang,
            translated_text=None if lang == 'en' else text,
            sentiment_score=sentiment_score,
            confidence=confidence,
            language_confidence=lang_confidence
        )


# Test function
if __name__ == "__main__":
    analyzer = MultiLingualSentimentAnalyzer()
    
    # Test texts
    test_texts = [
        ("Bitcoin is going to the moon! 🚀", "English"),
        ("比特币涨势强劲，看多！", "Chinese"),
        ("ビットコインが上がっています", "Japanese"),
        ("비트코인 상승세", "Korean"),
        ("Bitcoin está subiendo mucho", "Spanish")
    ]
    
    for text, expected_lang in test_texts:
        result = analyzer.analyze(text)
        print(f"\n{expected_lang}:")
        print(f"  Text: {text}")
        print(f"  Detected: {result.detected_language}")
        print(f"  Translated: {result.translated_text}")
        print(f"  Sentiment: {result.sentiment_score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")
