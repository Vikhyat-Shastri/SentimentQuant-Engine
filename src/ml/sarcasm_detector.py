"""
Sarcasm Detection Module

Detects sarcastic and ironic sentiment in text using:
- Sentiment contradiction (positive words + negative context)
- Linguistic patterns (intensifiers, punctuation)
- Contextual analysis
- Machine learning models
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger


@dataclass
class SarcasmResult:
    """Sarcasm detection result"""
    text: str
    is_sarcastic: bool
    confidence: float  # 0-1
    indicators: List[str]  # What indicated sarcasm
    adjusted_sentiment: float  # Sentiment after sarcasm correction


class SarcasmDetector:
    """
    Detect sarcasm and irony in text
    
    Sarcasm indicators:
    - Sentiment contradiction (positive words in negative context)
    - Excessive punctuation (!!!, ???)
    - Intensifiers with mild sentiment
    - Quotation marks around positive words
    - Common sarcastic phrases
    """
    
    def __init__(self, use_model: bool = True):
        """
        Initialize sarcasm detector
        
        Args:
            use_model: Whether to use ML model (slower but more accurate)
        """
        self.use_model = use_model
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize model if requested
        self.model = None
        if use_model:
            try:
                # Use a model fine-tuned on sarcasm detection
                self.model = pipeline(
                    "text-classification",
                    model="mrm8488/t5-base-finetuned-sarcasm-twitter",
                    top_k=1
                )
                logger.info("Sarcasm detection model loaded")
            except Exception as e:
                logger.warning(f"Could not load sarcasm model: {e}")
                self.use_model = False
        
        # Sarcastic phrases
        self.sarcastic_phrases = [
            "oh great", "oh wonderful", "just great", "just wonderful",
            "yeah right", "sure thing", "how nice", "real nice",
            "thanks a lot", "big surprise", "shocker",
            "who would have thought", "color me surprised",
            "well done", "good job", "nice one", "brilliant"
        ]
        
        # Intensifiers
        self.intensifiers = [
            "so", "very", "really", "extremely", "incredibly",
            "absolutely", "totally", "completely", "utterly"
        ]
        
        logger.info("SarcasmDetector initialized")
    
    def _check_sentiment_contradiction(self, text: str) -> Tuple[bool, float]:
        """
        Check for sentiment contradiction
        
        Sarcasm often uses positive words in negative context
        
        Returns:
            (has_contradiction, contradiction_strength)
        """
        # Get VADER sentiment
        scores = self.vader.polarity_scores(text)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        if len(sentences) < 2:
            return False, 0.0
        
        # Check each sentence sentiment
        sentence_sentiments = []
        for sent in sentences:
            if sent.strip():
                sent_scores = self.vader.polarity_scores(sent)
                sentence_sentiments.append(sent_scores['compound'])
        
        if len(sentence_sentiments) < 2:
            return False, 0.0
        
        # Check for alternating sentiment
        sentiment_changes = 0
        for i in range(len(sentence_sentiments) - 1):
            if (sentence_sentiments[i] > 0.3 and sentence_sentiments[i+1] < -0.3) or \
               (sentence_sentiments[i] < -0.3 and sentence_sentiments[i+1] > 0.3):
                sentiment_changes += 1
        
        contradiction_strength = sentiment_changes / max(len(sentence_sentiments) - 1, 1)
        has_contradiction = contradiction_strength > 0.5
        
        return has_contradiction, contradiction_strength
    
    def _check_linguistic_patterns(self, text: str) -> Tuple[List[str], float]:
        """
        Check for linguistic sarcasm indicators
        
        Returns:
            (indicators, pattern_score)
        """
        indicators = []
        score = 0.0
        text_lower = text.lower()
        
        # Excessive punctuation
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        if exclamation_count >= 3:
            indicators.append(f"excessive_exclamation ({exclamation_count})")
            score += 0.3
        
        if question_count >= 2:
            indicators.append(f"multiple_questions ({question_count})")
            score += 0.2
        
        # Quotation marks around positive words
        quoted_text = re.findall(r'["\']([^"\']+)["\']', text)
        if quoted_text:
            for quote in quoted_text:
                quote_sentiment = self.vader.polarity_scores(quote)['compound']
                if quote_sentiment > 0.3:
                    indicators.append(f"quoted_positive: '{quote}'")
                    score += 0.4
        
        # Sarcastic phrases
        for phrase in self.sarcastic_phrases:
            if phrase in text_lower:
                indicators.append(f"sarcastic_phrase: '{phrase}'")
                score += 0.5
        
        # Intensifiers with weak sentiment
        has_intensifier = any(word in text_lower for word in self.intensifiers)
        overall_sentiment = abs(self.vader.polarity_scores(text)['compound'])
        
        if has_intensifier and overall_sentiment < 0.3:
            indicators.append("intensifier_with_mild_sentiment")
            score += 0.3
        
        # All caps words (shouting)
        caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
        if len(caps_words) >= 2:
            indicators.append(f"caps_words: {', '.join(caps_words[:3])}")
            score += 0.2
        
        # Ellipsis
        if '...' in text:
            indicators.append("ellipsis")
            score += 0.1
        
        return indicators, min(score, 1.0)
    
    def _check_contextual_cues(self, text: str) -> Tuple[List[str], float]:
        """
        Check for contextual sarcasm cues
        
        Returns:
            (indicators, context_score)
        """
        indicators = []
        score = 0.0
        text_lower = text.lower()
        
        # Hyperbole detection (exaggerated statements)
        hyperbole_words = [
            'never', 'always', 'everyone', 'nobody', 'nothing',
            'everything', 'best', 'worst', 'ever', 'impossible'
        ]
        
        hyperbole_count = sum(1 for word in hyperbole_words if word in text_lower)
        if hyperbole_count >= 2:
            indicators.append(f"hyperbole ({hyperbole_count} words)")
            score += 0.3
        
        # Rhetorical questions
        if '?' in text:
            rhetorical_patterns = [
                r'what could go wrong',
                r'who would have thought',
                r'what a surprise',
                r'why am i not surprised',
                r'seriously\?'
            ]
            
            for pattern in rhetorical_patterns:
                if re.search(pattern, text_lower):
                    indicators.append(f"rhetorical_question: {pattern}")
                    score += 0.4
        
        # Emoji contradiction (positive emoji with negative text or vice versa)
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
        emojis = re.findall(emoji_pattern, text)
        
        if emojis:
            text_sentiment = self.vader.polarity_scores(text)['compound']
            
            # Common positive emojis
            positive_emojis = ['😂', '🤣', '😅', '😏', '🙄']
            has_positive_emoji = any(e in emojis for e in positive_emojis)
            
            if has_positive_emoji and text_sentiment < -0.2:
                indicators.append("emoji_sentiment_mismatch")
                score += 0.4
        
        return indicators, min(score, 1.0)
    
    def detect(self, text: str) -> SarcasmResult:
        """
        Detect sarcasm in text
        
        Args:
            text: Text to analyze
        
        Returns:
            SarcasmResult
        """
        if not text or len(text.strip()) < 3:
            return SarcasmResult(
                text=text,
                is_sarcastic=False,
                confidence=0.0,
                indicators=[],
                adjusted_sentiment=0.0
            )
        
        all_indicators = []
        total_score = 0.0
        
        # Check sentiment contradiction
        has_contradiction, contradiction_score = self._check_sentiment_contradiction(text)
        if has_contradiction:
            all_indicators.append(f"sentiment_contradiction ({contradiction_score:.2f})")
            total_score += contradiction_score * 0.4
        
        # Check linguistic patterns
        ling_indicators, ling_score = self._check_linguistic_patterns(text)
        all_indicators.extend(ling_indicators)
        total_score += ling_score * 0.3
        
        # Check contextual cues
        context_indicators, context_score = self._check_contextual_cues(text)
        all_indicators.extend(context_indicators)
        total_score += context_score * 0.3
        
        # Use ML model if available
        model_score = 0.0
        if self.use_model and self.model is not None:
            try:
                prediction = self.model(text[:512])[0]  # Truncate to model max length
                if prediction[0]['label'].lower() in ['sarcasm', 'sarcastic', 'label_1']:
                    model_score = prediction[0]['score']
                    all_indicators.append(f"model_prediction ({model_score:.2f})")
                    total_score += model_score * 0.3
            except Exception as e:
                logger.debug(f"Model prediction failed: {e}")
        
        # Final sarcasm determination
        confidence = min(total_score, 1.0)
        is_sarcastic = confidence > 0.5
        
        # Calculate adjusted sentiment
        original_sentiment = self.vader.polarity_scores(text)['compound']
        
        if is_sarcastic:
            # Flip sentiment if sarcastic
            adjusted_sentiment = -original_sentiment * confidence
        else:
            adjusted_sentiment = original_sentiment
        
        return SarcasmResult(
            text=text,
            is_sarcastic=is_sarcastic,
            confidence=confidence,
            indicators=all_indicators,
            adjusted_sentiment=adjusted_sentiment
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SarcasmResult]:
        """
        Analyze multiple texts for sarcasm
        
        Args:
            texts: List of texts
        
        Returns:
            List of SarcasmResult objects
        """
        results = []
        for text in texts:
            result = self.detect(text)
            results.append(result)
        
        return results


# Test function
if __name__ == "__main__":
    # Test cases
    test_texts = [
        # Obvious sarcasm
        "Oh great, another Monday morning!",
        "Yeah right, like that's ever going to happen...",
        "Well that went EXACTLY as planned 🙄",
        "What a 'surprise' that it broke again!!!",
        
        # Subtle sarcasm
        "Thanks a lot for the help",
        "I just LOVE waiting in line for hours",
        "Brilliant idea, really brilliant",
        
        # Rhetorical sarcasm
        "Who would have thought that would fail?",
        "What could possibly go wrong?",
        
        # Not sarcastic
        "I had a great day at work today!",
        "Thank you so much for your help, I really appreciate it.",
        "The weather is nice today.",
        
        # Context-dependent
        "Bitcoin is doing so well today... down 10%",
        "Love how everyone is panic selling 😂"
    ]
    
    print("\n" + "="*80)
    print("SARCASM DETECTION TEST")
    print("="*80)
    
    # Initialize detector (without ML model for speed)
    detector = SarcasmDetector(use_model=False)
    
    for i, text in enumerate(test_texts, 1):
        result = detector.detect(text)
        
        print(f"\n{i}. Text: \"{text}\"")
        print(f"   Sarcastic: {'YES' if result.is_sarcastic else 'NO'} "
              f"(confidence: {result.confidence:.3f})")
        
        if result.indicators:
            print(f"   Indicators: {', '.join(result.indicators)}")
        
        print(f"   Original sentiment: {detector.vader.polarity_scores(text)['compound']:.3f}")
        print(f"   Adjusted sentiment: {result.adjusted_sentiment:.3f}")
    
    # Statistics
    results = detector.analyze_batch(test_texts)
    sarcastic_count = sum(1 for r in results if r.is_sarcastic)
    
    print(f"\n" + "="*80)
    print(f"SUMMARY: {sarcastic_count}/{len(test_texts)} texts detected as sarcastic")
    print("="*80)
