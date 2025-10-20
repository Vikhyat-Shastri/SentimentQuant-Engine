"""
Text preprocessing pipeline for sentiment analysis.
Handles cleaning, normalization, and preparation of text data.
"""

import re
from typing import List, Dict, Any, Optional
import emoji
from dataclasses import dataclass
from .ner import FinancialNER, Entity


@dataclass
class ProcessedText:
    """
    Container for processed text with metadata.
    
    Attributes:
        original: Original text
        cleaned: Cleaned text
        tokens: List of tokens
        entities: Extracted entities (tickers, etc.)
        metadata: Additional processing metadata
    """
    original: str
    cleaned: str
    tokens: List[str]
    entities: Dict[str, List[str]]
    metadata: Dict[str, Any]


class TextPreprocessor:
    """
    Preprocessing pipeline for financial social media text.
    Optimized for Twitter, Reddit, and news content.
    """
    
    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        lowercase: bool = True,
        min_length: int = 10,
        max_length: int = 500,
        use_ner: bool = True,
        use_spacy: bool = False
    ):
        """
        Initialize TextPreprocessor.
        
        Args:
            remove_urls: Whether to remove URLs
            remove_mentions: Whether to remove @mentions
            remove_hashtags: Whether to remove #hashtags
            lowercase: Whether to convert to lowercase
            min_length: Minimum text length to accept
            max_length: Maximum text length (truncate)
            use_ner: Whether to extract named entities
            use_spacy: Whether to use spaCy for NER (requires spaCy installation)
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.lowercase = lowercase
        self.min_length = min_length
        self.max_length = max_length
        self.use_ner = use_ner
        
        # Initialize NER if enabled
        self.ner = None
        if use_ner:
            self.ner = FinancialNER(use_spacy=use_spacy)
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.ticker_pattern = re.compile(r'\$([A-Z]{2,5})\b')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Financial stopwords (in addition to standard stopwords)
        self.financial_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were',
            'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when',
            'where', 'why', 'how', 'rt'  # retweet
        }
        
        # Sentiment-bearing emojis (keep these)
        self.sentiment_emojis = {
            '🚀', '📈', '💎', '🌙', '💰', '🔥',  # Positive/Bullish
            '📉', '💩', '🐻', '⚠️', '❌', '😱'   # Negative/Bearish
        }
    
    def preprocess(self, text: str) -> ProcessedText:
        """
        Process text through full preprocessing pipeline.
        
        Args:
            text: Raw text to process
            
        Returns:
            ProcessedText object with all processing results
        """
        original = text
        metadata = {}
        
        # Extract entities before cleaning
        entities = self._extract_entities(text)
        metadata['entity_count'] = sum(len(v) for v in entities.values())
        
        # Extract emoji sentiment
        emoji_sentiment = self._extract_emoji_sentiment(text)
        metadata['emoji_sentiment'] = emoji_sentiment
        
        # Clean text
        cleaned = self._clean_text(text)
        
        # Validate length
        if len(cleaned) < self.min_length:
            metadata['too_short'] = True
        
        # Truncate if too long
        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length]
            metadata['truncated'] = True
        
        # Tokenize
        tokens = self._tokenize(cleaned)
        metadata['token_count'] = len(tokens)
        
        return ProcessedText(
            original=original,
            cleaned=cleaned,
            tokens=tokens,
            entities=entities,
            metadata=metadata
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = self.mention_pattern.sub('', text)
        
        # Remove hashtags (but keep the text)
        if self.remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        else:
            # Remove # but keep the word
            text = text.replace('#', '')
        
        # Convert emojis to text (keep sentiment)
        text = self._process_emojis(text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s$.,!?@#-]', ' ', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types and values
        """
        # Use advanced NER if available
        if self.ner:
            ner_entities = self.ner.extract_entities(text)
            entity_summary = self.ner.get_entity_summary(ner_entities)
            
            # Also extract basic entities for backward compatibility
            entities = {
                'tickers': entity_summary.get('ticker', []),
                'cryptos': entity_summary.get('crypto', []),
                'companies': entity_summary.get('company', []),
                'exchanges': entity_summary.get('exchange', []),
                'financial_terms': entity_summary.get('financial_term', []),
                'numbers': entity_summary.get('number', []),
                'percentages': entity_summary.get('percentage', []),
                'persons': entity_summary.get('person', []),
            }
            
            # Extract hashtags and mentions (not in NER)
            hashtags = self.hashtag_pattern.findall(text)
            entities['hashtags'] = [h.lower() for h in hashtags]
            
            mentions = self.mention_pattern.findall(text)
            entities['mentions'] = [m.lower() for m in mentions]
            
            return entities
        
        # Fallback to basic entity extraction
        entities = {
            'tickers': [],
            'hashtags': [],
            'mentions': []
        }
        
        # Extract tickers ($BTC, $ETH)
        tickers = self.ticker_pattern.findall(text.upper())
        entities['tickers'] = list(set(tickers))
        
        # Extract hashtags
        hashtags = self.hashtag_pattern.findall(text)
        entities['hashtags'] = [h.lower() for h in hashtags]
        
        # Extract mentions
        mentions = self.mention_pattern.findall(text)
        entities['mentions'] = [m.lower() for m in mentions]
        
        # Also detect crypto names without $
        crypto_keywords = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'btc': 'BTC',
            'eth': 'ETH',
            'binance': 'BNB',
            'cardano': 'ADA',
            'solana': 'SOL',
            'ripple': 'XRP',
            'dogecoin': 'DOGE',
            'doge': 'DOGE'
        }
        
        text_lower = text.lower()
        for keyword, ticker in crypto_keywords.items():
            if keyword in text_lower:
                if ticker not in entities['tickers']:
                    entities['tickers'].append(ticker)
        
        return entities
    
    def _extract_emoji_sentiment(self, text: str) -> float:
        """
        Extract sentiment from emojis.
        
        Args:
            text: Text containing emojis
            
        Returns:
            Emoji sentiment score (-1 to 1)
        """
        # Positive emojis
        positive_emojis = {'🚀', '📈', '💎', '🌙', '💰', '🔥', '✨', '💪', '🎉', '😄', '😊', '👍'}
        # Negative emojis
        negative_emojis = {'📉', '💩', '🐻', '⚠️', '❌', '😱', '😰', '😭', '👎', '💔'}
        
        positive_count = sum(1 for char in text if char in positive_emojis)
        negative_count = sum(1 for char in text if char in negative_emojis)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    def _process_emojis(self, text: str) -> str:
        """
        Convert emojis to text descriptions.
        
        Args:
            text: Text with emojis
            
        Returns:
            Text with emoji descriptions
        """
        # Keep sentiment-bearing emojis, remove others
        result = []
        for char in text:
            if char in emoji.EMOJI_DATA:
                if char in self.sentiment_emojis:
                    # Keep important emojis
                    result.append(char)
            else:
                result.append(char)
        
        return ''.join(result)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization
        tokens = text.split()
        
        # Remove stopwords
        tokens = [
            token for token in tokens
            if token.lower() not in self.financial_stopwords
            and len(token) > 1  # Remove single characters
        ]
        
        return tokens
    
    def batch_preprocess(self, texts: List[str]) -> List[ProcessedText]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of ProcessedText objects
        """
        return [self.preprocess(text) for text in texts]
    
    def is_valid_text(self, processed: ProcessedText) -> bool:
        """
        Check if processed text is valid for analysis.
        
        Args:
            processed: ProcessedText object
            
        Returns:
            True if text is valid
        """
        # Check minimum length
        if len(processed.cleaned) < self.min_length:
            return False
        
        # Check if there are any tokens
        if len(processed.tokens) == 0:
            return False
        
        # Check if text has any meaningful content
        if not any(c.isalnum() for c in processed.cleaned):
            return False
        
        return True


def create_preprocessor_from_config(config: Dict[str, Any]) -> TextPreprocessor:
    """
    Create TextPreprocessor from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured TextPreprocessor instance
    """
    processing_config = config.get('processing', {})
    
    return TextPreprocessor(
        remove_urls=processing_config.get('remove_urls', True),
        remove_mentions=processing_config.get('remove_mentions', True),
        remove_hashtags=processing_config.get('remove_hashtags', False),
        lowercase=processing_config.get('lowercase', True),
        min_length=processing_config.get('min_text_length', 10),
        max_length=processing_config.get('max_text_length', 500)
    )
