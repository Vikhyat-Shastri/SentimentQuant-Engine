"""
Named Entity Recognition (NER) for Financial Text.
Identifies companies, cryptocurrencies, stock tickers, people, and financial terms.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of financial entities."""
    TICKER = "ticker"
    CRYPTO = "crypto"
    COMPANY = "company"
    PERSON = "person"
    EXCHANGE = "exchange"
    FINANCIAL_TERM = "financial_term"
    CURRENCY = "currency"
    NUMBER = "number"
    PERCENTAGE = "percentage"


@dataclass
class Entity:
    """
    Represents a named entity extracted from text.
    
    Attributes:
        text: The entity text
        type: EntityType
        start: Start position in original text
        end: End position in original text
        confidence: Confidence score (0-1)
        metadata: Additional entity metadata
    """
    text: str
    type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FinancialNER:
    """
    Named Entity Recognition for financial text.
    Identifies tickers, cryptos, companies, exchanges, and financial terms.
    """
    
    def __init__(self, use_spacy: bool = False):
        """
        Initialize FinancialNER.
        
        Args:
            use_spacy: Whether to use spaCy for enhanced NER (optional)
        """
        self.use_spacy = use_spacy
        self.nlp = None
        
        # Try to load spaCy if requested
        if use_spacy:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy model: en_core_web_sm")
                except OSError:
                    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                    self.use_spacy = False
            except ImportError:
                logger.warning("spaCy not installed. Using pattern-based NER only.")
                self.use_spacy = False
        
        # Compile regex patterns
        self._compile_patterns()
        
        # Load entity dictionaries
        self._load_entities()
    
    def _compile_patterns(self):
        """Compile regex patterns for entity extraction."""
        # Stock ticker pattern: $AAPL, TSLA, etc.
        self.ticker_pattern = re.compile(r'\$?([A-Z]{1,5})\b')
        
        # Crypto pattern: BTC, Bitcoin, ETH, Ethereum, etc.
        self.crypto_pattern = re.compile(
            r'\b(BTC|ETH|XRP|ADA|SOL|DOGE|SHIB|MATIC|DOT|AVAX|LINK|UNI|'
            r'bitcoin|ethereum|ripple|cardano|solana|dogecoin|polygon|'
            r'polkadot|avalanche|chainlink|uniswap)\b',
            re.IGNORECASE
        )
        
        # Dollar amounts: $1000, $1.5M, $2.3B, etc.
        self.dollar_pattern = re.compile(
            r'\$\s?(\d+(?:,\d{3})*(?:\.\d+)?)\s?([KMBTkmbt])?'
        )
        
        # Percentage: 10%, -5.2%, +15%, etc.
        self.percentage_pattern = re.compile(
            r'[+-]?\d+(?:\.\d+)?%'
        )
        
        # Price pattern: Trading at $150, price of $50k, etc.
        self.price_pattern = re.compile(
            r'(?:price|trading|valued?)\s+(?:at|of)?\s*\$?\s?(\d+[kKmMbBtT]?)',
            re.IGNORECASE
        )
        
        # Cashtag pattern: $BTC, $AAPL (Twitter style)
        self.cashtag_pattern = re.compile(r'\$([A-Z]{2,5})\b')
        
        # Person pattern: CEO, founder, by Name, etc.
        self.person_pattern = re.compile(
            r'\b(?:CEO|CTO|CFO|founder|president|chairman|analyst|trader|investor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            re.IGNORECASE
        )
    
    def _load_entities(self):
        """Load known entities dictionaries."""
        # Major cryptocurrencies with aliases
        self.crypto_map = {
            'btc': 'Bitcoin',
            'bitcoin': 'Bitcoin',
            'eth': 'Ethereum',
            'ethereum': 'Ethereum',
            'xrp': 'Ripple',
            'ripple': 'Ripple',
            'ada': 'Cardano',
            'cardano': 'Cardano',
            'sol': 'Solana',
            'solana': 'Solana',
            'doge': 'Dogecoin',
            'dogecoin': 'Dogecoin',
            'shib': 'Shiba Inu',
            'matic': 'Polygon',
            'polygon': 'Polygon',
            'dot': 'Polkadot',
            'polkadot': 'Polkadot',
            'avax': 'Avalanche',
            'avalanche': 'Avalanche',
            'link': 'Chainlink',
            'chainlink': 'Chainlink',
            'uni': 'Uniswap',
            'uniswap': 'Uniswap',
            'bnb': 'Binance Coin',
            'ltc': 'Litecoin',
            'bch': 'Bitcoin Cash',
            'xlm': 'Stellar',
            'trx': 'Tron',
            'atom': 'Cosmos',
            'algo': 'Algorand',
            'vet': 'VeChain',
            'ftm': 'Fantom',
            'sand': 'The Sandbox',
            'mana': 'Decentraland',
            'grt': 'The Graph',
            'aave': 'Aave',
            'mkr': 'Maker',
            'snx': 'Synthetix'
        }
        
        # Major stock tickers (tech companies commonly mentioned)
        self.stock_tickers = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Alphabet',
            'GOOG': 'Alphabet',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'NVDA': 'NVIDIA',
            'AMD': 'AMD',
            'NFLX': 'Netflix',
            'COIN': 'Coinbase',
            'SQ': 'Block',
            'PYPL': 'PayPal',
            'V': 'Visa',
            'MA': 'Mastercard',
            'JPM': 'JPMorgan',
            'BAC': 'Bank of America',
            'GS': 'Goldman Sachs',
            'MS': 'Morgan Stanley',
            'C': 'Citigroup',
            'WFC': 'Wells Fargo',
            'SPY': 'S&P 500 ETF',
            'QQQ': 'Nasdaq ETF',
            'DIA': 'Dow Jones ETF',
            'IWM': 'Russell 2000 ETF'
        }
        
        # Exchanges
        self.exchanges = {
            'binance', 'coinbase', 'kraken', 'gemini', 'bitstamp',
            'ftx', 'kucoin', 'huobi', 'okx', 'bitfinex', 'bybit',
            'nyse', 'nasdaq', 'chicago', 'cboe', 'cme'
        }
        
        # Financial terms
        self.financial_terms = {
            # Market movements
            'bull', 'bullish', 'bear', 'bearish', 'rally', 'crash',
            'dump', 'pump', 'moon', 'lambo', 'hodl', 'fud', 'fomo',
            'dip', 'correction', 'breakout', 'breakdown', 'reversal',
            
            # Trading terms
            'long', 'short', 'leverage', 'margin', 'liquidation',
            'position', 'entry', 'exit', 'profit', 'loss', 'pnl',
            'stop-loss', 'take-profit', 'resistance', 'support',
            
            # Technical indicators
            'rsi', 'macd', 'ema', 'sma', 'bollinger', 'fibonacci',
            'volume', 'volatility', 'momentum', 'trend',
            
            # Crypto-specific
            'defi', 'nft', 'dao', 'staking', 'mining', 'halving',
            'blockchain', 'wallet', 'exchange', 'token', 'coin',
            'altcoin', 'whale', 'airdrop', 'ath', 'atl',
            
            # Market cap
            'marketcap', 'market cap', 'mcap', 'valuation',
            
            # Sentiment
            'bullish', 'bearish', 'neutral', 'optimistic', 'pessimistic',
            'confident', 'uncertain', 'fearful', 'greedy'
        }
        
        # Currency symbols
        self.currencies = {
            'usd', 'eur', 'gbp', 'jpy', 'cny', 'chf', 'aud', 'cad',
            'dollar', 'euro', 'pound', 'yen', 'yuan'
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract all named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of Entity objects
        """
        entities = []
        
        # Extract different entity types
        entities.extend(self._extract_tickers(text))
        entities.extend(self._extract_cryptos(text))
        entities.extend(self._extract_exchanges(text))
        entities.extend(self._extract_financial_terms(text))
        entities.extend(self._extract_numbers(text))
        entities.extend(self._extract_percentages(text))
        
        # Use spaCy for person/organization extraction if available
        if self.use_spacy and self.nlp:
            entities.extend(self._extract_spacy_entities(text))
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda e: e.start)
        
        return entities
    
    def _extract_tickers(self, text: str) -> List[Entity]:
        """Extract stock ticker symbols."""
        entities = []
        
        # Extract cashtags ($AAPL style)
        for match in self.cashtag_pattern.finditer(text):
            ticker = match.group(1)
            # Check if it's a known stock ticker or crypto
            if ticker in self.stock_tickers or ticker in [k.upper() for k in self.crypto_map.keys()]:
                entity = Entity(
                    text=ticker,
                    type=EntityType.TICKER,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    metadata={'symbol': ticker}
                )
                if ticker in self.stock_tickers:
                    entity.metadata['company'] = self.stock_tickers[ticker]
                entities.append(entity)
        
        # Extract standalone tickers (careful to avoid false positives)
        words = text.split()
        for i, word in enumerate(words):
            # Clean word
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word.isupper() and 2 <= len(clean_word) <= 5:
                if clean_word in self.stock_tickers:
                    # Find position in original text
                    start = text.find(word)
                    if start != -1:
                        entity = Entity(
                            text=clean_word,
                            type=EntityType.TICKER,
                            start=start,
                            end=start + len(word),
                            confidence=0.85,
                            metadata={
                                'symbol': clean_word,
                                'company': self.stock_tickers[clean_word]
                            }
                        )
                        entities.append(entity)
        
        return entities
    
    def _extract_cryptos(self, text: str) -> List[Entity]:
        """Extract cryptocurrency mentions."""
        entities = []
        
        for match in self.crypto_pattern.finditer(text):
            crypto_text = match.group(1)
            crypto_normalized = self.crypto_map.get(crypto_text.lower(), crypto_text)
            
            entity = Entity(
                text=crypto_text,
                type=EntityType.CRYPTO,
                start=match.start(),
                end=match.end(),
                confidence=0.9,
                metadata={
                    'crypto': crypto_normalized,
                    'symbol': crypto_text.upper()
                }
            )
            entities.append(entity)
        
        return entities
    
    def _extract_exchanges(self, text: str) -> List[Entity]:
        """Extract exchange mentions."""
        entities = []
        text_lower = text.lower()
        
        for exchange in self.exchanges:
            pattern = r'\b' + re.escape(exchange) + r'\b'
            for match in re.finditer(pattern, text_lower):
                entity = Entity(
                    text=exchange.capitalize(),
                    type=EntityType.EXCHANGE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                    metadata={'exchange': exchange}
                )
                entities.append(entity)
        
        return entities
    
    def _extract_financial_terms(self, text: str) -> List[Entity]:
        """Extract financial terms and jargon."""
        entities = []
        text_lower = text.lower()
        
        for term in self.financial_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, text_lower):
                entity = Entity(
                    text=match.group(0),
                    type=EntityType.FINANCIAL_TERM,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                    metadata={'term': term}
                )
                entities.append(entity)
        
        return entities
    
    def _extract_numbers(self, text: str) -> List[Entity]:
        """Extract monetary amounts."""
        entities = []
        
        for match in self.dollar_pattern.finditer(text):
            amount = match.group(1).replace(',', '')
            multiplier = match.group(2)
            
            # Calculate actual value
            value = float(amount)
            if multiplier:
                mult_map = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
                value *= mult_map.get(multiplier.upper(), 1)
            
            entity = Entity(
                text=match.group(0),
                type=EntityType.NUMBER,
                start=match.start(),
                end=match.end(),
                confidence=0.95,
                metadata={
                    'value': value,
                    'formatted': match.group(0)
                }
            )
            entities.append(entity)
        
        return entities
    
    def _extract_percentages(self, text: str) -> List[Entity]:
        """Extract percentage values."""
        entities = []
        
        for match in self.percentage_pattern.finditer(text):
            percentage_str = match.group(0)
            # Extract numeric value
            value = float(re.search(r'[+-]?\d+(?:\.\d+)?', percentage_str).group())
            
            entity = Entity(
                text=percentage_str,
                type=EntityType.PERCENTAGE,
                start=match.start(),
                end=match.end(),
                confidence=0.95,
                metadata={
                    'value': value,
                    'direction': 'up' if value > 0 else 'down' if value < 0 else 'neutral'
                }
            )
            entities.append(entity)
        
        return entities
    
    def _extract_spacy_entities(self, text: str) -> List[Entity]:
        """Extract entities using spaCy NER."""
        entities = []
        
        if not self.nlp:
            return entities
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity_type = None
            
            if ent.label_ in ['PERSON']:
                entity_type = EntityType.PERSON
            elif ent.label_ in ['ORG', 'COMPANY']:
                entity_type = EntityType.COMPANY
            elif ent.label_ in ['MONEY']:
                entity_type = EntityType.NUMBER
            elif ent.label_ in ['PERCENT']:
                entity_type = EntityType.PERCENTAGE
            
            if entity_type:
                entity = Entity(
                    text=ent.text,
                    type=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8,
                    metadata={'spacy_label': ent.label_}
                )
                entities.append(entity)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities (overlapping spans)."""
        if not entities:
            return []
        
        # Sort by start position, then by confidence (descending)
        entities.sort(key=lambda e: (e.start, -e.confidence))
        
        deduplicated = []
        for entity in entities:
            # Check if this entity overlaps with any already added
            overlaps = False
            for existing in deduplicated:
                if (entity.start >= existing.start and entity.start < existing.end) or \
                   (entity.end > existing.start and entity.end <= existing.end):
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated
    
    def get_entity_summary(self, entities: List[Entity]) -> Dict[str, List[str]]:
        """
        Get summary of entities by type.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Dictionary mapping entity types to lists of entity texts
        """
        summary = {}
        for entity in entities:
            entity_type = entity.type.value
            if entity_type not in summary:
                summary[entity_type] = []
            summary[entity_type].append(entity.text)
        
        # Remove duplicates
        for entity_type in summary:
            summary[entity_type] = list(set(summary[entity_type]))
        
        return summary
    
    def tag_text(self, text: str) -> str:
        """
        Tag entities in text with XML-style tags.
        
        Args:
            text: Input text
            
        Returns:
            Text with entities tagged
        """
        entities = self.extract_entities(text)
        
        if not entities:
            return text
        
        # Sort entities by start position in reverse order
        entities.sort(key=lambda e: e.start, reverse=True)
        
        tagged_text = text
        for entity in entities:
            tag = entity.type.value.upper()
            tagged_text = (
                tagged_text[:entity.start] +
                f"<{tag}>{entity.text}</{tag}>" +
                tagged_text[entity.end:]
            )
        
        return tagged_text


def demo_ner():
    """Demonstration of NER capabilities."""
    print("=" * 80)
    print("FINANCIAL NER DEMONSTRATION")
    print("=" * 80)
    
    # Initialize NER
    ner = FinancialNER(use_spacy=True)
    
    # Test texts
    test_texts = [
        "Bitcoin is up 15% today! BTC breaking $50k resistance. 🚀",
        "$AAPL and $TSLA are rallying. Apple up 5.2%, Tesla gains 8%",
        "Ethereum crashed to $2000. ETH down -12% on Binance and Coinbase",
        "Bought 100 DOGE at $0.15. Dogecoin to the moon! 💎",
        "CEO Elon Musk tweeted about crypto. Market reacting bullish.",
        "Sold my $NVDA position for +25% profit. NVIDIA looking bearish.",
        "DeFi tokens pumping: UNI +18%, AAVE +22%, LINK +15%",
        "S&P 500 ($SPY) correlation with Bitcoin is increasing",
        "FUD spreading on Reddit. Whale dumped $2.5M worth of SOL"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text}")
        print("-" * 80)
        
        # Extract entities
        entities = ner.extract_entities(text)
        
        # Show entities by type
        summary = ner.get_entity_summary(entities)
        
        for entity_type, items in summary.items():
            print(f"  {entity_type.upper()}: {', '.join(items)}")
        
        # Show tagged text
        tagged = ner.tag_text(text)
        print(f"\n  Tagged: {tagged}")
    
    print("\n" + "=" * 80)
    print("NER Demo Complete!")


if __name__ == "__main__":
    demo_ner()
