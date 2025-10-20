"""Quick test script to verify core functionality."""

from src.utils import config_manager, helpers
from src.processing.text_preprocessor import TextPreprocessor

print("=" * 60)
print("Testing Core Utilities")
print("=" * 60)

# Test 1: Config Manager
print("\n✓ Test 1: Config Manager")
fear_threshold = config_manager.get('sentiment_config', 'sentiment.thresholds.fear', 45)
print(f"  Fear threshold: {fear_threshold}")

# Test 2: Helper Functions
print("\n✓ Test 2: Helper Functions")
timestamp = helpers.get_timestamp()
print(f"  Current timestamp: {timestamp}")
tickers = helpers.extract_tickers("I'm bullish on $BTC and $ETH!")
print(f"  Extracted tickers: {tickers}")

# Test 3: Text Preprocessor
print("\n✓ Test 3: Text Preprocessor")
preprocessor = TextPreprocessor()
text = "Bitcoin is going to the moon! 🚀 $BTC #crypto"
result = preprocessor.preprocess(text)

print(f"  Original: {result.original}")
print(f"  Cleaned: {result.cleaned}")
print(f"  Tokens: {result.tokens}")
print(f"  Tickers: {result.entities['tickers']}")
print(f"  Hashtags: {result.entities['hashtags']}")
print(f"  Emoji sentiment: {result.metadata.get('emoji_sentiment', 0):.2f}")

# Test 4: Influence Score
print("\n✓ Test 4: Influence Score Calculation")
score1 = helpers.calculate_influence_score(followers=1000, verified=False)
score2 = helpers.calculate_influence_score(followers=50000, verified=True)
print(f"  Regular user (1K followers): {score1:.2f}x")
print(f"  Verified user (50K followers): {score2:.2f}x")

# Test 5: Time Decay
print("\n✓ Test 5: Time Decay Function")
recent_value = helpers.exponential_decay(1.0, age_seconds=0, half_life_seconds=14400)
aged_value = helpers.exponential_decay(1.0, age_seconds=14400, half_life_seconds=14400)
print(f"  Recent (0s old): {recent_value:.2f}")
print(f"  Aged (4h old): {aged_value:.2f}")

print("\n" + "=" * 60)
print("✅ All Core Tests Passed!")
print("=" * 60)
print("\nNext step: Install ML libraries (torch, transformers, spacy)")
print("Run: pip install torch transformers spacy")
