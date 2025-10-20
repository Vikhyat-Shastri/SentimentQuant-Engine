"""
Utility functions for the sentiment analysis system.
Common helper functions used across modules.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import numpy as np


def get_timestamp() -> float:
    """
    Get current Unix timestamp.
    
    Returns:
        Current Unix timestamp (seconds since epoch)
    """
    return datetime.now(timezone.utc).timestamp()


def timestamp_to_datetime(timestamp: float) -> datetime:
    """
    Convert Unix timestamp to datetime object.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Datetime object in UTC
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def hash_text(text: str) -> str:
    """
    Generate hash of text for deduplication.
    
    Args:
        text: Text to hash
        
    Returns:
        MD5 hash of text
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove null bytes
    text = text.replace('\x00', '')
    return text.strip()


def extract_tickers(text: str) -> List[str]:
    """
    Extract cryptocurrency/stock tickers from text.
    
    Args:
        text: Text to extract tickers from
        
    Returns:
        List of found tickers (e.g., ['BTC', 'ETH'])
    """
    # Pattern for $TICKER or common crypto names
    ticker_pattern = r'\$([A-Z]{2,5})\b'
    tickers = re.findall(ticker_pattern, text.upper())
    
    # Common crypto keywords
    crypto_map = {
        'BITCOIN': 'BTC',
        'ETHEREUM': 'ETH',
        'BINANCE': 'BNB',
        'CARDANO': 'ADA',
        'SOLANA': 'SOL',
        'RIPPLE': 'XRP',
        'DOGECOIN': 'DOGE',
        'POLKADOT': 'DOT'
    }
    
    text_upper = text.upper()
    for keyword, ticker in crypto_map.items():
        if keyword in text_upper:
            tickers.append(ticker)
    
    # Remove duplicates and return
    return list(set(tickers))


def calculate_influence_score(
    followers: int = 0,
    karma: int = 0,
    verified: bool = False,
    log_base: int = 10
) -> float:
    """
    Calculate user influence score based on social metrics.
    
    Args:
        followers: Number of followers (Twitter)
        karma: Reddit karma score
        verified: Whether user is verified
        log_base: Base for logarithmic scaling
        
    Returns:
        Influence multiplier (1.0 = baseline)
    """
    base_score = 1.0
    
    # Verified accounts get bonus
    if verified:
        base_score *= 1.2
    
    # Logarithmic scaling of followers/karma
    if followers > 100:
        follower_score = np.log(followers) / np.log(log_base)
        base_score *= (1.0 + follower_score * 0.1)
    
    if karma > 100:
        karma_score = np.log(karma) / np.log(log_base)
        base_score *= (1.0 + karma_score * 0.1)
    
    # Cap maximum influence
    return min(base_score, 3.0)


def exponential_decay(
    value: float,
    age_seconds: float,
    half_life_seconds: float = 14400.0  # 4 hours default
) -> float:
    """
    Apply exponential time decay to a value.
    
    Args:
        value: Original value
        age_seconds: Age of value in seconds
        half_life_seconds: Half-life for decay
        
    Returns:
        Decayed value
    """
    if age_seconds <= 0:
        return value
    
    decay_factor = np.exp(-np.log(2) * age_seconds / half_life_seconds)
    return value * decay_factor


def calculate_moving_average(
    values: List[float],
    window: int
) -> float:
    """
    Calculate simple moving average.
    
    Args:
        values: List of values
        window: Window size
        
    Returns:
        Moving average of last 'window' values
    """
    if not values:
        return 0.0
    
    recent_values = values[-window:]
    return np.mean(recent_values)


def calculate_momentum(
    values: List[float],
    period: int = 14
) -> float:
    """
    Calculate momentum (rate of change).
    
    Args:
        values: List of values
        period: Period for momentum calculation
        
    Returns:
        Momentum value
    """
    if len(values) < period + 1:
        return 0.0
    
    current = values[-1]
    previous = values[-(period + 1)]
    
    if previous == 0:
        return 0.0
    
    return (current - previous) / abs(previous)


def normalize_score(
    value: float,
    min_val: float = -1.0,
    max_val: float = 1.0,
    target_min: float = 0.0,
    target_max: float = 100.0
) -> float:
    """
    Normalize value to target range.
    
    Args:
        value: Value to normalize
        min_val: Minimum of input range
        max_val: Maximum of input range
        target_min: Minimum of target range
        target_max: Maximum of target range
        
    Returns:
        Normalized value
    """
    # Clamp value to input range
    value = max(min_val, min(max_val, value))
    
    # Normalize to 0-1
    normalized = (value - min_val) / (max_val - min_val)
    
    # Scale to target range
    return target_min + normalized * (target_max - target_min)


def weighted_average(
    values: List[float],
    weights: List[float]
) -> float:
    """
    Calculate weighted average.
    
    Args:
        values: List of values
        weights: List of weights (same length as values)
        
    Returns:
        Weighted average
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight


def detect_outliers(
    values: List[float],
    threshold: float = 3.0
) -> List[bool]:
    """
    Detect outliers using Z-score method.
    
    Args:
        values: List of values
        threshold: Z-score threshold for outlier
        
    Returns:
        Boolean list indicating outliers
    """
    if len(values) < 3:
        return [False] * len(values)
    
    values_array = np.array(values)
    mean = np.mean(values_array)
    std = np.std(values_array)
    
    if std == 0:
        return [False] * len(values)
    
    z_scores = np.abs((values_array - mean) / std)
    return (z_scores > threshold).tolist()


def format_large_number(number: float) -> str:
    """
    Format large numbers for display (e.g., 1.5M, 2.3K).
    
    Args:
        number: Number to format
        
    Returns:
        Formatted string
    """
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.1f}K"
    else:
        return f"{number:.0f}"


def calculate_percentile(
    value: float,
    values: List[float]
) -> float:
    """
    Calculate percentile rank of value in list.
    
    Args:
        value: Value to rank
        values: List of values
        
    Returns:
        Percentile (0-100)
    """
    if not values:
        return 50.0
    
    values_sorted = sorted(values)
    rank = sum(1 for v in values_sorted if v <= value)
    return (rank / len(values)) * 100
