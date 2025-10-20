"""
Signal Generation Module

This module provides trading signal generation based on sentiment analysis.
It converts Fear & Greed Index and sentiment scores into actionable trading signals.
"""

from .signal_generator import SignalGenerator

__all__ = ['SignalGenerator']
