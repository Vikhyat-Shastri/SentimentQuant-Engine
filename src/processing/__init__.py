"""
Text processing module for sentiment analysis.
Handles text cleaning, normalization, and named entity recognition.
"""

from .text_preprocessor import TextPreprocessor, ProcessedText
from .ner import FinancialNER, Entity, EntityType

__all__ = [
    'TextPreprocessor',
    'ProcessedText',
    'FinancialNER',
    'Entity',
    'EntityType'
]
