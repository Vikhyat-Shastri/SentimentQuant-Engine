"""
Deep Learning models for sentiment analysis.

Supports:
- BERT-based models (FinBERT for financial sentiment)
- RoBERTa (twitter-roberta-base-sentiment)
- DistilBERT (lightweight, faster)
- LSTM with GloVe embeddings (custom training)
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    FINBERT = "ProsusAI/finbert"  # Financial sentiment (3-class)
    TWITTER_ROBERTA = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Twitter sentiment
    DISTILBERT = "distilbert-base-uncased-finetuned-sst-2-english"  # General sentiment
    CRYPTO_BERT = "ElKulako/cryptobert"  # Crypto-specific sentiment
    

class SentimentModel:
    """
    Deep learning sentiment analysis model.
    
    Uses pre-trained transformers for financial/crypto sentiment analysis.
    Much more accurate than VADER for domain-specific text.
    """
    
    def __init__(
        self, 
        model_type: ModelType = ModelType.FINBERT,
        device: str = "auto",
        cache_dir: str = "models/cache"
    ):
        """
        Initialize sentiment model.
        
        Args:
            model_type: Type of model to use
            device: Device to run on ('cpu', 'cuda', or 'auto')
            cache_dir: Directory to cache models
        """
        self.model_type = model_type
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing {model_type.value} on {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
        
        logger.info(f"Model loaded successfully: {model_type.name}")
    
    def _load_model(self):
        """Load pre-trained model and tokenizer."""
        try:
            model_name = self.model_type.value
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def analyze(
        self, 
        text: str,
        return_all_scores: bool = False
    ) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            return_all_scores: Return scores for all classes
        
        Returns:
            Dictionary with sentiment analysis results:
            - score: Normalized sentiment score (-1 to 1)
            - confidence: Confidence of prediction (0 to 1)
            - label: Predicted label (positive/negative/neutral)
            - raw_scores: Raw model outputs (if return_all_scores=True)
        """
        if not text or not text.strip():
            return {
                'score': 0.0,
                'confidence': 0.0,
                'label': 'neutral',
                'raw_scores': {}
            }
        
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length * 4:  # Rough estimate
                text = text[:max_length * 4]
            
            # Get prediction
            result = self.pipeline(text, truncation=True, max_length=max_length)[0]
            
            # Extract label and confidence
            label = result['label'].lower()
            confidence = result['score']
            
            # Normalize to -1 to 1 scale
            score = self._normalize_score(label, confidence)
            
            output = {
                'score': score,
                'confidence': confidence,
                'label': label,
                'model': self.model_type.name
            }
            
            if return_all_scores:
                # Get all class scores
                with torch.no_grad():
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding=True
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                    raw_scores = {}
                    for idx, prob in enumerate(probs[0].cpu().numpy()):
                        raw_scores[f"class_{idx}"] = float(prob)
                    
                    output['raw_scores'] = raw_scores
            
            return output
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'label': 'neutral',
                'error': str(e)
            }
    
    def _normalize_score(self, label: str, confidence: float) -> float:
        """
        Normalize model output to -1 to 1 scale.
        
        Args:
            label: Predicted label
            confidence: Model confidence
        
        Returns:
            Normalized score (-1 to 1)
        """
        # Handle different label formats
        label_lower = label.lower()
        
        if 'positive' in label_lower or label_lower == 'pos':
            return confidence
        elif 'negative' in label_lower or label_lower == 'neg':
            return -confidence
        elif 'neutral' in label_lower:
            return 0.0
        else:
            # Unknown label, try to infer from confidence
            return confidence if confidence > 0.5 else -confidence
    
    def analyze_batch(
        self, 
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """
        Analyze multiple texts efficiently.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
        
        Returns:
            List of sentiment results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                batch_results = self.pipeline(
                    batch,
                    truncation=True,
                    max_length=512,
                    batch_size=batch_size
                )
                
                for result in batch_results:
                    label = result['label'].lower()
                    confidence = result['score']
                    score = self._normalize_score(label, confidence)
                    
                    results.append({
                        'score': score,
                        'confidence': confidence,
                        'label': label,
                        'model': self.model_type.name
                    })
            
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                # Return neutral for failed items
                for _ in batch:
                    results.append({
                        'score': 0.0,
                        'confidence': 0.0,
                        'label': 'neutral',
                        'error': str(e)
                    })
        
        return results
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding from model.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        """
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                outputs = self.model.base_model(**inputs)
                
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                return embedding.flatten()
        
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros(768)  # Return zero vector


class EnsembleModel:
    """
    Ensemble of multiple sentiment models for robust predictions.
    
    Combines predictions from multiple models using weighted averaging.
    """
    
    def __init__(
        self,
        models: List[SentimentModel],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of SentimentModel instances
            weights: Weights for each model (normalized to sum to 1)
        """
        self.models = models
        
        if weights is None:
            # Equal weights
            weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        self.weights = weights
        
        logger.info(f"Initialized ensemble with {len(models)} models")
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using ensemble.
        
        Args:
            text: Input text
        
        Returns:
            Aggregated sentiment results
        """
        results = []
        
        for model in self.models:
            result = model.analyze(text)
            results.append(result)
        
        # Weighted average of scores
        score = sum(r['score'] * w for r, w in zip(results, self.weights))
        
        # Average confidence
        confidence = sum(r['confidence'] * w for r, w in zip(results, self.weights))
        
        # Majority vote for label
        labels = [r['label'] for r in results]
        label = max(set(labels), key=labels.count)
        
        return {
            'score': score,
            'confidence': confidence,
            'label': label,
            'model': 'ensemble',
            'individual_results': results
        }


# Global model cache to avoid reloading
_model_cache: Dict[ModelType, SentimentModel] = {}


def get_model(
    model_type: ModelType = ModelType.FINBERT,
    device: str = "auto"
) -> SentimentModel:
    """
    Get or create sentiment model (with caching).
    
    Args:
        model_type: Type of model
        device: Device to run on
    
    Returns:
        SentimentModel instance
    """
    cache_key = (model_type, device)
    
    if cache_key not in _model_cache:
        _model_cache[cache_key] = SentimentModel(model_type, device)
    
    return _model_cache[cache_key]
