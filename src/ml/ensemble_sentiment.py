"""
Ensemble Sentiment Analysis

Combines multiple sentiment models for improved accuracy:
- FinBERT (financial domain)
- VADER (rule-based, fast)
- Twitter-RoBERTa (social media)
- TextBlob (simple baseline)

Ensemble strategies:
- Weighted voting
- Stacking (meta-learner)
- Confidence-based selection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class EnsembleSentimentResult:
    """Result from ensemble sentiment analysis"""
    text: str
    ensemble_score: float
    ensemble_label: str
    confidence: float
    individual_scores: Dict[str, float]
    individual_labels: Dict[str, str]
    method: str  # 'voting', 'weighted', 'stacking', 'confidence'


class EnsembleSentimentAnalyzer:
    """
    Ensemble sentiment analyzer combining multiple models
    
    Models:
    1. FinBERT - Financial domain expertise
    2. VADER - Fast rule-based
    3. Twitter-RoBERTa - Social media optimized
    4. TextBlob - Simple baseline
    """
    
    def __init__(
        self,
        models: List[str] = None,
        weights: Dict[str, float] = None,
        use_stacking: bool = False
    ):
        """
        Initialize ensemble analyzer
        
        Args:
            models: List of models to use (default: all available)
            weights: Model weights for weighted voting
            use_stacking: Use stacking meta-learner
        """
        self.available_models = ['finbert', 'vader', 'twitter-roberta', 'textblob']
        self.models = models or self.available_models
        
        # Default weights (can be tuned based on validation)
        self.weights = weights or {
            'finbert': 0.4,          # Highest weight for financial domain
            'vader': 0.2,            # Lower weight, but fast baseline
            'twitter-roberta': 0.3,  # Good for social media
            'textblob': 0.1          # Lowest weight, simple baseline
        }
        
        self.use_stacking = use_stacking
        
        # Initialize models
        self.model_instances = {}
        self._initialize_models()
        
        logger.info(f"EnsembleSentimentAnalyzer initialized with {len(self.model_instances)} models")
    
    def _initialize_models(self):
        """Initialize sentiment models"""
        # FinBERT
        if 'finbert' in self.models:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                model_name = "ProsusAI/finbert"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                self.model_instances['finbert'] = {
                    'tokenizer': tokenizer,
                    'model': model,
                    'type': 'transformer'
                }
                logger.info("✓ FinBERT loaded")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {e}")
        
        # VADER
        if 'vader' in self.models:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                
                analyzer = SentimentIntensityAnalyzer()
                self.model_instances['vader'] = {
                    'analyzer': analyzer,
                    'type': 'rule-based'
                }
                logger.info("✓ VADER loaded")
            except Exception as e:
                logger.warning(f"Failed to load VADER: {e}")
        
        # Twitter-RoBERTa
        if 'twitter-roberta' in self.models:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                
                model_name = "cardiffnlp/twitter-roberta-base-sentiment"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                self.model_instances['twitter-roberta'] = {
                    'tokenizer': tokenizer,
                    'model': model,
                    'type': 'transformer'
                }
                logger.info("✓ Twitter-RoBERTa loaded")
            except Exception as e:
                logger.warning(f"Failed to load Twitter-RoBERTa: {e}")
        
        # TextBlob
        if 'textblob' in self.models:
            try:
                from textblob import TextBlob
                
                self.model_instances['textblob'] = {
                    'analyzer': TextBlob,
                    'type': 'simple'
                }
                logger.info("✓ TextBlob loaded")
            except Exception as e:
                logger.warning(f"Failed to load TextBlob: {e}")
    
    def analyze_finbert(self, text: str) -> Tuple[float, str, float]:
        """
        Analyze with FinBERT
        
        Returns:
            (score, label, confidence)
        """
        if 'finbert' not in self.model_instances:
            return 0.0, 'neutral', 0.0
        
        try:
            import torch
            
            model_data = self.model_instances['finbert']
            tokenizer = model_data['tokenizer']
            model = model_data['model']
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT labels: [positive, negative, neutral]
            probs = probabilities[0].numpy()
            label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
            
            predicted_label_idx = np.argmax(probs)
            label = label_map[predicted_label_idx]
            confidence = float(probs[predicted_label_idx])
            
            # Convert to score [-1, 1]
            if label == 'positive':
                score = confidence
            elif label == 'negative':
                score = -confidence
            else:
                score = 0.0
            
            return score, label, confidence
            
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return 0.0, 'neutral', 0.0
    
    def analyze_vader(self, text: str) -> Tuple[float, str, float]:
        """
        Analyze with VADER
        
        Returns:
            (score, label, confidence)
        """
        if 'vader' not in self.model_instances:
            return 0.0, 'neutral', 0.0
        
        try:
            analyzer = self.model_instances['vader']['analyzer']
            scores = analyzer.polarity_scores(text)
            
            compound = scores['compound']
            
            # Label based on compound score
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Confidence is absolute value
            confidence = abs(compound)
            
            return compound, label, confidence
            
        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")
            return 0.0, 'neutral', 0.0
    
    def analyze_twitter_roberta(self, text: str) -> Tuple[float, str, float]:
        """
        Analyze with Twitter-RoBERTa
        
        Returns:
            (score, label, confidence)
        """
        if 'twitter-roberta' not in self.model_instances:
            return 0.0, 'neutral', 0.0
        
        try:
            import torch
            
            model_data = self.model_instances['twitter-roberta']
            tokenizer = model_data['tokenizer']
            model = model_data['model']
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Labels: [negative, neutral, positive]
            probs = probabilities[0].numpy()
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            
            predicted_label_idx = np.argmax(probs)
            label = label_map[predicted_label_idx]
            confidence = float(probs[predicted_label_idx])
            
            # Convert to score [-1, 1]
            if label == 'positive':
                score = confidence
            elif label == 'negative':
                score = -confidence
            else:
                score = 0.0
            
            return score, label, confidence
            
        except Exception as e:
            logger.error(f"Twitter-RoBERTa analysis failed: {e}")
            return 0.0, 'neutral', 0.0
    
    def analyze_textblob(self, text: str) -> Tuple[float, str, float]:
        """
        Analyze with TextBlob
        
        Returns:
            (score, label, confidence)
        """
        if 'textblob' not in self.model_instances:
            return 0.0, 'neutral', 0.0
        
        try:
            TextBlob = self.model_instances['textblob']['analyzer']
            blob = TextBlob(text)
            
            polarity = blob.sentiment.polarity  # [-1, 1]
            
            # Label
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Confidence
            confidence = abs(polarity)
            
            return polarity, label, confidence
            
        except Exception as e:
            logger.error(f"TextBlob analysis failed: {e}")
            return 0.0, 'neutral', 0.0
    
    def analyze_all_models(self, text: str) -> Dict[str, Tuple[float, str, float]]:
        """
        Run all available models
        
        Returns:
            Dict of model_name -> (score, label, confidence)
        """
        results = {}
        
        if 'finbert' in self.model_instances:
            results['finbert'] = self.analyze_finbert(text)
        
        if 'vader' in self.model_instances:
            results['vader'] = self.analyze_vader(text)
        
        if 'twitter-roberta' in self.model_instances:
            results['twitter-roberta'] = self.analyze_twitter_roberta(text)
        
        if 'textblob' in self.model_instances:
            results['textblob'] = self.analyze_textblob(text)
        
        return results
    
    def ensemble_voting(self, results: Dict[str, Tuple[float, str, float]]) -> EnsembleSentimentResult:
        """
        Simple majority voting ensemble
        
        Args:
            results: Dict of model results
        
        Returns:
            EnsembleSentimentResult
        """
        labels = [label for _, label, _ in results.values()]
        
        # Count votes
        label_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for label in labels:
            label_counts[label] += 1
        
        # Majority label
        ensemble_label = max(label_counts, key=label_counts.get)
        
        # Average score from models with same label
        matching_scores = [
            score for model, (score, label, _) in results.items()
            if label == ensemble_label
        ]
        ensemble_score = np.mean(matching_scores) if matching_scores else 0.0
        
        # Confidence based on agreement
        confidence = label_counts[ensemble_label] / len(labels)
        
        # Extract individual scores and labels
        individual_scores = {model: score for model, (score, _, _) in results.items()}
        individual_labels = {model: label for model, (_, label, _) in results.items()}
        
        return EnsembleSentimentResult(
            text="",
            ensemble_score=ensemble_score,
            ensemble_label=ensemble_label,
            confidence=confidence,
            individual_scores=individual_scores,
            individual_labels=individual_labels,
            method='voting'
        )
    
    def ensemble_weighted(self, results: Dict[str, Tuple[float, str, float]]) -> EnsembleSentimentResult:
        """
        Weighted average ensemble
        
        Args:
            results: Dict of model results
        
        Returns:
            EnsembleSentimentResult
        """
        weighted_score = 0.0
        total_weight = 0.0
        
        for model, (score, label, conf) in results.items():
            weight = self.weights.get(model, 1.0)
            weighted_score += score * weight
            total_weight += weight
        
        ensemble_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine label from ensemble score
        if ensemble_score > 0.05:
            ensemble_label = 'positive'
        elif ensemble_score < -0.05:
            ensemble_label = 'negative'
        else:
            ensemble_label = 'neutral'
        
        # Confidence based on score magnitude and model agreement
        confidence = min(abs(ensemble_score), 1.0)
        
        # Extract individual scores and labels
        individual_scores = {model: score for model, (score, _, _) in results.items()}
        individual_labels = {model: label for model, (_, label, _) in results.items()}
        
        return EnsembleSentimentResult(
            text="",
            ensemble_score=ensemble_score,
            ensemble_label=ensemble_label,
            confidence=confidence,
            individual_scores=individual_scores,
            individual_labels=individual_labels,
            method='weighted'
        )
    
    def ensemble_confidence(self, results: Dict[str, Tuple[float, str, float]]) -> EnsembleSentimentResult:
        """
        Confidence-based selection (pick most confident model)
        
        Args:
            results: Dict of model results
        
        Returns:
            EnsembleSentimentResult
        """
        # Find model with highest confidence
        best_model = None
        best_confidence = 0.0
        best_score = 0.0
        best_label = 'neutral'
        
        for model, (score, label, conf) in results.items():
            if conf > best_confidence:
                best_confidence = conf
                best_score = score
                best_label = label
                best_model = model
        
        # Extract individual scores and labels
        individual_scores = {model: score for model, (score, _, _) in results.items()}
        individual_labels = {model: label for model, (_, label, _) in results.items()}
        
        return EnsembleSentimentResult(
            text="",
            ensemble_score=best_score,
            ensemble_label=best_label,
            confidence=best_confidence,
            individual_scores=individual_scores,
            individual_labels=individual_labels,
            method=f'confidence_{best_model}'
        )
    
    def analyze(
        self,
        text: str,
        method: str = 'weighted'
    ) -> EnsembleSentimentResult:
        """
        Analyze text with ensemble
        
        Args:
            text: Text to analyze
            method: 'voting', 'weighted', or 'confidence'
        
        Returns:
            EnsembleSentimentResult
        """
        # Run all models
        results = self.analyze_all_models(text)
        
        if not results:
            logger.warning("No models available for analysis")
            return EnsembleSentimentResult(
                text=text,
                ensemble_score=0.0,
                ensemble_label='neutral',
                confidence=0.0,
                individual_scores={},
                individual_labels={},
                method='none'
            )
        
        # Apply ensemble method
        if method == 'voting':
            result = self.ensemble_voting(results)
        elif method == 'weighted':
            result = self.ensemble_weighted(results)
        elif method == 'confidence':
            result = self.ensemble_confidence(results)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        result.text = text
        return result


# Test function
if __name__ == "__main__":
    # Test texts
    test_texts = [
        "Bitcoin is soaring to new highs! Great investment opportunity! 🚀",
        "Market crash incoming. Sell everything before it's too late.",
        "BTC trading sideways, consolidating at $45k support.",
        "Ethereum network upgrade successful, community excited about future prospects."
    ]
    
    # Initialize ensemble (only use available models)
    try:
        analyzer = EnsembleSentimentAnalyzer(
            models=['vader', 'textblob'],  # Start with simple models
            weights={'vader': 0.6, 'textblob': 0.4}
        )
        
        print("\n" + "="*80)
        print("ENSEMBLE SENTIMENT ANALYSIS")
        print("="*80)
        
        for text in test_texts:
            print(f"\nText: {text}")
            print("-" * 80)
            
            # Weighted ensemble
            result = analyzer.analyze(text, method='weighted')
            
            print(f"Ensemble Score: {result.ensemble_score:.3f}")
            print(f"Ensemble Label: {result.ensemble_label}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Method: {result.method}")
            
            print("\nIndividual Models:")
            for model, score in result.individual_scores.items():
                label = result.individual_labels[model]
                print(f"  {model:15s}: {score:+.3f} ({label})")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Install missing dependencies:")
        print("  pip install transformers vaderSentiment textblob")
