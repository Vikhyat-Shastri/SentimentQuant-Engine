"""
Market Regime Classification

Detects and classifies market regimes (bull, bear, sideways) using:
- Hidden Markov Models (HMM)
- K-means clustering
- Technical indicators
- Volatility analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeState:
    """Market regime state"""
    timestamp: datetime
    regime: str  # 'BULL', 'BEAR', 'SIDEWAYS'
    confidence: float
    indicators: Dict[str, float]
    volatility: float
    trend_strength: float


@dataclass
class RegimeTransition:
    """Regime change event"""
    timestamp: datetime
    from_regime: str
    to_regime: str
    confidence: float
    duration_previous: int  # days in previous regime


class MarketRegimeClassifier:
    """
    Classify market into regimes using multiple methods
    
    Methods:
    1. Technical indicators (trend + volatility)
    2. K-means clustering on price features
    3. Hidden Markov Model (if hmmlearn available)
    """
    
    def __init__(
        self,
        lookback_window: int = 50,
        regime_labels: Dict[int, str] = None
    ):
        """
        Initialize regime classifier
        
        Args:
            lookback_window: Days to look back for regime detection
            regime_labels: Mapping from cluster ID to regime name
        """
        self.lookback_window = lookback_window
        self.regime_labels = regime_labels or {
            0: 'SIDEWAYS',
            1: 'BULL',
            2: 'BEAR'
        }
        
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        
        # Try to import HMM
        try:
            from hmmlearn import hmm
            self.hmm_model = hmm.GaussianHMM(
                n_components=3,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            self.hmm_available = True
            logger.info("HMM available for regime detection")
        except ImportError:
            self.hmm_model = None
            self.hmm_available = False
            logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")
        
        logger.info(f"MarketRegimeClassifier initialized (lookback: {lookback_window} days)")
    
    def extract_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for regime classification
        
        Args:
            price_data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with extracted features
        """
        features = price_data.copy()
        
        # Returns
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        
        # Volatility (rolling std of returns)
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['volatility_50'] = features['returns'].rolling(50).std()
        
        # Trend indicators
        features['sma_20'] = features['close'].rolling(20).mean()
        features['sma_50'] = features['close'].rolling(50).mean()
        features['sma_200'] = features['close'].rolling(200).mean()
        
        # Price position relative to moving averages
        features['price_to_sma20'] = (features['close'] - features['sma_20']) / features['sma_20']
        features['price_to_sma50'] = (features['close'] - features['sma_50']) / features['sma_50']
        features['price_to_sma200'] = (features['close'] - features['sma_200']) / features['sma_200']
        
        # Moving average crossovers
        features['sma_20_50_cross'] = (features['sma_20'] - features['sma_50']) / features['sma_50']
        features['sma_50_200_cross'] = (features['sma_50'] - features['sma_200']) / features['sma_200']
        
        # Trend strength (ADX-like)
        features['trend_strength'] = self._calculate_trend_strength(features)
        
        # Volume indicators
        if 'volume' in features.columns:
            features['volume_sma'] = features['volume'].rolling(20).mean()
            features['volume_ratio'] = features['volume'] / features['volume_sma'].replace(0, 1)
        else:
            features['volume_ratio'] = 1.0
        
        # Momentum
        features['momentum_10'] = features['close'].pct_change(10)
        features['momentum_20'] = features['close'].pct_change(20)
        
        # High-Low range
        if 'high' in features.columns and 'low' in features.columns:
            features['hl_ratio'] = (features['high'] - features['low']) / features['close']
        else:
            features['hl_ratio'] = 0.0
        
        return features
    
    def _calculate_trend_strength(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate trend strength (simplified ADX)"""
        if 'high' not in data.columns or 'low' not in data.columns:
            return pd.Series(0, index=data.index)
        
        # True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional movement
        up_move = data['high'] - data['high'].shift()
        down_move = data['low'].shift() - data['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=data.index).rolling(period).mean()
        minus_dm = pd.Series(minus_dm, index=data.index).rolling(period).mean()
        
        # Directional indicators
        plus_di = 100 * plus_dm / atr.replace(0, 1)
        minus_di = 100 * minus_dm / atr.replace(0, 1)
        
        # ADX (trend strength)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def classify_technical(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify regime using technical indicators
        
        Logic:
        - BULL: Price > SMA50, SMA20 > SMA50, positive momentum
        - BEAR: Price < SMA50, SMA20 < SMA50, negative momentum
        - SIDEWAYS: Low trend strength, low volatility
        """
        features = self.extract_features(price_data)
        
        regimes = []
        
        for idx, row in features.iterrows():
            if pd.isna(row['sma_50']) or pd.isna(row['sma_20']):
                regime = 'SIDEWAYS'
                confidence = 0.5
            else:
                # Bull conditions
                bull_score = 0
                if row['close'] > row['sma_50']:
                    bull_score += 1
                if row['sma_20'] > row['sma_50']:
                    bull_score += 1
                if row['momentum_20'] > 0.05:  # 5% positive momentum
                    bull_score += 1
                if row['trend_strength'] > 25:  # Strong trend
                    bull_score += 1
                
                # Bear conditions
                bear_score = 0
                if row['close'] < row['sma_50']:
                    bear_score += 1
                if row['sma_20'] < row['sma_50']:
                    bear_score += 1
                if row['momentum_20'] < -0.05:  # 5% negative momentum
                    bear_score += 1
                if row['trend_strength'] > 25:  # Strong trend
                    bear_score += 1
                
                # Determine regime
                if bull_score >= 3:
                    regime = 'BULL'
                    confidence = bull_score / 4.0
                elif bear_score >= 3:
                    regime = 'BEAR'
                    confidence = bear_score / 4.0
                else:
                    regime = 'SIDEWAYS'
                    confidence = 1.0 - max(bull_score, bear_score) / 4.0
            
            regimes.append({
                'timestamp': idx,
                'regime': regime,
                'confidence': confidence,
                'volatility': row.get('volatility_20', 0),
                'trend_strength': row.get('trend_strength', 0)
            })
        
        return pd.DataFrame(regimes)
    
    def classify_kmeans(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify regime using K-means clustering
        
        Features: returns, volatility, momentum, trend
        """
        features = self.extract_features(price_data)
        
        # Select features for clustering
        feature_cols = [
            'returns', 'volatility_20', 'momentum_20',
            'price_to_sma50', 'trend_strength'
        ]
        
        # Drop NaN and prepare data
        X = features[feature_cols].dropna()
        
        if len(X) < 10:
            logger.warning("Insufficient data for K-means clustering")
            return pd.DataFrame()
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-means
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Assign regime labels based on cluster characteristics
        cluster_profiles = []
        for cluster_id in range(3):
            cluster_mask = clusters == cluster_id
            cluster_data = X[cluster_mask]
            
            profile = {
                'cluster_id': cluster_id,
                'avg_returns': cluster_data['returns'].mean(),
                'avg_volatility': cluster_data['volatility_20'].mean(),
                'avg_momentum': cluster_data['momentum_20'].mean()
            }
            cluster_profiles.append(profile)
        
        # Sort by returns to assign labels
        cluster_profiles.sort(key=lambda x: x['avg_returns'])
        
        label_mapping = {
            cluster_profiles[0]['cluster_id']: 'BEAR',    # Lowest returns
            cluster_profiles[1]['cluster_id']: 'SIDEWAYS', # Middle returns
            cluster_profiles[2]['cluster_id']: 'BULL'      # Highest returns
        }
        
        # Create results
        results = []
        for idx, (timestamp, cluster) in enumerate(zip(X.index, clusters)):
            regime = label_mapping[cluster]
            
            # Calculate confidence based on distance to cluster center
            center = self.kmeans.cluster_centers_[cluster]
            distance = np.linalg.norm(X_scaled[idx] - center)
            confidence = 1.0 / (1.0 + distance)
            
            results.append({
                'timestamp': timestamp,
                'regime': regime,
                'confidence': confidence,
                'cluster_id': int(cluster)
            })
        
        return pd.DataFrame(results)
    
    def classify_hmm(self, price_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Classify regime using Hidden Markov Model
        
        States: 3 hidden states mapped to regimes
        Observations: returns and volatility
        """
        if not self.hmm_available:
            logger.warning("HMM not available")
            return None
        
        features = self.extract_features(price_data)
        
        # Prepare observations (returns and volatility)
        obs_cols = ['returns', 'volatility_20']
        X = features[obs_cols].dropna()
        
        if len(X) < 20:
            logger.warning("Insufficient data for HMM")
            return None
        
        # Fit HMM
        try:
            X_array = X.values
            self.hmm_model.fit(X_array)
            
            # Predict hidden states
            states = self.hmm_model.predict(X_array)
            
            # Map states to regimes based on mean returns in each state
            state_profiles = []
            for state_id in range(3):
                state_mask = states == state_id
                state_returns = X['returns'][state_mask].mean()
                state_profiles.append((state_id, state_returns))
            
            state_profiles.sort(key=lambda x: x[1])
            
            state_mapping = {
                state_profiles[0][0]: 'BEAR',
                state_profiles[1][0]: 'SIDEWAYS',
                state_profiles[2][0]: 'BULL'
            }
            
            # Create results
            results = []
            for timestamp, state in zip(X.index, states):
                regime = state_mapping[state]
                
                # Confidence from state probability
                proba = self.hmm_model.predict_proba(X_array)
                confidence = proba[len(results), state]
                
                results.append({
                    'timestamp': timestamp,
                    'regime': regime,
                    'confidence': confidence,
                    'state_id': int(state)
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"HMM classification failed: {e}")
            return None
    
    def classify(
        self,
        price_data: pd.DataFrame,
        method: str = 'ensemble'
    ) -> pd.DataFrame:
        """
        Classify market regime
        
        Args:
            price_data: DataFrame with OHLCV data
            method: 'technical', 'kmeans', 'hmm', or 'ensemble'
        
        Returns:
            DataFrame with regime classifications
        """
        if method == 'technical':
            return self.classify_technical(price_data)
        elif method == 'kmeans':
            return self.classify_kmeans(price_data)
        elif method == 'hmm':
            result = self.classify_hmm(price_data)
            return result if result is not None else self.classify_technical(price_data)
        elif method == 'ensemble':
            return self.classify_ensemble(price_data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def classify_ensemble(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensemble classification using voting from multiple methods
        """
        # Get classifications from each method
        technical_results = self.classify_technical(price_data)
        kmeans_results = self.classify_kmeans(price_data)
        hmm_results = self.classify_hmm(price_data)
        
        # Merge results
        merged = technical_results.set_index('timestamp')
        
        if not kmeans_results.empty:
            kmeans_merged = kmeans_results.set_index('timestamp')[['regime', 'confidence']]
            kmeans_merged.columns = ['regime_kmeans', 'confidence_kmeans']
            merged = merged.join(kmeans_merged, how='left')
        
        if hmm_results is not None and not hmm_results.empty:
            hmm_merged = hmm_results.set_index('timestamp')[['regime', 'confidence']]
            hmm_merged.columns = ['regime_hmm', 'confidence_hmm']
            merged = merged.join(hmm_merged, how='left')
        
        # Voting
        ensemble_results = []
        for timestamp, row in merged.iterrows():
            votes = {'BULL': 0, 'BEAR': 0, 'SIDEWAYS': 0}
            confidences = []
            
            # Technical vote
            votes[row['regime']] += row['confidence']
            confidences.append(row['confidence'])
            
            # K-means vote
            if 'regime_kmeans' in row and pd.notna(row['regime_kmeans']):
                votes[row['regime_kmeans']] += row['confidence_kmeans']
                confidences.append(row['confidence_kmeans'])
            
            # HMM vote
            if 'regime_hmm' in row and pd.notna(row['regime_hmm']):
                votes[row['regime_hmm']] += row['confidence_hmm']
                confidences.append(row['confidence_hmm'])
            
            # Determine winner
            winning_regime = max(votes, key=votes.get)
            ensemble_confidence = np.mean(confidences)
            
            ensemble_results.append({
                'timestamp': timestamp,
                'regime': winning_regime,
                'confidence': ensemble_confidence,
                'volatility': row['volatility'],
                'trend_strength': row['trend_strength']
            })
        
        return pd.DataFrame(ensemble_results)
    
    def detect_transitions(self, regime_data: pd.DataFrame) -> List[RegimeTransition]:
        """Detect regime change transitions"""
        transitions = []
        
        for i in range(1, len(regime_data)):
            current = regime_data.iloc[i]
            previous = regime_data.iloc[i-1]
            
            if current['regime'] != previous['regime']:
                # Find duration of previous regime
                duration = 1
                for j in range(i-2, -1, -1):
                    if regime_data.iloc[j]['regime'] == previous['regime']:
                        duration += 1
                    else:
                        break
                
                transition = RegimeTransition(
                    timestamp=current['timestamp'],
                    from_regime=previous['regime'],
                    to_regime=current['regime'],
                    confidence=current['confidence'],
                    duration_previous=duration
                )
                transitions.append(transition)
        
        return transitions


# Test function
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', end='2024-10-01', freq='D')
    
    # Simulate price data with regime changes
    prices = []
    current_price = 100
    regime_periods = [
        ('BULL', 100, 0.02),    # Bull for 100 days, 2% daily drift
        ('SIDEWAYS', 80, 0.0),  # Sideways for 80 days
        ('BEAR', 80, -0.015),   # Bear for 80 days, -1.5% daily drift
        ('BULL', 40, 0.025)     # Bull for 40 days
    ]
    
    for regime, days, drift in regime_periods:
        for _ in range(min(days, len(dates) - len(prices))):
            returns = np.random.normal(drift, 0.02)
            current_price *= (1 + returns)
            prices.append(current_price)
    
    price_data = pd.DataFrame({
        'close': prices[:len(dates)],
        'high': np.array(prices[:len(dates)]) * 1.01,
        'low': np.array(prices[:len(dates)]) * 0.99,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Classify regimes
    classifier = MarketRegimeClassifier()
    
    print("\n=== Technical Classification ===")
    technical_regimes = classifier.classify(price_data, method='technical')
    print(technical_regimes.tail(10))
    
    print("\n=== K-means Classification ===")
    kmeans_regimes = classifier.classify(price_data, method='kmeans')
    print(kmeans_regimes.tail(10))
    
    print("\n=== Ensemble Classification ===")
    ensemble_regimes = classifier.classify(price_data, method='ensemble')
    print(ensemble_regimes.tail(10))
    
    # Detect transitions
    transitions = classifier.detect_transitions(ensemble_regimes)
    print(f"\n=== Detected {len(transitions)} Regime Transitions ===")
    for trans in transitions[:5]:
        print(f"{trans.timestamp.date()}: {trans.from_regime} → {trans.to_regime} "
              f"(confidence: {trans.confidence:.2f}, prev duration: {trans.duration_previous} days)")
