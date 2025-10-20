"""
Predictive Price Modeling

Uses sentiment signals and historical price data to predict future price movements.
Implements LSTM/GRU models for time series forecasting.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class PricePrediction:
    """Price prediction result"""
    symbol: str
    current_price: float
    predicted_price: float
    predicted_change: float  # Percentage change
    confidence: float
    prediction_horizon: str  # e.g., "1h", "4h", "24h"
    features_used: Dict[str, float]


class LSTMPricePredictor(nn.Module):
    """
    LSTM-based price prediction model
    
    Features:
    - Sentiment scores (multiple time aggregations)
    - Price features (returns, volatility, volume)
    - Technical indicators (RSI, MACD, Bollinger Bands)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Predict price change percentage
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
        
        Returns:
            Predicted price change percentage
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        prediction = self.fc(last_output)
        
        return prediction


class PricePredictor:
    """
    Main price prediction engine
    
    Combines sentiment signals with market data to predict future prices.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        sequence_length: int = 24,  # Look back 24 hours
        prediction_horizons: List[str] = None
    ):
        """
        Initialize price predictor
        
        Args:
            model_path: Path to saved model weights
            sequence_length: Number of historical timesteps to use
            prediction_horizons: Time horizons to predict (e.g., ["1h", "4h", "24h"])
        """
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons or ["1h", "4h", "24h"]
        
        # Feature configuration
        self.feature_columns = [
            'sentiment_1h', 'sentiment_4h', 'sentiment_24h',  # Sentiment aggregations
            'returns_1h', 'returns_4h', 'returns_24h',  # Price returns
            'volatility_24h',  # Volatility
            'volume_ratio',  # Volume relative to average
            'rsi_14',  # RSI indicator
            'macd', 'macd_signal',  # MACD indicators
            'bb_position'  # Position within Bollinger Bands
        ]
        
        self.input_size = len(self.feature_columns)
        
        # Initialize models for each prediction horizon
        self.models = {}
        for horizon in self.prediction_horizons:
            model = LSTMPricePredictor(
                input_size=self.input_size,
                hidden_size=128,
                num_layers=2,
                dropout=0.2
            )
            
            if model_path:
                model_file = model_path / f"price_predictor_{horizon}.pth"
                if model_file.exists():
                    model.load_state_dict(torch.load(model_file))
                    logger.info(f"Loaded model for {horizon} from {model_file}")
            
            model.eval()
            self.models[horizon] = model
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for model in self.models.values():
            model.to(self.device)
        
        logger.info(f"PricePredictor initialized with {len(self.models)} models")
        logger.info(f"Device: {self.device}")
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prepare features for prediction
        
        Args:
            df: Price data with OHLCV columns
            sentiment_data: Optional sentiment data with timestamps
        
        Returns:
            DataFrame with engineered features
        """
        features = df.copy()
        
        # Price returns
        features['returns_1h'] = df['close'].pct_change(1)
        features['returns_4h'] = df['close'].pct_change(4)
        features['returns_24h'] = df['close'].pct_change(24)
        
        # Volatility
        features['volatility_24h'] = df['close'].pct_change().rolling(24).std()
        
        # Volume ratio
        avg_volume = df['volume'].rolling(24).mean()
        features['volume_ratio'] = df['volume'] / avg_volume.replace(0, 1)
        
        # RSI
        features['rsi_14'] = self._calculate_rsi(df['close'], period=14)
        
        # MACD
        macd, signal = self._calculate_macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        
        # Bollinger Bands position
        features['bb_position'] = self._calculate_bb_position(df['close'])
        
        # Sentiment features
        if sentiment_data is not None:
            # Merge sentiment data
            sentiment_agg = self._aggregate_sentiment(sentiment_data)
            features = features.join(sentiment_agg, how='left')
        else:
            # Use zeros if no sentiment data
            features['sentiment_1h'] = 0.0
            features['sentiment_4h'] = 0.0
            features['sentiment_24h'] = 0.0
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line"""
        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period).mean()
        
        return macd, signal
    
    def _calculate_bb_position(
        self,
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> pd.Series:
        """
        Calculate position within Bollinger Bands
        
        Returns value between 0 (lower band) and 1 (upper band)
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        band_width = (upper_band - lower_band).replace(0, 1)
        position = (prices - lower_band) / band_width
        
        return position.clip(0, 1)
    
    def _aggregate_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment data over different time windows
        
        Args:
            sentiment_df: DataFrame with timestamp and sentiment columns
        
        Returns:
            DataFrame with aggregated sentiment
        """
        if 'timestamp' not in sentiment_df.columns:
            return pd.DataFrame()
        
        sentiment_df = sentiment_df.set_index('timestamp')
        
        # Aggregate over different windows
        agg_1h = sentiment_df['sentiment'].rolling('1H').mean()
        agg_4h = sentiment_df['sentiment'].rolling('4H').mean()
        agg_24h = sentiment_df['sentiment'].rolling('24H').mean()
        
        result = pd.DataFrame({
            'sentiment_1h': agg_1h,
            'sentiment_4h': agg_4h,
            'sentiment_24h': agg_24h
        })
        
        return result
    
    def predict(
        self,
        price_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame] = None,
        symbol: str = "BTC-USD"
    ) -> List[PricePrediction]:
        """
        Make price predictions for all horizons
        
        Args:
            price_data: Historical price data
            sentiment_data: Optional sentiment data
            symbol: Trading symbol
        
        Returns:
            List of predictions for each horizon
        """
        # Prepare features
        features = self.prepare_features(price_data, sentiment_data)
        
        # Get latest sequence
        if len(features) < self.sequence_length:
            logger.warning(f"Insufficient data: {len(features)} < {self.sequence_length}")
            return []
        
        latest_sequence = features[self.feature_columns].iloc[-self.sequence_length:]
        
        # Convert to tensor
        X = torch.FloatTensor(latest_sequence.values).unsqueeze(0).to(self.device)
        
        # Current price
        current_price = price_data['close'].iloc[-1]
        
        # Make predictions for each horizon
        predictions = []
        
        with torch.no_grad():
            for horizon, model in self.models.items():
                # Predict price change percentage
                pred_change = model(X).item()
                
                # Calculate predicted price
                predicted_price = current_price * (1 + pred_change / 100)
                
                # Confidence based on model certainty (simplified)
                confidence = 1.0 - min(abs(pred_change) / 10.0, 0.5)  # Placeholder
                
                # Get feature values for transparency
                features_dict = {
                    col: float(latest_sequence[col].iloc[-1])
                    for col in self.feature_columns[:5]  # Top 5 features
                }
                
                prediction = PricePrediction(
                    symbol=symbol,
                    current_price=current_price,
                    predicted_price=predicted_price,
                    predicted_change=pred_change,
                    confidence=confidence,
                    prediction_horizon=horizon,
                    features_used=features_dict
                )
                
                predictions.append(prediction)
        
        return predictions


# Test function
if __name__ == "__main__":
    # Test with random data
    predictor = PricePredictor()
    
    # Generate random price data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1H')
    price_data = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.randn(100).cumsum() + 50000,
        'volume': np.random.rand(100) * 1000000
    })
    
    # Generate random sentiment data
    sentiment_data = pd.DataFrame({
        'timestamp': dates,
        'sentiment': np.random.randn(100) * 0.3
    })
    
    # Make predictions
    predictions = predictor.predict(price_data, sentiment_data, symbol="BTC-USD")
    
    for pred in predictions:
        print(f"\n{pred.prediction_horizon} Prediction:")
        print(f"  Current: ${pred.current_price:.2f}")
        print(f"  Predicted: ${pred.predicted_price:.2f}")
        print(f"  Change: {pred.predicted_change:+.2f}%")
        print(f"  Confidence: {pred.confidence:.2f}")
