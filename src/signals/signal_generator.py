"""
Signal Generator Module

Generates trading signals based on sentiment analysis and Fear & Greed Index.
Provides BUY/SELL/HOLD recommendations with confidence scores and position sizing.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import yaml
import pandas as pd

from src.sentiment.fear_greed_index import FearGreedIndexCalculator

try:
    from src.correlation.fund_flow import FundFlowAnalyzer
    FUND_FLOW_AVAILABLE = True
except ImportError:
    FUND_FLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class SignalAction(Enum):
    """Trading signal actions"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    timestamp: datetime
    symbol: str
    action: SignalAction
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    position_size: float  # 0.0 to 1.0 (percentage of portfolio)
    fear_greed_index: float
    sentiment_score: float
    reasoning: str
    metadata: Dict


class SignalGenerator:
    """
    Generates trading signals from sentiment data.
    
    Signal Generation Logic:
    - FGI < 25: STRONG_BUY (extreme fear = opportunity)
    - FGI 25-40: BUY (fear = buy the dip)
    - FGI 40-60: HOLD (neutral = wait)
    - FGI 60-75: SELL (greed = take profits)
    - FGI > 75: STRONG_SELL (extreme greed = danger)
    
    Position Sizing:
    - Based on signal strength and confidence
    - Higher confidence = larger position
    - Risk management with max position limits
    """
    
    def __init__(self, config_path: str = "config/signal_config.yaml"):
        """
        Initialize signal generator.
        
        Args:
            config_path: Path to signal configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize Fear & Greed Index calculator
        self.fgi_calculator = FearGreedIndexCalculator()
        
        # Initialize Fund Flow Analyzer if available
        if FUND_FLOW_AVAILABLE:
            try:
                self.fund_flow_analyzer = FundFlowAnalyzer()
                logger.info("Fund Flow Analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Fund Flow Analyzer: {e}")
                self.fund_flow_analyzer = None
        else:
            self.fund_flow_analyzer = None
        
        # Statistics
        self.signals_generated = 0
        self.last_signal_time = None
        self.signal_history: List[TradingSignal] = []
        self.max_history = 100
        
        # Thread control
        self._running = False
        self._thread = None
        
        # Benchmark reference (set externally if benchmarking)
        self.benchmark = None
        
        logger.info("SignalGenerator initialized")
    
    def _load_config(self) -> Dict:
        """Load signal configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract signals section if it exists
            if 'signals' in config:
                config = config['signals']
            
            # Ensure we have the default structure if config is incomplete
            default_config = self._get_default_config()
            
            # Merge with defaults (config takes precedence)
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            
            logger.info(f"Loaded signal config from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'thresholds': {
                'extreme_fear': 25,
                'fear': 40,
                'neutral': 60,
                'greed': 75
            },
            'position_sizing': {
                'min_position': 0.02,  # 2% minimum (reduced from 5%)
                'max_position': 0.10,  # 10% maximum (reduced from 25%)
                'base_position': 0.05  # 5% base (reduced from 10%)
            },
            'confidence': {
                'min_confidence': 0.75,  # INCREASED from 0.3 to 0.75 - only take high-confidence trades
                'high_confidence': 0.85   # INCREASED from 0.7 to 0.85 - stricter quality filter
            },
            'risk_management': {
                'max_daily_trades': 5,     # REDUCED from 10 to 5 - fewer trades
                'cooldown_minutes': 60     # INCREASED from 5 to 60 - 1 hour between trades
            }
        }
    
    def generate_signal(self, sentiment_data) -> Optional[TradingSignal]:
        """
        Generate trading signal from sentiment data.
        
        Args:
            sentiment_data: DataPacket with sentiment data OR dictionary containing:
                - fear_greed_index: float (0-100)
                - sentiment_score: float (-1 to 1)
                - symbol: str
                - source_count: int
                - metadata: dict
        
        Returns:
            TradingSignal or None if no signal generated
        """
        try:
            # Handle DataPacket objects (from actual pipeline)
            if hasattr(sentiment_data, 'data'):
                data = sentiment_data.data
                sentiment_score = data.get('sentiment_score', 0)
                symbol = data.get('asset', 'BTC-USD')
                source_count = 1  # From single packet
                
                # Check if FGI is pre-calculated, otherwise calculate it
                if 'fear_greed_index' in data:
                    fgi = data['fear_greed_index']
                else:
                    # Calculate Fear & Greed Index from sentiment score
                    # Map sentiment score (-1 to 1) to FGI (0 to 100)
                    # -1 (fear) => 0, 0 (neutral) => 50, 1 (greed) => 100
                    fgi = ((sentiment_score + 1) / 2) * 100
                
            # Handle dictionary objects (from tests and backtesting)
            else:
                data = sentiment_data
                # Use provided FGI if available, otherwise calculate from sentiment
                if 'fear_greed_index' in data:
                    fgi = data['fear_greed_index']
                else:
                    sentiment_score = data.get('sentiment_score', 0)
                    fgi = ((sentiment_score + 1) / 2) * 100
                    
                sentiment_score = data.get('sentiment_score', 0)
                symbol = data.get('symbol', data.get('asset', 'BTC-USD'))
                source_count = data.get('source_count', 1)
            
            # Determine signal action based on Fear & Greed Index
            action, base_strength = self._determine_action(fgi)
            
            # Calculate confidence based on sentiment consistency
            confidence = self._calculate_confidence(fgi, sentiment_score, data)
            
            # Integrate fund flow analysis if available (disabled for backtesting to avoid API rate limits)
            flow_signal = None
            flow_metrics = None
            if False and self.fund_flow_analyzer is not None:  # Temporarily disabled
                try:
                    # Extract symbol without exchange suffix (e.g., BTC-USD -> BTC)
                    base_symbol = symbol.split('-')[0] if '-' in symbol else symbol
                    
                    # Get fund flow signal
                    flow_signal = self.fund_flow_analyzer.get_flow_signal(base_symbol)
                    flow_metrics = flow_signal.get('metrics')
                    
                    # Enhance confidence with flow signal strength
                    flow_strength = flow_signal.get('signal_strength', 0.5)
                    confidence = confidence * 0.6 + flow_strength * 0.4
                    
                    # Adjust action if flow signal strongly contradicts sentiment
                    flow_rec = flow_signal.get('recommendation', 'NEUTRAL')
                    if flow_rec in ['STRONG_BUY', 'BUY'] and action in [SignalAction.SELL, SignalAction.STRONG_SELL]:
                        # Flow suggests buying but sentiment suggests selling
                        if flow_strength > 0.7:  # Strong flow signal
                            action = SignalAction.HOLD  # Downgrade to hold
                            logger.info(f"Flow signal adjusted action from {action.value} to HOLD")
                    elif flow_rec in ['STRONG_SELL', 'SELL'] and action in [SignalAction.BUY, SignalAction.STRONG_BUY]:
                        # Flow suggests selling but sentiment suggests buying
                        if flow_strength > 0.7:  # Strong flow signal
                            action = SignalAction.HOLD  # Downgrade to hold
                            logger.info(f"Flow signal adjusted action from {action.value} to HOLD")
                    
                except Exception as e:
                    logger.warning(f"Failed to get fund flow signal: {e}")
                    flow_signal = None
            
            # Skip signal if confidence too low
            min_confidence = self.config['confidence']['min_confidence']
            if confidence < min_confidence:
                logger.debug(f"Signal confidence {confidence:.2f} below threshold {min_confidence}")
                return None
            
            # Calculate position size
            position_size = self._calculate_position_size(base_strength, confidence)
            
            # Generate reasoning (include flow metrics)
            reasoning = self._generate_reasoning(action, fgi, sentiment_score, confidence, flow_signal)
            
            # Create signal
            signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                strength=base_strength,
                confidence=confidence,
                position_size=position_size,
                fear_greed_index=fgi,
                sentiment_score=sentiment_score,
                reasoning=reasoning,
                metadata=data.get('metadata', {})
            )
            
            # Update statistics
            self.signals_generated += 1
            self.last_signal_time = datetime.now()
            
            # Add to history (keep limited size)
            self.signal_history.append(signal)
            if len(self.signal_history) > self.max_history:
                self.signal_history.pop(0)
            
            logger.info(
                f"Generated signal: {action.value} for {symbol} "
                f"(FGI: {fgi:.1f}, Confidence: {confidence:.2f}, Position: {position_size:.1%})"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return None
    
    def _determine_action(self, fgi: float) -> Tuple[SignalAction, float]:
        """
        Determine signal action and base strength from Fear & Greed Index.
        
        IMPROVED TREND-FOLLOWING STRATEGY:
        Instead of contrarian (buy fear, sell greed), we follow momentum:
        - High FGI (greed) = bullish momentum → BUY
        - Low FGI (fear) = bearish momentum → SELL
        - Neutral = HOLD (wait for clear trend)
        
        This aligns with market psychology where trends persist.
        
        Args:
            fgi: Fear & Greed Index (0-100)
        
        Returns:
            Tuple of (SignalAction, strength)
        """
        # Get thresholds with defaults
        if 'thresholds' in self.config and isinstance(self.config['thresholds'], dict):
            thresholds = self.config['thresholds']
            # Handle both config formats
            if 'extreme_fear' in thresholds:
                # Default format - INVERTED for trend-following
                extreme_fear = thresholds.get('extreme_fear', 25)
                fear = thresholds.get('fear', 40)
                neutral = thresholds.get('neutral', 60)
                greed = thresholds.get('greed', 75)
            else:
                # Config file format with buy/sell/hold - map to our thresholds
                extreme_fear = 25
                fear = thresholds.get('buy', {}).get('fear_greed_max', 30) + 10
                neutral = thresholds.get('hold', {}).get('fear_greed_max', 75) - 15
                greed = thresholds.get('sell', {}).get('fear_greed_min', 75)
        else:
            # Fallback to defaults
            extreme_fear, fear, neutral, greed = 25, 40, 60, 75
        
        # TREND-FOLLOWING LOGIC (INVERTED from original contrarian strategy)
        if fgi > greed:
            # Extreme greed (>75) = strong bullish momentum → STRONG_BUY
            strength = (fgi - greed) / (100 - greed)
            return SignalAction.STRONG_BUY, min(strength, 1.0)
        
        elif fgi > neutral:
            # Greed (60-75) = bullish momentum → BUY
            strength = (fgi - neutral) / (greed - neutral)
            return SignalAction.BUY, strength * 0.8  # Moderate strength
        
        elif fgi > fear:
            # Neutral (40-60) = no clear trend → HOLD
            return SignalAction.HOLD, 0.5
        
        elif fgi > extreme_fear:
            # Fear (25-40) = bearish momentum → SELL
            strength = 1.0 - ((fgi - extreme_fear) / (fear - extreme_fear))
            return SignalAction.SELL, strength * 0.8
        
        else:
            # Extreme fear (<25) = strong bearish momentum → STRONG_SELL
            strength = 1.0 - (fgi / extreme_fear)  # Higher strength for lower FGI
            return SignalAction.STRONG_SELL, min(strength, 1.0)
    
    def _calculate_confidence(self, fgi: float, sentiment_score: float, 
                             sentiment_data: Dict) -> float:
        """
        Calculate signal confidence based on multiple factors.
        
        Factors:
        - Consistency between FGI and sentiment score
        - Number of data sources
        - Volatility of sentiment
        
        Args:
            fgi: Fear & Greed Index
            sentiment_score: Sentiment score (-1 to 1)
            sentiment_data: Full sentiment data
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Factor 1: FGI and sentiment alignment
        # Convert FGI to sentiment scale (-1 to 1)
        fgi_sentiment = (fgi - 50) / 50  # 0 -> -1, 100 -> 1
        
        # Calculate alignment (1.0 = perfect alignment, 0.0 = opposite)
        alignment = 1.0 - (abs(fgi_sentiment - sentiment_score) / 2.0)
        
        # Factor 2: Source count (more sources = higher confidence)
        source_count = sentiment_data.get('source_count', 1)
        source_factor = min(source_count / 10.0, 1.0)  # Cap at 10 sources
        
        # Factor 3: Sentiment extremeness (more extreme = higher confidence)
        extremeness = abs(sentiment_score)
        
        # Combined confidence (weighted average)
        confidence = (
            alignment * 0.4 +           # 40% weight on alignment
            source_factor * 0.3 +       # 30% weight on source count
            extremeness * 0.3           # 30% weight on extremeness
        )
        
        return min(confidence, 1.0)
    
    def _calculate_position_size(self, strength: float, confidence: float) -> float:
        """
        Calculate recommended position size.
        
        Position sizing based on Kelly Criterion approximation:
        - Higher confidence = larger position
        - Risk management with min/max limits
        
        Args:
            strength: Signal strength (0.0 to 1.0)
            confidence: Signal confidence (0.0 to 1.0)
        
        Returns:
            Position size as percentage of portfolio (0.0 to 1.0)
        """
        sizing = self.config['position_sizing']
        
        # Base position adjusted by strength and confidence
        position = sizing['base_position'] * strength * confidence
        
        # Apply min/max limits
        position = max(position, sizing['min_position'])
        position = min(position, sizing['max_position'])
        
        return position
    
    def _generate_reasoning(self, action: SignalAction, fgi: float, 
                           sentiment_score: float, confidence: float, 
                           flow_signal: Optional[Dict] = None) -> str:
        """
        Generate human-readable reasoning for the signal.
        
        Args:
            action: Signal action
            fgi: Fear & Greed Index
            sentiment_score: Sentiment score
            confidence: Signal confidence
            flow_signal: Optional fund flow signal data
        
        Returns:
            Reasoning string
        """
        # FGI interpretation
        if fgi < 25:
            fgi_desc = "extreme fear"
        elif fgi < 40:
            fgi_desc = "fear"
        elif fgi < 60:
            fgi_desc = "neutral"
        elif fgi < 75:
            fgi_desc = "greed"
        else:
            fgi_desc = "extreme greed"
        
        # Sentiment interpretation
        if sentiment_score < -0.3:
            sent_desc = "very negative"
        elif sentiment_score < -0.1:
            sent_desc = "negative"
        elif sentiment_score < 0.1:
            sent_desc = "neutral"
        elif sentiment_score < 0.3:
            sent_desc = "positive"
        else:
            sent_desc = "very positive"
        
        # Confidence interpretation
        if confidence > 0.7:
            conf_desc = "high"
        elif confidence > 0.5:
            conf_desc = "moderate"
        else:
            conf_desc = "low"
        
        reasoning = (
            f"Market shows {fgi_desc} (FGI: {fgi:.1f}) with {sent_desc} sentiment "
            f"(score: {sentiment_score:.2f}). "
        )
        
        # Add fund flow context if available
        if flow_signal and flow_signal.get('metrics'):
            metrics = flow_signal['metrics']
            flow_rec = flow_signal.get('recommendation', 'NEUTRAL')
            
            # Net flow interpretation
            net_flow = metrics.get('net_flow', 0)
            if net_flow > 1000000:
                flow_desc = "strong institutional buying"
            elif net_flow > 0:
                flow_desc = "institutional accumulation"
            elif net_flow > -1000000:
                flow_desc = "institutional distribution"
            else:
                flow_desc = "heavy institutional selling"
            
            # Correlation interpretation
            correlation = metrics.get('correlation', 0)
            if abs(correlation) > 0.7:
                corr_desc = "strong price-flow correlation"
            elif abs(correlation) > 0.4:
                corr_desc = "moderate price-flow correlation"
            else:
                corr_desc = "weak price-flow correlation"
            
            reasoning += (
                f"Fund flow shows {flow_desc} (net: ${net_flow:,.0f}) with {corr_desc} "
                f"({correlation:.2f}). Flow recommendation: {flow_rec}. "
            )
        
        reasoning += (
            f"{action.value} signal generated with {conf_desc} confidence ({confidence:.2f})."
        )
        
        return reasoning
    
    def start(self, sentiment_queue: queue.Queue, signal_queue: queue.Queue):
        """
        Start signal generation thread.
        
        Args:
            sentiment_queue: Input queue with sentiment data
            signal_queue: Output queue for generated signals
        """
        if self._running:
            logger.warning("SignalGenerator already running")
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._process_signals,
            args=(sentiment_queue, signal_queue),
            daemon=True,
            name="SignalGenerator"
        )
        self._thread.start()
        logger.info("SignalGenerator thread started")
    
    def _process_signals(self, sentiment_queue: queue.Queue, signal_queue: queue.Queue):
        """
        Process sentiment data and generate signals.
        
        Args:
            sentiment_queue: Input queue with sentiment data
            signal_queue: Output queue for signals
        """
        logger.info("Signal generation processing started")
        
        while self._running:
            try:
                # Get sentiment data (block with timeout)
                sentiment_data = sentiment_queue.get(timeout=1.0)
                
                # Generate signal with timing if benchmarking
                if self.benchmark:
                    start_time = time.perf_counter()
                    signal = self.generate_signal(sentiment_data)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    self.benchmark.record_signal_latency(elapsed_ms)
                else:
                    signal = self.generate_signal(sentiment_data)
                
                if signal:
                    # Convert to dictionary for queue
                    signal_dict = {
                        'timestamp': signal.timestamp.isoformat(),
                        'symbol': signal.symbol,
                        'action': signal.action.value,
                        'strength': signal.strength,
                        'confidence': signal.confidence,
                        'position_size': signal.position_size,
                        'fear_greed_index': signal.fear_greed_index,
                        'sentiment_score': signal.sentiment_score,
                        'reasoning': signal.reasoning,
                        'metadata': signal.metadata
                    }
                    
                    # Put signal in queue
                    signal_queue.put(signal_dict)
                    logger.debug(f"Signal queued: {signal.action.value} for {signal.symbol}")
                
                sentiment_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing signal: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("Signal generation processing stopped")
    
    def stop(self):
        """Stop signal generation"""
        if not self._running:
            return
        
        logger.info("Stopping signal generator...")
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5.0)
        
        # Save signals before stopping
        self.save_signals()
        
        logger.info("Signal generator stopped")
    
    def save_signals(self, filepath: Optional[str] = None) -> str:
        """
        Save generated signals to CSV file.
        
        Args:
            filepath: Optional custom filepath. If None, auto-generates filename.
            
        Returns:
            Path to saved file
        """
        if not self.signal_history:
            logger.warning("No signals to save")
            return None
        
        # Create output directory
        output_dir = Path("data/signals")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"signals_{timestamp}.csv"
        else:
            filepath = Path(filepath)
        
        # Convert signals to DataFrame
        signal_records = []
        for signal in self.signal_history:
            signal_records.append({
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'action': signal.action.value,
                'confidence': signal.confidence,
                'position_size': signal.position_size,
                'fear_greed_index': signal.fear_greed_index,
                'sentiment_score': signal.sentiment_score,
                'reasoning': signal.reasoning
            })
        
        df = pd.DataFrame(signal_records)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Saved {len(signal_records)} signals to {filepath}")
        return str(filepath)
    
    @property
    def stats(self) -> Dict:
        """Get signal generation statistics"""
        return {
            'signals_generated': self.signals_generated,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'signal_history_size': len(self.signal_history),
            'running': self._running
        }
    
    def get_signal_summary(self) -> Dict:
        """Get summary of recent signals"""
        if not self.signal_history:
            return {'total': 0, 'by_action': {}}
        
        # Count by action
        action_counts = {}
        for signal in self.signal_history:
            action = signal.action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(s.confidence for s in self.signal_history) / len(self.signal_history)
        
        return {
            'total': len(self.signal_history),
            'by_action': action_counts,
            'avg_confidence': avg_confidence,
            'last_signal': self.signal_history[-1].action.value if self.signal_history else None
        }
