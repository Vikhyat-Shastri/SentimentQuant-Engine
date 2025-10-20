"""
Fund Flow Correlation Analyzer.

Analyzes correlation between institutional money flows and crypto prices.
Uses on-chain metrics and exchange data to track smart money movements.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from loguru import logger

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available. Install with: pip install yfinance")


@dataclass
class FundFlowMetrics:
    """Container for fund flow metrics."""
    timestamp: float
    symbol: str
    net_flow: float  # Net institutional flow in USD
    inflow: float    # Institutional buying in USD
    outflow: float   # Institutional selling in USD
    whale_transactions: int  # Number of large transactions (>$100k)
    exchange_netflow: float  # Net flow to/from exchanges
    correlation_score: float  # Correlation with price movement (-1 to 1)


class FundFlowAnalyzer:
    """
    Analyzes fund flows and correlations with cryptocurrency prices.
    
    Tracks:
    - Institutional money flows
    - Whale wallet movements
    - Exchange inflows/outflows
    - Correlation with price action
    """
    
    def __init__(
        self,
        lookback_period: int = 7,  # days
        correlation_window: int = 24  # hours
    ):
        """
        Initialize fund flow analyzer.
        
        Args:
            lookback_period: Days of historical data to analyze
            lookback_period: Days of historical data to analyze
            correlation_window: Hours for correlation calculation
        """
        self.lookback_period = lookback_period
        self.correlation_window = correlation_window
        self.flow_history: Dict[str, List[FundFlowMetrics]] = {}
        
        logger.info(f"FundFlowAnalyzer initialized (lookback: {lookback_period}d, window: {correlation_window}h)")
    
    def analyze_fund_flow(self, symbol: str = "BTC") -> Optional[FundFlowMetrics]:
        """
        Analyze current fund flows for a symbol.
        
        Args:
            symbol: Cryptocurrency symbol (BTC, ETH, etc.)
            
        Returns:
            FundFlowMetrics object or None if data unavailable
        """
        try:
            # Get price data
            ticker = f"{symbol}-USD"
            price_data = self._get_price_data(ticker)
            
            if price_data is None:
                logger.warning(f"No price data returned for {symbol}")
                return None
                
            if len(price_data) < 24:
                logger.warning(f"Insufficient price data for {symbol}: {len(price_data)} hours")
                return None
            
            # Calculate volume-based flow proxy
            # In production, this would use on-chain data from APIs like:
            # - Glassnode, IntoTheBlock, CryptoQuant
            # For now, we use volume and price changes as proxies
            
            recent_volume = price_data['Volume'].tail(24)
            recent_prices = price_data['Close'].tail(24)
            price_changes = recent_prices.pct_change()
            
            # Estimate institutional flow based on volume and price movement
            # Positive correlation suggests buying, negative suggests selling
            avg_volume = recent_volume.mean().item() if hasattr(recent_volume.mean(), 'item') else float(recent_volume.mean())
            current_volume = recent_volume.iloc[-1].item() if hasattr(recent_volume.iloc[-1], 'item') else float(recent_volume.iloc[-1])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate net flow (volume-weighted)
            price_momentum = price_changes.mean().item() if hasattr(price_changes.mean(), 'item') else float(price_changes.mean())
            net_flow = float(current_volume * price_momentum * volume_ratio)
            
            # Estimate inflow/outflow
            if net_flow > 0:
                inflow = abs(net_flow)
                outflow = 0.0
            else:
                inflow = 0.0
                outflow = abs(net_flow)
            
            # Count "whale" transactions (large volume spikes)
            volume_threshold = float(avg_volume * 2.0)  # 2x average = whale
            whale_sum = (recent_volume > volume_threshold).sum()
            whale_transactions = whale_sum.item() if hasattr(whale_sum, 'item') else int(whale_sum)
            
            # Calculate exchange netflow proxy
            # Positive = flowing to exchanges (potential selling)
            # Negative = flowing out of exchanges (potential holding)
            volatility_val = price_changes.std()
            volatility = volatility_val.item() if hasattr(volatility_val, 'item') else float(volatility_val)
            exchange_netflow = float(-net_flow * volatility)  # Inverse relationship
            
            # Calculate correlation between volume and price
            correlation = float(np.corrcoef(recent_volume, recent_prices)[0, 1])
            
            metrics = FundFlowMetrics(
                timestamp=time.time(),
                symbol=symbol,
                net_flow=net_flow,
                inflow=inflow,
                outflow=outflow,
                whale_transactions=whale_transactions,
                exchange_netflow=exchange_netflow,
                correlation_score=correlation if not np.isnan(correlation) else 0.0
            )
            
            # Store in history
            if symbol not in self.flow_history:
                self.flow_history[symbol] = []
            self.flow_history[symbol].append(metrics)
            
            # Keep only recent history
            max_history = self.lookback_period * 24  # hourly data
            if len(self.flow_history[symbol]) > max_history:
                self.flow_history[symbol] = self.flow_history[symbol][-max_history:]
            
            logger.debug(f"Fund flow for {symbol}: net=${net_flow:,.0f}, whales={whale_transactions}, corr={correlation:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing fund flow for {symbol}: {e}")
            return None
    
    def get_flow_signal(self, symbol: str = "BTC") -> Dict[str, float]:
        """
        Get trading signal based on fund flows.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Dictionary with flow signal metrics
        """
        metrics = self.analyze_fund_flow(symbol)
        
        if metrics is None:
            return {
                'flow_score': 0.0,
                'signal_strength': 0.0,
                'recommendation': 'NEUTRAL'
            }
        
        # Calculate flow score (-1 to 1)
        # Positive net flow + positive correlation = bullish
        # Negative net flow + negative correlation = bearish
        
        flow_direction = 1.0 if metrics.net_flow > 0 else -1.0
        flow_magnitude = min(abs(metrics.net_flow) / 1e9, 1.0)  # Normalize to billions
        correlation_factor = abs(metrics.correlation_score)
        
        flow_score = flow_direction * flow_magnitude * correlation_factor
        
        # Whale activity adds conviction
        whale_factor = min(metrics.whale_transactions / 10.0, 1.0)
        signal_strength = (correlation_factor + whale_factor) / 2.0
        
        # Generate recommendation
        if flow_score > 0.3 and signal_strength > 0.5:
            recommendation = 'STRONG_BUY'
        elif flow_score > 0.1:
            recommendation = 'BUY'
        elif flow_score < -0.3 and signal_strength > 0.5:
            recommendation = 'STRONG_SELL'
        elif flow_score < -0.1:
            recommendation = 'SELL'
        else:
            recommendation = 'NEUTRAL'
        
        return {
            'flow_score': flow_score,
            'signal_strength': signal_strength,
            'recommendation': recommendation,
            'net_flow': metrics.net_flow,
            'whale_transactions': metrics.whale_transactions,
            'correlation': metrics.correlation_score
        }
    
    def get_correlation_matrix(self, symbols: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix between symbols' fund flows.
        
        Args:
            symbols: List of symbols to correlate (default: BTC, ETH)
            
        Returns:
            Correlation matrix as nested dict
        """
        if symbols is None:
            symbols = ['BTC', 'ETH']
        
        matrix = {}
        
        for sym1 in symbols:
            matrix[sym1] = {}
            for sym2 in symbols:
                if sym1 == sym2:
                    matrix[sym1][sym2] = 1.0
                else:
                    # Calculate correlation between fund flows
                    corr = self._calculate_flow_correlation(sym1, sym2)
                    matrix[sym1][sym2] = corr
        
        return matrix
    
    def _calculate_flow_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Calculate correlation between two symbols' fund flows.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if symbol1 not in self.flow_history or symbol2 not in self.flow_history:
            return 0.0
        
        history1 = self.flow_history[symbol1]
        history2 = self.flow_history[symbol2]
        
        if len(history1) < 10 or len(history2) < 10:
            return 0.0
        
        # Get net flows
        flows1 = np.array([m.net_flow for m in history1[-24:]])
        flows2 = np.array([m.net_flow for m in history2[-24:]])
        
        # Ensure same length
        min_len = min(len(flows1), len(flows2))
        flows1 = flows1[-min_len:]
        flows2 = flows2[-min_len:]
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(flows1, flows2)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _get_price_data(self, ticker: str) -> Optional[any]:
        """
        Fetch price data from yfinance.
        
        Args:
            ticker: Ticker symbol (e.g., "BTC-USD")
            
        Returns:
            DataFrame with price data or None
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            # Download recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_period)
            
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='1h',
                progress=False
            )
            
            return data if len(data) > 0 else None
            
        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            return None
    
    def get_summary(self, symbol: str = "BTC") -> Dict:
        """
        Get summary of fund flow analysis.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Summary dictionary
        """
        if symbol not in self.flow_history or len(self.flow_history[symbol]) == 0:
            return {
                'symbol': symbol,
                'available': False,
                'message': 'No fund flow data available'
            }
        
        recent_metrics = self.flow_history[symbol][-1]
        signal = self.get_flow_signal(symbol)
        
        # Calculate 24h change
        if len(self.flow_history[symbol]) >= 24:
            flow_24h_ago = self.flow_history[symbol][-24].net_flow
            flow_change = ((recent_metrics.net_flow - flow_24h_ago) / abs(flow_24h_ago) * 100
                          if flow_24h_ago != 0 else 0)
        else:
            flow_change = 0
        
        return {
            'symbol': symbol,
            'available': True,
            'current_flow': recent_metrics.net_flow,
            'flow_24h_change_pct': flow_change,
            'whale_transactions': recent_metrics.whale_transactions,
            'correlation': recent_metrics.correlation_score,
            'signal': signal,
            'timestamp': recent_metrics.timestamp
        }


if __name__ == "__main__":
    # Test the fund flow analyzer
    logger.info("Testing FundFlowAnalyzer...")
    
    analyzer = FundFlowAnalyzer(lookback_period=7, correlation_window=24)
    
    # Analyze BTC
    logger.info("\n=== Bitcoin (BTC) Fund Flow Analysis ===")
    btc_metrics = analyzer.analyze_fund_flow("BTC")
    if btc_metrics:
        logger.info(f"Net Flow: ${btc_metrics.net_flow:,.2f}")
        logger.info(f"Whale Transactions: {btc_metrics.whale_transactions}")
        logger.info(f"Correlation: {btc_metrics.correlation_score:.3f}")
        
        signal = analyzer.get_flow_signal("BTC")
        logger.info(f"\nSignal: {signal['recommendation']}")
        logger.info(f"Flow Score: {signal['flow_score']:.3f}")
        logger.info(f"Signal Strength: {signal['signal_strength']:.3f}")
    
    # Analyze ETH
    logger.info("\n=== Ethereum (ETH) Fund Flow Analysis ===")
    eth_metrics = analyzer.analyze_fund_flow("ETH")
    if eth_metrics:
        logger.info(f"Net Flow: ${eth_metrics.net_flow:,.2f}")
        logger.info(f"Whale Transactions: {eth_metrics.whale_transactions}")
        logger.info(f"Correlation: {eth_metrics.correlation_score:.3f}")
    
    # Get correlation matrix
    logger.info("\n=== Correlation Matrix ===")
    corr_matrix = analyzer.get_correlation_matrix(['BTC', 'ETH'])
    for sym1, correlations in corr_matrix.items():
        for sym2, corr in correlations.items():
            logger.info(f"{sym1} vs {sym2}: {corr:.3f}")
    
    # Get summary
    logger.info("\n=== Summary ===")
    summary = analyzer.get_summary("BTC")
    logger.info(f"BTC Summary: {summary}")
