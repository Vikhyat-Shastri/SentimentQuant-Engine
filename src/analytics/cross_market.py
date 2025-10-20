"""
Cross-Market Analysis

Analyzes correlations and spillover effects between different markets:
- Crypto-stock correlations
- Bitcoin dominance effects
- Traditional market spillover
- Cross-asset sentiment transfer
- Risk-on/risk-off regime detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from loguru import logger


@dataclass
class MarketCorrelation:
    """Correlation between two markets"""
    market1: str
    market2: str
    correlation: float
    p_value: float
    lag: int  # lag in hours
    sample_size: int


@dataclass
class SpilloverEffect:
    """Spillover effect from one market to another"""
    source_market: str
    target_market: str
    spillover_strength: float  # 0-1
    direction: str  # 'positive' or 'negative'
    lag_hours: int
    significance: float


@dataclass
class RiskRegime:
    """Risk-on/risk-off regime"""
    regime: str  # 'risk_on', 'risk_off', 'neutral'
    confidence: float
    btc_correlation: float
    stock_correlation: float
    volatility_level: float


@dataclass
class CrossMarketMetrics:
    """Aggregate cross-market metrics"""
    timestamp: datetime
    btc_stock_correlation: float
    btc_dominance: float
    market_coupling: float  # 0-1, degree of synchronization
    spillover_index: float  # 0-1, overall spillover strength
    risk_regime: RiskRegime


class CrossMarketAnalyzer:
    """
    Analyze cross-market relationships and spillovers
    
    Features:
    - Calculate correlations with lags
    - Detect spillover effects
    - Measure Bitcoin dominance
    - Identify risk-on/risk-off regimes
    - Track sentiment transfer between markets
    """
    
    def __init__(self):
        """Initialize cross-market analyzer"""
        self.correlations: Dict[Tuple[str, str], MarketCorrelation] = {}
        self.spillovers: List[SpilloverEffect] = []
        
        logger.info("CrossMarketAnalyzer initialized")
    
    def calculate_correlation(
        self,
        data1: pd.Series,
        data2: pd.Series,
        max_lag: int = 24
    ) -> Tuple[float, int]:
        """
        Calculate correlation with optimal lag
        
        Args:
            data1: First time series
            data2: Second time series
            max_lag: Maximum lag to test (in hours)
        
        Returns:
            (best_correlation, optimal_lag)
        """
        if len(data1) < 2 or len(data2) < 2:
            return 0.0, 0
        
        # Align series
        common_index = data1.index.intersection(data2.index)
        if len(common_index) < 2:
            return 0.0, 0
        
        s1 = data1.loc[common_index]
        s2 = data2.loc[common_index]
        
        best_corr = 0.0
        best_lag = 0
        
        # Test different lags
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr, _ = stats.pearsonr(s1, s2)
            elif lag > 0:
                # s2 leads s1
                if len(s1) > lag:
                    corr, _ = stats.pearsonr(s1[lag:], s2[:-lag])
                else:
                    continue
            else:
                # s1 leads s2
                pos_lag = abs(lag)
                if len(s2) > pos_lag:
                    corr, _ = stats.pearsonr(s1[:-pos_lag], s2[pos_lag:])
                else:
                    continue
            
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        return best_corr, best_lag
    
    def analyze_market_correlation(
        self,
        market1_data: pd.DataFrame,
        market2_data: pd.DataFrame,
        market1_name: str,
        market2_name: str,
        price_col: str = 'close'
    ) -> MarketCorrelation:
        """
        Analyze correlation between two markets
        
        Args:
            market1_data: First market OHLCV data
            market2_data: Second market OHLCV data
            market1_name: Name of first market
            market2_name: Name of second market
            price_col: Price column to use
        
        Returns:
            MarketCorrelation object
        """
        if price_col not in market1_data.columns or price_col not in market2_data.columns:
            logger.warning(f"Price column {price_col} not found")
            return MarketCorrelation(
                market1=market1_name,
                market2=market2_name,
                correlation=0.0,
                p_value=1.0,
                lag=0,
                sample_size=0
            )
        
        # Calculate returns
        returns1 = market1_data[price_col].pct_change().dropna()
        returns2 = market2_data[price_col].pct_change().dropna()
        
        # Find optimal correlation
        corr, lag = self.calculate_correlation(returns1, returns2, max_lag=24)
        
        # Calculate p-value
        common_index = returns1.index.intersection(returns2.index)
        if len(common_index) >= 3:
            aligned_returns1 = returns1.loc[common_index]
            aligned_returns2 = returns2.loc[common_index]
            _, p_value = stats.pearsonr(aligned_returns1, aligned_returns2)
        else:
            p_value = 1.0
        
        result = MarketCorrelation(
            market1=market1_name,
            market2=market2_name,
            correlation=corr,
            p_value=p_value,
            lag=lag,
            sample_size=len(common_index)
        )
        
        self.correlations[(market1_name, market2_name)] = result
        
        return result
    
    def detect_spillover(
        self,
        source_data: pd.DataFrame,
        target_data: pd.DataFrame,
        source_name: str,
        target_name: str,
        threshold: float = 0.5
    ) -> Optional[SpilloverEffect]:
        """
        Detect spillover effect from source to target market
        
        Spillover = source market changes predict target market changes
        
        Args:
            source_data: Source market data
            target_data: Target market data
            source_name: Source market name
            target_name: Target market name
            threshold: Minimum correlation for spillover
        
        Returns:
            SpilloverEffect if detected, None otherwise
        """
        if 'close' not in source_data.columns or 'close' not in target_data.columns:
            return None
        
        # Calculate returns
        source_returns = source_data['close'].pct_change().dropna()
        target_returns = target_data['close'].pct_change().dropna()
        
        # Test Granger causality (simplified)
        # Does lagged source predict target?
        best_spillover = 0.0
        best_lag = 0
        
        for lag in range(1, 25):  # Test lags 1-24 hours
            # Align data with lag
            common_index = source_returns.index.intersection(target_returns.index)
            if len(common_index) < lag + 10:
                continue
            
            source_lagged = source_returns.loc[common_index].shift(lag).dropna()
            target_current = target_returns.loc[common_index].loc[source_lagged.index]
            
            if len(source_lagged) < 10:
                continue
            
            # Calculate predictive correlation
            corr, p_val = stats.pearsonr(source_lagged, target_current)
            
            if abs(corr) > abs(best_spillover) and p_val < 0.05:
                best_spillover = corr
                best_lag = lag
        
        if abs(best_spillover) > threshold:
            direction = 'positive' if best_spillover > 0 else 'negative'
            
            spillover = SpilloverEffect(
                source_market=source_name,
                target_market=target_name,
                spillover_strength=abs(best_spillover),
                direction=direction,
                lag_hours=best_lag,
                significance=abs(best_spillover)
            )
            
            self.spillovers.append(spillover)
            return spillover
        
        return None
    
    def calculate_btc_dominance(
        self,
        btc_data: pd.DataFrame,
        alt_data_list: List[Tuple[str, pd.DataFrame]]
    ) -> float:
        """
        Calculate Bitcoin dominance effect
        
        Measures how much altcoins follow Bitcoin
        
        Args:
            btc_data: Bitcoin price data
            alt_data_list: List of (name, data) for altcoins
        
        Returns:
            Dominance score (0-1)
        """
        if 'close' not in btc_data.columns or not alt_data_list:
            return 0.0
        
        btc_returns = btc_data['close'].pct_change().dropna()
        
        correlations = []
        for alt_name, alt_data in alt_data_list:
            if 'close' not in alt_data.columns:
                continue
            
            alt_returns = alt_data['close'].pct_change().dropna()
            
            # Calculate correlation
            common_index = btc_returns.index.intersection(alt_returns.index)
            if len(common_index) >= 10:
                btc_aligned = btc_returns.loc[common_index]
                alt_aligned = alt_returns.loc[common_index]
                
                corr, _ = stats.pearsonr(btc_aligned, alt_aligned)
                correlations.append(abs(corr))
        
        if not correlations:
            return 0.0
        
        # Average correlation = dominance
        dominance = np.mean(correlations)
        return min(dominance, 1.0)
    
    def detect_risk_regime(
        self,
        btc_data: pd.DataFrame,
        stock_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None
    ) -> RiskRegime:
        """
        Detect risk-on/risk-off regime
        
        Risk-on: Stocks and crypto moving together (positive correlation)
        Risk-off: Flight to safety (negative correlation or high volatility)
        
        Args:
            btc_data: Bitcoin data
            stock_data: Stock market data (e.g., SPY)
            vix_data: Optional VIX data
        
        Returns:
            RiskRegime
        """
        if 'close' not in btc_data.columns or 'close' not in stock_data.columns:
            return RiskRegime(
                regime='neutral',
                confidence=0.0,
                btc_correlation=0.0,
                stock_correlation=0.0,
                volatility_level=0.0
            )
        
        # Calculate returns
        btc_returns = btc_data['close'].pct_change().dropna()
        stock_returns = stock_data['close'].pct_change().dropna()
        
        # BTC-Stock correlation
        common_index = btc_returns.index.intersection(stock_returns.index)
        if len(common_index) >= 10:
            btc_aligned = btc_returns.loc[common_index]
            stock_aligned = stock_returns.loc[common_index]
            
            btc_stock_corr, _ = stats.pearsonr(btc_aligned, stock_aligned)
        else:
            btc_stock_corr = 0.0
        
        # Volatility
        btc_vol = btc_returns.std() * np.sqrt(24)  # Hourly to daily
        stock_vol = stock_returns.std() * np.sqrt(24)
        
        # VIX level
        if vix_data is not None and 'close' in vix_data.columns:
            vix_level = vix_data['close'].iloc[-1]
            normalized_vix = min(vix_level / 50.0, 1.0)  # Normalize to 0-1
        else:
            normalized_vix = (btc_vol + stock_vol) / 2
        
        # Determine regime
        if btc_stock_corr > 0.4 and normalized_vix < 0.4:
            regime = 'risk_on'
            confidence = min(btc_stock_corr + (1 - normalized_vix), 1.0)
        elif btc_stock_corr < -0.2 or normalized_vix > 0.6:
            regime = 'risk_off'
            confidence = min(abs(btc_stock_corr) + normalized_vix, 1.0)
        else:
            regime = 'neutral'
            confidence = 1.0 - abs(btc_stock_corr)
        
        return RiskRegime(
            regime=regime,
            confidence=confidence,
            btc_correlation=btc_stock_corr,
            stock_correlation=1.0,  # Stock to itself
            volatility_level=normalized_vix
        )
    
    def calculate_market_coupling(
        self,
        market_data_dict: Dict[str, pd.DataFrame]
    ) -> float:
        """
        Calculate overall market coupling
        
        Measures synchronization across multiple markets
        
        Args:
            market_data_dict: Dict of market_name -> price data
        
        Returns:
            Coupling score (0-1)
        """
        if len(market_data_dict) < 2:
            return 0.0
        
        # Calculate all pairwise correlations
        markets = list(market_data_dict.keys())
        correlations = []
        
        for i, market1 in enumerate(markets):
            for market2 in markets[i+1:]:
                data1 = market_data_dict[market1]
                data2 = market_data_dict[market2]
                
                if 'close' not in data1.columns or 'close' not in data2.columns:
                    continue
                
                returns1 = data1['close'].pct_change().dropna()
                returns2 = data2['close'].pct_change().dropna()
                
                common_index = returns1.index.intersection(returns2.index)
                if len(common_index) >= 10:
                    r1 = returns1.loc[common_index]
                    r2 = returns2.loc[common_index]
                    
                    corr, _ = stats.pearsonr(r1, r2)
                    correlations.append(abs(corr))
        
        if not correlations:
            return 0.0
        
        # Average absolute correlation
        coupling = np.mean(correlations)
        return min(coupling, 1.0)
    
    def analyze(
        self,
        btc_data: pd.DataFrame,
        stock_data: pd.DataFrame,
        alt_coins: List[Tuple[str, pd.DataFrame]] = None,
        vix_data: Optional[pd.DataFrame] = None
    ) -> CrossMarketMetrics:
        """
        Comprehensive cross-market analysis
        
        Args:
            btc_data: Bitcoin price data
            stock_data: Stock market data
            alt_coins: Optional list of (name, data) for altcoins
            vix_data: Optional VIX data
        
        Returns:
            CrossMarketMetrics
        """
        timestamp = datetime.now()
        
        # BTC-Stock correlation
        btc_stock_corr = self.analyze_market_correlation(
            btc_data, stock_data, 'BTC', 'SPY'
        )
        
        # BTC dominance
        if alt_coins:
            btc_dominance = self.calculate_btc_dominance(btc_data, alt_coins)
        else:
            btc_dominance = 0.0
        
        # Market coupling
        market_dict = {'BTC': btc_data, 'SPY': stock_data}
        if alt_coins:
            for name, data in alt_coins:
                market_dict[name] = data
        
        coupling = self.calculate_market_coupling(market_dict)
        
        # Detect spillovers
        btc_to_stock = self.detect_spillover(btc_data, stock_data, 'BTC', 'SPY')
        stock_to_btc = self.detect_spillover(stock_data, btc_data, 'SPY', 'BTC')
        
        # Spillover index
        spillover_strengths = [s.spillover_strength for s in self.spillovers]
        spillover_index = np.mean(spillover_strengths) if spillover_strengths else 0.0
        
        # Risk regime
        risk_regime = self.detect_risk_regime(btc_data, stock_data, vix_data)
        
        metrics = CrossMarketMetrics(
            timestamp=timestamp,
            btc_stock_correlation=btc_stock_corr.correlation,
            btc_dominance=btc_dominance,
            market_coupling=coupling,
            spillover_index=spillover_index,
            risk_regime=risk_regime
        )
        
        return metrics


# Test function
if __name__ == "__main__":
    # Generate sample market data
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='h')
    
    # Bitcoin data
    btc_price = 50000 * (1 + np.cumsum(np.random.normal(0, 0.01, len(dates))))
    btc_data = pd.DataFrame({
        'timestamp': dates,
        'close': btc_price,
        'volume': np.random.uniform(1e9, 5e9, len(dates))
    }).set_index('timestamp')
    
    # Stock data (correlated with BTC)
    stock_returns = btc_data['close'].pct_change() * 0.5 + np.random.normal(0, 0.005, len(dates))
    stock_price = 400 * (1 + stock_returns).cumprod()
    stock_data = pd.DataFrame({
        'timestamp': dates,
        'close': stock_price,
        'volume': np.random.uniform(1e8, 5e8, len(dates))
    }).set_index('timestamp')
    
    # Altcoins
    eth_returns = btc_data['close'].pct_change() * 0.8 + np.random.normal(0, 0.015, len(dates))
    eth_price = 3000 * (1 + eth_returns).cumprod()
    eth_data = pd.DataFrame({
        'timestamp': dates,
        'close': eth_price
    }).set_index('timestamp')
    
    alt_coins = [('ETH', eth_data)]
    
    # VIX data
    vix_price = 20 + 15 * np.abs(np.random.normal(0, 1, len(dates)))
    vix_data = pd.DataFrame({
        'timestamp': dates,
        'close': vix_price
    }).set_index('timestamp')
    
    # Analyze
    analyzer = CrossMarketAnalyzer()
    
    print("\n" + "="*80)
    print("CROSS-MARKET ANALYSIS")
    print("="*80)
    
    metrics = analyzer.analyze(btc_data, stock_data, alt_coins, vix_data)
    
    print(f"\nTimestamp: {metrics.timestamp}")
    print(f"BTC-Stock Correlation: {metrics.btc_stock_correlation:.3f}")
    print(f"BTC Dominance: {metrics.btc_dominance:.3f}")
    print(f"Market Coupling: {metrics.market_coupling:.3f}")
    print(f"Spillover Index: {metrics.spillover_index:.3f}")
    
    print(f"\n📊 Risk Regime: {metrics.risk_regime.regime.upper()}")
    print(f"  Confidence: {metrics.risk_regime.confidence:.3f}")
    print(f"  BTC Correlation: {metrics.risk_regime.btc_correlation:.3f}")
    print(f"  Volatility Level: {metrics.risk_regime.volatility_level:.3f}")
    
    print(f"\n🔗 Market Correlations:")
    for (m1, m2), corr in analyzer.correlations.items():
        print(f"  {m1} <-> {m2}: {corr.correlation:.3f} (lag: {corr.lag}h, p={corr.p_value:.3f})")
    
    print(f"\n💫 Spillover Effects: {len(analyzer.spillovers)}")
    for spillover in analyzer.spillovers:
        print(f"  {spillover.source_market} -> {spillover.target_market}: "
              f"{spillover.spillover_strength:.3f} ({spillover.direction}, lag: {spillover.lag_hours}h)")
