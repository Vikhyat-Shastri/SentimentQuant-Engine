"""
Regime-Specific Backtesting

Test trading strategy performance in different market regimes:
- Bull market regime
- Bear market regime
- Sideways market regime
- High volatility regime
- Low volatility regime
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

# Import regime classifier
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from analytics.regime_detection import MarketRegimeClassifier, RegimeState


@dataclass
class RegimePerformance:
    """Performance metrics for a specific regime"""
    regime: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    avg_return_per_trade: float
    sharpe_ratio: float
    max_drawdown: float
    time_in_regime: float  # Percentage of total time


@dataclass
class RegimeBacktestResult:
    """Complete regime-specific backtest results"""
    overall_performance: Dict[str, float]
    regime_performances: Dict[str, RegimePerformance]
    regime_transitions: int
    best_regime: str
    worst_regime: str
    regime_distribution: Dict[str, float]


class RegimeBacktester:
    """
    Backtest trading strategy with regime-specific analysis
    
    Features:
    - Separate performance metrics per regime
    - Regime transition analysis
    - Optimal regime identification
    - Regime-adaptive parameters
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001
    ):
        """
        Initialize regime backtester
        
        Args:
            initial_capital: Starting capital
            commission: Trading commission (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        
        self.regime_classifier = MarketRegimeClassifier()
        
        logger.info("RegimeBacktester initialized")
    
    def classify_regimes(
        self,
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Classify market regimes for entire dataset
        
        Args:
            price_data: OHLCV data
        
        Returns:
            DataFrame with regime column added
        """
        regimes = []
        
        for i in range(len(price_data)):
            # Get window of data
            window_start = max(0, i - 100)
            window_data = price_data.iloc[window_start:i+1]
            
            if len(window_data) < 10:
                regimes.append(RegimeState.SIDEWAYS.value)
                continue
            
            # Classify regime
            regime = self.regime_classifier.classify_ensemble(window_data)
            regimes.append(regime.value)
        
        price_data['regime'] = regimes
        
        return price_data
    
    def execute_strategy(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        regime_adaptive: bool = False
    ) -> pd.DataFrame:
        """
        Execute trading strategy and track regime-specific performance
        
        Args:
            price_data: OHLCV data with regime column
            signals: Trading signals (-1, 0, 1)
            regime_adaptive: Whether to adapt strategy to regime
        
        Returns:
            DataFrame with trades and performance
        """
        capital = self.initial_capital
        position = 0  # Current position size
        entry_price = 0
        
        trades = []
        
        for i in range(len(price_data)):
            current_price = price_data['close'].iloc[i]
            current_regime = price_data['regime'].iloc[i]
            timestamp = price_data.index[i]
            
            # Get signal
            if i < len(signals):
                signal = signals['signal'].iloc[i]
            else:
                signal = 0
            
            # Regime-adaptive position sizing
            if regime_adaptive:
                if current_regime == RegimeState.BULL.value:
                    position_multiplier = 1.5  # Larger positions in bull
                elif current_regime == RegimeState.BEAR.value:
                    position_multiplier = 0.5  # Smaller positions in bear
                else:
                    position_multiplier = 1.0
            else:
                position_multiplier = 1.0
            
            # Execute trades
            if signal == 1 and position == 0:
                # Buy
                position_size = (capital * position_multiplier) / current_price
                position = position_size
                entry_price = current_price
                capital -= position_size * current_price * (1 + self.commission)
                
                trades.append({
                    'timestamp': timestamp,
                    'type': 'BUY',
                    'price': current_price,
                    'size': position_size,
                    'regime': current_regime,
                    'capital': capital
                })
                
            elif signal == -1 and position > 0:
                # Sell
                exit_price = current_price
                pnl = position * (exit_price - entry_price)
                pnl_pct = (exit_price - entry_price) / entry_price
                
                capital += position * exit_price * (1 - self.commission)
                
                trades.append({
                    'timestamp': timestamp,
                    'type': 'SELL',
                    'price': current_price,
                    'size': position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'regime': current_regime,
                    'capital': capital
                })
                
                position = 0
                entry_price = 0
        
        # Close any open position
        if position > 0:
            exit_price = price_data['close'].iloc[-1]
            pnl = position * (exit_price - entry_price)
            pnl_pct = (exit_price - entry_price) / entry_price
            
            capital += position * exit_price * (1 - self.commission)
            
            trades.append({
                'timestamp': price_data.index[-1],
                'type': 'SELL',
                'price': exit_price,
                'size': position,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'regime': price_data['regime'].iloc[-1],
                'capital': capital
            })
        
        return pd.DataFrame(trades)
    
    def calculate_regime_performance(
        self,
        trades: pd.DataFrame,
        price_data: pd.DataFrame
    ) -> Dict[str, RegimePerformance]:
        """
        Calculate performance metrics for each regime
        
        Args:
            trades: Trade history
            price_data: Price data with regimes
        
        Returns:
            Dict of regime -> RegimePerformance
        """
        regime_performances = {}
        
        # Get all regimes
        regimes = [RegimeState.BULL.value, RegimeState.BEAR.value, RegimeState.SIDEWAYS.value]
        
        for regime in regimes:
            # Filter trades in this regime
            regime_trades = trades[trades['regime'] == regime]
            
            if len(regime_trades) == 0:
                regime_performances[regime] = RegimePerformance(
                    regime=regime,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    total_return=0.0,
                    avg_return_per_trade=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    time_in_regime=0.0
                )
                continue
            
            # Get sell trades (complete trades)
            sell_trades = regime_trades[regime_trades['type'] == 'SELL']
            
            if len(sell_trades) == 0:
                continue
            
            # Calculate metrics
            total_trades = len(sell_trades)
            winning_trades = len(sell_trades[sell_trades['pnl'] > 0])
            losing_trades = len(sell_trades[sell_trades['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            returns = sell_trades['pnl_pct'].values
            total_return = (1 + returns).prod() - 1
            avg_return = returns.mean()
            
            # Sharpe ratio
            if len(returns) > 1 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            else:
                sharpe = 0.0
            
            # Max drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
            
            # Time in regime
            regime_time = (price_data['regime'] == regime).sum()
            total_time = len(price_data)
            time_pct = regime_time / total_time * 100
            
            regime_performances[regime] = RegimePerformance(
                regime=regime,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_return=total_return,
                avg_return_per_trade=avg_return,
                sharpe_ratio=sharpe,
                max_drawdown=max_drawdown,
                time_in_regime=time_pct
            )
        
        return regime_performances
    
    def backtest(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        regime_adaptive: bool = False
    ) -> RegimeBacktestResult:
        """
        Run complete regime-specific backtest
        
        Args:
            price_data: OHLCV data
            signals: Trading signals
            regime_adaptive: Use regime-adaptive position sizing
        
        Returns:
            RegimeBacktestResult
        """
        logger.info("Starting regime-specific backtest...")
        
        # Classify regimes
        price_data_with_regimes = self.classify_regimes(price_data.copy())
        
        # Execute strategy
        trades = self.execute_strategy(
            price_data_with_regimes,
            signals,
            regime_adaptive
        )
        
        if len(trades) == 0:
            logger.warning("No trades executed")
            return RegimeBacktestResult(
                overall_performance={},
                regime_performances={},
                regime_transitions=0,
                best_regime='',
                worst_regime='',
                regime_distribution={}
            )
        
        # Calculate overall performance
        final_capital = trades['capital'].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        sell_trades = trades[trades['type'] == 'SELL']
        win_rate = len(sell_trades[sell_trades['pnl'] > 0]) / len(sell_trades) if len(sell_trades) > 0 else 0
        
        overall_performance = {
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': len(sell_trades),
            'win_rate': win_rate
        }
        
        # Calculate regime-specific performance
        regime_performances = self.calculate_regime_performance(trades, price_data_with_regimes)
        
        # Find best/worst regimes
        regime_returns = {r: p.total_return for r, p in regime_performances.items() if p.total_trades > 0}
        best_regime = max(regime_returns, key=regime_returns.get) if regime_returns else ''
        worst_regime = min(regime_returns, key=regime_returns.get) if regime_returns else ''
        
        # Regime distribution
        regime_distribution = {}
        for regime in [RegimeState.BULL.value, RegimeState.BEAR.value, RegimeState.SIDEWAYS.value]:
            regime_time = (price_data_with_regimes['regime'] == regime).sum()
            regime_distribution[regime] = regime_time / len(price_data_with_regimes) * 100
        
        # Count regime transitions
        regime_changes = (price_data_with_regimes['regime'] != price_data_with_regimes['regime'].shift()).sum()
        
        result = RegimeBacktestResult(
            overall_performance=overall_performance,
            regime_performances=regime_performances,
            regime_transitions=regime_changes,
            best_regime=best_regime,
            worst_regime=worst_regime,
            regime_distribution=regime_distribution
        )
        
        logger.info("Regime-specific backtest complete")
        
        return result


# Test function
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='h')
    
    # Simulate different regimes
    prices = []
    current_price = 50000
    
    for i in range(len(dates)):
        # Change regime every 1000 hours
        if i < 1000:
            # Bull market
            change = np.random.normal(0.0005, 0.01)
        elif i < 2000:
            # Bear market
            change = np.random.normal(-0.0003, 0.01)
        else:
            # Sideways
            change = np.random.normal(0, 0.008)
        
        current_price *= (1 + change)
        prices.append(current_price)
    
    price_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': np.array(prices) * 1.01,
        'low': np.array(prices) * 0.99,
        'close': prices,
        'volume': np.random.uniform(1e9, 5e9, len(dates))
    }).set_index('timestamp')
    
    # Generate simple signals (trend following)
    returns = price_data['close'].pct_change()
    ma_short = price_data['close'].rolling(20).mean()
    ma_long = price_data['close'].rolling(50).mean()
    
    signals = pd.DataFrame({
        'signal': 0
    }, index=price_data.index)
    
    signals.loc[ma_short > ma_long, 'signal'] = 1
    signals.loc[ma_short < ma_long, 'signal'] = -1
    
    print("\n" + "="*80)
    print("REGIME-SPECIFIC BACKTESTING")
    print("="*80)
    
    # Run backtest
    backtester = RegimeBacktester(initial_capital=10000)
    
    print("\n⚙️  Running standard backtest...")
    result_standard = backtester.backtest(price_data, signals, regime_adaptive=False)
    
    print("\n⚙️  Running regime-adaptive backtest...")
    result_adaptive = backtester.backtest(price_data, signals, regime_adaptive=True)
    
    # Print results
    for name, result in [("STANDARD", result_standard), ("REGIME-ADAPTIVE", result_adaptive)]:
        print(f"\n{'='*80}")
        print(f"{name} STRATEGY RESULTS")
        print(f"{'='*80}")
        
        print(f"\n📊 Overall Performance:")
        print(f"   Final Capital: ${result.overall_performance['final_capital']:,.2f}")
        print(f"   Total Return: {result.overall_performance['total_return']:.2%}")
        print(f"   Total Trades: {result.overall_performance['total_trades']}")
        print(f"   Win Rate: {result.overall_performance['win_rate']:.2%}")
        
        print(f"\n📈 Regime Distribution:")
        for regime, pct in result.regime_distribution.items():
            print(f"   {regime}: {pct:.1f}%")
        
        print(f"\n🔄 Regime Transitions: {result.regime_transitions}")
        
        print(f"\n💰 Performance by Regime:")
        for regime, perf in result.regime_performances.items():
            if perf.total_trades > 0:
                print(f"\n   {regime.upper()}:")
                print(f"      Trades: {perf.total_trades}")
                print(f"      Win Rate: {perf.win_rate:.2%}")
                print(f"      Total Return: {perf.total_return:.2%}")
                print(f"      Avg Return/Trade: {perf.avg_return_per_trade:.2%}")
                print(f"      Sharpe Ratio: {perf.sharpe_ratio:.2f}")
                print(f"      Max Drawdown: {perf.max_drawdown:.2%}")
        
        if result.best_regime:
            print(f"\n🏆 Best Regime: {result.best_regime}")
            print(f"⚠️  Worst Regime: {result.worst_regime}")
