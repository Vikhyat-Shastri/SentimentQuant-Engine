"""
Backtesting Engine for Strategy Evaluation

This module provides comprehensive backtesting capabilities for evaluating
trading strategies based on sentiment-driven signals.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import yfinance as yf
from pathlib import Path
import json


@dataclass
class Trade:
    """Represents a single trade execution"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    action: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: Optional[float]
    position_size: float  # Percentage of portfolio
    quantity: float  # Number of units
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    confidence: float = 0.0
    fear_greed_index: float = 50.0
    status: str = 'OPEN'  # 'OPEN' or 'CLOSED'
    
    def close_trade(self, exit_time: datetime, exit_price: float) -> None:
        """Close the trade and calculate P&L"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = 'CLOSED'
        
        if self.action in ['BUY', 'STRONG_BUY']:
            # Long position: profit when price goes up
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_percent = ((exit_price / self.entry_price) - 1) * 100
        else:
            # Short position: profit when price goes down
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_percent = ((self.entry_price / exit_price) - 1) * 100
    
    def to_dict(self) -> Dict:
        """Convert trade to dictionary"""
        return {
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'symbol': self.symbol,
            'action': self.action,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'position_size': self.position_size,
            'quantity': self.quantity,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'confidence': self.confidence,
            'fear_greed_index': self.fear_greed_index,
            'status': self.status
        }


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_percent: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: timedelta
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'total_return_percent': self.total_return_percent,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_percent': self.max_drawdown_percent,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_trade_duration': str(self.avg_trade_duration),
            'trades': [t.to_dict() for t in self.trades]
        }
    
    def summary(self) -> str:
        """Generate a text summary of the backtest results"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║               BACKTESTING RESULTS SUMMARY                    ║
╚══════════════════════════════════════════════════════════════╝

📅 Period: {self.start_date.date()} to {self.end_date.date()}
💰 Initial Capital: ${self.initial_capital:,.2f}
💰 Final Capital: ${self.final_capital:,.2f}

📈 PERFORMANCE METRICS
   Total Return: ${self.total_return:,.2f} ({self.total_return_percent:+.2f}%)
   Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_percent:.2f}%)
   Sharpe Ratio: {self.sharpe_ratio:.2f}
   Sortino Ratio: {self.sortino_ratio:.2f}

📊 TRADE STATISTICS
   Total Trades: {self.total_trades}
   Winning Trades: {self.winning_trades} ({self.win_rate:.1f}%)
   Losing Trades: {self.losing_trades} ({100-self.win_rate:.1f}%)
   Profit Factor: {self.profit_factor:.2f}
   
   Average Win: ${self.avg_win:,.2f}
   Average Loss: ${self.avg_loss:,.2f}
   Largest Win: ${self.largest_win:,.2f}
   Largest Loss: ${self.largest_loss:,.2f}
   
   Avg Trade Duration: {self.avg_trade_duration}

═══════════════════════════════════════════════════════════════
"""


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies
    
    Features:
    - Historical data download and management
    - Signal-based trade execution simulation
    - Position management and P&L tracking
    - Comprehensive performance metrics
    - Transaction cost modeling
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1% commission
        slippage: float = 0.0005,  # 0.05% slippage
        max_position_size: float = 0.25,  # Max 25% per position
        risk_free_rate: float = 0.04,  # 4% annual risk-free rate
        stop_loss_pct: float = 0.02,  # 2% stop-loss
        take_profit_pct: float = 0.06  # 6% take-profit (3:1 reward:risk)
    ):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital in USD
            commission: Commission rate (0.001 = 0.1%)
            slippage: Slippage rate (0.0005 = 0.05%)
            max_position_size: Maximum position size as fraction of capital
            risk_free_rate: Annual risk-free rate for Sharpe ratio
            stop_loss_pct: Stop-loss percentage (0.02 = 2%)
            take_profit_pct: Take-profit percentage (0.06 = 6%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # State variables
        self.cash = initial_capital
        self.positions: Dict[str, Trade] = {}  # Open positions by symbol
        self.closed_trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
        logger.info(f"BacktestEngine initialized with ${initial_capital:,.2f} capital")
        logger.info(f"Risk Management: Stop-Loss={stop_loss_pct:.1%}, Take-Profit={take_profit_pct:.1%}")
    
    def download_historical_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1h'
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical OHLCV data
        
        Args:
            symbols: List of symbols (e.g., ['BTC-USD', 'ETH-USD'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
        
        Returns:
            Dictionary mapping symbol to DataFrame with OHLCV data
        """
        logger.info(f"Downloading historical data for {len(symbols)} symbols...")
        logger.info(f"Period: {start_date} to {end_date}, Interval: {interval}")
        
        historical_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
                
                if df.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Standardize column names
                df.columns = [col.lower() for col in df.columns]
                historical_data[symbol] = df
                
                logger.info(f"✓ {symbol}: {len(df)} bars downloaded")
                
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
        
        return historical_data
    
    def simulate_trade(
        self,
        timestamp: datetime,
        symbol: str,
        action: str,
        price: float,
        position_size: float,
        confidence: float = 0.0,
        fear_greed_index: float = 50.0
    ) -> Optional[Trade]:
        """
        Simulate a trade execution
        
        Args:
            timestamp: Trade timestamp
            symbol: Trading symbol
            action: Signal action (BUY, SELL, STRONG_BUY, STRONG_SELL, HOLD)
            price: Current market price
            position_size: Position size as fraction of capital
            confidence: Signal confidence (0-1)
            fear_greed_index: Fear & Greed Index value
        
        Returns:
            Trade object if executed, None otherwise
        """
        # Skip HOLD signals
        if action == 'HOLD':
            return None
        
        # Apply position size limits
        position_size = min(position_size, self.max_position_size)
        
        # Check if we have an open position
        if symbol in self.positions:
            existing_trade = self.positions[symbol]
            
            # Close position if signal reverses
            if (existing_trade.action in ['BUY', 'STRONG_BUY'] and action in ['SELL', 'STRONG_SELL']) or \
               (existing_trade.action in ['SELL', 'STRONG_SELL'] and action in ['BUY', 'STRONG_BUY']):
                
                # Apply slippage and commission
                exit_price = price * (1 - self.slippage) if action in ['SELL', 'STRONG_SELL'] else price * (1 + self.slippage)
                
                # Close the position
                existing_trade.close_trade(timestamp, exit_price)
                
                # Update cash (return capital + P&L)
                position_value = existing_trade.entry_price * existing_trade.quantity
                self.cash += position_value + existing_trade.pnl
                
                # Apply commission
                commission_cost = abs(exit_price * existing_trade.quantity * self.commission)
                self.cash -= commission_cost
                
                # Record closed trade
                self.closed_trades.append(existing_trade)
                del self.positions[symbol]
                
                logger.debug(f"Closed {existing_trade.action} position in {symbol}: P&L ${existing_trade.pnl:,.2f}")
        
        # Open new position if we have cash
        if action in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
            # Calculate position value
            total_equity = self.cash + sum(
                pos.entry_price * pos.quantity for pos in self.positions.values()
            )
            position_value = total_equity * position_size
            
            if position_value > self.cash:
                logger.debug(f"Insufficient cash for {symbol} position (need ${position_value:,.2f}, have ${self.cash:,.2f})")
                return None
            
            # Apply slippage
            entry_price = price * (1 + self.slippage) if action in ['BUY', 'STRONG_BUY'] else price * (1 - self.slippage)
            
            # Calculate quantity
            quantity = position_value / entry_price
            
            # Apply commission
            commission_cost = position_value * self.commission
            
            # Update cash
            self.cash -= (position_value + commission_cost)
            
            # Create trade
            trade = Trade(
                entry_time=timestamp,
                exit_time=None,
                symbol=symbol,
                action=action,
                entry_price=entry_price,
                exit_price=None,
                position_size=position_size,
                quantity=quantity,
                confidence=confidence,
                fear_greed_index=fear_greed_index
            )
            
            self.positions[symbol] = trade
            logger.debug(f"Opened {action} position in {symbol}: {quantity:.4f} units @ ${entry_price:.2f}")
            
            return trade
        
        return None
    
    def run_backtest(
        self,
        signals: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame]
    ) -> BacktestResult:
        """
        Run backtest with given signals and price data
        
        Args:
            signals: DataFrame with columns [timestamp, symbol, action, position_size, confidence, fear_greed_index]
            price_data: Dictionary mapping symbol to price DataFrame
        
        Returns:
            BacktestResult with performance metrics
        """
        logger.info("Starting backtest...")
        logger.info(f"Signals: {len(signals)}, Period: {signals['timestamp'].min()} to {signals['timestamp'].max()}")
        
        # Reset state
        self.cash = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_history = []
        
        # Sort signals by timestamp
        signals = signals.sort_values('timestamp').reset_index(drop=True)
        
        # Process each signal
        for idx, signal in signals.iterrows():
            timestamp = signal['timestamp']
            symbol = signal['symbol']
            action = signal['action']
            position_size = signal.get('position_size', 0.1)
            confidence = signal.get('confidence', 0.0)
            fgi = signal.get('fear_greed_index', 50.0)
            
            # Get current price
            if symbol not in price_data:
                continue
            
            price_df = price_data[symbol]
            
            # Handle timezone-aware timestamps
            search_timestamp = timestamp
            if price_df.index.tz is not None:
                # Make timestamp timezone-aware if price data is
                if timestamp.tzinfo is None:
                    search_timestamp = timestamp.tz_localize('UTC')
            
            price_row = price_df[price_df.index <= search_timestamp].tail(1)
            
            if price_row.empty:
                continue
            
            current_price = price_row['close'].iloc[0]
            
            # CHECK STOP-LOSS AND TAKE-PROFIT FOR EXISTING POSITIONS
            for pos_symbol, position in list(self.positions.items()):
                if pos_symbol == symbol:  # Only check the symbol we're trading
                    # Calculate current P&L percentage
                    if position.action in ['BUY', 'STRONG_BUY']:
                        pnl_pct = (current_price / position.entry_price) - 1
                    else:  # SELL or STRONG_SELL
                        pnl_pct = (position.entry_price / current_price) - 1
                    
                    # Check stop-loss (hit max loss)
                    if pnl_pct <= -self.stop_loss_pct:
                        logger.debug(f"Stop-loss triggered for {pos_symbol}: {pnl_pct:.2%}")
                        exit_price = current_price * (1 - self.slippage) if position.action in ['BUY', 'STRONG_BUY'] else current_price * (1 + self.slippage)
                        position.close_trade(timestamp, exit_price)
                        
                        position_value = position.entry_price * position.quantity
                        self.cash += position_value + position.pnl
                        commission_cost = abs(exit_price * position.quantity * self.commission)
                        self.cash -= commission_cost
                        
                        self.closed_trades.append(position)
                        del self.positions[pos_symbol]
                        continue
                    
                    # Check take-profit (hit profit target)
                    if pnl_pct >= self.take_profit_pct:
                        logger.debug(f"Take-profit triggered for {pos_symbol}: {pnl_pct:.2%}")
                        exit_price = current_price * (1 - self.slippage) if position.action in ['BUY', 'STRONG_BUY'] else current_price * (1 + self.slippage)
                        position.close_trade(timestamp, exit_price)
                        
                        position_value = position.entry_price * position.quantity
                        self.cash += position_value + position.pnl
                        commission_cost = abs(exit_price * position.quantity * self.commission)
                        self.cash -= commission_cost
                        
                        self.closed_trades.append(position)
                        del self.positions[pos_symbol]
                        continue
            
            # Execute trade (if signal says so)
            self.simulate_trade(
                timestamp=timestamp,
                symbol=symbol,
                action=action,
                price=current_price,
                position_size=position_size,
                confidence=confidence,
                fear_greed_index=fgi
            )
            
            # Record equity
            total_equity = self.cash + sum(
                pos.entry_price * pos.quantity * (current_price / pos.entry_price)
                for pos in self.positions.values()
            )
            self.equity_history.append((timestamp, total_equity))
        
        # Close all remaining positions at final price
        if self.positions:
            final_timestamp = signals['timestamp'].max()
            for symbol, position in list(self.positions.items()):
                price_df = price_data[symbol]
                final_price = price_df['close'].iloc[-1]
                position.close_trade(final_timestamp, final_price)
                
                position_value = position.entry_price * position.quantity
                self.cash += position_value + position.pnl
                self.closed_trades.append(position)
            
            self.positions = {}
        
        # Calculate metrics
        result = self._calculate_metrics(
            start_date=signals['timestamp'].min(),
            end_date=signals['timestamp'].max()
        )
        
        logger.info("Backtest completed!")
        logger.info(f"Total Return: {result.total_return_percent:+.2f}%")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Win Rate: {result.win_rate:.1f}%")
        
        return result
    
    def _calculate_metrics(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate performance metrics from closed trades"""
        
        # Basic metrics
        final_capital = self.cash
        total_return = final_capital - self.initial_capital
        total_return_percent = (total_return / self.initial_capital) * 100
        
        # Trade statistics
        total_trades = len(self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl and t.pnl < 0]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
        
        # Average trade duration
        durations = []
        for trade in self.closed_trades:
            if trade.exit_time and trade.entry_time:
                durations.append(trade.exit_time - trade.entry_time)
        avg_duration = sum(durations, timedelta()) / len(durations) if durations else timedelta()
        
        # Equity curve
        equity_df = pd.DataFrame(self.equity_history, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
        equity_df['drawdown_pct'] = (equity_df['drawdown'] / equity_df['peak']) * 100
        
        max_drawdown = equity_df['drawdown'].min()
        max_drawdown_percent = abs(equity_df['drawdown_pct'].min())
        
        # Sharpe ratio
        if len(equity_df) > 1:
            equity_df['returns'] = equity_df['equity'].pct_change()
            returns = equity_df['returns'].dropna()
            
            if len(returns) > 0 and returns.std() > 0:
                # Annualize returns (assuming hourly data)
                periods_per_year = 24 * 365
                excess_returns = returns.mean() * periods_per_year - self.risk_free_rate
                sharpe_ratio = excess_returns / (returns.std() * np.sqrt(periods_per_year))
                
                # Sortino ratio (using downside deviation)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
                    sortino_ratio = excess_returns / downside_std if downside_std > 0 else 0
                else:
                    sortino_ratio = sharpe_ratio
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_percent=total_return_percent,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_duration,
            trades=self.closed_trades,
            equity_curve=equity_df
        )
    
    def save_results(self, result: BacktestResult, output_dir: str = 'data/backtesting') -> None:
        """Save backtest results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary as JSON
        summary_file = output_path / f'backtest_{timestamp}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Save equity curve
        equity_file = output_path / f'backtest_{timestamp}_equity.csv'
        result.equity_curve.to_csv(equity_file)
        
        # Save trades
        trades_file = output_path / f'backtest_{timestamp}_trades.csv'
        trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
        trades_df.to_csv(trades_file, index=False)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"  - Summary: {summary_file.name}")
        logger.info(f"  - Equity: {equity_file.name}")
        logger.info(f"  - Trades: {trades_file.name}")
