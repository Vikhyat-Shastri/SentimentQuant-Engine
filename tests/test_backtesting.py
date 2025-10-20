"""
Unit tests for backtesting engine
"""
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.backtesting import BacktestEngine, Trade, BacktestResult


class TestTrade(unittest.TestCase):
    """Test Trade class"""
    
    def test_trade_initialization(self):
        """Test trade creation"""
        trade = Trade(
            entry_time=datetime.now(),
            exit_time=None,
            symbol='BTC-USD',
            action='BUY',
            entry_price=50000.0,
            exit_price=None,
            position_size=0.1,
            quantity=0.2,
            confidence=0.85,
            fear_greed_index=35.0
        )
        
        self.assertEqual(trade.symbol, 'BTC-USD')
        self.assertEqual(trade.action, 'BUY')
        self.assertEqual(trade.status, 'OPEN')
        self.assertIsNone(trade.pnl)
    
    def test_close_long_trade_profit(self):
        """Test closing a profitable long trade"""
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(hours=2)
        
        trade = Trade(
            entry_time=entry_time,
            exit_time=None,
            symbol='BTC-USD',
            action='BUY',
            entry_price=50000.0,
            exit_price=None,
            position_size=0.1,
            quantity=0.2
        )
        
        # Close at higher price (profit)
        trade.close_trade(exit_time, 52000.0)
        
        self.assertEqual(trade.status, 'CLOSED')
        self.assertEqual(trade.exit_price, 52000.0)
        self.assertAlmostEqual(trade.pnl, 400.0, places=2)  # (52000 - 50000) * 0.2
        self.assertAlmostEqual(trade.pnl_percent, 4.0, places=2)
    
    def test_close_long_trade_loss(self):
        """Test closing a losing long trade"""
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(hours=2)
        
        trade = Trade(
            entry_time=entry_time,
            exit_time=None,
            symbol='BTC-USD',
            action='BUY',
            entry_price=50000.0,
            exit_price=None,
            position_size=0.1,
            quantity=0.2
        )
        
        # Close at lower price (loss)
        trade.close_trade(exit_time, 48000.0)
        
        self.assertEqual(trade.status, 'CLOSED')
        self.assertAlmostEqual(trade.pnl, -400.0, places=2)  # (48000 - 50000) * 0.2
        self.assertAlmostEqual(trade.pnl_percent, -4.0, places=2)
    
    def test_close_short_trade_profit(self):
        """Test closing a profitable short trade"""
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(hours=2)
        
        trade = Trade(
            entry_time=entry_time,
            exit_time=None,
            symbol='BTC-USD',
            action='SELL',
            entry_price=50000.0,
            exit_price=None,
            position_size=0.1,
            quantity=0.2
        )
        
        # Close at lower price (profit for short)
        trade.close_trade(exit_time, 48000.0)
        
        self.assertEqual(trade.status, 'CLOSED')
        self.assertAlmostEqual(trade.pnl, 400.0, places=2)  # (50000 - 48000) * 0.2
    
    def test_trade_to_dict(self):
        """Test trade serialization"""
        trade = Trade(
            entry_time=datetime(2024, 10, 1, 12, 0),
            exit_time=None,
            symbol='BTC-USD',
            action='BUY',
            entry_price=50000.0,
            exit_price=None,
            position_size=0.1,
            quantity=0.2
        )
        
        trade_dict = trade.to_dict()
        
        self.assertIn('symbol', trade_dict)
        self.assertIn('action', trade_dict)
        self.assertIn('entry_price', trade_dict)
        self.assertEqual(trade_dict['symbol'], 'BTC-USD')


class TestBacktestEngine(unittest.TestCase):
    """Test BacktestEngine class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = BacktestEngine(
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0005,
            max_position_size=0.25
        )
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.initial_capital, 100000.0)
        self.assertEqual(self.engine.cash, 100000.0)
        self.assertEqual(len(self.engine.positions), 0)
        self.assertEqual(len(self.engine.closed_trades), 0)
    
    def test_simulate_buy_trade(self):
        """Test simulating a buy trade"""
        timestamp = datetime.now()
        
        trade = self.engine.simulate_trade(
            timestamp=timestamp,
            symbol='BTC-USD',
            action='BUY',
            price=50000.0,
            position_size=0.1,
            confidence=0.8,
            fear_greed_index=35.0
        )
        
        self.assertIsNotNone(trade)
        self.assertEqual(trade.symbol, 'BTC-USD')
        self.assertEqual(trade.action, 'BUY')
        self.assertLess(self.engine.cash, 100000.0)  # Cash reduced
        self.assertIn('BTC-USD', self.engine.positions)
    
    def test_simulate_hold_trade(self):
        """Test that HOLD signals don't create trades"""
        timestamp = datetime.now()
        
        trade = self.engine.simulate_trade(
            timestamp=timestamp,
            symbol='BTC-USD',
            action='HOLD',
            price=50000.0,
            position_size=0.1
        )
        
        self.assertIsNone(trade)
        self.assertEqual(self.engine.cash, 100000.0)  # Cash unchanged
        self.assertEqual(len(self.engine.positions), 0)
    
    def test_close_position_on_reverse_signal(self):
        """Test closing position when signal reverses"""
        timestamp = datetime.now()
        
        # Open long position
        self.engine.simulate_trade(
            timestamp=timestamp,
            symbol='BTC-USD',
            action='BUY',
            price=50000.0,
            position_size=0.1
        )
        
        self.assertIn('BTC-USD', self.engine.positions)
        initial_trades = len(self.engine.closed_trades)
        
        # Reverse to sell signal
        self.engine.simulate_trade(
            timestamp=timestamp + timedelta(hours=1),
            symbol='BTC-USD',
            action='SELL',
            price=52000.0,
            position_size=0.1
        )
        
        # Position should be closed
        self.assertEqual(len(self.engine.closed_trades), initial_trades + 1)
    
    def test_max_position_size_enforcement(self):
        """Test that max position size is enforced"""
        timestamp = datetime.now()
        
        trade = self.engine.simulate_trade(
            timestamp=timestamp,
            symbol='BTC-USD',
            action='BUY',
            price=50000.0,
            position_size=0.5,  # Exceeds max of 0.25
            confidence=0.9
        )
        
        # Position size should be capped at max_position_size
        self.assertIsNotNone(trade)
        self.assertLessEqual(trade.position_size, self.engine.max_position_size)
    
    def test_insufficient_cash(self):
        """Test that cash is properly managed"""
        timestamp = datetime.now()
        
        initial_cash = self.engine.cash
        
        # Open a large position
        trade1 = self.engine.simulate_trade(
            timestamp=timestamp,
            symbol='BTC-USD',
            action='BUY',
            price=50000.0,
            position_size=0.25  # 25% of capital
        )
        
        self.assertIsNotNone(trade1)
        self.assertLess(self.engine.cash, initial_cash)  # Cash reduced
        
        # Cash should be tracked correctly
        remaining_cash = self.engine.cash
        self.assertGreater(remaining_cash, 0)  # Still have cash left
    
    def test_run_backtest_basic(self):
        """Test basic backtest execution"""
        # Create sample signals
        signals = pd.DataFrame([
            {
                'timestamp': pd.Timestamp('2024-10-01 12:00:00'),
                'symbol': 'BTC-USD',
                'action': 'BUY',
                'position_size': 0.1,
                'confidence': 0.8,
                'fear_greed_index': 35.0
            },
            {
                'timestamp': pd.Timestamp('2024-10-01 14:00:00'),
                'symbol': 'BTC-USD',
                'action': 'SELL',
                'position_size': 0.1,
                'confidence': 0.75,
                'fear_greed_index': 65.0
            }
        ])
        
        # Create sample price data
        dates = pd.date_range('2024-10-01', '2024-10-02', freq='1h')
        price_data = {
            'BTC-USD': pd.DataFrame({
                'open': np.random.uniform(49000, 51000, len(dates)),
                'high': np.random.uniform(50000, 52000, len(dates)),
                'low': np.random.uniform(48000, 50000, len(dates)),
                'close': np.random.uniform(49000, 51000, len(dates)),
                'volume': np.random.uniform(1000, 5000, len(dates))
            }, index=dates)
        }
        
        # Run backtest
        result = self.engine.run_backtest(signals, price_data)
        
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.initial_capital, 100000.0)
        self.assertIsNotNone(result.final_capital)
        self.assertIsNotNone(result.total_return)


if __name__ == '__main__':
    unittest.main()
