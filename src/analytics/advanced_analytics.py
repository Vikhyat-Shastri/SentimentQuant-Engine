"""
Advanced analytics for sentiment-price correlation and alpha generation.

Features:
- Sentiment-price correlation analysis
- Signal accuracy tracking
- Alpha generation metrics (Sharpe ratio, max drawdown)
- Win rate and profit factor calculations
- Rolling performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsMetrics:
    """Advanced analytics metrics."""
    
    # Sentiment-price correlation
    sentiment_price_correlation: float = 0.0
    correlation_significance: str = "none"  # none, weak, moderate, strong
    correlation_direction: str = "neutral"   # positive, negative, neutral
    
    # Signal accuracy
    total_signals: int = 0
    correct_signals: int = 0
    signal_accuracy: float = 0.0
    
    # Trading performance
    win_rate: float = 0.0
    loss_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Alpha metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    
    # Risk metrics
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0
    
    # Metadata
    timestamp: str = ""
    analysis_period_days: int = 0
    num_trades: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Analytics metrics saved to {filepath}")


class AdvancedAnalytics:
    """Advanced analytics engine."""
    
    def __init__(self):
        """Initialize analytics engine."""
        self.sentiment_history: List[Dict] = []
        self.price_history: List[Dict] = []
        self.signal_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        logger.info("AdvancedAnalytics initialized")
    
    def calculate_sentiment_price_correlation(
        self,
        sentiment_scores: List[float],
        price_changes: List[float]
    ) -> Tuple[float, str, str]:
        """
        Calculate correlation between sentiment and price changes.
        
        Args:
            sentiment_scores: List of sentiment scores
            price_changes: List of price changes (returns)
        
        Returns:
            Tuple of (correlation, significance, direction)
        """
        if len(sentiment_scores) < 10 or len(price_changes) < 10:
            return 0.0, "insufficient_data", "neutral"
        
        # Ensure equal length
        min_len = min(len(sentiment_scores), len(price_changes))
        sentiment_scores = sentiment_scores[-min_len:]
        price_changes = price_changes[-min_len:]
        
        # Calculate Pearson correlation
        correlation = np.corrcoef(sentiment_scores, price_changes)[0, 1]
        
        if np.isnan(correlation):
            return 0.0, "none", "neutral"
        
        # Determine significance
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            significance = "strong"
        elif abs_corr >= 0.4:
            significance = "moderate"
        elif abs_corr >= 0.2:
            significance = "weak"
        else:
            significance = "none"
        
        # Determine direction
        if correlation > 0.1:
            direction = "positive"
        elif correlation < -0.1:
            direction = "negative"
        else:
            direction = "neutral"
        
        return correlation, significance, direction
    
    def calculate_signal_accuracy(
        self,
        signals: List[Dict],
        actual_outcomes: List[bool]
    ) -> Tuple[float, int, int]:
        """
        Calculate signal accuracy.
        
        Args:
            signals: List of signal dictionaries
            actual_outcomes: List of True/False for correct/incorrect
        
        Returns:
            Tuple of (accuracy, correct_count, total_count)
        """
        if not signals or not actual_outcomes:
            return 0.0, 0, 0
        
        total = min(len(signals), len(actual_outcomes))
        correct = sum(actual_outcomes[:total])
        
        accuracy = correct / total if total > 0 else 0.0
        
        return accuracy, correct, total
    
    def calculate_sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: List of period returns
            risk_free_rate: Annual risk-free rate (default: 2%)
            periods_per_year: Trading periods per year (default: 252 for daily)
        
        Returns:
            Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        
        # Calculate excess returns
        period_rf_rate = risk_free_rate / periods_per_year
        excess_returns = returns_array - period_rf_rate
        
        # Calculate Sharpe ratio
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            return 0.0
        
        sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
        
        return float(sharpe)
    
    def calculate_max_drawdown(
        self,
        equity_curve: List[float]
    ) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration.
        
        Args:
            equity_curve: List of equity values over time
        
        Returns:
            Tuple of (max_drawdown_pct, duration_in_periods)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0, 0
        
        equity_array = np.array(equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_array)
        
        # Calculate drawdowns
        drawdowns = (equity_array - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = float(np.min(drawdowns))
        max_dd_idx = np.argmin(drawdowns)
        
        # Calculate duration (periods from last peak to recovery)
        peak_idx = np.argmax(equity_array[:max_dd_idx + 1])
        
        # Find recovery point (if any)
        recovery_idx = max_dd_idx
        peak_value = equity_array[peak_idx]
        
        for i in range(max_dd_idx, len(equity_array)):
            if equity_array[i] >= peak_value:
                recovery_idx = i
                break
        else:
            recovery_idx = len(equity_array) - 1
        
        duration = recovery_idx - peak_idx
        
        return max_dd, duration
    
    def calculate_win_rate(
        self,
        trades: List[Dict]
    ) -> Tuple[float, float, int, int]:
        """
        Calculate win rate from trades.
        
        Args:
            trades: List of trade dictionaries with 'pnl' field
        
        Returns:
            Tuple of (win_rate, loss_rate, wins, losses)
        """
        if not trades:
            return 0.0, 0.0, 0, 0
        
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losses = sum(1 for t in trades if t.get('pnl', 0) < 0)
        total = wins + losses
        
        if total == 0:
            return 0.0, 0.0, 0, 0
        
        win_rate = wins / total
        loss_rate = losses / total
        
        return win_rate, loss_rate, wins, losses
    
    def calculate_profit_factor(
        self,
        trades: List[Dict]
    ) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Args:
            trades: List of trade dictionaries with 'pnl' field
        
        Returns:
            Profit factor
        """
        if not trades:
            return 0.0
        
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calculate_var(
        self,
        returns: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
        
        Args:
            returns: List of returns
            confidence: Confidence level (default: 0.95)
        
        Returns:
            Tuple of (VaR, CVaR)
        """
        if not returns or len(returns) < 10:
            return 0.0, 0.0
        
        returns_array = np.array(sorted(returns))
        
        # Calculate VaR (negative of percentile because returns are losses)
        var_idx = int((1 - confidence) * len(returns_array))
        var = -returns_array[var_idx]
        
        # Calculate CVaR (average of returns below VaR)
        tail_returns = returns_array[:var_idx + 1]
        cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else 0.0
        
        return float(var), float(cvar)
    
    def analyze_from_backtest(
        self,
        backtest_trades_path: str,
        backtest_equity_path: str
    ) -> AnalyticsMetrics:
        """
        Perform analytics from backtest results.
        
        Args:
            backtest_trades_path: Path to trades CSV
            backtest_equity_path: Path to equity CSV
        
        Returns:
            AnalyticsMetrics object
        """
        try:
            # Load data
            trades_df = pd.read_csv(backtest_trades_path)
            equity_df = pd.read_csv(backtest_equity_path)
            
            # Convert to list of dicts
            trades = trades_df.to_dict('records')
            
            # Extract equity curve (try different column names)
            equity_col = None
            for col_name in ['equity', 'portfolio_value', 'value', 'balance']:
                if col_name in equity_df.columns:
                    equity_col = col_name
                    break
            
            if equity_col is None:
                logger.error(f"No equity column found in {backtest_equity_path}")
                return AnalyticsMetrics()
            
            equity_curve = equity_df[equity_col].tolist()
            
            # Calculate returns
            returns = equity_df[equity_col].pct_change().dropna().tolist()
            
            # Calculate metrics
            metrics = AnalyticsMetrics(
                timestamp=datetime.now().isoformat(),
                num_trades=len(trades),
                analysis_period_days=len(equity_curve)
            )
            
            # Win rate and profit factor
            win_rate, loss_rate, wins, losses = self.calculate_win_rate(trades)
            metrics.win_rate = float(win_rate)
            metrics.loss_rate = float(loss_rate)
            
            profit_factor = self.calculate_profit_factor(trades)
            metrics.profit_factor = float(profit_factor)
            
            # Sharpe ratio
            sharpe = self.calculate_sharpe_ratio(returns)
            metrics.sharpe_ratio = float(sharpe)
            
            # Max drawdown
            max_dd, dd_duration = self.calculate_max_drawdown(equity_curve)
            metrics.max_drawdown = float(max_dd)
            metrics.max_drawdown_duration_days = int(dd_duration)
            
            # Returns
            total_return = (equity_curve[-1] / equity_curve[0]) - 1 if equity_curve else 0.0
            metrics.total_return = float(total_return)
            
            # Annualized return (assuming daily data)
            days = len(equity_curve)
            if days > 0:
                metrics.annualized_return = float(((1 + total_return) ** (365 / days)) - 1)
            
            # Volatility
            if returns:
                metrics.volatility = float(np.std(returns) * np.sqrt(252))  # Annualized
            
            # VaR and CVaR
            var, cvar = self.calculate_var(returns)
            metrics.value_at_risk_95 = float(var)
            metrics.conditional_var_95 = float(cvar)
            
            # Sentiment-price correlation (if we have sentiment data)
            if 'sentiment_score' in trades_df.columns and 'pnl' in trades_df.columns:
                sentiment_scores = trades_df['sentiment_score'].dropna().tolist()
                pnls = trades_df['pnl'].dropna().tolist()
                
                if sentiment_scores and pnls:
                    corr, sig, direction = self.calculate_sentiment_price_correlation(
                        sentiment_scores,
                        pnls
                    )
                    metrics.sentiment_price_correlation = float(corr)
                    metrics.correlation_significance = str(sig)
                    metrics.correlation_direction = str(direction)
            
            logger.info(f"Analytics complete: Sharpe={sharpe:.2f}, Win Rate={win_rate:.2%}, Max DD={max_dd:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing backtest results: {e}", exc_info=True)
            return AnalyticsMetrics()
    
    def print_report(self, metrics: AnalyticsMetrics):
        """
        Print analytics report.
        
        Args:
            metrics: AnalyticsMetrics object
        """
        print("\n" + "="*70)
        print("ADVANCED ANALYTICS REPORT")
        print("="*70)
        print(f"Timestamp: {metrics.timestamp}")
        print(f"Analysis Period: {metrics.analysis_period_days} days")
        print(f"Number of Trades: {metrics.num_trades}")
        print()
        
        print("SENTIMENT-PRICE CORRELATION")
        print("-" * 70)
        print(f"Correlation:    {metrics.sentiment_price_correlation:>10.4f}")
        print(f"Significance:   {metrics.correlation_significance:>10}")
        print(f"Direction:      {metrics.correlation_direction:>10}")
        print()
        
        print("SIGNAL ACCURACY")
        print("-" * 70)
        print(f"Total Signals:  {metrics.total_signals:>10,}")
        print(f"Correct:        {metrics.correct_signals:>10,}")
        print(f"Accuracy:       {metrics.signal_accuracy:>10.2%}")
        print()
        
        print("TRADING PERFORMANCE")
        print("-" * 70)
        print(f"Win Rate:       {metrics.win_rate:>10.2%}")
        print(f"Loss Rate:      {metrics.loss_rate:>10.2%}")
        print(f"Profit Factor:  {metrics.profit_factor:>10.2f}")
        print()
        
        print("ALPHA METRICS")
        print("-" * 70)
        print(f"Sharpe Ratio:   {metrics.sharpe_ratio:>10.2f}")
        print(f"Max Drawdown:   {metrics.max_drawdown:>10.2%}")
        print(f"DD Duration:    {metrics.max_drawdown_duration_days:>10} days")
        print()
        
        print("RETURN METRICS")
        print("-" * 70)
        print(f"Total Return:   {metrics.total_return:>10.2%}")
        print(f"Annual Return:  {metrics.annualized_return:>10.2%}")
        print(f"Volatility:     {metrics.volatility:>10.2%}")
        print()
        
        print("RISK METRICS")
        print("-" * 70)
        print(f"VaR (95%):      {metrics.value_at_risk_95:>10.2%}")
        print(f"CVaR (95%):     {metrics.conditional_var_95:>10.2%}")
        print()
        
        print("PERFORMANCE RATING")
        print("-" * 70)
        
        # Rating based on Sharpe ratio
        if metrics.sharpe_ratio > 2.0:
            rating = "🌟 Excellent"
        elif metrics.sharpe_ratio > 1.0:
            rating = "✅ Good"
        elif metrics.sharpe_ratio > 0.5:
            rating = "⚠️  Fair"
        else:
            rating = "❌ Poor"
        
        print(f"Overall Rating: {rating}")
        print("="*70)
        print()
    
    def save_report(self, metrics: AnalyticsMetrics, output_dir: str = "data/analytics"):
        """
        Save analytics report.
        
        Args:
            metrics: AnalyticsMetrics object
            output_dir: Output directory
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"{output_dir}/analytics_{timestamp}.json"
        metrics.to_json(json_path)
        
        # Save text report
        txt_path = f"{output_dir}/analytics_{timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            
            self.print_report(metrics)
            
            sys.stdout = old_stdout
            content = buffer.getvalue()
            
            f.write(content)
        
        logger.info(f"Analytics report saved to {output_dir}")
