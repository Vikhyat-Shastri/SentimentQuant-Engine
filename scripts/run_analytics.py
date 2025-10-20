"""
Run advanced analytics on backtest results.

Usage:
    python run_analytics.py --trades data/backtesting/backtest_20251016_213849_trades.csv --equity data/backtesting/backtest_20251016_213849_equity.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
import glob

from src.analytics.advanced_analytics import AdvancedAnalytics
from src.analytics.behavioral_analysis import BehavioralBiasDetector
from src.analytics.regime_detection import MarketRegimeClassifier
from src.analytics.crowd_psychology import CrowdPsychologyAnalyzer
from src.analytics.cross_market import CrossMarketAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def find_latest_backtest() -> tuple:
    """Find latest backtest files."""
    backtest_dir = Path("data/backtesting")
    
    if not backtest_dir.exists():
        return None, None
    
    # Find all trade files
    trade_files = list(backtest_dir.glob("backtest_*_trades.csv"))
    
    if not trade_files:
        return None, None
    
    # Get most recent
    latest_trade = max(trade_files, key=lambda p: p.stat().st_mtime)
    
    # Find corresponding equity file
    timestamp = latest_trade.stem.replace("_trades", "")
    equity_file = backtest_dir / f"{timestamp}_equity.csv"
    
    if not equity_file.exists():
        return None, None
    
    return str(latest_trade), str(equity_file)


def run_analytics(trades_path: str = None, equity_path: str = None):
    """
    Run advanced analytics.
    
    Args:
        trades_path: Path to trades CSV
        equity_path: Path to equity CSV
    """
    # If no paths provided, find latest backtest
    if not trades_path or not equity_path:
        logger.info("No paths provided, searching for latest backtest...")
        trades_path, equity_path = find_latest_backtest()
        
        if not trades_path:
            logger.error("No backtest files found in data/backtesting/")
            return 1
        
        logger.info(f"Found latest backtest: {Path(trades_path).stem}")
    
    # Verify files exist
    if not Path(trades_path).exists():
        logger.error(f"Trades file not found: {trades_path}")
        return 1
    
    if not Path(equity_path).exists():
        logger.error(f"Equity file not found: {equity_path}")
        return 1
    
    logger.info(f"Analyzing backtest results:")
    logger.info(f"  Trades: {trades_path}")
    logger.info(f"  Equity: {equity_path}")
    
    # Load data
    trades_df = pd.read_csv(trades_path)
    equity_df = pd.read_csv(equity_path)
    
    # Initialize analytics
    logger.info("\n🚀 Initializing advanced analytics modules...")
    analytics = AdvancedAnalytics()
    behavior_analyzer = BehavioralBiasDetector()
    regime_classifier = MarketRegimeClassifier()
    crowd_psychology = CrowdPsychologyAnalyzer()
    cross_market = CrossMarketAnalyzer()
    logger.info("✅ All analytics modules initialized")
    
    # Run baseline analysis
    logger.info("\n📊 Running baseline analysis...")
    metrics = analytics.analyze_from_backtest(trades_path, equity_path)
    
    # BONUS FEATURE: Behavioral Analysis (HIGH priority)
    logger.info("\n🧠 Running behavioral bias detection...")
    try:
        biases = behavior_analyzer.analyze_trading_behavior(trades_df)
        logger.info("  Behavioral Biases Detected:")
        logger.info(f"    Loss Aversion Score: {biases.loss_aversion_score:.2f}")
        logger.info(f"    Overconfidence Score: {biases.overconfidence_score:.2f}")
        logger.info(f"    Recency Bias Score: {biases.recency_bias_score:.2f}")
        logger.info(f"    Confirmation Bias: {biases.confirmation_bias_score:.2f}")
        logger.info(f"    Disposition Effect: {biases.disposition_effect_score:.2f}")
        
        if biases.loss_aversion_score > 0.7:
            logger.warning("  ⚠️  HIGH loss aversion detected - consider reviewing stop losses")
        if biases.overconfidence_score > 0.7:
            logger.warning("  ⚠️  HIGH overconfidence detected - reduce position sizes")
        if biases.recency_bias_score > 0.7:
            logger.warning("  ⚠️  HIGH recency bias detected - review historical performance")
    except Exception as e:
        logger.error(f"  Behavioral analysis failed: {e}")
    
    # BONUS FEATURE: Regime Classification (MEDIUM priority)
    logger.info("\n📈 Running market regime analysis...")
    try:
        if 'close' in equity_df.columns or 'equity' in equity_df.columns:
            price_series = equity_df['equity'].values if 'equity' in equity_df.columns else equity_df['close'].values
            
            # Classify regime for entire period
            regime = regime_classifier.classify_current_regime(price_series)
            logger.info(f"  Overall Market Regime: {regime}")
            
            # Analyze regime transitions
            regimes = regime_classifier.classify_regimes(price_series, window=20)
            regime_counts = pd.Series(regimes).value_counts()
            logger.info(f"  Regime Distribution:")
            for r, count in regime_counts.items():
                pct = count / len(regimes) * 100
                logger.info(f"    {r}: {count} periods ({pct:.1f}%)")
            
            # Performance by regime
            if len(trades_df) > 0 and 'pnl' in trades_df.columns:
                logger.info(f"  Strategy performed in {regime} market")
    except Exception as e:
        logger.error(f"  Regime analysis failed: {e}")
    
    # BONUS FEATURE: Crowd Psychology (MEDIUM priority)
    logger.info("\n👥 Running crowd psychology analysis...")
    try:
        if 'sentiment_score' in trades_df.columns or 'confidence' in trades_df.columns:
            # Create sentiment data from trades
            sentiment_data = trades_df.copy()
            if 'sentiment_score' not in sentiment_data.columns and 'confidence' in sentiment_data.columns:
                sentiment_data['sentiment_score'] = sentiment_data['confidence'] * 2 - 1  # Convert to -1 to 1
            
            crowd_metrics = crowd_psychology.analyze(sentiment_data)
            logger.info(f"  Crowd Psychology Metrics:")
            logger.info(f"    FOMO Score: {crowd_metrics.fomo_score:.2f}")
            logger.info(f"    Panic Score: {crowd_metrics.panic_score:.2f}")
            logger.info(f"    Herd Behavior: {crowd_metrics.herd_behavior_score:.2f}")
            logger.info(f"    Euphoria Level: {crowd_metrics.euphoria_level:.2f}")
            logger.info(f"    Contrarian Signals: {crowd_metrics.contrarian_signal_count}")
            
            if crowd_metrics.fomo_score > 0.75:
                logger.warning("  ⚠️  EXTREME FOMO detected - market may be overheated")
            if crowd_metrics.panic_score > 0.75:
                logger.warning("  ⚠️  EXTREME PANIC detected - potential buying opportunity")
    except Exception as e:
        logger.error(f"  Crowd psychology analysis failed: {e}")
    
    # BONUS FEATURE: Cross-Market Analysis (MEDIUM priority)
    logger.info("\n🔗 Running cross-market correlation analysis...")
    try:
        if 'symbol' in trades_df.columns and len(trades_df['symbol'].unique()) > 1:
            # Analyze correlations between symbols
            correlations = cross_market.analyze_sentiment_correlation(trades_df)
            if correlations:
                logger.info(f"  Cross-Asset Correlations:")
                for (asset1, asset2), corr in list(correlations.items())[:5]:
                    logger.info(f"    {asset1} ↔ {asset2}: {corr:.2f}")
            else:
                logger.info("  Insufficient data for cross-market analysis")
    except Exception as e:
        logger.error(f"  Cross-market analysis failed: {e}")
    
    # Print report
    analytics.print_report(metrics)
    
    # Save report
    analytics.save_report(metrics)
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYTICS SUMMARY")
    print("="*70)
    print(f"✅ Analysis completed successfully!")
    print(f"📊 Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"📈 Win Rate: {metrics.win_rate:.1%}")
    print(f"💰 Profit Factor: {metrics.profit_factor:.2f}")
    print(f"📉 Max Drawdown: {metrics.max_drawdown:.1%}")
    
    # Performance rating
    if metrics.sharpe_ratio > 2.0:
        print(f"🌟 Rating: Excellent")
    elif metrics.sharpe_ratio > 1.0:
        print(f"✅ Rating: Good")
    elif metrics.sharpe_ratio > 0.5:
        print(f"⚠️  Rating: Fair")
    else:
        print(f"❌ Rating: Poor")
    
    print("="*70)
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run advanced analytics')
    parser.add_argument(
        '--trades',
        type=str,
        help='Path to trades CSV file (auto-detects if not provided)'
    )
    parser.add_argument(
        '--equity',
        type=str,
        help='Path to equity CSV file (auto-detects if not provided)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/analytics',
        help='Output directory for reports (default: data/analytics)'
    )
    
    args = parser.parse_args()
    
    try:
        return run_analytics(
            trades_path=args.trades,
            equity_path=args.equity
        )
    except Exception as e:
        logger.error(f"Analytics failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
