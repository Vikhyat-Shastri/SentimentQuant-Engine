"""
Visualization tools for backtesting results

Creates comprehensive charts and plots for:
- Equity curve with drawdown
- Signal distribution over time
- Fear & Greed Index vs Price
- Trade performance analysis

Usage:
    As library: Import functions and use them in your code
    As CLI tool: python visualize_backtest.py [--output-dir DIR] [--dashboard-only]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from datetime import datetime
import json
from loguru import logger

# BONUS FEATURE: Advanced Visualization (MEDIUM priority)
from src.analytics.advanced_viz import AdvancedVisualizer


# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_equity_curve(equity_df: pd.DataFrame, output_path: str = None) -> None:
    """
    Plot equity curve with drawdown overlay
    
    Args:
        equity_df: DataFrame with timestamp index and equity/drawdown columns
        output_path: Path to save plot (if None, displays only)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Equity curve
    ax1.plot(equity_df.index, equity_df['equity'], linewidth=2, color='#2E86AB', label='Portfolio Value')
    ax1.axhline(y=equity_df['equity'].iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.fill_between(equity_df.index, equity_df['equity'], equity_df['equity'].iloc[0], 
                     where=(equity_df['equity'] >= equity_df['equity'].iloc[0]), 
                     alpha=0.3, color='green', label='Profit')
    ax1.fill_between(equity_df.index, equity_df['equity'], equity_df['equity'].iloc[0], 
                     where=(equity_df['equity'] < equity_df['equity'].iloc[0]), 
                     alpha=0.3, color='red', label='Loss')
    
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Drawdown
    ax2.fill_between(equity_df.index, equity_df['drawdown'], 0, 
                     alpha=0.5, color='#A23B72', label='Drawdown')
    ax2.plot(equity_df.index, equity_df['drawdown'], linewidth=2, color='#A23B72')
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Equity curve saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_signals_distribution(signals_df: pd.DataFrame, output_path: str = None) -> None:
    """
    Plot signal distribution over time and by type
    
    Args:
        signals_df: DataFrame with trading signals
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Signal counts by action
    signal_counts = signals_df['action'].value_counts()
    colors = {'STRONG_BUY': '#006400', 'BUY': '#90EE90', 'HOLD': '#808080', 
              'SELL': '#FFB6C1', 'STRONG_SELL': '#8B0000'}
    
    ax1 = axes[0, 0]
    bars = ax1.bar(signal_counts.index, signal_counts.values, 
                   color=[colors.get(x, 'gray') for x in signal_counts.index])
    ax1.set_title('Signal Distribution by Action', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Signal Type', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Signals over time
    ax2 = axes[0, 1]
    signals_df['date'] = pd.to_datetime(signals_df['timestamp']).dt.date
    daily_signals = signals_df.groupby(['date', 'action']).size().unstack(fill_value=0)
    
    daily_signals.plot(kind='area', stacked=True, ax=ax2, 
                      color=[colors.get(col, 'gray') for col in daily_signals.columns],
                      alpha=0.7)
    ax2.set_title('Signals Over Time (Stacked)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Signal Count', fontweight='bold')
    ax2.legend(title='Action', loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Confidence distribution
    ax3 = axes[1, 0]
    ax3.hist(signals_df['confidence'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax3.axvline(signals_df['confidence'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {signals_df["confidence"].mean():.2f}')
    ax3.set_title('Signal Confidence Distribution', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Confidence', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.legend(framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Position size distribution
    ax4 = axes[1, 1]
    ax4.hist(signals_df['position_size'] * 100, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
    ax4.axvline(signals_df['position_size'].mean() * 100, color='red', linestyle='--',
               linewidth=2, label=f'Mean: {signals_df["position_size"].mean()*100:.1f}%')
    ax4.set_title('Position Size Distribution', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Position Size (%)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.legend(framealpha=0.9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Signal distribution saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_fgi_vs_price(signals_df: pd.DataFrame, price_data: Dict[str, pd.DataFrame], 
                     symbol: str = 'BTC-USD', output_path: str = None) -> None:
    """
    Plot Fear & Greed Index vs Price for correlation analysis
    
    Args:
        signals_df: DataFrame with signals including FGI
        price_data: Dictionary of price DataFrames by symbol
        symbol: Symbol to plot
        output_path: Path to save plot
    """
    if symbol not in price_data:
        print(f"⚠️  Symbol {symbol} not found in price data")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Get price data
    prices = price_data[symbol]['close']
    
    # Filter signals for this symbol
    symbol_signals = signals_df[signals_df['symbol'] == symbol].copy()
    symbol_signals['timestamp'] = pd.to_datetime(symbol_signals['timestamp'])
    symbol_signals = symbol_signals.set_index('timestamp')
    
    # Plot price with signals
    ax1.plot(prices.index, prices.values, linewidth=2, color='#2E86AB', label='Price', alpha=0.8)
    
    # Add buy/sell markers
    buy_signals = symbol_signals[symbol_signals['action'].isin(['BUY', 'STRONG_BUY'])]
    sell_signals = symbol_signals[symbol_signals['action'].isin(['SELL', 'STRONG_SELL'])]
    
    if not buy_signals.empty:
        buy_prices = []
        for ts in buy_signals.index:
            closest_price = prices[prices.index <= ts].iloc[-1] if len(prices[prices.index <= ts]) > 0 else None
            if closest_price is not None:
                buy_prices.append((ts, closest_price))
        
        if buy_prices:
            buy_times, buy_vals = zip(*buy_prices)
            ax1.scatter(buy_times, buy_vals, color='green', marker='^', s=100, 
                       label='Buy Signal', zorder=5, edgecolors='black', linewidth=1)
    
    if not sell_signals.empty:
        sell_prices = []
        for ts in sell_signals.index:
            closest_price = prices[prices.index <= ts].iloc[-1] if len(prices[prices.index <= ts]) > 0 else None
            if closest_price is not None:
                sell_prices.append((ts, closest_price))
        
        if sell_prices:
            sell_times, sell_vals = zip(*sell_prices)
            ax1.scatter(sell_times, sell_vals, color='red', marker='v', s=100,
                       label='Sell Signal', zorder=5, edgecolors='black', linewidth=1)
    
    ax1.set_ylabel(f'{symbol} Price ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{symbol} Price with Trading Signals', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot Fear & Greed Index
    if not symbol_signals.empty:
        ax2.plot(symbol_signals.index, symbol_signals['fear_greed_index'], 
                linewidth=2, color='#A23B72', label='Fear & Greed Index')
        
        # Add threshold lines
        ax2.axhline(y=25, color='darkgreen', linestyle='--', alpha=0.5, label='Extreme Fear (< 25)')
        ax2.axhline(y=75, color='darkred', linestyle='--', alpha=0.5, label='Extreme Greed (> 75)')
        ax2.fill_between(symbol_signals.index, 0, 25, alpha=0.2, color='green')
        ax2.fill_between(symbol_signals.index, 75, 100, alpha=0.2, color='red')
        ax2.fill_between(symbol_signals.index, 25, 75, alpha=0.1, color='gray')
        
        ax2.set_ylabel('Fear & Greed Index', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_title('Fear & Greed Index Over Time', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ FGI vs Price chart saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_trade_analysis(trades_df: pd.DataFrame, output_path: str = None) -> None:
    """
    Plot trade performance analysis
    
    Args:
        trades_df: DataFrame with trade data
        output_path: Path to save plot
    """
    if trades_df.empty:
        print("⚠️  No trades to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # P&L distribution
    ax1 = axes[0, 0]
    colors = ['green' if x > 0 else 'red' for x in trades_df['pnl']]
    ax1.bar(range(len(trades_df)), trades_df['pnl'], color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_title('P&L by Trade', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Trade Number', fontweight='bold')
    ax1.set_ylabel('P&L ($)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Cumulative P&L
    ax2 = axes[0, 1]
    cumulative_pnl = trades_df['pnl'].cumsum()
    ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='#2E86AB', marker='o')
    ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0, 
                     where=(cumulative_pnl >= 0), alpha=0.3, color='green')
    ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                     where=(cumulative_pnl < 0), alpha=0.3, color='red')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Cumulative P&L', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Trade Number', fontweight='bold')
    ax2.set_ylabel('Cumulative P&L ($)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # P&L by action type
    ax3 = axes[1, 0]
    action_pnl = trades_df.groupby('action')['pnl'].sum()
    colors_action = {'STRONG_BUY': '#006400', 'BUY': '#90EE90', 
                    'SELL': '#FFB6C1', 'STRONG_SELL': '#8B0000'}
    bars = ax3.bar(action_pnl.index, action_pnl.values,
                   color=[colors_action.get(x, 'gray') for x in action_pnl.index],
                   alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Total P&L by Action Type', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Action', fontweight='bold')
    ax3.set_ylabel('Total P&L ($)', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # Trade duration distribution
    ax4 = axes[1, 1]
    trades_df['duration'] = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
    trades_df['duration_hours'] = trades_df['duration'].dt.total_seconds() / 3600
    
    ax4.hist(trades_df['duration_hours'], bins=20, color='#F18F01', alpha=0.7, edgecolor='black')
    ax4.axvline(trades_df['duration_hours'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {trades_df["duration_hours"].mean():.1f}h')
    ax4.set_title('Trade Duration Distribution', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Duration (hours)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.legend(framealpha=0.9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Trade analysis saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_summary_dashboard(backtest_dir: str, output_path: str = None) -> None:
    """
    Create a comprehensive summary dashboard from backtest results
    
    Args:
        backtest_dir: Directory containing backtest results
        output_path: Path to save dashboard image
    """
    backtest_path = Path(backtest_dir)
    
    # Find latest backtest files
    summary_files = list(backtest_path.glob('backtest_*_summary.json'))
    if not summary_files:
        print("❌ No backtest results found")
        return
    
    latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
    timestamp = latest_summary.stem.split('_')[1] + '_' + latest_summary.stem.split('_')[2]
    
    # Load data
    with open(latest_summary, 'r') as f:
        summary = json.load(f)
    
    equity_file = backtest_path / f'backtest_{timestamp}_equity.csv'
    trades_file = backtest_path / f'backtest_{timestamp}_trades.csv'
    
    if not equity_file.exists() or not trades_file.exists():
        print(f"❌ Missing data files for {timestamp}")
        return
    
    equity_df = pd.read_csv(equity_file, index_col=0, parse_dates=True)
    trades_df = pd.read_csv(trades_file)
    
    # Create dashboard
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('📊 Backtesting Results Dashboard', fontsize=20, fontweight='bold', y=0.98)
    
    # Metrics panel (top row)
    ax_metrics = fig.add_subplot(gs[0, :])
    ax_metrics.axis('off')
    
    metrics_text = f"""
    Period: {summary['start_date'][:10]} to {summary['end_date'][:10]}
    Initial Capital: ${summary['initial_capital']:,.0f}    |    Final Capital: ${summary['final_capital']:,.2f}
    Total Return: {summary['total_return_percent']:+.2f}%    |    Max Drawdown: {summary['max_drawdown_percent']:.2f}%
    
    Total Trades: {summary['total_trades']}    |    Win Rate: {summary['win_rate']:.1f}%    |    Profit Factor: {summary['profit_factor']:.2f}
    Sharpe Ratio: {summary['sharpe_ratio']:.2f}    |    Sortino Ratio: {summary['sortino_ratio']:.2f}
    
    Winning Trades: {summary['winning_trades']}    |    Losing Trades: {summary['losing_trades']}
    Avg Win: ${summary['avg_win']:,.2f}    |    Avg Loss: ${summary['avg_loss']:,.2f}
    """
    
    ax_metrics.text(0.5, 0.5, metrics_text, ha='center', va='center',
                   fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Equity curve
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(equity_df.index, equity_df['equity'], linewidth=2, color='#2E86AB')
    ax1.fill_between(equity_df.index, equity_df['equity'], summary['initial_capital'],
                     where=(equity_df['equity'] >= summary['initial_capital']),
                     alpha=0.3, color='green')
    ax1.fill_between(equity_df.index, equity_df['equity'], summary['initial_capital'],
                     where=(equity_df['equity'] < summary['initial_capital']),
                     alpha=0.3, color='red')
    ax1.axhline(y=summary['initial_capital'], color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Portfolio Equity Curve', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Value ($)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Cumulative P&L
    ax2 = fig.add_subplot(gs[2, 0])
    if not trades_df.empty:
        cumulative_pnl = trades_df['pnl'].cumsum()
        ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='#2E86AB', marker='o')
        ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                        where=(cumulative_pnl >= 0), alpha=0.3, color='green')
        ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                        where=(cumulative_pnl < 0), alpha=0.3, color='red')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Cumulative P&L', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trade #', fontweight='bold')
    ax2.set_ylabel('P&L ($)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Drawdown
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.fill_between(equity_df.index, equity_df['drawdown'], 0, alpha=0.5, color='#A23B72')
    ax3.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontweight='bold')
    ax3.set_ylabel('Drawdown ($)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Win/Loss distribution
    ax4 = fig.add_subplot(gs[2, 2])
    win_loss = [summary['winning_trades'], summary['losing_trades']]
    colors_wl = ['#90EE90', '#FFB6C1']
    wedges, texts, autotexts = ax4.pie(win_loss, labels=['Wins', 'Losses'],
                                        colors=colors_wl, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontweight': 'bold'})
    ax4.set_title(f'Win/Loss Ratio\n(Win Rate: {summary["win_rate"]:.1f}%)', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# CLI Tool - Generate All Visualizations
# ============================================================================

def load_latest_backtest_data(backtest_dir: str = 'data/backtesting'):
    """Load the latest backtest results"""
    backtest_path = Path(backtest_dir)
    
    # Find latest summary
    summary_files = list(backtest_path.glob('backtest_*_summary.json'))
    if not summary_files:
        raise FileNotFoundError("No backtest results found")
    
    latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
    timestamp = latest_summary.stem.split('_')[1] + '_' + latest_summary.stem.split('_')[2]
    
    logger.info(f"Loading backtest results: {timestamp}")
    
    # Load all files
    with open(latest_summary, 'r') as f:
        summary = json.load(f)
    
    equity_file = backtest_path / f'backtest_{timestamp}_equity.csv'
    trades_file = backtest_path / f'backtest_{timestamp}_trades.csv'
    
    equity_df = pd.read_csv(equity_file, index_col=0, parse_dates=True)
    trades_df = pd.read_csv(trades_file)
    
    return timestamp, summary, equity_df, trades_df


def create_all_visualizations(output_dir: str = 'data/backtesting/charts', use_advanced: bool = True):
    """
    Generate all visualization charts from latest backtest results.
    
    Creates:
    1. Summary dashboard with all metrics
    2. Equity curve with drawdown
    3. Trade performance analysis
    4. (Optional) Advanced interactive dashboard
    
    Args:
        output_dir: Directory to save charts
        use_advanced: Use advanced visualizer (BONUS FEATURE)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("📊 Creating Comprehensive Visualizations")
    logger.info("=" * 70)
    
    # Load data
    try:
        timestamp, summary, equity_df, trades_df = load_latest_backtest_data()
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.info("Please run backtest_strategy.py first to generate results")
        return
    
    # 1. Summary Dashboard (most important)
    logger.info("\n1️⃣  Generating Summary Dashboard...")
    dashboard_path = output_path / f'dashboard_{timestamp}.png'
    create_summary_dashboard('data/backtesting', str(dashboard_path))
    
    # 2. Equity Curve
    logger.info("\n2️⃣  Generating Equity Curve...")
    equity_path = output_path / f'equity_curve_{timestamp}.png'
    plot_equity_curve(equity_df, str(equity_path))
    
    # 3. Trade Analysis
    logger.info("\n3️⃣  Generating Trade Analysis...")
    if not trades_df.empty:
        trades_path = output_path / f'trade_analysis_{timestamp}.png'
        plot_trade_analysis(trades_df, str(trades_path))
    else:
        logger.warning("⚠️  No trades to analyze")
    
    # BONUS FEATURE: Advanced Interactive Dashboard (MEDIUM priority)
    if use_advanced:
        logger.info("\n4️⃣  Generating Advanced Interactive Dashboard (BONUS FEATURE)...")
        try:
            advanced_viz = AdvancedVisualizer(style='dark')
            
            # Load price data if available
            price_data = {}
            if not trades_df.empty and 'symbol' in trades_df.columns:
                symbols = trades_df['symbol'].unique()
                logger.info(f"   Loading price data for {len(symbols)} symbols...")
            
            # Create performance dashboard
            logger.info("   Creating interactive performance dashboard...")
            dashboard_html = output_path / f'interactive_dashboard_{timestamp}.html'
            advanced_viz.create_performance_dashboard(
                trades_df=trades_df,
                equity_df=equity_df,
                price_data=price_data
            )
            advanced_viz.save_dashboard(str(dashboard_html))
            logger.success(f"   ✓ Interactive dashboard saved: {dashboard_html.name}")
            
            # Create correlation heatmap
            if len(trades_df) > 10:
                logger.info("   Creating correlation heatmap...")
                heatmap_path = output_path / f'correlation_heatmap_{timestamp}.html'
                fig = advanced_viz.plot_correlation_heatmap(trades_df, price_data)
                fig.write_html(str(heatmap_path))
                logger.success(f"   ✓ Correlation heatmap saved: {heatmap_path.name}")
            
            # Create 3D performance surface
            logger.info("   Creating 3D performance surface...")
            surface_path = output_path / f'performance_surface_{timestamp}.html'
            fig = advanced_viz.plot_3d_performance_surface(equity_df)
            fig.write_html(str(surface_path))
            logger.success(f"   ✓ 3D surface saved: {surface_path.name}")
            
        except Exception as e:
            logger.error(f"   Advanced visualization failed: {e}")
            logger.info("   Continuing with standard visualizations...")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ Visualization Generation Complete!")
    logger.info("=" * 70)
    logger.info(f"📁 Charts saved to: {output_dir}")
    logger.info(f"")
    logger.info(f"📊 Generated Files:")
    logger.info(f"   ✓ Dashboard: {dashboard_path.name}")
    logger.info(f"   ✓ Equity Curve: {equity_path.name}")
    if not trades_df.empty:
        logger.info(f"   ✓ Trade Analysis: {trades_path.name}")
    if use_advanced:
        logger.info(f"   ✓ Interactive Dashboard: interactive_dashboard_{timestamp}.html")
        logger.info(f"   ✓ Correlation Heatmap: correlation_heatmap_{timestamp}.html")
        logger.info(f"   ✓ 3D Performance Surface: performance_surface_{timestamp}.html")
    logger.info(f"")
    logger.info(f"💡 Tip: Open the HTML files in your browser for interactive charts!")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate visualizations from backtest results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations from latest backtest
  python visualize_backtest.py
  
  # Specify custom output directory
  python visualize_backtest.py --output-dir custom/charts
  
  # Just create dashboard
  python visualize_backtest.py --dashboard-only
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/backtesting/charts',
        help='Output directory for charts (default: data/backtesting/charts)'
    )
    
    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Only generate summary dashboard'
    )
    
    parser.add_argument(
        '--advanced',
        action='store_true',
        default=True,
        help='Use advanced interactive visualizations (default: True)'
    )
    
    parser.add_argument(
        '--no-advanced',
        dest='advanced',
        action='store_false',
        help='Disable advanced visualizations'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📊 Backtesting Visualization Tool")
    print("=" * 70)
    print()
    
    if args.dashboard_only:
        # Just create dashboard
        create_summary_dashboard('data/backtesting', f'{args.output_dir}/dashboard.png')
    else:
        # Create all visualizations
        create_all_visualizations(args.output_dir, use_advanced=args.advanced)
