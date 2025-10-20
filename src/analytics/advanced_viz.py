"""
Advanced Visualization

Creates interactive dashboards and advanced visualizations:
- Real-time sentiment heatmaps
- Interactive Plotly dashboards
- Network graphs
- 3D visualizations
- Correlation matrices
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
from loguru import logger


class AdvancedVisualizer:
    """
    Create advanced interactive visualizations
    
    Features:
    - Real-time sentiment heatmaps
    - Multi-panel dashboards
    - Network graphs
    - 3D scatter plots
    - Correlation matrices
    """
    
    def __init__(self):
        """Initialize advanced visualizer"""
        self.theme = 'plotly_dark'
        logger.info("AdvancedVisualizer initialized")
    
    def create_sentiment_heatmap(
        self,
        sentiment_data: pd.DataFrame,
        title: str = "Sentiment Heatmap Over Time"
    ) -> go.Figure:
        """
        Create real-time sentiment heatmap
        
        Args:
            sentiment_data: DataFrame with columns:
                - timestamp: datetime
                - source: data source (twitter, reddit, news)
                - sentiment: sentiment score
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Pivot data for heatmap
        if 'source' not in sentiment_data.columns or 'sentiment' not in sentiment_data.columns:
            logger.warning("Missing required columns for heatmap")
            return go.Figure()
        
        # Group by time and source
        sentiment_data['hour'] = pd.to_datetime(sentiment_data['timestamp']).dt.floor('h')
        
        heatmap_data = sentiment_data.pivot_table(
            index='source',
            columns='hour',
            values='sentiment',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title="Sentiment"),
            hovertemplate='Source: %{y}<br>Time: %{x}<br>Sentiment: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Source",
            template=self.theme,
            height=400
        )
        
        return fig
    
    def create_dashboard(
        self,
        price_data: pd.DataFrame,
        sentiment_data: pd.DataFrame,
        signals: pd.DataFrame = None
    ) -> go.Figure:
        """
        Create comprehensive interactive dashboard
        
        Args:
            price_data: Price OHLCV data
            sentiment_data: Sentiment data with timestamp and score
            signals: Optional trading signals
        
        Returns:
            Multi-panel Plotly dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Price Action', 'Volume',
                'Sentiment Score', 'Sentiment Distribution',
                'Returns Distribution', 'Volatility',
                'Cumulative Returns', 'Signal Performance'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Price Action
        if 'close' in price_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['close'],
                    name='Close Price',
                    line=dict(color='cyan', width=2)
                ),
                row=1, col=1
            )
        
        # 2. Volume
        if 'volume' in price_data.columns:
            fig.add_trace(
                go.Bar(
                    x=price_data.index,
                    y=price_data['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
        
        # 3. Sentiment Score
        if 'sentiment' in sentiment_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data['timestamp'] if 'timestamp' in sentiment_data.columns else sentiment_data.index,
                    y=sentiment_data['sentiment'],
                    name='Sentiment',
                    line=dict(color='gold', width=2),
                    fill='tozeroy'
                ),
                row=2, col=1
            )
            
            # Add neutral line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # 4. Sentiment Distribution
        if 'sentiment' in sentiment_data.columns:
            fig.add_trace(
                go.Histogram(
                    x=sentiment_data['sentiment'],
                    name='Sentiment Dist',
                    marker_color='gold',
                    nbinsx=50
                ),
                row=2, col=2
            )
        
        # 5. Returns Distribution
        if 'close' in price_data.columns:
            returns = price_data['close'].pct_change().dropna()
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name='Returns Dist',
                    marker_color='lightgreen',
                    nbinsx=50
                ),
                row=3, col=1
            )
        
        # 6. Volatility
        if 'close' in price_data.columns:
            volatility = price_data['close'].pct_change().rolling(24).std() * np.sqrt(24)
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=volatility,
                    name='Volatility',
                    line=dict(color='red', width=2)
                ),
                row=3, col=2
            )
        
        # 7. Cumulative Returns
        if 'close' in price_data.columns:
            cum_returns = (1 + price_data['close'].pct_change()).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=cum_returns,
                    name='Cumulative Returns',
                    line=dict(color='lime', width=2)
                ),
                row=4, col=1
            )
        
        # 8. Signal Performance
        if signals is not None and 'signal' in signals.columns:
            signal_returns = []
            for i in range(len(signals)):
                if signals['signal'].iloc[i] != 0 and i < len(price_data) - 1:
                    entry_price = price_data['close'].iloc[i]
                    exit_price = price_data['close'].iloc[i+1]
                    ret = (exit_price - entry_price) / entry_price * signals['signal'].iloc[i]
                    signal_returns.append(ret)
            
            if signal_returns:
                cum_signal_returns = np.cumprod(1 + np.array(signal_returns))
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(cum_signal_returns))),
                        y=cum_signal_returns,
                        name='Signal Returns',
                        line=dict(color='orange', width=2)
                    ),
                    row=4, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Sentiment Trading Dashboard",
            template=self.theme,
            height=1200,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_correlation_matrix(
        self,
        data_dict: Dict[str, pd.Series],
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        """
        Create interactive correlation matrix
        
        Args:
            data_dict: Dictionary of {name: time_series}
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Align all series
        df = pd.DataFrame(data_dict)
        corr_matrix = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            width=700
        )
        
        return fig
    
    def create_3d_scatter(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        color_col: Optional[str] = None,
        title: str = "3D Scatter Plot"
    ) -> go.Figure:
        """
        Create 3D scatter plot
        
        Args:
            data: DataFrame with data
            x_col: Column for X axis
            y_col: Column for Y axis
            z_col: Column for Z axis
            color_col: Optional column for color coding
            title: Chart title
        
        Returns:
            Plotly 3D figure
        """
        if color_col and color_col in data.columns:
            color = data[color_col]
            colorbar_title = color_col
        else:
            color = 'cyan'
            colorbar_title = None
        
        fig = go.Figure(data=[go.Scatter3d(
            x=data[x_col],
            y=data[y_col],
            z=data[z_col],
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                colorscale='Viridis' if color_col else None,
                showscale=bool(color_col),
                colorbar=dict(title=colorbar_title) if colorbar_title else None
            ),
            text=[f"{x_col}: {x:.2f}<br>{y_col}: {y:.2f}<br>{z_col}: {z:.2f}" 
                  for x, y, z in zip(data[x_col], data[y_col], data[z_col])],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            template=self.theme,
            height=700
        )
        
        return fig
    
    def create_network_graph(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        title: str = "Network Graph"
    ) -> go.Figure:
        """
        Create network graph visualization
        
        Args:
            nodes: List of node names
            edges: List of (source, target, weight) tuples
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Create node positions using force-directed layout (simplified)
        np.random.seed(42)
        node_positions = {node: (np.random.rand(), np.random.rand()) for node in nodes}
        
        # Create edge traces
        edge_traces = []
        for source, target, weight in edges:
            if source in node_positions and target in node_positions:
                x0, y0 = node_positions[source]
                x1, y1 = node_positions[target]
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(
                        width=weight * 2,
                        color='gray'
                    ),
                    hoverinfo='none',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = [node_positions[node][0] for node in nodes]
        node_y = [node_positions[node][1] for node in nodes]
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='white')
            ),
            text=nodes,
            textposition='top center',
            hoverinfo='text',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def create_candlestick_with_signals(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame = None,
        title: str = "Price with Trading Signals"
    ) -> go.Figure:
        """
        Create candlestick chart with trading signals overlay
        
        Args:
            price_data: OHLCV data
            signals: Optional trading signals
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Candlestick
        fig = go.Figure(data=[go.Candlestick(
            x=price_data.index,
            open=price_data['open'] if 'open' in price_data.columns else price_data['close'],
            high=price_data['high'] if 'high' in price_data.columns else price_data['close'],
            low=price_data['low'] if 'low' in price_data.columns else price_data['close'],
            close=price_data['close'],
            name='Price'
        )])
        
        # Add signals
        if signals is not None and 'signal' in signals.columns:
            buy_signals = signals[signals['signal'] == 1]
            sell_signals = signals[signals['signal'] == -1]
            
            if len(buy_signals) > 0:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=price_data.loc[buy_signals.index, 'close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='lime',
                        line=dict(width=2, color='white')
                    ),
                    name='Buy Signal'
                ))
            
            if len(sell_signals) > 0:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=price_data.loc[sell_signals.index, 'close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(width=2, color='white')
                    ),
                    name='Sell Signal'
                ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        
        return fig


# Test function
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='h')
    
    # Price data
    price = 50000 * (1 + np.cumsum(np.random.normal(0, 0.01, len(dates))))
    price_data = pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.normal(0, 100, len(dates)),
        'high': price + np.abs(np.random.normal(200, 100, len(dates))),
        'low': price - np.abs(np.random.normal(200, 100, len(dates))),
        'close': price,
        'volume': np.random.uniform(1e9, 5e9, len(dates))
    }).set_index('timestamp')
    
    # Sentiment data
    sentiment_data = pd.DataFrame({
        'timestamp': np.repeat(dates, 3),
        'source': ['twitter', 'reddit', 'news'] * len(dates),
        'sentiment': np.random.normal(0, 0.3, len(dates) * 3)
    })
    
    # Signals
    signal_indices = np.random.choice(len(dates), 20, replace=False)
    signals = pd.DataFrame({
        'signal': np.random.choice([-1, 0, 1], len(dates))
    }, index=dates)
    
    # Create visualizer
    viz = AdvancedVisualizer()
    
    print("\n" + "="*80)
    print("ADVANCED VISUALIZATION")
    print("="*80)
    
    print("\n✨ Creating visualizations...")
    
    # 1. Sentiment Heatmap
    fig_heatmap = viz.create_sentiment_heatmap(sentiment_data)
    print("  ✓ Sentiment heatmap created")
    
    # 2. Dashboard
    fig_dashboard = viz.create_dashboard(price_data, sentiment_data, signals)
    print("  ✓ Interactive dashboard created")
    
    # 3. Correlation Matrix
    corr_data = {
        'Price': price_data['close'],
        'Volume': price_data['volume'],
        'Sentiment': sentiment_data.groupby('timestamp')['sentiment'].mean()
    }
    fig_corr = viz.create_correlation_matrix(corr_data)
    print("  ✓ Correlation matrix created")
    
    # 4. 3D Scatter
    scatter_data = pd.DataFrame({
        'price': price_data['close'][:100],
        'volume': price_data['volume'][:100],
        'returns': price_data['close'][:100].pct_change(),
        'sentiment': sentiment_data.groupby('timestamp')['sentiment'].mean()[:100]
    }).dropna()
    
    fig_3d = viz.create_3d_scatter(
        scatter_data,
        'price', 'volume', 'returns',
        color_col='sentiment',
        title="Price-Volume-Returns 3D View"
    )
    print("  ✓ 3D scatter plot created")
    
    # 5. Network Graph
    nodes = ['BTC', 'ETH', 'Twitter', 'Reddit', 'News']
    edges = [
        ('BTC', 'ETH', 0.8),
        ('Twitter', 'BTC', 0.6),
        ('Reddit', 'BTC', 0.5),
        ('News', 'BTC', 0.7),
        ('Twitter', 'Reddit', 0.4)
    ]
    fig_network = viz.create_network_graph(nodes, edges, "Sentiment Network")
    print("  ✓ Network graph created")
    
    # 6. Candlestick
    fig_candle = viz.create_candlestick_with_signals(price_data, signals)
    print("  ✓ Candlestick chart created")
    
    print(f"\n📊 All visualizations ready!")
    print(f"   - Heatmap: {len(sentiment_data)} sentiment points")
    print(f"   - Dashboard: {len(price_data)} price points, {len(signals[signals['signal'] != 0])} signals")
    print(f"   - 3D Scatter: {len(scatter_data)} points")
    print(f"   - Network: {len(nodes)} nodes, {len(edges)} edges")
    
    # To display figures, uncomment:
    # fig_heatmap.show()
    # fig_dashboard.show()
    # fig_corr.show()
    # fig_3d.show()
    # fig_network.show()
    # fig_candle.show()
