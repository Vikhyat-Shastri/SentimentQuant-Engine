"""
Market Impact Modeling

Models the effect of trade execution on market price:
- Slippage calculation
- Liquidity modeling
- Order book dynamics
- Temporary vs permanent impact
- Optimal execution strategies
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class MarketImpactResult:
    """Result of market impact calculation"""
    intended_price: float
    execution_price: float
    slippage: float  # In price units
    slippage_pct: float  # As percentage
    impact_cost: float  # Total cost of impact
    temporary_impact: float
    permanent_impact: float


@dataclass
class OrderBookState:
    """Snapshot of order book"""
    bids: List[Tuple[float, float]]  # (price, volume) pairs
    asks: List[Tuple[float, float]]
    mid_price: float
    spread: float
    depth: float  # Total volume in top levels


class MarketImpactModel:
    """
    Model market impact of trades
    
    Features:
    - Price impact from order size
    - Slippage estimation
    - Liquidity-based impact
    - Temporary vs permanent impact
    - Optimal execution sizing
    """
    
    def __init__(
        self,
        impact_coefficient: float = 0.1,
        temporary_impact_decay: float = 0.5,
        min_liquidity: float = 1000.0
    ):
        """
        Initialize market impact model
        
        Args:
            impact_coefficient: Market impact strength (higher = more impact)
            temporary_impact_decay: Decay rate of temporary impact (0-1)
            min_liquidity: Minimum liquidity threshold
        """
        self.impact_coefficient = impact_coefficient
        self.temporary_impact_decay = temporary_impact_decay
        self.min_liquidity = min_liquidity
        
        logger.info("MarketImpactModel initialized")
    
    def estimate_liquidity(
        self,
        volume_data: pd.Series,
        window: int = 20
    ) -> float:
        """
        Estimate market liquidity from volume
        
        Args:
            volume_data: Historical volume data
            window: Lookback window
        
        Returns:
            Estimated liquidity
        """
        recent_volume = volume_data.tail(window)
        avg_volume = recent_volume.mean()
        volume_volatility = recent_volume.std()
        
        # Liquidity is high when volume is high and stable
        liquidity = avg_volume / (1 + volume_volatility / avg_volume)
        
        return max(liquidity, self.min_liquidity)
    
    def calculate_price_impact(
        self,
        order_size: float,
        liquidity: float,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Calculate price impact using market microstructure model
        
        Based on Almgren-Chriss model:
        Impact = γ * (order_size / liquidity)^α * volatility
        
        Args:
            order_size: Size of order
            liquidity: Market liquidity
            volatility: Price volatility
        
        Returns:
            (temporary_impact, permanent_impact)
        """
        # Normalize order size by liquidity
        relative_size = order_size / liquidity
        
        # Temporary impact (recovers over time)
        # Higher for larger orders relative to liquidity
        alpha = 0.6  # Power law exponent
        temporary_impact = (
            self.impact_coefficient * 
            (relative_size ** alpha) * 
            volatility
        )
        
        # Permanent impact (information effect)
        # Smaller portion that doesn't recover
        permanent_impact = temporary_impact * (1 - self.temporary_impact_decay)
        
        return temporary_impact, permanent_impact
    
    def simulate_order_book(
        self,
        mid_price: float,
        spread_pct: float = 0.001,
        depth: float = 10000.0
    ) -> OrderBookState:
        """
        Simulate order book state
        
        Args:
            mid_price: Current mid price
            spread_pct: Bid-ask spread as percentage
            depth: Total depth in base units
        
        Returns:
            OrderBookState
        """
        spread = mid_price * spread_pct
        
        # Generate bid levels (descending price)
        bids = []
        for i in range(10):
            level_price = mid_price - spread/2 - i * spread * 0.1
            level_volume = depth * (1 - i * 0.08)  # Decreasing volume
            bids.append((level_price, level_volume))
        
        # Generate ask levels (ascending price)
        asks = []
        for i in range(10):
            level_price = mid_price + spread/2 + i * spread * 0.1
            level_volume = depth * (1 - i * 0.08)
            asks.append((level_price, level_volume))
        
        return OrderBookState(
            bids=bids,
            asks=asks,
            mid_price=mid_price,
            spread=spread,
            depth=depth
        )
    
    def execute_market_order(
        self,
        order_size: float,
        order_book: OrderBookState,
        side: str = 'buy'
    ) -> MarketImpactResult:
        """
        Execute market order and calculate impact
        
        Args:
            order_size: Size of order
            order_book: Current order book state
            side: 'buy' or 'sell'
        
        Returns:
            MarketImpactResult
        """
        intended_price = order_book.mid_price
        
        # Select side of book
        if side == 'buy':
            levels = order_book.asks
            direction = 1
        else:
            levels = order_book.bids
            direction = -1
        
        # Walk through order book levels
        remaining_size = order_size
        total_cost = 0.0
        executed_volume = 0.0
        
        for price, volume in levels:
            if remaining_size <= 0:
                break
            
            # Execute against this level
            executed_at_level = min(remaining_size, volume)
            total_cost += executed_at_level * price
            executed_volume += executed_at_level
            remaining_size -= executed_at_level
        
        # Average execution price
        if executed_volume > 0:
            execution_price = total_cost / executed_volume
        else:
            execution_price = intended_price
        
        # Slippage
        slippage = (execution_price - intended_price) * direction
        slippage_pct = slippage / intended_price
        
        # Impact cost
        impact_cost = abs(slippage * order_size)
        
        # Estimate temporary vs permanent impact
        liquidity = order_book.depth
        volatility = order_book.spread / order_book.mid_price  # Proxy for volatility
        
        temp_impact, perm_impact = self.calculate_price_impact(
            order_size, liquidity, volatility
        )
        
        return MarketImpactResult(
            intended_price=intended_price,
            execution_price=execution_price,
            slippage=slippage,
            slippage_pct=slippage_pct,
            impact_cost=impact_cost,
            temporary_impact=temp_impact * intended_price,
            permanent_impact=perm_impact * intended_price
        )
    
    def optimal_execution_slices(
        self,
        total_size: float,
        time_horizon: int,
        risk_aversion: float = 1.0
    ) -> List[float]:
        """
        Calculate optimal execution schedule (TWAP-like)
        
        Based on Almgren-Chriss framework
        
        Args:
            total_size: Total order size
            time_horizon: Number of time periods
            risk_aversion: Risk aversion parameter (higher = more conservative)
        
        Returns:
            List of order sizes per period
        """
        if time_horizon <= 0:
            return [total_size]
        
        # Simple linear TWAP (Time-Weighted Average Price)
        if risk_aversion < 0.5:
            # Low risk aversion: uniform slices
            slice_size = total_size / time_horizon
            return [slice_size] * time_horizon
        
        # Higher risk aversion: front-load execution
        # Exponential decay schedule
        decay_rate = 0.5 * risk_aversion
        
        slices = []
        remaining = total_size
        
        for t in range(time_horizon):
            # Exponentially decreasing slice size
            fraction = np.exp(-decay_rate * t / time_horizon)
            slice_size = remaining * fraction / sum(np.exp(-decay_rate * i / time_horizon) 
                                                     for i in range(t, time_horizon))
            
            slices.append(slice_size)
            remaining -= slice_size
        
        # Normalize to exactly match total size
        slices = [s * total_size / sum(slices) for s in slices]
        
        return slices
    
    def simulate_vwap_execution(
        self,
        total_size: float,
        price_data: pd.DataFrame,
        volume_data: pd.Series
    ) -> Dict[str, float]:
        """
        Simulate VWAP (Volume-Weighted Average Price) execution
        
        Args:
            total_size: Total order size
            price_data: Historical price data
            volume_data: Historical volume data
        
        Returns:
            Execution statistics
        """
        # Calculate VWAP
        vwap = (price_data['close'] * volume_data).sum() / volume_data.sum()
        
        # Simulate execution at VWAP
        execution_price = vwap
        benchmark_price = price_data['close'].iloc[0]  # Arrival price
        
        slippage = execution_price - benchmark_price
        slippage_pct = slippage / benchmark_price
        
        # Estimate costs
        liquidity = volume_data.mean()
        impact_cost = self.impact_coefficient * (total_size / liquidity) * benchmark_price
        
        return {
            'benchmark_price': benchmark_price,
            'execution_price': execution_price,
            'vwap': vwap,
            'slippage': slippage,
            'slippage_pct': slippage_pct,
            'impact_cost': impact_cost,
            'total_cost': abs(slippage * total_size) + impact_cost
        }


# Test function
if __name__ == "__main__":
    print("\n" + "="*80)
    print("MARKET IMPACT MODELING")
    print("="*80)
    
    # Initialize model
    model = MarketImpactModel(
        impact_coefficient=0.1,
        temporary_impact_decay=0.5
    )
    
    # Test 1: Order book execution
    print("\n📊 Test 1: Market Order Execution")
    print("-" * 80)
    
    mid_price = 50000.0
    order_book = model.simulate_order_book(
        mid_price=mid_price,
        spread_pct=0.001,  # 0.1% spread
        depth=100000.0
    )
    
    print(f"Order Book:")
    print(f"   Mid Price: ${mid_price:,.2f}")
    print(f"   Spread: ${order_book.spread:.2f} ({order_book.spread/mid_price:.3%})")
    print(f"   Depth: {order_book.depth:,.0f}")
    
    # Execute different order sizes
    order_sizes = [1000, 5000, 20000, 50000]
    
    for size in order_sizes:
        result = model.execute_market_order(size, order_book, side='buy')
        
        print(f"\n   Order Size: {size:,.0f}")
        print(f"      Intended: ${result.intended_price:,.2f}")
        print(f"      Executed: ${result.execution_price:,.2f}")
        print(f"      Slippage: ${result.slippage:.2f} ({result.slippage_pct:.4%})")
        print(f"      Impact Cost: ${result.impact_cost:,.2f}")
        print(f"      Temp Impact: ${result.temporary_impact:.2f}")
        print(f"      Perm Impact: ${result.permanent_impact:.2f}")
    
    # Test 2: Optimal execution slicing
    print("\n\n📈 Test 2: Optimal Execution Schedule")
    print("-" * 80)
    
    total_order = 100000
    time_periods = 10
    
    for risk_aversion in [0.3, 1.0, 2.0]:
        slices = model.optimal_execution_slices(
            total_order,
            time_periods,
            risk_aversion
        )
        
        print(f"\nRisk Aversion: {risk_aversion}")
        print(f"   Slices: {[f'{s:,.0f}' for s in slices[:5]]}...")
        print(f"   Total: {sum(slices):,.0f}")
        print(f"   Max Slice: {max(slices):,.0f}")
        print(f"   Min Slice: {min(slices):,.0f}")
    
    # Test 3: VWAP execution
    print("\n\n💹 Test 3: VWAP Execution Simulation")
    print("-" * 80)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    prices = 50000 * (1 + np.cumsum(np.random.normal(0, 0.001, 100)))
    volumes = np.random.uniform(1e6, 5e6, 100)
    
    price_data = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    volume_data = pd.Series(volumes, index=dates)
    
    vwap_result = model.simulate_vwap_execution(
        total_size=10000,
        price_data=price_data,
        volume_data=volume_data
    )
    
    print(f"Order Size: 10,000")
    print(f"   Benchmark Price: ${vwap_result['benchmark_price']:,.2f}")
    print(f"   VWAP: ${vwap_result['vwap']:,.2f}")
    print(f"   Execution Price: ${vwap_result['execution_price']:,.2f}")
    print(f"   Slippage: ${vwap_result['slippage']:.2f} ({vwap_result['slippage_pct']:.4%})")
    print(f"   Impact Cost: ${vwap_result['impact_cost']:,.2f}")
    print(f"   Total Cost: ${vwap_result['total_cost']:,.2f}")
    
    # Test 4: Liquidity estimation
    print("\n\n💧 Test 4: Liquidity Estimation")
    print("-" * 80)
    
    liquidity = model.estimate_liquidity(volume_data, window=20)
    print(f"Estimated Liquidity: {liquidity:,.0f}")
    print(f"Avg Volume (20 periods): {volume_data.tail(20).mean():,.0f}")
    print(f"Volume Volatility: {volume_data.tail(20).std():,.0f}")
    
    print(f"\n✅ Market impact modeling complete!")
