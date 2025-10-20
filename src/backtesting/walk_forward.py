"""
Walk-Forward Analysis

Implements rolling window optimization for robust strategy validation.
Prevents overfitting by testing on out-of-sample data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import json


@dataclass
class WalkForwardWindow:
    """Single walk-forward window"""
    window_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime
    optimal_params: Dict
    in_sample_return: float
    out_of_sample_return: float
    in_sample_sharpe: float
    out_of_sample_sharpe: float


@dataclass
class WalkForwardResults:
    """Complete walk-forward analysis results"""
    windows: List[WalkForwardWindow]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    parameter_stability: float  # 0-1, higher is more stable
    efficiency_ratio: float  # OOS performance / IS performance


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis Engine
    
    Features:
    - Rolling window optimization
    - Parameter stability analysis
    - Out-of-sample validation
    - Efficiency ratio calculation
    """
    
    def __init__(
        self,
        in_sample_days: int = 90,
        out_of_sample_days: int = 30,
        step_days: int = 30,
        optimization_metric: str = 'sharpe'
    ):
        """
        Initialize walk-forward analyzer
        
        Args:
            in_sample_days: Days for parameter optimization
            out_of_sample_days: Days for out-of-sample testing
            step_days: Days to step forward between windows
            optimization_metric: Metric to optimize ('sharpe', 'return', 'sortino')
        """
        self.in_sample_days = in_sample_days
        self.out_of_sample_days = out_of_sample_days
        self.step_days = step_days
        self.optimization_metric = optimization_metric
        
        logger.info(f"WalkForwardAnalyzer initialized:")
        logger.info(f"  In-sample: {in_sample_days} days")
        logger.info(f"  Out-of-sample: {out_of_sample_days} days")
        logger.info(f"  Step: {step_days} days")
    
    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward windows
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
        
        Returns:
            List of (is_start, is_end, oos_start, oos_end) tuples
        """
        windows = []
        current_start = start_date
        
        while True:
            # In-sample period
            is_start = current_start
            is_end = is_start + timedelta(days=self.in_sample_days)
            
            # Out-of-sample period
            oos_start = is_end
            oos_end = oos_start + timedelta(days=self.out_of_sample_days)
            
            if oos_end > end_date:
                break
            
            windows.append((is_start, is_end, oos_start, oos_end))
            
            # Step forward
            current_start += timedelta(days=self.step_days)
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List]
    ) -> Tuple[Dict, float]:
        """
        Optimize strategy parameters on in-sample data
        
        Args:
            data: Historical data for in-sample period
            strategy_func: Function that runs strategy with given params
                          Should return dict with 'returns' and 'sharpe'
            param_grid: Dictionary of parameter names to list of values
                       e.g., {'threshold': [0.5, 0.6, 0.7]}
        
        Returns:
            Tuple of (optimal_params, optimal_score)
        """
        best_score = -np.inf
        best_params = None
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Simple grid search
        import itertools
        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            
            try:
                # Run strategy with these parameters
                result = strategy_func(data, params)
                
                # Get score based on optimization metric
                if self.optimization_metric == 'sharpe':
                    score = result.get('sharpe', 0)
                elif self.optimization_metric == 'return':
                    score = result.get('returns', 0)
                elif self.optimization_metric == 'sortino':
                    score = result.get('sortino', 0)
                else:
                    score = result.get('sharpe', 0)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Error testing params {params}: {e}")
                continue
        
        if best_params is None:
            # Use default parameters
            best_params = {k: v[0] for k, v in param_grid.items()}
            best_score = 0.0
        
        return best_params, best_score
    
    def test_parameters(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        params: Dict
    ) -> Dict:
        """
        Test parameters on out-of-sample data
        
        Args:
            data: Historical data for out-of-sample period
            strategy_func: Strategy function
            params: Parameters to test
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            result = strategy_func(data, params)
            return result
        except Exception as e:
            logger.error(f"Error testing OOS: {e}")
            return {'returns': 0, 'sharpe': 0}
    
    def run_analysis(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List],
        output_dir: Optional[Path] = None
    ) -> WalkForwardResults:
        """
        Run complete walk-forward analysis
        
        Args:
            data: Complete historical data with datetime index
            strategy_func: Strategy function to test
            param_grid: Parameter grid for optimization
            output_dir: Optional directory to save results
        
        Returns:
            WalkForwardResults with complete analysis
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # Generate windows
        start_date = data.index[0]
        end_date = data.index[-1]
        windows = self.generate_windows(start_date, end_date)
        
        # Run analysis for each window
        wf_windows = []
        all_oos_returns = []
        
        for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Extract data
            is_data = data[is_start:is_end]
            oos_data = data[oos_start:oos_end]
            
            if len(is_data) < 10 or len(oos_data) < 5:
                logger.warning(f"Insufficient data in window {i+1}, skipping")
                continue
            
            # Optimize on in-sample
            optimal_params, is_score = self.optimize_parameters(
                is_data, strategy_func, param_grid
            )
            
            # Get in-sample metrics
            is_result = strategy_func(is_data, optimal_params)
            is_return = is_result.get('returns', 0)
            is_sharpe = is_result.get('sharpe', 0)
            
            # Test on out-of-sample
            oos_result = self.test_parameters(oos_data, strategy_func, optimal_params)
            oos_return = oos_result.get('returns', 0)
            oos_sharpe = oos_result.get('sharpe', 0)
            
            # Store window results
            wf_window = WalkForwardWindow(
                window_id=i+1,
                in_sample_start=is_start,
                in_sample_end=is_end,
                out_of_sample_start=oos_start,
                out_of_sample_end=oos_end,
                optimal_params=optimal_params,
                in_sample_return=is_return,
                out_of_sample_return=oos_return,
                in_sample_sharpe=is_sharpe,
                out_of_sample_sharpe=oos_sharpe
            )
            
            wf_windows.append(wf_window)
            all_oos_returns.append(oos_return)
            
            logger.info(f"  Optimal params: {optimal_params}")
            logger.info(f"  IS: {is_return:.2%} return, {is_sharpe:.2f} Sharpe")
            logger.info(f"  OOS: {oos_return:.2%} return, {oos_sharpe:.2f} Sharpe")
        
        # Calculate aggregate metrics
        total_return = sum(all_oos_returns)
        
        # Calculate Sharpe ratio
        if len(all_oos_returns) > 1:
            sharpe_ratio = np.mean(all_oos_returns) / (np.std(all_oos_returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        cumulative_returns = np.cumprod([1 + r for r in all_oos_returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Calculate parameter stability
        parameter_stability = self._calculate_parameter_stability(wf_windows)
        
        # Calculate efficiency ratio (OOS / IS performance)
        avg_oos_sharpe = np.mean([w.out_of_sample_sharpe for w in wf_windows])
        avg_is_sharpe = np.mean([w.in_sample_sharpe for w in wf_windows])
        efficiency_ratio = avg_oos_sharpe / (avg_is_sharpe + 1e-6)
        
        # Create results
        results = WalkForwardResults(
            windows=wf_windows,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            parameter_stability=parameter_stability,
            efficiency_ratio=efficiency_ratio
        )
        
        # Save results
        if output_dir:
            self._save_results(results, output_dir)
        
        return results
    
    def _calculate_parameter_stability(self, windows: List[WalkForwardWindow]) -> float:
        """
        Calculate stability of optimal parameters across windows
        
        Returns value between 0 (unstable) and 1 (stable)
        """
        if len(windows) < 2:
            return 1.0
        
        # Get all parameter keys
        param_keys = set()
        for w in windows:
            param_keys.update(w.optimal_params.keys())
        
        # Calculate stability for each parameter
        stabilities = []
        
        for key in param_keys:
            values = [w.optimal_params.get(key, 0) for w in windows]
            
            # Normalize values
            min_val = min(values)
            max_val = max(values)
            
            if max_val == min_val:
                stability = 1.0
            else:
                # Calculate coefficient of variation
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / (abs(mean_val) + 1e-6)
                
                # Convert to stability score (lower CV = higher stability)
                stability = 1.0 / (1.0 + cv)
            
            stabilities.append(stability)
        
        return np.mean(stabilities)
    
    def _save_results(self, results: WalkForwardResults, output_dir: Path):
        """Save analysis results to file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = {
            'total_return': results.total_return,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'parameter_stability': results.parameter_stability,
            'efficiency_ratio': results.efficiency_ratio,
            'num_windows': len(results.windows)
        }
        
        summary_file = output_dir / 'walk_forward_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")
        
        # Save detailed windows
        windows_data = []
        for w in results.windows:
            windows_data.append({
                'window_id': w.window_id,
                'in_sample_start': w.in_sample_start.isoformat(),
                'in_sample_end': w.in_sample_end.isoformat(),
                'out_of_sample_start': w.out_of_sample_start.isoformat(),
                'out_of_sample_end': w.out_of_sample_end.isoformat(),
                'optimal_params': w.optimal_params,
                'in_sample_return': w.in_sample_return,
                'out_of_sample_return': w.out_of_sample_return,
                'in_sample_sharpe': w.in_sample_sharpe,
                'out_of_sample_sharpe': w.out_of_sample_sharpe
            })
        
        windows_file = output_dir / 'walk_forward_windows.json'
        with open(windows_file, 'w') as f:
            json.dump(windows_data, f, indent=2)
        
        logger.info(f"Saved window details to {windows_file}")


# Example usage
if __name__ == "__main__":
    # Example strategy function
    def example_strategy(data: pd.DataFrame, params: Dict) -> Dict:
        """Simple threshold-based strategy"""
        threshold = params.get('threshold', 0.5)
        
        # Simulate returns based on threshold
        returns = np.random.randn() * 0.01
        sharpe = np.random.randn()
        
        return {
            'returns': returns,
            'sharpe': sharpe
        }
    
    # Generate example data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    data = pd.DataFrame({
        'price': np.random.randn(len(dates)).cumsum() + 100
    }, index=dates)
    
    # Run walk-forward analysis
    analyzer = WalkForwardAnalyzer(
        in_sample_days=90,
        out_of_sample_days=30,
        step_days=30
    )
    
    param_grid = {
        'threshold': [0.5, 0.6, 0.7, 0.8]
    }
    
    results = analyzer.run_analysis(data, example_strategy, param_grid)
    
    print(f"\nWalk-Forward Results:")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Parameter Stability: {results.parameter_stability:.2f}")
    print(f"Efficiency Ratio: {results.efficiency_ratio:.2f}")
