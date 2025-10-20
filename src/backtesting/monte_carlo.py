"""
Monte Carlo Simulation for Backtesting

Simulates thousands of potential outcomes to understand risk and probability distributions.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import json


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation"""
    num_simulations: int
    final_returns: np.ndarray  # Array of final returns from all simulations
    drawdown_paths: List[np.ndarray]  # Drawdown curves
    mean_return: float
    median_return: float
    std_return: float
    confidence_intervals: Dict[int, tuple]  # e.g., {95: (lower, upper)}
    probability_profit: float
    probability_loss_gt_10: float  # Probability of >10% loss
    worst_case: float
    best_case: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (Expected Shortfall)


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for strategy backtesting
    
    Features:
    - Bootstrap resampling of historical returns
    - Path-dependent simulations
    - Risk metrics (VaR, CVaR)
    - Confidence intervals
    """
    
    def __init__(
        self,
        num_simulations: int = 10000,
        confidence_levels: List[int] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo simulator
        
        Args:
            num_simulations: Number of simulation paths
            confidence_levels: Confidence levels for intervals (e.g., [90, 95, 99])
            random_seed: Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.confidence_levels = confidence_levels or [90, 95, 99]
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.info(f"MonteCarloSimulator initialized:")
        logger.info(f"  Simulations: {num_simulations}")
        logger.info(f"  Confidence levels: {self.confidence_levels}")
    
    def simulate_bootstrap(
        self,
        historical_returns: np.ndarray,
        num_periods: Optional[int] = None
    ) -> MonteCarloResults:
        """
        Run bootstrap Monte Carlo simulation
        
        Randomly samples from historical returns to generate possible future paths.
        
        Args:
            historical_returns: Array of historical period returns
            num_periods: Number of periods to simulate (defaults to len of historical)
        
        Returns:
            MonteCarloResults with statistics
        """
        if num_periods is None:
            num_periods = len(historical_returns)
        
        logger.info(f"Running {self.num_simulations} bootstrap simulations...")
        
        # Storage for results
        final_returns = np.zeros(self.num_simulations)
        drawdown_paths = []
        
        for i in range(self.num_simulations):
            # Bootstrap: sample with replacement
            simulated_returns = np.random.choice(
                historical_returns,
                size=num_periods,
                replace=True
            )
            
            # Calculate cumulative returns
            cumulative = np.cumprod(1 + simulated_returns)
            final_return = cumulative[-1] - 1
            final_returns[i] = final_return
            
            # Calculate drawdowns
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            
            # Store some drawdown paths for visualization
            if i < 100:  # Store first 100 paths
                drawdown_paths.append(drawdowns)
        
        # Calculate statistics
        results = self._calculate_statistics(final_returns, drawdown_paths)
        
        logger.info(f"Simulation complete. Mean return: {results.mean_return:.2%}")
        
        return results
    
    def simulate_parametric(
        self,
        mean_return: float,
        std_return: float,
        num_periods: int,
        distribution: str = 'normal'
    ) -> MonteCarloResults:
        """
        Run parametric Monte Carlo simulation
        
        Assumes returns follow a specific distribution.
        
        Args:
            mean_return: Expected period return
            std_return: Standard deviation of returns
            num_periods: Number of periods to simulate
            distribution: 'normal' or 'student_t'
        
        Returns:
            MonteCarloResults with statistics
        """
        logger.info(f"Running {self.num_simulations} parametric simulations...")
        
        final_returns = np.zeros(self.num_simulations)
        drawdown_paths = []
        
        for i in range(self.num_simulations):
            # Generate returns based on distribution
            if distribution == 'normal':
                simulated_returns = np.random.normal(
                    mean_return,
                    std_return,
                    num_periods
                )
            elif distribution == 'student_t':
                # Student's t-distribution (heavier tails)
                df = 5  # degrees of freedom
                simulated_returns = np.random.standard_t(df, num_periods) * std_return + mean_return
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
            
            # Calculate cumulative returns
            cumulative = np.cumprod(1 + simulated_returns)
            final_return = cumulative[-1] - 1
            final_returns[i] = final_return
            
            # Calculate drawdowns
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            
            if i < 100:
                drawdown_paths.append(drawdowns)
        
        results = self._calculate_statistics(final_returns, drawdown_paths)
        
        logger.info(f"Simulation complete. Mean return: {results.mean_return:.2%}")
        
        return results
    
    def simulate_with_correlation(
        self,
        returns_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
        num_periods: int
    ) -> MonteCarloResults:
        """
        Run Monte Carlo with correlated assets
        
        Args:
            returns_matrix: Historical returns for multiple assets (T x N)
            correlation_matrix: Correlation matrix (N x N)
            num_periods: Number of periods to simulate
        
        Returns:
            MonteCarloResults for portfolio
        """
        logger.info(f"Running {self.num_simulations} correlated simulations...")
        
        # Calculate mean and covariance
        mean_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)
        
        final_returns = np.zeros(self.num_simulations)
        drawdown_paths = []
        
        for i in range(self.num_simulations):
            # Generate correlated returns
            simulated_returns = np.random.multivariate_normal(
                mean_returns,
                cov_matrix,
                size=num_periods
            )
            
            # Portfolio return (equal weight for simplicity)
            portfolio_returns = simulated_returns.mean(axis=1)
            
            # Calculate cumulative returns
            cumulative = np.cumprod(1 + portfolio_returns)
            final_return = cumulative[-1] - 1
            final_returns[i] = final_return
            
            # Calculate drawdowns
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            
            if i < 100:
                drawdown_paths.append(drawdowns)
        
        results = self._calculate_statistics(final_returns, drawdown_paths)
        
        return results
    
    def _calculate_statistics(
        self,
        final_returns: np.ndarray,
        drawdown_paths: List[np.ndarray]
    ) -> MonteCarloResults:
        """Calculate statistics from simulation results"""
        
        # Basic statistics
        mean_return = np.mean(final_returns)
        median_return = np.median(final_returns)
        std_return = np.std(final_returns)
        
        # Confidence intervals
        confidence_intervals = {}
        for level in self.confidence_levels:
            lower_pct = (100 - level) / 2
            upper_pct = 100 - lower_pct
            
            lower = np.percentile(final_returns, lower_pct)
            upper = np.percentile(final_returns, upper_pct)
            
            confidence_intervals[level] = (lower, upper)
        
        # Probabilities
        probability_profit = np.mean(final_returns > 0)
        probability_loss_gt_10 = np.mean(final_returns < -0.1)
        
        # Best/worst cases
        worst_case = np.min(final_returns)
        best_case = np.max(final_returns)
        
        # Value at Risk (VaR) - 95th percentile loss
        var_95 = -np.percentile(final_returns, 5)
        
        # Conditional VaR (CVaR) - expected loss in worst 5% of cases
        worst_5_pct = final_returns[final_returns <= np.percentile(final_returns, 5)]
        cvar_95 = -np.mean(worst_5_pct) if len(worst_5_pct) > 0 else 0.0
        
        return MonteCarloResults(
            num_simulations=self.num_simulations,
            final_returns=final_returns,
            drawdown_paths=drawdown_paths,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            confidence_intervals=confidence_intervals,
            probability_profit=probability_profit,
            probability_loss_gt_10=probability_loss_gt_10,
            worst_case=worst_case,
            best_case=best_case,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def plot_results(
        self,
        results: MonteCarloResults,
        output_path: Optional[Path] = None
    ):
        """
        Plot Monte Carlo results
        
        Args:
            results: MonteCarloResults to plot
            output_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Distribution of final returns
        ax = axes[0, 0]
        ax.hist(results.final_returns * 100, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(results.mean_return * 100, color='r', linestyle='--', label=f'Mean: {results.mean_return:.2%}')
        ax.axvline(results.median_return * 100, color='g', linestyle='--', label=f'Median: {results.median_return:.2%}')
        ax.set_xlabel('Final Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Final Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Confidence intervals
        ax = axes[0, 1]
        levels = sorted(results.confidence_intervals.keys())
        lower_bounds = [results.confidence_intervals[l][0] * 100 for l in levels]
        upper_bounds = [results.confidence_intervals[l][1] * 100 for l in levels]
        
        x = np.arange(len(levels))
        width = 0.35
        
        ax.bar(x, upper_bounds, width, label='Upper Bound', alpha=0.7)
        ax.bar(x, lower_bounds, width, label='Lower Bound', alpha=0.7)
        ax.set_ylabel('Return (%)')
        ax.set_title('Confidence Intervals')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{l}%' for l in levels])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Drawdown paths
        ax = axes[1, 0]
        for dd_path in results.drawdown_paths[:50]:  # Plot first 50
            ax.plot(dd_path * 100, alpha=0.1, color='blue')
        
        # Plot mean drawdown
        mean_dd = np.mean([dd for dd in results.drawdown_paths], axis=0) * 100
        ax.plot(mean_dd, color='red', linewidth=2, label='Mean Drawdown')
        
        ax.set_xlabel('Period')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Simulated Drawdown Paths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Risk metrics
        ax = axes[1, 1]
        metrics = {
            'Probability\nof Profit': results.probability_profit * 100,
            'Probability\nLoss > 10%': results.probability_loss_gt_10 * 100,
            'VaR 95%': results.var_95 * 100,
            'CVaR 95%': results.cvar_95 * 100
        }
        
        bars = ax.bar(metrics.keys(), metrics.values(), alpha=0.7)
        bars[0].set_color('green')
        bars[1].set_color('red')
        bars[2].set_color('orange')
        bars[3].set_color('darkred')
        
        ax.set_ylabel('Value (%)')
        ax.set_title('Risk Metrics')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(
        self,
        results: MonteCarloResults,
        output_dir: Path
    ):
        """Save simulation results to file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = {
            'num_simulations': results.num_simulations,
            'mean_return': float(results.mean_return),
            'median_return': float(results.median_return),
            'std_return': float(results.std_return),
            'confidence_intervals': {
                str(k): [float(v[0]), float(v[1])]
                for k, v in results.confidence_intervals.items()
            },
            'probability_profit': float(results.probability_profit),
            'probability_loss_gt_10': float(results.probability_loss_gt_10),
            'worst_case': float(results.worst_case),
            'best_case': float(results.best_case),
            'var_95': float(results.var_95),
            'cvar_95': float(results.cvar_95)
        }
        
        summary_file = output_dir / 'monte_carlo_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved results to {summary_file}")


# Example usage
if __name__ == "__main__":
    # Generate example historical returns
    np.random.seed(42)
    historical_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    
    # Run bootstrap simulation
    simulator = MonteCarloSimulator(num_simulations=10000)
    results = simulator.simulate_bootstrap(historical_returns, num_periods=252)
    
    # Print results
    print("\nMonte Carlo Simulation Results:")
    print(f"Number of simulations: {results.num_simulations}")
    print(f"Mean return: {results.mean_return:.2%}")
    print(f"Median return: {results.median_return:.2%}")
    print(f"Std deviation: {results.std_return:.2%}")
    print(f"\nConfidence Intervals:")
    for level, (lower, upper) in results.confidence_intervals.items():
        print(f"  {level}%: [{lower:.2%}, {upper:.2%}]")
    print(f"\nRisk Metrics:")
    print(f"Probability of profit: {results.probability_profit:.2%}")
    print(f"Probability of >10% loss: {results.probability_loss_gt_10:.2%}")
    print(f"VaR 95%: {results.var_95:.2%}")
    print(f"CVaR 95%: {results.cvar_95:.2%}")
    print(f"Worst case: {results.worst_case:.2%}")
    print(f"Best case: {results.best_case:.2%}")
    
    # Plot results
    simulator.plot_results(results)
