"""
SIMD Vectorization

Optimized vectorized operations using NumPy and numba:
- Batch sentiment scoring
- Vectorized signal generation
- Fast correlation calculations
- Parallel array operations
- JIT compilation
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time
from loguru import logger

# Try to import numba for JIT compilation
try:
    from numba import jit, prange, vectorize, float64
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logger.warning("numba not available - JIT compilation disabled")
    
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def vectorize(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@dataclass
class VectorizationBenchmark:
    """Benchmark results for vectorization"""
    operation: str
    vectorized_time: float
    scalar_time: float
    speedup: float
    data_size: int


class SIMDOperations:
    """
    SIMD-optimized operations for sentiment analysis
    
    Features:
    - Vectorized sentiment scoring
    - Fast correlation calculations
    - Batch signal generation
    - JIT-compiled functions
    """
    
    def __init__(self):
        """Initialize SIMD operations"""
        self.has_numba = HAS_NUMBA
        logger.info(f"SIMDOperations initialized (numba: {self.has_numba})")
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def vectorized_sentiment_score(
        texts_features: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized sentiment scoring
        
        Args:
            texts_features: Feature matrix (N x F)
            weights: Weight vector (F,)
        
        Returns:
            Sentiment scores (N,)
        """
        scores = np.zeros(texts_features.shape[0])
        
        for i in prange(texts_features.shape[0]):
            score = 0.0
            for j in range(texts_features.shape[1]):
                score += texts_features[i, j] * weights[j]
            scores[i] = np.tanh(score)  # Normalize to [-1, 1]
        
        return scores
    
    @staticmethod
    @jit(nopython=True)
    def fast_moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """
        Fast moving average using cumsum trick
        
        Args:
            data: Input array
            window: Window size
        
        Returns:
            Moving average
        """
        cumsum = np.zeros(len(data) + 1)
        cumsum[1:] = np.cumsum(data)
        
        result = np.zeros(len(data))
        for i in range(window - 1, len(data)):
            result[i] = (cumsum[i + 1] - cumsum[i - window + 1]) / window
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_correlation_matrix(data: np.ndarray) -> np.ndarray:
        """
        Fast correlation matrix calculation
        
        Args:
            data: Data matrix (N x M)
        
        Returns:
            Correlation matrix (M x M)
        """
        n, m = data.shape
        
        # Standardize data
        means = np.zeros(m)
        stds = np.zeros(m)
        
        for j in range(m):
            means[j] = np.mean(data[:, j])
            stds[j] = np.std(data[:, j])
        
        standardized = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if stds[j] > 0:
                    standardized[i, j] = (data[i, j] - means[j]) / stds[j]
        
        # Compute correlation
        corr = np.zeros((m, m))
        for i in prange(m):
            for j in range(i, m):
                corr_val = 0.0
                for k in range(n):
                    corr_val += standardized[k, i] * standardized[k, j]
                corr_val /= n
                corr[i, j] = corr_val
                corr[j, i] = corr_val
        
        return corr
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def batch_signal_generation(
        prices: np.ndarray,
        sentiments: np.ndarray,
        short_ma: int = 20,
        long_ma: int = 50,
        sentiment_threshold: float = 0.3
    ) -> np.ndarray:
        """
        Vectorized signal generation
        
        Args:
            prices: Price array
            sentiments: Sentiment array
            short_ma: Short MA period
            long_ma: Long MA period
            sentiment_threshold: Sentiment threshold
        
        Returns:
            Signal array (-1, 0, 1)
        """
        n = len(prices)
        signals = np.zeros(n)
        
        # Calculate moving averages
        short_ma_values = np.zeros(n)
        long_ma_values = np.zeros(n)
        
        for i in range(n):
            if i >= short_ma - 1:
                short_sum = 0.0
                for j in range(short_ma):
                    short_sum += prices[i - j]
                short_ma_values[i] = short_sum / short_ma
            
            if i >= long_ma - 1:
                long_sum = 0.0
                for j in range(long_ma):
                    long_sum += prices[i - j]
                long_ma_values[i] = long_sum / long_ma
        
        # Generate signals
        for i in prange(long_ma, n):
            # Trend signal
            trend_signal = 0
            if short_ma_values[i] > long_ma_values[i]:
                trend_signal = 1
            elif short_ma_values[i] < long_ma_values[i]:
                trend_signal = -1
            
            # Sentiment signal
            sentiment_signal = 0
            if sentiments[i] > sentiment_threshold:
                sentiment_signal = 1
            elif sentiments[i] < -sentiment_threshold:
                sentiment_signal = -1
            
            # Combined signal (both must agree)
            if trend_signal == sentiment_signal and trend_signal != 0:
                signals[i] = trend_signal
        
        return signals
    
    @staticmethod
    @jit(nopython=True)
    def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Fast RSI calculation
        
        Args:
            prices: Price array
            period: RSI period
        
        Returns:
            RSI values
        """
        n = len(prices)
        rsi = np.zeros(n)
        
        if n < period + 1:
            return rsi
        
        # Calculate price changes
        changes = np.zeros(n - 1)
        for i in range(n - 1):
            changes[i] = prices[i + 1] - prices[i]
        
        # Calculate RSI
        for i in range(period, n):
            gains = 0.0
            losses = 0.0
            
            for j in range(period):
                change = changes[i - period + j]
                if change > 0:
                    gains += change
                else:
                    losses -= change
            
            avg_gain = gains / period
            avg_loss = losses / period
            
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    @staticmethod
    def vectorized_batch_normalize(data: np.ndarray) -> np.ndarray:
        """
        Vectorized batch normalization (pure NumPy)
        
        Args:
            data: Input array
        
        Returns:
            Normalized array
        """
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        normalized = (data - mean) / std
        
        return normalized
    
    @staticmethod
    def vectorized_softmax(x: np.ndarray) -> np.ndarray:
        """
        Vectorized softmax (pure NumPy)
        
        Args:
            x: Input array
        
        Returns:
            Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def benchmark_vectorization():
    """Benchmark vectorized vs scalar operations"""
    print("\n⚡ Benchmarking Vectorization")
    print("-" * 80)
    
    ops = SIMDOperations()
    results = []
    
    # Benchmark 1: Sentiment scoring
    print("\n1. Sentiment Scoring:")
    n_texts = 10000
    n_features = 50
    
    features = np.random.randn(n_texts, n_features).astype(np.float64)
    weights = np.random.randn(n_features).astype(np.float64)
    
    # Vectorized
    start = time.perf_counter()
    scores_vec = ops.vectorized_sentiment_score(features, weights)
    vec_time = time.perf_counter() - start
    
    # Scalar
    start = time.perf_counter()
    scores_scalar = np.zeros(n_texts)
    for i in range(n_texts):
        scores_scalar[i] = np.tanh(np.dot(features[i], weights))
    scalar_time = time.perf_counter() - start
    
    speedup = scalar_time / vec_time
    print(f"   Vectorized: {vec_time*1000:.2f} ms")
    print(f"   Scalar: {scalar_time*1000:.2f} ms")
    print(f"   Speedup: {speedup:.2f}x")
    
    results.append(VectorizationBenchmark(
        "Sentiment Scoring",
        vec_time,
        scalar_time,
        speedup,
        n_texts
    ))
    
    # Benchmark 2: Moving Average
    print("\n2. Moving Average:")
    data = np.random.randn(100000).astype(np.float64)
    window = 20
    
    # Vectorized
    start = time.perf_counter()
    ma_vec = ops.fast_moving_average(data, window)
    vec_time = time.perf_counter() - start
    
    # Using pandas (baseline)
    start = time.perf_counter()
    ma_pandas = pd.Series(data).rolling(window).mean().values
    pandas_time = time.perf_counter() - start
    
    speedup = pandas_time / vec_time
    print(f"   Vectorized: {vec_time*1000:.2f} ms")
    print(f"   Pandas: {pandas_time*1000:.2f} ms")
    print(f"   Speedup: {speedup:.2f}x")
    
    results.append(VectorizationBenchmark(
        "Moving Average",
        vec_time,
        pandas_time,
        speedup,
        len(data)
    ))
    
    # Benchmark 3: Correlation Matrix
    print("\n3. Correlation Matrix:")
    n_samples = 1000
    n_vars = 50
    data = np.random.randn(n_samples, n_vars).astype(np.float64)
    
    # Vectorized
    start = time.perf_counter()
    corr_vec = ops.fast_correlation_matrix(data)
    vec_time = time.perf_counter() - start
    
    # NumPy baseline
    start = time.perf_counter()
    corr_numpy = np.corrcoef(data.T)
    numpy_time = time.perf_counter() - start
    
    speedup = numpy_time / vec_time
    print(f"   Vectorized: {vec_time*1000:.2f} ms")
    print(f"   NumPy: {numpy_time*1000:.2f} ms")
    print(f"   Speedup: {speedup:.2f}x")
    
    results.append(VectorizationBenchmark(
        "Correlation Matrix",
        vec_time,
        numpy_time,
        speedup,
        n_samples
    ))
    
    # Benchmark 4: Signal Generation
    print("\n4. Batch Signal Generation:")
    n_points = 10000
    prices = 50000 + np.cumsum(np.random.randn(n_points) * 100)
    sentiments = np.random.randn(n_points) * 0.5
    
    # Vectorized
    start = time.perf_counter()
    signals_vec = ops.batch_signal_generation(prices, sentiments)
    vec_time = time.perf_counter() - start
    
    print(f"   Vectorized: {vec_time*1000:.2f} ms")
    print(f"   Signals generated: {np.sum(signals_vec != 0)}")
    
    # Benchmark 5: RSI Calculation
    print("\n5. Fast RSI:")
    prices = 50000 + np.cumsum(np.random.randn(10000) * 100)
    
    start = time.perf_counter()
    rsi = ops.fast_rsi(prices, period=14)
    vec_time = time.perf_counter() - start
    
    print(f"   Vectorized: {vec_time*1000:.2f} ms")
    print(f"   RSI range: [{rsi[rsi > 0].min():.2f}, {rsi.max():.2f}]")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SIMD VECTORIZATION TEST")
    print("="*80)
    
    if not HAS_NUMBA:
        print("\n⚠️  Warning: numba not installed")
        print("   Install with: pip install numba")
        print("   Performance will be degraded without JIT compilation")
    
    # Run benchmarks
    results = benchmark_vectorization()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_speedup = np.mean([r.speedup for r in results])
    print(f"\nAverage Speedup: {total_speedup:.2f}x")
    
    print(f"\nAll operations:")
    for r in results:
        print(f"   {r.operation:20s}: {r.speedup:6.2f}x  "
              f"({r.data_size:,} items)")
    
    print(f"\n✅ SIMD vectorization complete!")
    print(f"   Using: {'numba JIT' if HAS_NUMBA else 'Pure NumPy'}")
