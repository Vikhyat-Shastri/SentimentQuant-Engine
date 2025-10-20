"""
Performance benchmarking for sentiment analysis system.

Measures:
- Sentiment analysis latency (target: <100ms per document)
- Signal generation latency (target: <500ms)
- End-to-end processing time
- System throughput (target: >10,000 posts/min)
- Memory usage
- CPU utilization
"""

import time
import psutil
import statistics
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import functools

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Metrics from performance benchmark."""
    
    # Timing metrics (milliseconds)
    sentiment_analysis_latency_avg: float = 0.0
    sentiment_analysis_latency_p50: float = 0.0
    sentiment_analysis_latency_p95: float = 0.0
    sentiment_analysis_latency_p99: float = 0.0
    sentiment_analysis_latency_max: float = 0.0
    
    signal_generation_latency_avg: float = 0.0
    signal_generation_latency_p50: float = 0.0
    signal_generation_latency_p95: float = 0.0
    signal_generation_latency_p99: float = 0.0
    signal_generation_latency_max: float = 0.0
    
    end_to_end_latency_avg: float = 0.0
    end_to_end_latency_p50: float = 0.0
    end_to_end_latency_p95: float = 0.0
    end_to_end_latency_p99: float = 0.0
    end_to_end_latency_max: float = 0.0
    
    # Throughput metrics
    throughput_items_per_sec: float = 0.0
    throughput_items_per_min: float = 0.0
    total_items_processed: int = 0
    total_duration_sec: float = 0.0
    
    # Resource metrics
    memory_usage_mb: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_percent_avg: float = 0.0
    cpu_percent_peak: float = 0.0
    
    # Target compliance
    meets_latency_target: bool = False
    meets_throughput_target: bool = False
    
    # Metadata
    timestamp: str = ""
    duration_seconds: int = 0
    num_measurements: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Benchmark metrics saved to {filepath}")


class PerformanceBenchmark:
    """Performance benchmarking tool."""
    
    # Performance targets
    SENTIMENT_LATENCY_TARGET_MS = 100  # <100ms per document
    SIGNAL_LATENCY_TARGET_MS = 500     # <500ms per signal
    THROUGHPUT_TARGET_PER_MIN = 10000  # >10,000 posts/min
    
    def __init__(self):
        """Initialize benchmark."""
        self.sentiment_timings: List[float] = []
        self.signal_timings: List[float] = []
        self.e2e_timings: List[float] = []
        
        self.start_time = None
        self.end_time = None
        
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        
        self.process = psutil.Process()
        
        logger.info("PerformanceBenchmark initialized")
    
    def timing_decorator(self, metric_list: str):
        """
        Decorator to measure function execution time.
        
        Args:
            metric_list: Name of timing list ('sentiment', 'signal', 'e2e')
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                # Store timing
                if metric_list == 'sentiment':
                    self.sentiment_timings.append(elapsed_ms)
                elif metric_list == 'signal':
                    self.signal_timings.append(elapsed_ms)
                elif metric_list == 'e2e':
                    self.e2e_timings.append(elapsed_ms)
                
                return result
            return wrapper
        return decorator
    
    def record_sentiment_latency(self, latency_ms: float):
        """Record sentiment analysis latency."""
        self.sentiment_timings.append(latency_ms)
    
    def record_signal_latency(self, latency_ms: float):
        """Record signal generation latency."""
        self.signal_timings.append(latency_ms)
    
    def record_e2e_latency(self, latency_ms: float):
        """Record end-to-end latency."""
        self.e2e_timings.append(latency_ms)
    
    def sample_resources(self):
        """Sample current resource usage."""
        try:
            # Memory usage in MB
            mem_info = self.process.memory_info()
            memory_mb = mem_info.rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            
            # CPU percentage
            cpu_percent = self.process.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)
            
        except Exception as e:
            logger.warning(f"Error sampling resources: {e}")
    
    def start_benchmark(self):
        """Start benchmark timer."""
        self.start_time = time.time()
        self.sample_resources()
        logger.info("Benchmark started")
    
    def stop_benchmark(self):
        """Stop benchmark timer."""
        self.end_time = time.time()
        self.sample_resources()
        logger.info("Benchmark stopped")
    
    def calculate_percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile from data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (percentile / 100)
        f = int(k)
        c = f + 1
        
        if c >= len(sorted_data):
            return sorted_data[-1]
        
        d0 = sorted_data[f] * (c - k)
        d1 = sorted_data[c] * (k - f)
        return d0 + d1
    
    def get_metrics(self) -> BenchmarkMetrics:
        """
        Calculate and return benchmark metrics.
        
        Returns:
            BenchmarkMetrics object
        """
        if not self.start_time or not self.end_time:
            logger.warning("Benchmark not started/stopped properly")
            return BenchmarkMetrics()
        
        duration = self.end_time - self.start_time
        total_items = len(self.sentiment_timings)
        
        # Calculate latency metrics
        metrics = BenchmarkMetrics(
            timestamp=datetime.now().isoformat(),
            duration_seconds=int(duration),
            num_measurements=total_items
        )
        
        # Sentiment analysis latency
        if self.sentiment_timings:
            metrics.sentiment_analysis_latency_avg = statistics.mean(self.sentiment_timings)
            metrics.sentiment_analysis_latency_p50 = self.calculate_percentile(self.sentiment_timings, 50)
            metrics.sentiment_analysis_latency_p95 = self.calculate_percentile(self.sentiment_timings, 95)
            metrics.sentiment_analysis_latency_p99 = self.calculate_percentile(self.sentiment_timings, 99)
            metrics.sentiment_analysis_latency_max = max(self.sentiment_timings)
        
        # Signal generation latency
        if self.signal_timings:
            metrics.signal_generation_latency_avg = statistics.mean(self.signal_timings)
            metrics.signal_generation_latency_p50 = self.calculate_percentile(self.signal_timings, 50)
            metrics.signal_generation_latency_p95 = self.calculate_percentile(self.signal_timings, 95)
            metrics.signal_generation_latency_p99 = self.calculate_percentile(self.signal_timings, 99)
            metrics.signal_generation_latency_max = max(self.signal_timings)
        
        # End-to-end latency
        if self.e2e_timings:
            metrics.end_to_end_latency_avg = statistics.mean(self.e2e_timings)
            metrics.end_to_end_latency_p50 = self.calculate_percentile(self.e2e_timings, 50)
            metrics.end_to_end_latency_p95 = self.calculate_percentile(self.e2e_timings, 95)
            metrics.end_to_end_latency_p99 = self.calculate_percentile(self.e2e_timings, 99)
            metrics.end_to_end_latency_max = max(self.e2e_timings)
        
        # Throughput metrics
        if duration > 0:
            metrics.throughput_items_per_sec = total_items / duration
            metrics.throughput_items_per_min = (total_items / duration) * 60
            metrics.total_items_processed = total_items
            metrics.total_duration_sec = duration
        
        # Resource metrics
        if self.memory_samples:
            metrics.memory_usage_mb = statistics.mean(self.memory_samples)
            metrics.memory_peak_mb = max(self.memory_samples)
        
        if self.cpu_samples:
            metrics.cpu_percent_avg = statistics.mean(self.cpu_samples)
            metrics.cpu_percent_peak = max(self.cpu_samples)
        
        # Check target compliance
        metrics.meets_latency_target = (
            metrics.sentiment_analysis_latency_p95 < self.SENTIMENT_LATENCY_TARGET_MS and
            metrics.signal_generation_latency_p95 < self.SIGNAL_LATENCY_TARGET_MS
        )
        metrics.meets_throughput_target = (
            metrics.throughput_items_per_min > self.THROUGHPUT_TARGET_PER_MIN
        )
        
        return metrics
    
    def print_report(self, metrics: Optional[BenchmarkMetrics] = None):
        """
        Print benchmark report.
        
        Args:
            metrics: BenchmarkMetrics object (calculates if not provided)
        """
        if metrics is None:
            metrics = self.get_metrics()
        
        print("\n" + "="*70)
        print("PERFORMANCE BENCHMARK REPORT")
        print("="*70)
        print(f"Timestamp: {metrics.timestamp}")
        print(f"Duration: {metrics.duration_seconds}s")
        print(f"Items Processed: {metrics.total_items_processed:,}")
        print()
        
        print("LATENCY METRICS (milliseconds)")
        print("-" * 70)
        print(f"{'Metric':<30} {'Avg':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'Max':>10}")
        print("-" * 70)
        print(f"{'Sentiment Analysis':<30} "
              f"{metrics.sentiment_analysis_latency_avg:>10.2f} "
              f"{metrics.sentiment_analysis_latency_p50:>10.2f} "
              f"{metrics.sentiment_analysis_latency_p95:>10.2f} "
              f"{metrics.sentiment_analysis_latency_p99:>10.2f} "
              f"{metrics.sentiment_analysis_latency_max:>10.2f}")
        
        print(f"{'Signal Generation':<30} "
              f"{metrics.signal_generation_latency_avg:>10.2f} "
              f"{metrics.signal_generation_latency_p50:>10.2f} "
              f"{metrics.signal_generation_latency_p95:>10.2f} "
              f"{metrics.signal_generation_latency_p99:>10.2f} "
              f"{metrics.signal_generation_latency_max:>10.2f}")
        
        print(f"{'End-to-End':<30} "
              f"{metrics.end_to_end_latency_avg:>10.2f} "
              f"{metrics.end_to_end_latency_p50:>10.2f} "
              f"{metrics.end_to_end_latency_p95:>10.2f} "
              f"{metrics.end_to_end_latency_p99:>10.2f} "
              f"{metrics.end_to_end_latency_max:>10.2f}")
        print()
        
        print("THROUGHPUT METRICS")
        print("-" * 70)
        print(f"Items/second:  {metrics.throughput_items_per_sec:>10.2f}")
        print(f"Items/minute:  {metrics.throughput_items_per_min:>10,.0f}")
        print()
        
        print("RESOURCE USAGE")
        print("-" * 70)
        print(f"Memory (avg):  {metrics.memory_usage_mb:>10.2f} MB")
        print(f"Memory (peak): {metrics.memory_peak_mb:>10.2f} MB")
        print(f"CPU (avg):     {metrics.cpu_percent_avg:>10.2f}%")
        print(f"CPU (peak):    {metrics.cpu_percent_peak:>10.2f}%")
        print()
        
        print("TARGET COMPLIANCE")
        print("-" * 70)
        print(f"Sentiment latency target (<{self.SENTIMENT_LATENCY_TARGET_MS}ms p95): "
              f"{'[PASS]' if metrics.meets_latency_target else '[FAIL]'}")
        print(f"Signal latency target (<{self.SIGNAL_LATENCY_TARGET_MS}ms p95): "
              f"{'[PASS]' if metrics.meets_latency_target else '[FAIL]'}")
        print(f"Throughput target (>{self.THROUGHPUT_TARGET_PER_MIN:,}/min): "
              f"{'[PASS]' if metrics.meets_throughput_target else '[FAIL]'}")
        print()
        
        print("ACTUAL vs TARGET")
        print("-" * 70)
        print(f"Sentiment P95: {metrics.sentiment_analysis_latency_p95:.2f}ms "
              f"(target: {self.SENTIMENT_LATENCY_TARGET_MS}ms)")
        print(f"Signal P95:    {metrics.signal_generation_latency_p95:.2f}ms "
              f"(target: {self.SIGNAL_LATENCY_TARGET_MS}ms)")
        print(f"Throughput:    {metrics.throughput_items_per_min:,.0f}/min "
              f"(target: {self.THROUGHPUT_TARGET_PER_MIN:,}/min)")
        print("="*70)
        print()
    
    def save_report(self, output_dir: str = "data/benchmarking"):
        """
        Save benchmark report to file.
        
        Args:
            output_dir: Directory to save report
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        metrics = self.get_metrics()
        
        # Save JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"{output_dir}/benchmark_{timestamp}.json"
        metrics.to_json(json_path)
        
        # Save text report
        txt_path = f"{output_dir}/benchmark_{timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            import sys
            from io import StringIO
            
            # Redirect stdout to capture print output
            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            
            self.print_report(metrics)
            
            # Restore stdout and get content
            sys.stdout = old_stdout
            content = buffer.getvalue()
            
            f.write(content)
        
        logger.info(f"Benchmark report saved to {output_dir}")
        
        return metrics


def benchmark_function(func: Callable, *args, **kwargs) -> tuple:
    """
    Benchmark a single function call.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Tuple of (result, latency_ms)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    latency_ms = (time.perf_counter() - start) * 1000
    
    return result, latency_ms
