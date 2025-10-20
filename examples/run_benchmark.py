"""
Run performance benchmark on sentiment analysis system.

Usage:
    python run_benchmark.py --duration 120 --mode simulation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import queue
import time
import threading
from datetime import datetime

from src.benchmarking.performance import PerformanceBenchmark
from src.ingestion.twitter_stream import TwitterStream
from src.ingestion.reddit_stream import RedditStream
from src.ingestion.news_stream import NewsStream
from src.ingestion.market_data import MarketDataFeed
from src.processing.worker import ProcessingWorker
from src.signals.signal_generator import SignalGenerator
from src.utils import thread_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def run_benchmark(duration: int = 120, mode: str = "simulation"):
    """
    Run performance benchmark.
    
    Args:
        duration: Benchmark duration in seconds
        mode: 'simulation' or 'live'
    """
    logger.info(f"Starting performance benchmark (duration: {duration}s, mode: {mode})")
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark()
    
    # Create queues from thread_manager (they're already initialized)
    raw_queue = thread_manager.queues['raw_data']
    sentiment_queue = thread_manager.queues['sentiment_queue']
    signal_queue = thread_manager.queues['signal_queue']
    
    # Initialize components
    logger.info("Initializing system components...")
    
    # Data streams
    twitter = TwitterStream(mode=mode)
    reddit = RedditStream(mode=mode)
    news = NewsStream(mode=mode)
    market = MarketDataFeed()
    
    # Workers with benchmarking (use ProcessingWorker, not SentimentWorker)
    workers = []
    for i in range(2):
        worker = ProcessingWorker(
            worker_id=i + 1,
            batch_size=10,
            batch_timeout=1.0
        )
        workers.append(worker)
    
    # Signal generator
    signal_gen = SignalGenerator()
    
    # Start benchmark timer
    benchmark.start_benchmark()
    
    # Create stop event for data streams
    stop_event = threading.Event()
    
    # Start all threads
    logger.info("Starting threads...")
    
    # Start data streams in threads (they need stop_event, not queue)
    twitter_thread = threading.Thread(
        target=twitter.start,
        args=(stop_event,),
        daemon=True,
        name="TwitterStream"
    )
    twitter_thread.start()
    
    reddit_thread = threading.Thread(
        target=reddit.start,
        args=(stop_event,),
        daemon=True,
        name="RedditStream"
    )
    reddit_thread.start()
    
    news_thread = threading.Thread(
        target=news.start,
        args=(stop_event,),
        daemon=True,
        name="NewsStream"
    )
    news_thread.start()
    
    market_thread = threading.Thread(
        target=market.start,
        args=(stop_event,),
        daemon=True,
        name="MarketDataFeed"
    )
    market_thread.start()
    
    for worker in workers:
        worker.start()
    
    signal_gen.start(sentiment_queue, signal_queue)
    
    # Resource sampling thread
    def sample_resources():
        while not resource_stop_event.is_set():
            benchmark.sample_resources()
            time.sleep(1)
    
    resource_stop_event = threading.Event()
    resource_thread = threading.Thread(target=sample_resources, daemon=True)
    resource_thread.start()
    
    # Benchmark loop - just let the system run
    start_time = time.time()
    
    logger.info(f"Benchmark running for {duration} seconds...")
    logger.info("Workers will report actual processing times to benchmark...")
    
    # Pass benchmark to workers for latency recording
    for worker in workers:
        worker.benchmark = benchmark
    signal_gen.benchmark = benchmark
    
    items_processed = 0
    signals_generated = 0
    
    try:
        while time.time() - start_time < duration:
            # Just monitor queue sizes and count items
            try:
                # Count items from sentiment queue (non-blocking peek)
                current_sentiment_size = sentiment_queue.qsize()
                items_processed = max(items_processed, current_sentiment_size)
            except:
                pass
            
            try:
                # Count signals from signal queue (non-blocking peek)
                current_signal_size = signal_queue.qsize()
                signals_generated = max(signals_generated, current_signal_size)
            except:
                pass
            
            time.sleep(1)  # Check every second
    
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    
    finally:
        # Stop all components
        logger.info("Stopping system components...")
        stop_event.set()
        resource_stop_event.set()
        
        twitter.stop()
        reddit.stop()
        news.stop()
        market.stop()
        
        for worker in workers:
            worker.stop()
        
        signal_gen.stop()
        
        # Stop benchmark timer
        benchmark.stop_benchmark()
        
        # Wait a bit for threads to finish
        time.sleep(2)
        
        # Get actual counts from workers and signal generator
        items_processed = sum(worker.stats['processed_count'] for worker in workers)
        signals_generated = signal_gen.signals_generated
        
        # Calculate and print metrics
        logger.info(f"\nBenchmark complete! Processed {items_processed} items, generated {signals_generated} signals")
        
        metrics = benchmark.get_metrics()
        benchmark.print_report(metrics)
        
        # Save report
        benchmark.save_report()
        
        return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run performance benchmark')
    parser.add_argument(
        '--duration',
        type=int,
        default=120,
        help='Benchmark duration in seconds (default: 120)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='simulation',
        choices=['simulation', 'live'],
        help='Data source mode (default: simulation)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/benchmarking',
        help='Output directory for reports (default: data/benchmarking)'
    )
    
    args = parser.parse_args()
    
    try:
        metrics = run_benchmark(
            duration=args.duration,
            mode=args.mode
        )
        
        # Print summary
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"✅ Benchmark completed successfully!")
        print(f"📊 Total items processed: {metrics.total_items_processed:,}")
        print(f"⚡ Throughput: {metrics.throughput_items_per_min:,.0f} items/min")
        print(f"🎯 Latency targets: {'✅ MET' if metrics.meets_latency_target else '❌ MISSED'}")
        print(f"🚀 Throughput target: {'✅ MET' if metrics.meets_throughput_target else '❌ MISSED'}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
