"""Test the complete data ingestion system."""

import time
import signal
import threading
from loguru import logger

from src.utils import thread_manager
from src.ingestion import (
    start_twitter_stream,
    start_reddit_stream,
    start_market_feed
)

def main():
    """Test all data ingestion modules."""
    
    print("=" * 70)
    print("Data Ingestion System Test")
    print("=" * 70)
    print()
    print("This test will:")
    print("1. Start Twitter stream (simulated)")
    print("2. Start Reddit stream (simulated)")
    print("3. Start market data feed (simulated)")
    print("4. Collect data for 30 seconds")
    print("5. Display statistics")
    print()
    print("Press Ctrl+C to stop early")
    print("=" * 70)
    print()
    
    stop_event = threading.Event()
    
    def signal_handler(sig, frame):
        logger.info("Received stop signal")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start data ingestion threads
    logger.info("Starting data ingestion threads...")
    
    thread_manager.start_thread(
        name="twitter_stream",
        target=start_twitter_stream,
        args=()
    )
    
    thread_manager.start_thread(
        name="reddit_stream",
        target=start_reddit_stream,
        args=()
    )
    
    thread_manager.start_thread(
        name="market_feed",
        target=start_market_feed,
        args=()
    )
    
    logger.success("All data ingestion threads started!")
    print()
    
    # Monitor for 30 seconds
    try:
        for i in range(6):  # 6 x 5 seconds = 30 seconds
            if stop_event.is_set():
                break
            
            time.sleep(5)
            
            # Display statistics
            queue_sizes = thread_manager.get_all_queue_sizes()
            thread_status = thread_manager.get_thread_status()
            
            print(f"\n[{i*5+5}s] Status:")
            print(f"  Active threads: {sum(thread_status.values())}")
            print(f"  Raw data queue: {queue_sizes.get('raw_data', 0)} items")
            
            # Sample from queue without removing
            if queue_sizes.get('raw_data', 0) > 0:
                sample = thread_manager.get_data('raw_data', block=False)
                if sample:
                    print(f"  Latest: {sample.data_type.value} from {sample.source}")
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    finally:
        # Stop all threads
        logger.info("\nStopping all threads...")
        stop_event.set()
        thread_manager.stop_all_threads(timeout=5.0)
        
        # Final statistics
        print("\n" + "=" * 70)
        print("Final Statistics:")
        print("=" * 70)
        queue_sizes = thread_manager.get_all_queue_sizes()
        for queue_name, size in queue_sizes.items():
            print(f"  {queue_name}: {size} items")
        print("=" * 70)
        print("\n✅ Test complete!")
        print("\nNext steps:")
        print("1. Build sentiment analysis engine")
        print("2. Process data from raw_data queue")
        print("3. Generate sentiment scores")
        

if __name__ == "__main__":
    main()
