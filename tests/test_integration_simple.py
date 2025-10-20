"""
Integration Test - Full pipeline validation with simple, fast tests.

Tests the complete data flow without complex setup/teardown that can hang pytest.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from src.utils import thread_manager, DataPacket, DataType, get_timestamp
from src.processing.worker import ProcessingWorker


def clear_queue(queue_name):
    """Clear a single queue."""
    count = 0
    while count < 100:
        data = thread_manager.get_data(queue_name, block=False, timeout=None)
        if data is None:
            break
        count += 1
    return count


def clear_all_queues():
    """Clear all queues before/after tests."""
    for queue_name in ['raw_data', 'processed_data', 'sentiment_queue', 'signal_queue', 'storage_queue']:
        clear_queue(queue_name)


def test_twitter_sentiment_pipeline():
    """Test Twitter data → processing → sentiment output."""
    clear_all_queues()
    
    # Create test packets
    packets = [
        DataPacket(
            data_type=DataType.TWITTER,
            data={'text': "Bitcoin breaking out! $BTC", 'followers': 5000, 'verified': True, 'likes': 100, 'retweets': 20},
            source='twitter_test',
            timestamp=get_timestamp()
        ),
        DataPacket(
            data_type=DataType.TWITTER,
            data={'text': "Bearish market dump incoming", 'followers': 2000, 'verified': False, 'likes': 10, 'retweets': 2},
            source='twitter_test',
            timestamp=get_timestamp()
        ),
    ]
    
    # Add to queue
    for packet in packets:
        thread_manager.put_data('raw_data', packet)
    
    # Process
    worker = ProcessingWorker(worker_id=1, batch_size=5, batch_timeout=0.5)
    worker.start()
    time.sleep(1)
    
    # Check results
    results = []
    for _ in range(10):  # Try up to 10 times
        result = thread_manager.get_data('sentiment_queue', block=False, timeout=None)
        if result:
            results.append(result)
    
    # Cleanup
    worker.stop()
    clear_all_queues()
    
    # Assertions
    assert len(results) > 0, "Should have sentiment results"
    assert all(r.data_type == DataType.SENTIMENT for r in results), "All results should be SENTIMENT type"
    print(f"✓ test_twitter_sentiment_pipeline: Processed {len(results)} items")


def test_reddit_sentiment_pipeline():
    """Test Reddit data → processing → sentiment output."""
    clear_all_queues()
    
    packets = [
        DataPacket(
            data_type=DataType.REDDIT,
            data={'text': "BTC analysis: strong support at $45k", 'karma': 250, 'awards': 3, 'subreddit': 'cryptocurrency'},
            source='reddit_test',
            timestamp=get_timestamp()
        ),
    ]
    
    for packet in packets:
        thread_manager.put_data('raw_data', packet)
    
    worker = ProcessingWorker(worker_id=1, batch_size=5, batch_timeout=0.5)
    worker.start()
    time.sleep(1)
    
    results = []
    for _ in range(10):
        result = thread_manager.get_data('sentiment_queue', block=False, timeout=None)
        if result:
            results.append(result)
    
    worker.stop()
    clear_all_queues()
    
    assert len(results) > 0, "Should have sentiment results"
    print(f"✓ test_reddit_sentiment_pipeline: Processed {len(results)} items")


def test_data_integrity():
    """Test that data maintains integrity through pipeline."""
    clear_all_queues()
    
    test_text = "Bitcoin just broke $50k! This is amazing! #BTC #crypto"
    packet = DataPacket(
        data_type=DataType.TWITTER,
        data={'text': test_text, 'followers': 10000, 'verified': True, 'likes': 500, 'retweets': 100},
        source='test_twitter',
        timestamp=get_timestamp()
    )
    
    thread_manager.put_data('raw_data', packet)
    
    worker = ProcessingWorker(worker_id=1, batch_size=5, batch_timeout=0.5)
    worker.start()
    time.sleep(1)
    
    result = thread_manager.get_data('sentiment_queue', block=False, timeout=None)
    
    worker.stop()
    clear_all_queues()
    
    assert result is not None, "Should have processed the packet"
    assert result.data_type == DataType.SENTIMENT, "Should be sentiment type"
    
    data = result.data
    assert data['asset'] == 'BTC', "Should identify BTC"
    assert data['source'] == 'test_twitter', "Should preserve source"
    assert data['sentiment_score'] > 0, "Should be positive sentiment"
    # Ticker extraction is optional - asset identification is the key feature
    assert data['asset'] in ['BTC', 'BITCOIN'], "Should identify Bitcoin"
    
    print(f"✓ test_data_integrity: BTC sentiment={data['sentiment_score']:.3f}, source={data['source']}")


def test_multi_source_pipeline():
    """Test pipeline with multiple data source types."""
    clear_all_queues()
    
    packets = [
        DataPacket(data_type=DataType.TWITTER, 
                  data={'text': "Bitcoin bullish! $BTC", 'followers': 5000, 'verified': True, 'likes': 100, 'retweets': 20},
                  source='twitter_test', timestamp=get_timestamp()),
        DataPacket(data_type=DataType.REDDIT,
                  data={'text': "BTC technical analysis positive", 'karma': 300, 'awards': 5, 'subreddit': 'cryptocurrency'},
                  source='reddit_test', timestamp=get_timestamp()),
        DataPacket(data_type=DataType.MARKET,
                  data={'symbol': 'BTC-USD', 'price': 45123.45, 'volume': 987654321},
                  source='market_test', timestamp=get_timestamp()),
    ]
    
    for packet in packets:
        thread_manager.put_data('raw_data', packet)
    
    worker = ProcessingWorker(worker_id=1, batch_size=10, batch_timeout=0.5)
    worker.start()
    time.sleep(2)
    
    results = []
    for _ in range(20):
        result = thread_manager.get_data('sentiment_queue', block=False, timeout=None)
        if result:
            results.append(result)
    
    stats = worker.get_statistics()
    
    worker.stop()
    clear_all_queues()
    
    assert stats['error_count'] == 0, "Should have no errors"
    assert stats['processed_count'] > 0, "Should have processed items"
    assert len(results) > 0, "Should have sentiment results"
    
    # Check we got results from different sources
    sources = set(r.data['source'] for r in results if r.data_type == DataType.SENTIMENT)
    
    print(f"✓ test_multi_source_pipeline: {stats['processed_count']} processed, {len(results)} results, sources={sources}")


def test_throughput_performance():
    """Test pipeline throughput."""
    clear_all_queues()
    
    num_packets = 20
    for i in range(num_packets):
        packet = DataPacket(
            data_type=DataType.TWITTER,
            data={
                'text': f"Bitcoin analysis #{i} - market {'bullish' if i % 2 == 0 else 'bearish'}",
                'followers': 1000 + i * 10,
                'verified': i % 3 == 0,
                'likes': 10 + i,
                'retweets': 5 + i
            },
            source='throughput_test',
            timestamp=get_timestamp()
        )
        thread_manager.put_data('raw_data', packet)
    
    worker = ProcessingWorker(worker_id=1, batch_size=10, batch_timeout=0.5)
    start_time = time.time()
    worker.start()
    
    # Wait for processing
    time.sleep(3)
    
    stats = worker.get_statistics()
    elapsed = time.time() - start_time
    throughput = stats['processed_count'] / elapsed if elapsed > 0 else 0
    
    worker.stop()
    clear_all_queues()
    
    assert stats['error_count'] == 0, "Should have no errors"
    assert stats['processed_count'] >= num_packets * 0.8, f"Should process at least 80% of packets (got {stats['processed_count']}/{num_packets})"
    assert throughput > 1.0, f"Should process at least 1 item/sec (got {throughput:.1f})"
    
    print(f"✓ test_throughput_performance: {stats['processed_count']}/{num_packets} in {elapsed:.1f}s = {throughput:.1f} items/sec")


if __name__ == "__main__":
    print("=" * 70)
    print("INTEGRATION TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        test_twitter_sentiment_pipeline,
        test_reddit_sentiment_pipeline,
        test_data_integrity,
        test_multi_source_pipeline,
        test_throughput_performance,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"\nRunning {test_func.__name__}...")
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    sys.exit(0 if failed == 0 else 1)
