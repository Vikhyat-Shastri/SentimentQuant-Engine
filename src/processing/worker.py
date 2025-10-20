"""
Processing Worker - Consumes data from raw_data queue and processes through sentiment pipeline.

This worker thread:
1. Monitors raw_data queue for incoming data
2. Processes through sentiment pipeline
3. Outputs results to sentiment_queue
4. Handles errors gracefully with retry logic
"""

import time
import threading
from typing import Dict, Any, Optional
from queue import Empty
from loguru import logger

# Handle imports for both module and direct execution
try:
    from src.utils import DataPacket, DataType, thread_manager, get_timestamp
    from src.sentiment.processor import SentimentProcessor
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils import DataPacket, DataType, thread_manager, get_timestamp
    from src.sentiment.processor import SentimentProcessor


class ProcessingWorker:
    """
    Worker thread that processes data from raw_data queue through sentiment pipeline.
    
    Processes data from Twitter, Reddit, News, and Market feeds.
    Outputs processed sentiment to sentiment_queue for signal generation.
    """
    
    def __init__(
        self,
        worker_id: int = 1,
        batch_size: int = 10,
        batch_timeout: float = 1.0,
        error_retry_delay: float = 5.0
    ):
        """
        Initialize processing worker.
        
        Args:
            worker_id: Unique worker identifier
            batch_size: Maximum batch size for processing
            batch_timeout: Timeout for batch collection (seconds)
            error_retry_delay: Delay after error before retrying (seconds)
        """
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.error_retry_delay = error_retry_delay
        
        # Initialize sentiment processor
        self.processor = SentimentProcessor()
        
        # Worker state
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Benchmark reference (set externally if benchmarking)
        self.benchmark = None
        
        # Statistics
        self.stats = {
            'processed_count': 0,
            'error_count': 0,
            'batch_count': 0,
            'start_time': None,
            'last_process_time': None
        }
        
        logger.info(f"ProcessingWorker {worker_id} initialized (batch_size={batch_size})")
    
    @property
    def processed_count(self) -> int:
        """Get number of items processed."""
        return self.stats['processed_count']
    
    @property
    def error_count(self) -> int:
        """Get number of errors encountered."""
        return self.stats['error_count']
    
    def start(self) -> None:
        """Start the processing worker thread."""
        if self.running:
            logger.warning(f"Worker {self.worker_id} already running")
            return
        
        self.running = True
        self.stats['start_time'] = get_timestamp()
        self.thread = threading.Thread(
            target=self._process_loop,
            name=f"ProcessingWorker-{self.worker_id}",
            daemon=True
        )
        self.thread.start()
        logger.info(f"ProcessingWorker {self.worker_id} started")
    
    def stop(self) -> None:
        """Stop the processing worker thread gracefully."""
        if not self.running:
            return
        
        logger.info(f"Stopping ProcessingWorker {self.worker_id}...")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10.0)
            if self.thread.is_alive():
                logger.warning(f"Worker {self.worker_id} did not stop gracefully")
            else:
                logger.info(f"Worker {self.worker_id} stopped")
    
    def _process_loop(self) -> None:
        """Main processing loop - consumes from raw_data queue."""
        logger.info(f"Worker {self.worker_id} entering processing loop")
        
        while self.running:
            try:
                # Collect batch of data
                batch = self._collect_batch()
                
                if not batch:
                    # No data available, sleep briefly
                    time.sleep(0.1)
                    continue
                
                # Process batch
                self._process_batch(batch)
                self.stats['batch_count'] += 1
                
            except Exception as e:
                self.stats['error_count'] += 1
                logger.error(f"Worker {self.worker_id} error in processing loop: {e}", exc_info=True)
                time.sleep(self.error_retry_delay)
        
        logger.info(f"Worker {self.worker_id} exited processing loop")
    
    def _collect_batch(self) -> list:
        """
        Collect a batch of data packets from raw_data queue.
        
        Returns:
            List of DataPacket objects
        """
        batch = []
        start_time = time.time()
        
        while len(batch) < self.batch_size:
            # Check timeout
            if time.time() - start_time > self.batch_timeout:
                break
            
            try:
                # Get data from queue (non-blocking with timeout)
                packet = thread_manager.get_data(
                    queue_name='raw_data',
                    timeout=0.1
                )
                
                if packet:
                    batch.append(packet)
                    
            except Empty:
                # No data available
                if batch:
                    # We have some data, process it
                    break
                continue
            except Exception as e:
                logger.error(f"Error getting data from queue: {e}")
                break
        
        return batch
    
    def _process_batch(self, batch: list) -> None:
        """
        Process a batch of data packets through sentiment pipeline.
        
        Args:
            batch: List of DataPacket objects
        """
        logger.debug(f"Worker {self.worker_id} processing batch of {len(batch)} items")
        
        for packet in batch:
            try:
                # Measure actual processing time if benchmarking
                if self.benchmark:
                    start_time = time.perf_counter()
                    self._process_packet(packet)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    self.benchmark.record_sentiment_latency(elapsed_ms)
                else:
                    self._process_packet(packet)
                
                self.stats['processed_count'] += 1
                self.stats['last_process_time'] = get_timestamp()
                
            except Exception as e:
                self.stats['error_count'] += 1
                logger.error(f"Error processing packet: {e}", exc_info=False)
    
    def _process_packet(self, packet: DataPacket) -> None:
        """
        Process a single data packet through sentiment pipeline.
        
        Args:
            packet: DataPacket to process
        """
        # Extract data based on packet type
        if packet.data_type == DataType.TWITTER:
            self._process_twitter(packet)
        elif packet.data_type == DataType.REDDIT:
            self._process_reddit(packet)
        elif packet.data_type == DataType.NEWS:
            self._process_news(packet)
        elif packet.data_type == DataType.MARKET:
            self._process_market(packet)
        else:
            logger.warning(f"Unknown data type: {packet.data_type}")
    
    def _process_twitter(self, packet: DataPacket) -> None:
        """Process Twitter data packet."""
        data = packet.data
        
        # Extract tweet text and metadata (support both dict and dataclass)
        if isinstance(data, dict):
            text = data.get('text', '')
            metadata = {
                'followers': data.get('followers', 0),
                'verified': data.get('verified', False),
                'likes': data.get('likes', 0),
                'retweets': data.get('retweets', 0)
            }
        else:
            text = getattr(data, 'text', '')
            metadata = {
                'followers': getattr(data, 'followers', 0),
                'verified': getattr(data, 'verified', False),
                'likes': getattr(data, 'likes', 0),
                'retweets': getattr(data, 'retweets', 0)
            }
        
        if not text:
            logger.debug("Empty tweet text, skipping")
            return
        
        # Determine asset from text or default to BTC
        asset = self._extract_asset(text) or 'BTC'
        
        # Process through sentiment pipeline
        result = self.processor.process_and_aggregate(
            text=text,
            source=packet.source,
            asset=asset,
            metadata=metadata,
            get_aggregated=False  # Don't get aggregated for every tweet
        )
        
        if result['success']:
            # Put processed result in sentiment_queue
            self._output_result(result['processed'], packet.timestamp)
    
    def _process_reddit(self, packet: DataPacket) -> None:
        """Process Reddit data packet."""
        data = packet.data
        
        # Extract post text and metadata (support both dict and dataclass)
        if isinstance(data, dict):
            text = data.get('text', '')
            metadata = {
                'karma': data.get('karma', 0),
                'awards': data.get('awards', 0),
                'subreddit': data.get('subreddit', 'unknown')
            }
        else:
            text = getattr(data, 'text', '')
            metadata = {
                'karma': getattr(data, 'karma', 0),
                'awards': getattr(data, 'awards', 0),
                'subreddit': getattr(data, 'subreddit', 'unknown')
            }
        
        if not text:
            logger.debug("Empty reddit text, skipping")
            return
        
        # Determine asset
        asset = self._extract_asset(text) or 'BTC'
        
        # Process through sentiment pipeline
        result = self.processor.process_and_aggregate(
            text=text,
            source=packet.source,
            asset=asset,
            metadata=metadata,
            get_aggregated=False
        )
        
        if result['success']:
            self._output_result(result['processed'], packet.timestamp)
    
    def _process_news(self, packet: DataPacket) -> None:
        """Process news data packet."""
        data = packet.data
        
        # Extract headline/summary (support both dict and dataclass)
        if isinstance(data, dict):
            text = data.get('headline', '') + ' ' + data.get('summary', '')
            source_name = data.get('source', '')
        else:
            text = getattr(data, 'headline', '') + ' ' + getattr(data, 'summary', '')
            source_name = getattr(data, 'source', '')
        
        if not text.strip():
            logger.debug("Empty news text, skipping")
            return
        
        metadata = {
            'source': source_name,
            'source_credibility': 'high'  # News sources have high credibility
        }
        
        # Determine asset
        asset = self._extract_asset(text) or 'BTC'
        
        # Process through sentiment pipeline
        result = self.processor.process_and_aggregate(
            text=text,
            source=packet.source,
            asset=asset,
            metadata=metadata,
            get_aggregated=False
        )
        
        if result['success']:
            self._output_result(result['processed'], packet.timestamp)
    
    def _process_market(self, packet: DataPacket) -> None:
        """
        Process market data packet.
        
        Market data doesn't have text sentiment, but we track it
        for Fear & Greed Index calculation (volume, volatility, etc.)
        """
        data = packet.data
        
        # Extract market metrics (support both dict and dataclass)
        if isinstance(data, dict):
            symbol = data.get('symbol', 'BTC-USD')
            price = data.get('price', 0)
            volume = data.get('volume', 0)
        else:
            symbol = getattr(data, 'symbol', 'BTC-USD')
            price = getattr(data, 'price', 0)
            volume = getattr(data, 'volume', 0)
        
        asset = symbol.split('-')[0]  # Extract base asset (BTC from BTC-USD)
        
        # Store market data for later F&G Index calculation
        # This could be enhanced to track price/volume history
        logger.debug(f"Market data for {asset}: price={price}, volume={volume}")
    
    def _extract_asset(self, text: str) -> Optional[str]:
        """
        Extract primary asset from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Asset ticker or None
        """
        import re
        
        # Common crypto assets
        assets = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE', 'DOT', 'MATIC', 'AVAX']
        
        text_upper = text.upper()
        
        # Check for $ ticker mentions ($BTC, $ETH)
        ticker_pattern = re.compile(r'\$([A-Z]{2,5})\b')
        tickers = ticker_pattern.findall(text_upper)
        
        for ticker in tickers:
            if ticker in assets:
                return ticker
        
        # Check for asset mentions in text
        for asset in assets:
            if asset in text_upper or f'{asset.lower()}' in text.lower():
                return asset
        
        return None
    
    def _output_result(self, processed_sentiment, timestamp: float) -> None:
        """
        Output processed sentiment to sentiment_queue.
        
        Args:
            processed_sentiment: ProcessedSentiment object
            timestamp: Original data timestamp
        """
        # Create output packet
        output_data = {
            'asset': processed_sentiment.asset,
            'source': processed_sentiment.source,
            'sentiment_score': processed_sentiment.sentiment_score,
            'confidence': processed_sentiment.confidence,
            'entities': processed_sentiment.entities,
            'original_text': processed_sentiment.original_text[:100],  # Truncate for storage
            'processed_at': processed_sentiment.timestamp
        }
        
        packet = DataPacket(
            data_type=DataType.SENTIMENT,
            data=output_data,
            source=processed_sentiment.source,
            timestamp=timestamp
        )
        
        # Put in sentiment queue
        thread_manager.put_data('sentiment_queue', packet)
        
        logger.debug(
            f"Output sentiment: {processed_sentiment.asset} "
            f"({processed_sentiment.source}) = {processed_sentiment.sentiment_score:.3f}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get worker statistics.
        
        Returns:
            Dict with statistics
        """
        elapsed = 0
        if self.stats['start_time']:
            elapsed = get_timestamp() - self.stats['start_time']
        
        rate = self.stats['processed_count'] / elapsed if elapsed > 0 else 0
        
        return {
            'worker_id': self.worker_id,
            'running': self.running,
            'processed_count': self.stats['processed_count'],
            'error_count': self.stats['error_count'],
            'batch_count': self.stats['batch_count'],
            'processing_rate': rate,
            'elapsed_time': elapsed,
            'last_process_time': self.stats['last_process_time'],
            'processor_stats': self.processor.get_statistics()
        }


if __name__ == "__main__":
    # Test the processing worker
    print("=" * 70)
    print("Testing Processing Worker")
    print("=" * 70)
    
    # Create some test data packets
    test_packets = [
        DataPacket(
            data_type=DataType.TWITTER,
            data={
                'text': "Bitcoin breaking out! This could be huge! $BTC",
                'followers': 5000,
                'verified': True,
                'likes': 150,
                'retweets': 30
            },
            source='twitter_stream',
            timestamp=get_timestamp()
        ),
        DataPacket(
            data_type=DataType.REDDIT,
            data={
                'text': "Detailed analysis: BTC showing strong support at $45k",
                'karma': 250,
                'awards': 3,
                'subreddit': 'cryptocurrency'
            },
            source='reddit_stream',
            timestamp=get_timestamp()
        ),
        DataPacket(
            data_type=DataType.TWITTER,
            data={
                'text': "Market crash! Everyone panic selling! Get out now!",
                'followers': 1000,
                'verified': False,
                'likes': 20,
                'retweets': 5
            },
            source='twitter_stream',
            timestamp=get_timestamp()
        )
    ]
    
    # Add test packets to raw_data queue
    print(f"\nAdding {len(test_packets)} test packets to raw_data queue...")
    for packet in test_packets:
        thread_manager.put_data('raw_data', packet)
    
    # Create and start worker
    print("\nStarting processing worker...")
    worker = ProcessingWorker(worker_id=1, batch_size=5, batch_timeout=0.5)
    worker.start()
    
    # Let it process
    print("\nProcessing for 3 seconds...")
    time.sleep(3)
    
    # Check sentiment queue
    print("\nChecking sentiment_queue for results...")
    results_count = 0
    while True:
        try:
            result = thread_manager.get_data('sentiment_queue', timeout=0.1)
            if result:
                results_count += 1
                print(f"  Result {results_count}: {result.data['asset']} = {result.data['sentiment_score']:.3f}")
            else:
                break
        except Empty:
            break
    
    # Get statistics
    print("\n" + "=" * 70)
    print("Worker Statistics")
    print("=" * 70)
    stats = worker.get_statistics()
    print(f"Processed: {stats['processed_count']}")
    print(f"Errors: {stats['error_count']}")
    print(f"Batches: {stats['batch_count']}")
    print(f"Processing Rate: {stats['processing_rate']:.1f} items/sec")
    print(f"Processor Stats: {stats['processor_stats']}")
    
    # Stop worker
    print("\nStopping worker...")
    worker.stop()
    
    print("\n" + "=" * 70)
    print("✅ Processing Worker Test Complete!")
    print("=" * 70)
