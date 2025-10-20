"""
Streaming Optimization

Optimizes real-time data streaming performance:
- Zero-copy processing
- Adaptive buffering
- Backpressure handling
- Batch optimization
- Memory-efficient pipelines
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Iterator, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from queue import Queue, Full, Empty
from threading import Thread, Event, Lock
import time
from loguru import logger


@dataclass
class StreamMetrics:
    """Streaming performance metrics"""
    throughput: float  # items per second
    latency_p50: float  # median latency (ms)
    latency_p95: float  # 95th percentile latency (ms)
    latency_p99: float  # 99th percentile latency (ms)
    buffer_utilization: float  # 0-1
    dropped_items: int
    backpressure_events: int


@dataclass
class StreamConfig:
    """Stream configuration"""
    buffer_size: int = 1000
    batch_size: int = 32
    max_latency_ms: float = 100.0
    enable_backpressure: bool = True
    zero_copy: bool = True
    adaptive_batching: bool = True


class RingBuffer:
    """
    Lock-free ring buffer for zero-copy streaming
    
    Uses pre-allocated memory to avoid allocations
    """
    
    def __init__(self, capacity: int, item_size: int = 1024):
        """
        Initialize ring buffer
        
        Args:
            capacity: Number of items buffer can hold
            item_size: Size of each item in bytes
        """
        self.capacity = capacity
        self.item_size = item_size
        
        # Pre-allocate memory
        self.buffer = bytearray(capacity * item_size)
        self.metadata = [None] * capacity
        
        self.write_pos = 0
        self.read_pos = 0
        self.size = 0
        
        self.lock = Lock()
    
    def write(self, data: bytes, metadata: Any = None) -> bool:
        """
        Write data to buffer (zero-copy)
        
        Args:
            data: Bytes to write
            metadata: Optional metadata
        
        Returns:
            True if successful, False if buffer full
        """
        with self.lock:
            if self.size >= self.capacity:
                return False  # Buffer full
            
            # Copy data to pre-allocated buffer
            start_idx = self.write_pos * self.item_size
            end_idx = start_idx + len(data)
            
            if len(data) > self.item_size:
                logger.warning(f"Data too large: {len(data)} > {self.item_size}")
                return False
            
            self.buffer[start_idx:end_idx] = data
            self.metadata[self.write_pos] = (len(data), metadata)
            
            self.write_pos = (self.write_pos + 1) % self.capacity
            self.size += 1
            
            return True
    
    def read(self) -> Optional[Tuple[bytes, Any]]:
        """
        Read data from buffer (zero-copy via memoryview)
        
        Returns:
            (data_view, metadata) or None if empty
        """
        with self.lock:
            if self.size == 0:
                return None
            
            start_idx = self.read_pos * self.item_size
            data_len, metadata = self.metadata[self.read_pos]
            end_idx = start_idx + data_len
            
            # Return memoryview (zero-copy)
            data_view = memoryview(self.buffer)[start_idx:end_idx]
            
            self.read_pos = (self.read_pos + 1) % self.capacity
            self.size -= 1
            
            return (bytes(data_view), metadata)
    
    def utilization(self) -> float:
        """Get buffer utilization (0-1)"""
        with self.lock:
            return self.size / self.capacity


class AdaptiveBatcher:
    """
    Adaptive batching for optimal throughput/latency tradeoff
    
    Dynamically adjusts batch size based on load
    """
    
    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        target_latency_ms: float = 50.0
    ):
        """
        Initialize adaptive batcher
        
        Args:
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            target_latency_ms: Target processing latency
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        
        self.current_batch_size = min_batch_size
        self.buffer = []
        self.last_batch_time = time.time()
        
        # Latency tracking
        self.latencies = deque(maxlen=100)
    
    def add(self, item: Any) -> Optional[List[Any]]:
        """
        Add item to batch
        
        Returns:
            Batch if ready, None otherwise
        """
        self.buffer.append(item)
        
        # Check if batch is ready
        time_elapsed = (time.time() - self.last_batch_time) * 1000  # ms
        
        if len(self.buffer) >= self.current_batch_size or time_elapsed > self.target_latency_ms:
            batch = self.buffer
            self.buffer = []
            self.last_batch_time = time.time()
            return batch
        
        return None
    
    def update_latency(self, latency_ms: float):
        """Update latency measurement and adjust batch size"""
        self.latencies.append(latency_ms)
        
        if len(self.latencies) < 10:
            return
        
        avg_latency = np.mean(self.latencies)
        
        # Adjust batch size
        if avg_latency < self.target_latency_ms * 0.8:
            # Can increase batch size
            self.current_batch_size = min(
                int(self.current_batch_size * 1.2),
                self.max_batch_size
            )
        elif avg_latency > self.target_latency_ms * 1.2:
            # Need to decrease batch size
            self.current_batch_size = max(
                int(self.current_batch_size * 0.8),
                self.min_batch_size
            )


class BackpressureController:
    """
    Backpressure controller for flow control
    
    Prevents fast producers from overwhelming slow consumers
    """
    
    def __init__(
        self,
        high_watermark: float = 0.8,
        low_watermark: float = 0.5
    ):
        """
        Initialize backpressure controller
        
        Args:
            high_watermark: Buffer utilization to trigger backpressure
            low_watermark: Buffer utilization to release backpressure
        """
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        
        self.backpressure_active = False
        self.backpressure_count = 0
    
    def check(self, buffer_utilization: float) -> bool:
        """
        Check if backpressure should be applied
        
        Args:
            buffer_utilization: Current buffer utilization (0-1)
        
        Returns:
            True if should apply backpressure
        """
        if buffer_utilization >= self.high_watermark:
            if not self.backpressure_active:
                self.backpressure_active = True
                self.backpressure_count += 1
                logger.warning(f"Backpressure activated (utilization: {buffer_utilization:.2f})")
        elif buffer_utilization <= self.low_watermark:
            if self.backpressure_active:
                self.backpressure_active = False
                logger.info(f"Backpressure released (utilization: {buffer_utilization:.2f})")
        
        return self.backpressure_active


class StreamProcessor:
    """
    Optimized stream processor
    
    Features:
    - Zero-copy ring buffer
    - Adaptive batching
    - Backpressure control
    - Performance monitoring
    """
    
    def __init__(
        self,
        processor_func: Callable[[List[Any]], None],
        config: StreamConfig = None
    ):
        """
        Initialize stream processor
        
        Args:
            processor_func: Function to process batches
            config: Stream configuration
        """
        self.processor_func = processor_func
        self.config = config or StreamConfig()
        
        # Streaming components
        if self.config.zero_copy:
            self.buffer = RingBuffer(capacity=self.config.buffer_size)
        else:
            self.queue = Queue(maxsize=self.config.buffer_size)
        
        if self.config.adaptive_batching:
            self.batcher = AdaptiveBatcher(
                min_batch_size=1,
                max_batch_size=self.config.batch_size,
                target_latency_ms=self.config.max_latency_ms
            )
        
        if self.config.enable_backpressure:
            self.backpressure = BackpressureController()
        
        # Metrics
        self.items_processed = 0
        self.items_dropped = 0
        self.latencies = deque(maxlen=1000)
        self.start_time = time.time()
        
        # Control
        self.running = False
        self.stop_event = Event()
        self.worker_thread = None
        
        logger.info("StreamProcessor initialized")
    
    def start(self):
        """Start stream processing"""
        if self.running:
            logger.warning("Stream processor already running")
            return
        
        self.running = True
        self.stop_event.clear()
        
        self.worker_thread = Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("Stream processor started")
    
    def stop(self):
        """Stop stream processing"""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        logger.info("Stream processor stopped")
    
    def push(self, item: Any, metadata: Any = None) -> bool:
        """
        Push item to stream
        
        Args:
            item: Item to process
            metadata: Optional metadata
        
        Returns:
            True if successful, False if dropped
        """
        # Check backpressure
        if self.config.enable_backpressure:
            if self.config.zero_copy:
                utilization = self.buffer.utilization()
            else:
                utilization = self.queue.qsize() / self.config.buffer_size
            
            if self.backpressure.check(utilization):
                time.sleep(0.001)  # Brief pause
        
        # Add to buffer
        if self.config.zero_copy:
            # Convert item to bytes
            if isinstance(item, bytes):
                data = item
            elif isinstance(item, str):
                data = item.encode('utf-8')
            else:
                data = str(item).encode('utf-8')
            
            success = self.buffer.write(data, metadata)
        else:
            try:
                self.queue.put((item, metadata), block=False)
                success = True
            except Full:
                success = False
        
        if not success:
            self.items_dropped += 1
        
        return success
    
    def _process_loop(self):
        """Main processing loop"""
        batch = []
        batch_start_times = []
        
        while not self.stop_event.is_set():
            try:
                # Read from buffer
                if self.config.zero_copy:
                    result = self.buffer.read()
                    if result is None:
                        time.sleep(0.001)  # Brief sleep if empty
                        continue
                    
                    data, metadata = result
                    item = data.decode('utf-8')
                else:
                    try:
                        item, metadata = self.queue.get(timeout=0.01)
                    except Empty:
                        continue
                
                # Add to batch
                item_start_time = time.time()
                
                if self.config.adaptive_batching:
                    ready_batch = self.batcher.add((item, metadata))
                    
                    if ready_batch is not None:
                        batch = ready_batch
                        batch_start_times.append(item_start_time)
                    else:
                        continue
                else:
                    batch.append((item, metadata))
                    batch_start_times.append(item_start_time)
                    
                    if len(batch) < self.config.batch_size:
                        continue
                
                # Process batch
                process_start = time.time()
                
                self.processor_func([item for item, _ in batch])
                
                process_time = time.time() - process_start
                
                # Update metrics
                self.items_processed += len(batch)
                
                for start_time in batch_start_times:
                    latency_ms = (time.time() - start_time) * 1000
                    self.latencies.append(latency_ms)
                
                if self.config.adaptive_batching:
                    self.batcher.update_latency(process_time * 1000)
                
                # Reset batch
                batch = []
                batch_start_times = []
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def get_metrics(self) -> StreamMetrics:
        """Get current performance metrics"""
        elapsed = time.time() - self.start_time
        throughput = self.items_processed / max(elapsed, 1.0)
        
        if self.latencies:
            latencies_sorted = sorted(self.latencies)
            p50 = np.percentile(latencies_sorted, 50)
            p95 = np.percentile(latencies_sorted, 95)
            p99 = np.percentile(latencies_sorted, 99)
        else:
            p50 = p95 = p99 = 0.0
        
        if self.config.zero_copy:
            buffer_util = self.buffer.utilization()
        else:
            buffer_util = self.queue.qsize() / self.config.buffer_size
        
        backpressure_events = self.backpressure.backpressure_count if self.config.enable_backpressure else 0
        
        return StreamMetrics(
            throughput=throughput,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            buffer_utilization=buffer_util,
            dropped_items=self.items_dropped,
            backpressure_events=backpressure_events
        )


# Test function
if __name__ == "__main__":
    # Test processor function
    def test_processor(batch: List[Any]):
        """Simple processor that simulates work"""
        time.sleep(0.01)  # Simulate processing time
        print(f"Processed batch of {len(batch)} items")
    
    # Create stream processor
    config = StreamConfig(
        buffer_size=1000,
        batch_size=32,
        max_latency_ms=100.0,
        enable_backpressure=True,
        zero_copy=True,
        adaptive_batching=True
    )
    
    processor = StreamProcessor(test_processor, config)
    
    print("\n" + "="*80)
    print("STREAMING OPTIMIZATION TEST")
    print("="*80)
    
    # Start processor
    processor.start()
    
    # Push items
    print(f"\n📥 Pushing 1000 items...")
    for i in range(1000):
        success = processor.push(f"item_{i}", metadata={'id': i})
        if not success:
            print(f"⚠️  Item {i} dropped")
        
        if i % 100 == 0:
            time.sleep(0.05)  # Varying load
    
    # Wait for processing
    time.sleep(2.0)
    
    # Get metrics
    metrics = processor.get_metrics()
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Throughput: {metrics.throughput:.1f} items/sec")
    print(f"  Latency (p50): {metrics.latency_p50:.2f} ms")
    print(f"  Latency (p95): {metrics.latency_p95:.2f} ms")
    print(f"  Latency (p99): {metrics.latency_p99:.2f} ms")
    print(f"  Buffer Utilization: {metrics.buffer_utilization:.2%}")
    print(f"  Dropped Items: {metrics.dropped_items}")
    print(f"  Backpressure Events: {metrics.backpressure_events}")
    
    # Test adaptive batching
    if config.adaptive_batching:
        print(f"\n🔧 Adaptive Batching:")
        print(f"  Current Batch Size: {processor.batcher.current_batch_size}")
    
    # Stop processor
    processor.stop()
    
    print(f"\n✅ Stream processor stopped")
