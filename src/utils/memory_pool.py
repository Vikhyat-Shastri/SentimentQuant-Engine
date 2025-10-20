"""
Memory Pool Optimization

Pre-allocated memory pools for high-performance object reuse:
- Object pooling
- Memory pool allocation
- Zero-copy buffers
- Pool statistics
"""
import numpy as np
from typing import Any, List, Optional, Callable, Dict
from collections import deque
from dataclasses import dataclass
import time
from loguru import logger


@dataclass
class PoolStatistics:
    """Memory pool statistics"""
    total_allocations: int
    total_deallocations: int
    current_used: int
    current_free: int
    peak_used: int
    cache_hit_rate: float
    avg_allocation_time_ns: float


class MemoryPool:
    """
    Generic memory pool for object reuse
    
    Reduces allocation overhead by reusing objects
    """
    
    def __init__(
        self,
        factory: Callable[[], Any],
        initial_size: int = 10,
        max_size: int = 100
    ):
        """
        Initialize memory pool
        
        Args:
            factory: Function to create new objects
            initial_size: Initial pool size
            max_size: Maximum pool size
        """
        self.factory = factory
        self.max_size = max_size
        
        # Pre-allocate objects
        self._pool: deque = deque()
        for _ in range(initial_size):
            self._pool.append(factory())
        
        # Statistics
        self._total_allocations = 0
        self._total_deallocations = 0
        self._current_used = 0
        self._peak_used = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._allocation_times = []
        
        logger.debug(f"MemoryPool initialized (size: {initial_size}, max: {max_size})")
    
    def acquire(self) -> Any:
        """
        Acquire object from pool
        
        Returns:
            Object from pool or newly created
        """
        start_time = time.perf_counter_ns()
        
        if self._pool:
            # Reuse from pool (cache hit)
            obj = self._pool.pop()
            self._cache_hits += 1
        else:
            # Create new object (cache miss)
            obj = self.factory()
            self._cache_misses += 1
        
        self._current_used += 1
        self._total_allocations += 1
        self._peak_used = max(self._peak_used, self._current_used)
        
        elapsed_ns = time.perf_counter_ns() - start_time
        self._allocation_times.append(elapsed_ns)
        
        return obj
    
    def release(self, obj: Any):
        """
        Return object to pool
        
        Args:
            obj: Object to return
        """
        # Reset object state if needed
        if hasattr(obj, 'reset'):
            obj.reset()
        
        # Return to pool if not full
        if len(self._pool) < self.max_size:
            self._pool.append(obj)
        
        self._current_used -= 1
        self._total_deallocations += 1
    
    def get_statistics(self) -> PoolStatistics:
        """Get pool statistics"""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        avg_time = np.mean(self._allocation_times) if self._allocation_times else 0.0
        
        return PoolStatistics(
            total_allocations=self._total_allocations,
            total_deallocations=self._total_deallocations,
            current_used=self._current_used,
            current_free=len(self._pool),
            peak_used=self._peak_used,
            cache_hit_rate=cache_hit_rate,
            avg_allocation_time_ns=avg_time
        )


class BufferPool:
    """
    Memory pool for fixed-size buffers
    
    Efficient for network/stream processing
    """
    
    def __init__(
        self,
        buffer_size: int,
        pool_size: int = 100
    ):
        """
        Initialize buffer pool
        
        Args:
            buffer_size: Size of each buffer in bytes
            pool_size: Number of buffers to pre-allocate
        """
        self.buffer_size = buffer_size
        
        # Pre-allocate buffers
        self._pool: deque = deque()
        for _ in range(pool_size):
            buffer = bytearray(buffer_size)
            self._pool.append(buffer)
        
        self._total_acquired = 0
        self._total_released = 0
        self._current_used = 0
        
        logger.debug(f"BufferPool initialized ({buffer_size} bytes x {pool_size} buffers)")
    
    def acquire(self) -> bytearray:
        """
        Acquire buffer
        
        Returns:
            Buffer of specified size
        """
        if self._pool:
            buffer = self._pool.pop()
        else:
            # Create new buffer if pool exhausted
            buffer = bytearray(self.buffer_size)
        
        self._current_used += 1
        self._total_acquired += 1
        
        return buffer
    
    def release(self, buffer: bytearray):
        """
        Return buffer to pool
        
        Args:
            buffer: Buffer to return
        """
        # Clear buffer
        buffer[:] = b'\x00' * len(buffer)
        
        self._pool.append(buffer)
        self._current_used -= 1
        self._total_released += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get buffer pool statistics"""
        return {
            'buffer_size': self.buffer_size,
            'total_acquired': self._total_acquired,
            'total_released': self._total_released,
            'current_used': self._current_used,
            'current_free': len(self._pool)
        }


class ArrayPool:
    """
    Memory pool for NumPy arrays
    
    Reduces allocation overhead for numerical computations
    """
    
    def __init__(
        self,
        shape: tuple,
        dtype: np.dtype = np.float64,
        pool_size: int = 50
    ):
        """
        Initialize array pool
        
        Args:
            shape: Shape of arrays
            dtype: Data type
            pool_size: Number of arrays to pre-allocate
        """
        self.shape = shape
        self.dtype = dtype
        
        # Pre-allocate arrays
        self._pool: deque = deque()
        for _ in range(pool_size):
            array = np.zeros(shape, dtype=dtype)
            self._pool.append(array)
        
        self._total_acquired = 0
        self._total_released = 0
        
        logger.debug(f"ArrayPool initialized (shape: {shape}, dtype: {dtype}, size: {pool_size})")
    
    def acquire(self) -> np.ndarray:
        """
        Acquire array
        
        Returns:
            NumPy array of specified shape/dtype
        """
        if self._pool:
            array = self._pool.pop()
            # Zero out array
            array.fill(0)
        else:
            # Create new array if pool exhausted
            array = np.zeros(self.shape, dtype=self.dtype)
        
        self._total_acquired += 1
        
        return array
    
    def release(self, array: np.ndarray):
        """
        Return array to pool
        
        Args:
            array: Array to return
        """
        if array.shape == self.shape and array.dtype == self.dtype:
            self._pool.append(array)
            self._total_released += 1
        else:
            logger.warning(f"Array shape/dtype mismatch: {array.shape}/{array.dtype}")


# Example reusable object
class SentimentResult:
    """Example object for pooling"""
    
    def __init__(self):
        self.text = ""
        self.score = 0.0
        self.confidence = 0.0
        self.timestamp = 0.0
    
    def reset(self):
        """Reset object state"""
        self.text = ""
        self.score = 0.0
        self.confidence = 0.0
        self.timestamp = 0.0


# Test functions
def test_memory_pool():
    """Test generic memory pool"""
    print("\n🏊 Testing Memory Pool")
    print("-" * 80)
    
    # Create pool
    pool = MemoryPool(
        factory=SentimentResult,
        initial_size=10,
        max_size=50
    )
    
    # Acquire and release objects
    objects = []
    for i in range(20):
        obj = pool.acquire()
        obj.text = f"test_{i}"
        obj.score = np.random.random()
        objects.append(obj)
    
    print(f"Acquired: {len(objects)} objects")
    
    # Release half
    for obj in objects[:10]:
        pool.release(obj)
    
    print(f"Released: 10 objects")
    
    # Acquire more (should reuse)
    more_objects = []
    for _ in range(5):
        obj = pool.acquire()
        more_objects.append(obj)
    
    # Get statistics
    stats = pool.get_statistics()
    print(f"\nStatistics:")
    print(f"   Total Allocations: {stats.total_allocations}")
    print(f"   Total Deallocations: {stats.total_deallocations}")
    print(f"   Current Used: {stats.current_used}")
    print(f"   Current Free: {stats.current_free}")
    print(f"   Peak Used: {stats.peak_used}")
    print(f"   Cache Hit Rate: {stats.cache_hit_rate:.2%}")
    print(f"   Avg Allocation Time: {stats.avg_allocation_time_ns:.0f} ns")


def test_buffer_pool():
    """Test buffer pool"""
    print("\n📦 Testing Buffer Pool")
    print("-" * 80)
    
    # Create pool
    pool = BufferPool(buffer_size=1024, pool_size=50)
    
    # Acquire buffers
    buffers = []
    for i in range(30):
        buffer = pool.acquire()
        buffer[:10] = f"data_{i:03d}".encode()
        buffers.append(buffer)
    
    print(f"Acquired: {len(buffers)} buffers")
    
    # Release buffers
    for buffer in buffers[:20]:
        pool.release(buffer)
    
    print(f"Released: 20 buffers")
    
    # Get stats
    stats = pool.get_stats()
    print(f"\nStatistics:")
    print(f"   Buffer Size: {stats['buffer_size']} bytes")
    print(f"   Total Acquired: {stats['total_acquired']}")
    print(f"   Total Released: {stats['total_released']}")
    print(f"   Current Used: {stats['current_used']}")
    print(f"   Current Free: {stats['current_free']}")


def test_array_pool():
    """Test array pool"""
    print("\n🔢 Testing Array Pool")
    print("-" * 80)
    
    # Create pool
    pool = ArrayPool(shape=(100, 10), dtype=np.float32, pool_size=20)
    
    # Acquire arrays
    arrays = []
    for i in range(15):
        array = pool.acquire()
        array[:] = np.random.random((100, 10))
        arrays.append(array)
    
    print(f"Acquired: {len(arrays)} arrays")
    print(f"Array shape: {arrays[0].shape}")
    print(f"Array dtype: {arrays[0].dtype}")
    
    # Release arrays
    for array in arrays[:10]:
        pool.release(array)
    
    print(f"Released: 10 arrays")


def benchmark_pool_vs_allocation():
    """Benchmark pool vs direct allocation"""
    print("\n⚡ Benchmarking Pool vs Direct Allocation")
    print("-" * 80)
    
    pool = MemoryPool(factory=SentimentResult, initial_size=100, max_size=100)
    
    # Benchmark with pool
    start = time.perf_counter()
    for _ in range(1000):
        obj = pool.acquire()
        obj.score = 0.5
        pool.release(obj)
    pool_time = time.perf_counter() - start
    
    # Benchmark without pool
    start = time.perf_counter()
    for _ in range(1000):
        obj = SentimentResult()
        obj.score = 0.5
    direct_time = time.perf_counter() - start
    
    print(f"Pool allocation: {pool_time*1000:.2f} ms")
    print(f"Direct allocation: {direct_time*1000:.2f} ms")
    print(f"Speedup: {direct_time/pool_time:.2f}x")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MEMORY POOL OPTIMIZATION TEST")
    print("="*80)
    
    test_memory_pool()
    test_buffer_pool()
    test_array_pool()
    benchmark_pool_vs_allocation()
    
    print(f"\n✅ All tests complete!")
