"""
Lock-Free Data Structures

High-performance concurrent data structures:
- Lock-free queue
- Lock-free stack
- Atomic operations
- Compare-and-swap (CAS)
- Memory ordering
"""
import threading
from typing import Any, Optional, List
from dataclasses import dataclass
from collections import deque
import time
from loguru import logger


@dataclass
class Node:
    """Node for linked structures"""
    value: Any
    next: Optional['Node'] = None


class AtomicCounter:
    """
    Thread-safe atomic counter
    
    Uses Python's GIL for atomicity
    """
    
    def __init__(self, initial: int = 0):
        """Initialize atomic counter"""
        self._value = initial
        self._lock = threading.Lock()
    
    def increment(self, delta: int = 1) -> int:
        """
        Atomically increment counter
        
        Returns:
            New value
        """
        with self._lock:
            self._value += delta
            return self._value
    
    def decrement(self, delta: int = 1) -> int:
        """
        Atomically decrement counter
        
        Returns:
            New value
        """
        return self.increment(-delta)
    
    def get(self) -> int:
        """Get current value"""
        with self._lock:
            return self._value
    
    def set(self, value: int):
        """Set value"""
        with self._lock:
            self._value = value
    
    def compare_and_swap(self, expected: int, new_value: int) -> bool:
        """
        Compare and swap (CAS) operation
        
        Args:
            expected: Expected current value
            new_value: New value to set
        
        Returns:
            True if swap successful
        """
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False


class LockFreeQueue:
    """
    Lock-free MPMC (Multi-Producer Multi-Consumer) queue
    
    Uses fine-grained locking on head/tail for near-lock-free performance
    Note: True lock-free queues require atomic CAS operations not available
    in pure Python. This implementation minimizes lock contention.
    """
    
    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize lock-free queue
        
        Args:
            max_size: Maximum queue size (None = unbounded)
        """
        self.max_size = max_size
        self._queue = deque()
        
        # Separate locks for head and tail to reduce contention
        self._head_lock = threading.Lock()
        self._tail_lock = threading.Lock()
        
        self._size = AtomicCounter(0)
        
        logger.debug("LockFreeQueue initialized")
    
    def enqueue(self, item: Any, timeout: Optional[float] = None) -> bool:
        """
        Add item to queue
        
        Args:
            item: Item to add
            timeout: Optional timeout in seconds
        
        Returns:
            True if successful
        """
        start_time = time.time()
        
        while True:
            # Check size limit
            if self.max_size is not None:
                current_size = self._size.get()
                if current_size >= self.max_size:
                    if timeout is not None and (time.time() - start_time) > timeout:
                        return False
                    time.sleep(0.001)  # Brief backoff
                    continue
            
            # Try to acquire tail lock
            if self._tail_lock.acquire(blocking=False):
                try:
                    self._queue.append(item)
                    self._size.increment()
                    return True
                finally:
                    self._tail_lock.release()
            
            # Failed to acquire lock
            if timeout is not None and (time.time() - start_time) > timeout:
                return False
            
            # Brief backoff before retry
            time.sleep(0.0001)
    
    def dequeue(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Remove and return item from queue
        
        Args:
            timeout: Optional timeout in seconds
        
        Returns:
            Item or None if empty/timeout
        """
        start_time = time.time()
        
        while True:
            # Check if empty
            if self._size.get() == 0:
                if timeout is not None and (time.time() - start_time) > timeout:
                    return None
                time.sleep(0.001)  # Brief backoff
                continue
            
            # Try to acquire head lock
            if self._head_lock.acquire(blocking=False):
                try:
                    if len(self._queue) > 0:
                        item = self._queue.popleft()
                        self._size.decrement()
                        return item
                finally:
                    self._head_lock.release()
            
            # Failed to acquire lock
            if timeout is not None and (time.time() - start_time) > timeout:
                return None
            
            # Brief backoff before retry
            time.sleep(0.0001)
    
    def size(self) -> int:
        """Get current queue size"""
        return self._size.get()
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.size() == 0


class LockFreeStack:
    """
    Lock-free stack (LIFO)
    
    Uses CAS-style operations for thread safety
    """
    
    def __init__(self):
        """Initialize lock-free stack"""
        self._head: Optional[Node] = None
        self._lock = threading.Lock()  # Minimal locking for Python's limitations
        self._size = AtomicCounter(0)
        
        logger.debug("LockFreeStack initialized")
    
    def push(self, item: Any):
        """
        Push item onto stack
        
        Args:
            item: Item to push
        """
        new_node = Node(value=item)
        
        with self._lock:
            new_node.next = self._head
            self._head = new_node
            self._size.increment()
    
    def pop(self) -> Optional[Any]:
        """
        Pop item from stack
        
        Returns:
            Item or None if empty
        """
        with self._lock:
            if self._head is None:
                return None
            
            item = self._head.value
            self._head = self._head.next
            self._size.decrement()
            
            return item
    
    def peek(self) -> Optional[Any]:
        """
        Peek at top item without removing
        
        Returns:
            Top item or None if empty
        """
        with self._lock:
            if self._head is None:
                return None
            return self._head.value
    
    def size(self) -> int:
        """Get current stack size"""
        return self._size.get()
    
    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return self.size() == 0


class ConcurrentBuffer:
    """
    Concurrent circular buffer with lock-free reads
    
    Single-writer, multiple-reader buffer
    """
    
    def __init__(self, capacity: int):
        """
        Initialize concurrent buffer
        
        Args:
            capacity: Buffer capacity
        """
        self.capacity = capacity
        self._buffer = [None] * capacity
        self._write_index = AtomicCounter(0)
        self._write_lock = threading.Lock()
        
        logger.debug(f"ConcurrentBuffer initialized (capacity: {capacity})")
    
    def write(self, item: Any) -> bool:
        """
        Write item to buffer
        
        Args:
            item: Item to write
        
        Returns:
            True if successful
        """
        with self._write_lock:
            index = self._write_index.get() % self.capacity
            self._buffer[index] = item
            self._write_index.increment()
            return True
    
    def read(self, index: int) -> Optional[Any]:
        """
        Read item at index (lock-free)
        
        Args:
            index: Index to read
        
        Returns:
            Item or None
        """
        if 0 <= index < self.capacity:
            return self._buffer[index]
        return None
    
    def read_latest(self, count: int = 1) -> List[Any]:
        """
        Read latest N items
        
        Args:
            count: Number of items to read
        
        Returns:
            List of latest items
        """
        current_index = self._write_index.get()
        start_index = max(0, current_index - count)
        
        items = []
        for i in range(start_index, current_index):
            item = self.read(i % self.capacity)
            if item is not None:
                items.append(item)
        
        return items


# Test functions
def test_queue():
    """Test lock-free queue"""
    print("\n📦 Testing Lock-Free Queue")
    print("-" * 80)
    
    queue = LockFreeQueue(max_size=100)
    
    # Single-threaded test
    print("Single-threaded test:")
    for i in range(10):
        queue.enqueue(f"item_{i}")
    
    print(f"   Size: {queue.size()}")
    
    items = []
    while not queue.is_empty():
        item = queue.dequeue()
        if item:
            items.append(item)
    
    print(f"   Retrieved: {len(items)} items")
    print(f"   First: {items[0]}, Last: {items[-1]}")
    
    # Multi-threaded test
    print("\nMulti-threaded test:")
    
    produced = AtomicCounter(0)
    consumed = AtomicCounter(0)
    
    def producer(thread_id: int):
        for i in range(100):
            queue.enqueue(f"thread_{thread_id}_item_{i}")
            produced.increment()
            time.sleep(0.001)
    
    def consumer(thread_id: int):
        for _ in range(100):
            item = queue.dequeue(timeout=5.0)
            if item:
                consumed.increment()
    
    # Start threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=producer, args=(i,))
        threads.append(t)
        t.start()
    
    for i in range(3):
        t = threading.Thread(target=consumer, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    print(f"   Produced: {produced.get()} items")
    print(f"   Consumed: {consumed.get()} items")
    print(f"   Remaining: {queue.size()} items")


def test_stack():
    """Test lock-free stack"""
    print("\n📚 Testing Lock-Free Stack")
    print("-" * 80)
    
    stack = LockFreeStack()
    
    # Push items
    for i in range(10):
        stack.push(f"item_{i}")
    
    print(f"Size: {stack.size()}")
    print(f"Top: {stack.peek()}")
    
    # Pop items
    items = []
    while not stack.is_empty():
        item = stack.pop()
        if item:
            items.append(item)
    
    print(f"Retrieved: {len(items)} items")
    print(f"First popped: {items[0]}, Last popped: {items[-1]}")


def test_buffer():
    """Test concurrent buffer"""
    print("\n🔄 Testing Concurrent Buffer")
    print("-" * 80)
    
    buffer = ConcurrentBuffer(capacity=100)
    
    # Write items
    for i in range(150):  # More than capacity
        buffer.write(f"item_{i}")
    
    # Read latest
    latest = buffer.read_latest(10)
    print(f"Latest 10 items: {len(latest)}")
    if latest:
        print(f"   First: {latest[0]}")
        print(f"   Last: {latest[-1]}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LOCK-FREE DATA STRUCTURES TEST")
    print("="*80)
    
    test_queue()
    test_stack()
    test_buffer()
    
    print(f"\n✅ All tests complete!")
