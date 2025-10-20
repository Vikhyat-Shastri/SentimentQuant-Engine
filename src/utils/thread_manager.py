"""
Thread-safe queue manager for multi-threaded data processing.
Manages data flow between ingestion, processing, and analysis threads.
"""

import queue
import threading
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import time


class DataType(Enum):
    """Types of data flowing through the system."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    MARKET = "market"
    SENTIMENT = "sentiment"
    SIGNAL = "signal"


@dataclass
class DataPacket:
    """
    Standard data packet for inter-thread communication.
    
    Attributes:
        data_type: Type of data (twitter, reddit, news, etc.)
        timestamp: Unix timestamp when data was created
        source: Source identifier (e.g., "twitter_stream")
        data: Actual data payload
        metadata: Additional metadata
    """
    data_type: DataType
    timestamp: float
    source: str
    data: Any
    metadata: Optional[Dict[str, Any]] = None


class ThreadManager:
    """
    Manages multiple worker threads and their communication queues.
    Provides thread-safe data passing and lifecycle management.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize ThreadManager.
        
        Args:
            max_queue_size: Maximum items in each queue before blocking
        """
        self.max_queue_size = max_queue_size
        self.threads: Dict[str, threading.Thread] = {}
        self.queues: Dict[str, queue.Queue] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        self.running = False
        
        # Create main data queues
        self._initialize_queues()
    
    def _initialize_queues(self) -> None:
        """Initialize all data queues."""
        queue_names = [
            "raw_data",          # Raw ingested data
            "processed_data",    # Cleaned/normalized data
            "sentiment_queue",   # Sentiment analysis results
            "signal_queue",      # Trading signals
            "storage_queue"      # Data to be stored
        ]
        
        for queue_name in queue_names:
            self.queues[queue_name] = queue.Queue(maxsize=self.max_queue_size)
            logger.info(f"Initialized queue: {queue_name}")
    
    def start_thread(
        self,
        name: str,
        target: Callable,
        args: tuple = (),
        daemon: bool = True
    ) -> None:
        """
        Start a new worker thread.
        
        Args:
            name: Unique name for the thread
            target: Function to run in thread
            args: Arguments to pass to target function
            daemon: Whether thread should be daemon
        """
        if name in self.threads:
            logger.warning(f"Thread '{name}' already exists")
            return
        
        # Create stop event for this thread
        stop_event = threading.Event()
        self.stop_events[name] = stop_event
        
        # Create and start thread
        thread = threading.Thread(
            target=target,
            args=(*args, stop_event),
            name=name,
            daemon=daemon
        )
        self.threads[name] = thread
        thread.start()
        
        logger.info(f"Started thread: {name}")
    
    def stop_thread(self, name: str, timeout: float = 5.0) -> None:
        """
        Stop a specific thread gracefully.
        
        Args:
            name: Name of thread to stop
            timeout: Maximum time to wait for thread to stop
        """
        if name not in self.threads:
            logger.warning(f"Thread '{name}' not found")
            return
        
        # Signal thread to stop
        self.stop_events[name].set()
        
        # Wait for thread to finish
        self.threads[name].join(timeout=timeout)
        
        if self.threads[name].is_alive():
            logger.warning(f"Thread '{name}' did not stop within {timeout}s")
        else:
            logger.info(f"Stopped thread: {name}")
            del self.threads[name]
            del self.stop_events[name]
    
    def stop_all_threads(self, timeout: float = 10.0) -> None:
        """
        Stop all running threads gracefully.
        
        Args:
            timeout: Maximum time to wait for all threads
        """
        logger.info("Stopping all threads...")
        
        # Signal all threads to stop
        for stop_event in self.stop_events.values():
            stop_event.set()
        
        # Wait for all threads
        start_time = time.time()
        for name, thread in list(self.threads.items()):
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 0:
                thread.join(timeout=remaining_time)
                if not thread.is_alive():
                    logger.info(f"Thread '{name}' stopped")
            else:
                logger.warning(f"Timeout waiting for thread '{name}'")
        
        self.threads.clear()
        self.stop_events.clear()
        self.running = False
    
    def put_data(
        self,
        queue_name: str,
        data: Any,
        block: bool = True,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Put data into a queue.
        
        Args:
            queue_name: Name of queue
            data: Data to put in queue
            block: Whether to block if queue is full
            timeout: Timeout for blocking
            
        Returns:
            True if data was added, False otherwise
        """
        if queue_name not in self.queues:
            logger.error(f"Queue '{queue_name}' not found")
            return False
        
        try:
            self.queues[queue_name].put(data, block=block, timeout=timeout)
            return True
        except queue.Full:
            logger.warning(f"Queue '{queue_name}' is full, data dropped")
            return False
    
    def get_data(
        self,
        queue_name: str,
        block: bool = True,
        timeout: Optional[float] = None
    ) -> Optional[Any]:
        """
        Get data from a queue.
        
        Args:
            queue_name: Name of queue
            block: Whether to block if queue is empty
            timeout: Timeout for blocking
            
        Returns:
            Data from queue or None if empty/timeout
        """
        if queue_name not in self.queues:
            logger.error(f"Queue '{queue_name}' not found")
            return None
        
        try:
            return self.queues[queue_name].get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_queue_size(self, queue_name: str) -> int:
        """
        Get current size of a queue.
        
        Args:
            queue_name: Name of queue
            
        Returns:
            Number of items in queue
        """
        if queue_name not in self.queues:
            return 0
        return self.queues[queue_name].qsize()
    
    def get_all_queue_sizes(self) -> Dict[str, int]:
        """
        Get sizes of all queues.
        
        Returns:
            Dictionary mapping queue names to sizes
        """
        return {
            name: q.qsize() 
            for name, q in self.queues.items()
        }
    
    def is_thread_running(self, name: str) -> bool:
        """
        Check if a thread is running.
        
        Args:
            name: Name of thread
            
        Returns:
            True if thread is running
        """
        return name in self.threads and self.threads[name].is_alive()
    
    def get_thread_status(self) -> Dict[str, bool]:
        """
        Get status of all threads.
        
        Returns:
            Dictionary mapping thread names to running status
        """
        return {
            name: thread.is_alive()
            for name, thread in self.threads.items()
        }


# Global thread manager instance
thread_manager = ThreadManager()
