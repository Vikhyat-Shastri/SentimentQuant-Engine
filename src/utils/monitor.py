"""
Real-time monitoring dashboard for the sentiment analysis engine.
Displays live statistics in a terminal-based interface.
"""

import os
import sys
import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback to no colors
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = ''
    class Back:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ''


from src.utils import thread_manager, get_timestamp


class DashboardMonitor:
    """
    Real-time terminal dashboard for monitoring sentiment engine.
    """
    
    def __init__(self, refresh_interval: float = 1.0):
        """
        Initialize the dashboard monitor.
        
        Args:
            refresh_interval: Update interval in seconds
        """
        self.refresh_interval = refresh_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Statistics tracking
        self.start_time = time.time()
        self.last_processed_count = 0
        self.throughput_history = deque(maxlen=30)  # Last 30 measurements
        
        # Component references (to be set externally)
        self.workers: List = []
        self.data_streams: List = []
        
        # Screen dimensions
        self.width = 80
        self.height = 25
        
    def set_components(self, workers: List, data_streams: List):
        """
        Set references to system components for monitoring.
        
        Args:
            workers: List of ProcessingWorker instances
            data_streams: List of data stream instances
        """
        self.workers = workers
        self.data_streams = data_streams
    
    def start(self):
        """Start the monitoring dashboard in a background thread."""
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(
            target=self._monitor_loop,
            name="DashboardMonitor",
            daemon=True
        )
        self.thread.start()
    
    def stop(self):
        """Stop the monitoring dashboard."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Main monitoring loop that updates the dashboard."""
        while self.running:
            try:
                self._render_dashboard()
                time.sleep(self.refresh_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Don't let dashboard errors crash the system
                print(f"Dashboard error: {e}")
                time.sleep(self.refresh_interval)
    
    def _render_dashboard(self):
        """Render the complete dashboard."""
        # Clear screen (works on Windows and Unix)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Build dashboard content
        lines = []
        
        # Header
        lines.extend(self._render_header())
        lines.append("")
        
        # System Overview
        lines.extend(self._render_system_overview())
        lines.append("")
        
        # Queue Status
        lines.extend(self._render_queue_status())
        lines.append("")
        
        # Worker Status
        lines.extend(self._render_worker_status())
        lines.append("")
        
        # Data Streams Status
        lines.extend(self._render_stream_status())
        lines.append("")
        
        # Performance Metrics
        lines.extend(self._render_performance_metrics())
        lines.append("")
        
        # Footer
        lines.extend(self._render_footer())
        
        # Print all lines
        print("\n".join(lines))
    
    def _render_header(self) -> List[str]:
        """Render dashboard header."""
        lines = []
        border = "=" * self.width
        
        lines.append(f"{Fore.CYAN}{Style.BRIGHT}{border}")
        title = "🚀 SENTIMENT ENGINE MONITORING DASHBOARD"
        padding = (self.width - len(title)) // 2
        lines.append(f"{Fore.CYAN}{Style.BRIGHT}{' ' * padding}{title}")
        
        # Current time and runtime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        runtime = int(time.time() - self.start_time)
        runtime_str = f"Runtime: {runtime}s"
        time_line = f"{current_time}{' ' * (self.width - len(current_time) - len(runtime_str))}{runtime_str}"
        lines.append(f"{Fore.CYAN}{time_line}")
        lines.append(f"{Fore.CYAN}{border}")
        
        return lines
    
    def _render_system_overview(self) -> List[str]:
        """Render system overview section."""
        lines = []
        lines.append(f"{Fore.YELLOW}{Style.BRIGHT}📊 SYSTEM OVERVIEW")
        lines.append(f"{Fore.YELLOW}{'─' * self.width}")
        
        # Get statistics
        total_processed = sum(w.processed_count for w in self.workers if hasattr(w, 'processed_count'))
        total_errors = sum(w.error_count for w in self.workers if hasattr(w, 'error_count'))
        success_rate = ((total_processed - total_errors) / max(total_processed, 1)) * 100
        
        # Calculate current throughput
        current_throughput = self._calculate_throughput(total_processed)
        
        # Status color coding
        if success_rate >= 95:
            status_color = Fore.GREEN
            status_icon = "✓"
        elif success_rate >= 80:
            status_color = Fore.YELLOW
            status_icon = "⚠"
        else:
            status_color = Fore.RED
            status_icon = "✗"
        
        lines.append(f"  Status: {status_color}{Style.BRIGHT}{status_icon} {'HEALTHY' if success_rate >= 95 else 'WARNING' if success_rate >= 80 else 'CRITICAL'}")
        lines.append(f"  {Fore.WHITE}Processed: {Fore.GREEN}{total_processed:,} items")
        lines.append(f"  {Fore.WHITE}Errors: {Fore.RED if total_errors > 0 else Fore.GREEN}{total_errors} errors")
        lines.append(f"  {Fore.WHITE}Success Rate: {status_color}{success_rate:.2f}%")
        lines.append(f"  {Fore.WHITE}Throughput: {Fore.CYAN}{current_throughput:.2f} items/sec")
        
        return lines
    
    def _render_queue_status(self) -> List[str]:
        """Render queue status section."""
        lines = []
        lines.append(f"{Fore.YELLOW}{Style.BRIGHT}📦 QUEUE STATUS")
        lines.append(f"{Fore.YELLOW}{'─' * self.width}")
        
        queue_sizes = thread_manager.get_all_queue_sizes()
        
        queues = [
            ('raw_data', 'Raw Data'),
            ('processed_data', 'Processed'),
            ('sentiment_queue', 'Sentiment'),
            ('signal_queue', 'Signals'),
            ('storage_queue', 'Storage')
        ]
        
        for queue_id, queue_name in queues:
            size = queue_sizes.get(queue_id, 0)
            
            # Color coding based on queue depth
            if size == 0:
                color = Fore.GREEN
                bar = ""
            elif size < 10:
                color = Fore.CYAN
                bar = "█" * min(size, 20)
            elif size < 50:
                color = Fore.YELLOW
                bar = "█" * min(size // 3, 20)
            else:
                color = Fore.RED
                bar = "█" * 20
            
            lines.append(f"  {Fore.WHITE}{queue_name:12} │ {color}{size:4} items {bar}")
        
        return lines
    
    def _render_worker_status(self) -> List[str]:
        """Render worker status section."""
        lines = []
        lines.append(f"{Fore.YELLOW}{Style.BRIGHT}⚙️  PROCESSING WORKERS")
        lines.append(f"{Fore.YELLOW}{'─' * self.width}")
        
        if not self.workers:
            lines.append(f"  {Fore.RED}No workers available")
            return lines
        
        for worker in self.workers:
            worker_id = getattr(worker, 'worker_id', '?')
            is_running = getattr(worker, 'running', False)
            processed = getattr(worker, 'processed_count', 0)
            errors = getattr(worker, 'error_count', 0)
            
            status_color = Fore.GREEN if is_running else Fore.RED
            status_icon = "●" if is_running else "○"
            
            lines.append(f"  Worker {worker_id} {status_color}{status_icon} │ "
                        f"{Fore.WHITE}Processed: {Fore.CYAN}{processed:4} │ "
                        f"{Fore.WHITE}Errors: {Fore.RED if errors > 0 else Fore.GREEN}{errors:2}")
        
        return lines
    
    def _render_stream_status(self) -> List[str]:
        """Render data stream status section."""
        lines = []
        lines.append(f"{Fore.YELLOW}{Style.BRIGHT}📡 DATA STREAMS")
        lines.append(f"{Fore.YELLOW}{'─' * self.width}")
        
        if not self.data_streams:
            lines.append(f"  {Fore.RED}No data streams available")
            return lines
        
        stream_names = ['Twitter', 'Reddit', 'Market']
        for i, stream in enumerate(self.data_streams):
            name = stream_names[i] if i < len(stream_names) else f"Stream {i+1}"
            is_running = getattr(stream, 'running', False)
            
            status_color = Fore.GREEN if is_running else Fore.RED
            status_icon = "●" if is_running else "○"
            status_text = "ACTIVE" if is_running else "STOPPED"
            
            lines.append(f"  {name:10} {status_color}{status_icon} {status_text}")
        
        return lines
    
    def _render_performance_metrics(self) -> List[str]:
        """Render performance metrics section."""
        lines = []
        lines.append(f"{Fore.YELLOW}{Style.BRIGHT}📈 PERFORMANCE METRICS")
        lines.append(f"{Fore.YELLOW}{'─' * self.width}")
        
        # Throughput chart (last 30 seconds)
        if self.throughput_history:
            avg_throughput = sum(self.throughput_history) / len(self.throughput_history)
            max_throughput = max(self.throughput_history)
            min_throughput = min(self.throughput_history)
            
            lines.append(f"  {Fore.WHITE}Average: {Fore.CYAN}{avg_throughput:.2f} items/sec")
            lines.append(f"  {Fore.WHITE}Peak: {Fore.GREEN}{max_throughput:.2f} items/sec")
            lines.append(f"  {Fore.WHITE}Min: {Fore.YELLOW}{min_throughput:.2f} items/sec")
            
            # Simple sparkline
            sparkline = self._create_sparkline(list(self.throughput_history), width=50)
            lines.append(f"  {Fore.CYAN}Throughput: {sparkline}")
        else:
            lines.append(f"  {Fore.YELLOW}Collecting metrics...")
        
        return lines
    
    def _render_footer(self) -> List[str]:
        """Render dashboard footer."""
        lines = []
        border = "=" * self.width
        lines.append(f"{Fore.CYAN}{border}")
        
        footer_text = "Press Ctrl+C to stop monitoring"
        if COLORAMA_AVAILABLE:
            footer_text += " │ Colors: " + \
                          f"{Fore.GREEN}●{Fore.WHITE} Healthy  " + \
                          f"{Fore.YELLOW}●{Fore.WHITE} Warning  " + \
                          f"{Fore.RED}●{Fore.WHITE} Critical"
        
        lines.append(f"{Fore.CYAN}{footer_text}")
        lines.append(f"{Fore.CYAN}{border}")
        
        return lines
    
    def _calculate_throughput(self, current_processed: int) -> float:
        """
        Calculate current throughput.
        
        Args:
            current_processed: Total items processed so far
            
        Returns:
            Current throughput in items/sec
        """
        elapsed = time.time() - self.start_time
        if elapsed < 1:
            return 0.0
        
        # Calculate instantaneous throughput (items since last check)
        items_delta = current_processed - self.last_processed_count
        throughput = items_delta / self.refresh_interval
        
        self.last_processed_count = current_processed
        self.throughput_history.append(throughput)
        
        return throughput
    
    def _create_sparkline(self, data: List[float], width: int = 50) -> str:
        """
        Create a simple ASCII sparkline chart.
        
        Args:
            data: List of values to plot
            width: Width of the sparkline
            
        Returns:
            ASCII sparkline string
        """
        if not data:
            return ""
        
        # Normalize data to 0-7 range for block characters
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            return "▄" * min(len(data), width)
        
        # Use block characters for simple visualization
        blocks = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        
        # Take last 'width' points
        plot_data = data[-width:] if len(data) > width else data
        
        sparkline = ""
        for value in plot_data:
            normalized = (value - min_val) / (max_val - min_val)
            block_index = int(normalized * (len(blocks) - 1))
            sparkline += blocks[block_index]
        
        return sparkline


def create_monitor(workers: List, data_streams: List, refresh_interval: float = 1.0) -> DashboardMonitor:
    """
    Factory function to create and configure a dashboard monitor.
    
    Args:
        workers: List of ProcessingWorker instances
        data_streams: List of data stream instances
        refresh_interval: Update interval in seconds
        
    Returns:
        Configured DashboardMonitor instance
    """
    monitor = DashboardMonitor(refresh_interval=refresh_interval)
    monitor.set_components(workers, data_streams)
    return monitor


if __name__ == "__main__":
    # Test the dashboard
    print("Testing Dashboard Monitor...")
    print("This would normally show a live dashboard.")
    print("Run from main.py with --monitor flag to see it in action.")
