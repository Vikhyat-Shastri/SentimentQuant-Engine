"""
Main entry point for Fear & Greed Sentiment Analysis Engine.
Coordinates all system components and manages lifecycle.
"""

import sys
import signal
import time
import argparse
import threading
import queue
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import thread_manager, config_manager, create_monitor
from src.ingestion.twitter_stream import TwitterStream
from src.ingestion.reddit_stream import RedditStream
from src.ingestion.news_stream import NewsStream
from src.ingestion.market_data import MarketDataFeed
from src.processing.worker import ProcessingWorker
from src.signals.signal_generator import SignalGenerator

# BONUS FEATURES - Advanced ML & Analytics
from src.ml.sarcasm_detector import SarcasmDetector
from src.ml.ensemble_sentiment import EnsembleSentimentAnalyzer
from src.ml.multilingual_sentiment import MultiLingualSentimentAnalyzer
from src.ml.multimodal_analyzer import MultimodalAnalyzer
from src.analytics.crowd_psychology import CrowdPsychologyAnalyzer
from src.analytics.regime_detection import MarketRegimeClassifier
from src.analytics.cross_market import CrossMarketAnalyzer


class SentimentEngine:
    """
    Main sentiment analysis engine that coordinates all components.
    """
    
    def __init__(self, mode: str = "simulation", num_workers: int = 2, duration: Optional[int] = None, enable_monitor: bool = False):
        """
        Initialize the sentiment analysis engine.
        
        Args:
            mode: Operation mode ('simulation' or 'live')
            num_workers: Number of processing worker threads
            duration: Runtime duration in seconds (None = infinite)
            enable_monitor: Enable real-time dashboard monitoring
        """
        self.mode = mode
        self.num_workers = num_workers
        self.duration = duration
        self.enable_monitor = enable_monitor
        self.running = False
        self.stop_event = threading.Event()
        
        # Component lists
        self.data_streams: List = []
        self.workers: List[ProcessingWorker] = []
        self.signal_generator: Optional[SignalGenerator] = None
        self.signal_consumer_thread: Optional[threading.Thread] = None
        self.monitor = None
        
        # Advanced ML & Analytics Components (BONUS FEATURES)
        self.sarcasm_detector: Optional[SarcasmDetector] = None
        self.ensemble_sentiment: Optional[EnsembleSentimentAnalyzer] = None
        self.multilingual_analyzer: Optional[MultiLingualSentimentAnalyzer] = None
        self.multimodal_analyzer: Optional[MultimodalAnalyzer] = None
        self.crowd_psychology: Optional[CrowdPsychologyAnalyzer] = None
        self.regime_classifier: Optional[MarketRegimeClassifier] = None
        self.cross_market: Optional[CrossMarketAnalyzer] = None
        
        # Analytics tracking
        self.sentiment_history: List[Dict] = []
        self.market_data_history: List[Dict] = []
        
        self.setup_logging()
        self.load_configuration()
        self.initialize_advanced_components()
        
        logger.info("=" * 70)
        logger.info("🚀 Fear & Greed Sentiment Analysis Engine")
        logger.info("=" * 70)
        logger.info(f"Mode: {mode.upper()}")
        logger.info(f"Workers: {num_workers}")
        logger.info(f"Duration: {duration if duration else 'Infinite'} seconds")
        logger.info(f"Monitor: {'Enabled' if enable_monitor else 'Disabled'}")
        logger.info("=" * 70)
    
    def setup_logging(self) -> None:
        """Configure logging system."""
        # Remove default logger
        logger.remove()
        
        # Add console logger
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level="INFO"
        )
        
        # Add file logger
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        
        logger.add(
            "logs/sentiment_engine_{time:YYYY-MM-DD}.log",
            rotation="00:00",  # New file each day
            retention="7 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level="DEBUG"
        )
        
        logger.info("Logging system initialized")
    
    def load_configuration(self) -> None:
        """Load system configuration."""
        logger.info("Loading configuration files...")
        
        # Configuration is loaded automatically by config_manager
        sentiment_config = config_manager.get_section('sentiment_config', 'sentiment')
        signal_config = config_manager.get_section('signal_config', 'signals')
        
        if sentiment_config:
            logger.info(f"Loaded sentiment configuration")
        else:
            logger.warning("Sentiment configuration not found, using defaults")
        
        if signal_config:
            logger.info(f"Loaded signal configuration")
        else:
            logger.warning("Signal configuration not found, using defaults")
    
    def initialize_advanced_components(self) -> None:
        """Initialize advanced ML and analytics components (BONUS FEATURES)."""
        logger.info("🚀 Initializing advanced components...")
        
        try:
            # Sarcasm Detection (LOW priority bonus)
            self.sarcasm_detector = SarcasmDetector(use_model=False)  # Start without heavy model
            logger.success("   ✓ Sarcasm detector initialized")
            
            # Ensemble Sentiment Analysis (MEDIUM priority bonus)
            self.ensemble_sentiment = EnsembleSentimentAnalyzer()
            logger.success("   ✓ Ensemble sentiment analyzer initialized")
            
            # Multi-language Sentiment (HIGH priority bonus)
            self.multilingual_analyzer = MultiLingualSentimentAnalyzer()
            logger.success("   ✓ Multi-lingual analyzer initialized")
            
            # Multimodal Analysis (LOW priority bonus)
            self.multimodal_analyzer = MultimodalAnalyzer()
            logger.success("   ✓ Multimodal analyzer initialized")
            
            # Crowd Psychology (MEDIUM priority bonus)
            self.crowd_psychology = CrowdPsychologyAnalyzer()
            logger.success("   ✓ Crowd psychology analyzer initialized")
            
            # Regime Classification (MEDIUM priority bonus)
            self.regime_classifier = MarketRegimeClassifier()
            logger.success("   ✓ Market regime classifier initialized")
            
            # Cross-Market Analysis (MEDIUM priority bonus)
            self.cross_market = CrossMarketAnalyzer()
            logger.success("   ✓ Cross-market analyzer initialized")
            
            logger.info(f"   Total: 7 advanced components initialized")
            
        except Exception as e:
            logger.warning(f"Some advanced components failed to initialize: {e}")
            logger.info("System will continue with available components")
    
    def start(self) -> None:
        """Start the sentiment analysis engine."""
        logger.info("🔧 Starting sentiment analysis engine...")
        self.running = True
        
        try:
            # Start data ingestion threads
            logger.info("📡 Starting data ingestion streams...")
            self.start_data_ingestion()
            
            # Start processing worker threads
            logger.info("⚙️  Starting processing workers...")
            self.start_processing_workers()
            
            # Start signal generator
            logger.info("📊 Starting signal generator...")
            self.start_signal_generator()
            
            # Start signal consumer (displays signals in real-time)
            logger.info("📤 Starting signal consumer...")
            self.start_signal_consumer()
            
            # Start market data tracker for regime analysis
            logger.info("📈 Starting market data tracker...")
            self.start_market_tracker()
            
            # Start monitoring dashboard if enabled
            if self.enable_monitor:
                logger.info("📊 Starting monitoring dashboard...")
                self.monitor = create_monitor(
                    workers=self.workers,
                    data_streams=self.data_streams,
                    refresh_interval=1.0
                )
                self.monitor.start()
                logger.success("   ✓ Dashboard started")
                time.sleep(2)  # Give dashboard time to initialize
            
            logger.success("✅ All systems operational!")
            if not self.enable_monitor:
                logger.info("Press Ctrl+C to stop the engine")
                logger.info("")
            
            # Main loop
            self.main_loop()
            
            # Stop gracefully after duration completes
            self.stop()
            
        except KeyboardInterrupt:
            logger.info("\n⚠️  Received shutdown signal (Ctrl+C)")
            self.stop()
        except Exception as e:
            logger.exception(f"❌ Fatal error in main loop: {e}")
            self.stop()
            sys.exit(1)
    
    def main_loop(self) -> None:
        """
        Main processing loop.
        Monitors system health and displays statistics.
        """
        start_time = time.time()
        iteration = 0
        stats_interval = 10  # Display stats every 10 seconds
        
        while self.running and not self.stop_event.is_set():
            try:
                time.sleep(1)  # Check every second
                iteration += 1
                elapsed = time.time() - start_time
                
                # Check duration limit
                if self.duration and elapsed >= self.duration:
                    if not self.enable_monitor:
                        logger.info(f"⏰ Duration limit reached ({self.duration}s)")
                    break
                
                # Display statistics periodically (only if monitor not enabled)
                if not self.enable_monitor and iteration % stats_interval == 0:
                    self.display_statistics(elapsed)
                    self.display_advanced_analytics(elapsed)
                
            except KeyboardInterrupt:
                break
    
    def display_statistics(self, elapsed: float) -> None:
        """
        Display current system statistics.
        
        Args:
            elapsed: Elapsed time since start in seconds
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"📊 System Statistics - Runtime: {int(elapsed)}s")
        logger.info("=" * 70)
        
        # Queue sizes
        queue_sizes = thread_manager.get_all_queue_sizes()
        logger.info(f"📦 Queue Status:")
        logger.info(f"   raw_data: {queue_sizes.get('raw_data', 0)} items")
        logger.info(f"   processed_data: {queue_sizes.get('processed_data', 0)} items")
        logger.info(f"   sentiment_queue: {queue_sizes.get('sentiment_queue', 0)} items")
        logger.info(f"   signal_queue: {queue_sizes.get('signal_queue', 0)} items")
        logger.info(f"   storage_queue: {queue_sizes.get('storage_queue', 0)} items")
        
        # Worker status
        active_workers = sum(1 for w in self.workers if w.running)
        logger.info(f"⚙️  Workers: {active_workers}/{len(self.workers)} active")
        
        # Data streams status
        active_streams = sum(1 for s in self.data_streams if hasattr(s, 'running') and s.running)
        logger.info(f"📡 Data Streams: {active_streams}/{len(self.data_streams)} active")
        
        # Processing statistics (from workers)
        total_processed = sum(w.processed_count for w in self.workers if hasattr(w, 'processed_count'))
        total_errors = sum(w.error_count for w in self.workers if hasattr(w, 'error_count'))
        logger.info(f"📈 Processing: {total_processed} items, {total_errors} errors")
        
        if elapsed > 0:
            throughput = total_processed / elapsed
            logger.info(f"⚡ Throughput: {throughput:.2f} items/sec")
        
        # Signal generation statistics
        if self.signal_generator:
            signal_summary = self.signal_generator.get_signal_summary()
            logger.info(f"🎯 Signals Generated: {signal_summary.get('total', 0)}")
            if signal_summary.get('by_action'):
                actions_str = ", ".join([f"{k}: {v}" for k, v in signal_summary['by_action'].items()])
                logger.info(f"   {actions_str}")
            if signal_summary.get('avg_confidence'):
                logger.info(f"   Avg Confidence: {signal_summary['avg_confidence']:.2f}")
        
        logger.info("=" * 70)
        logger.info("")
    
    def display_advanced_analytics(self, elapsed: float) -> None:
        """
        Display advanced analytics from bonus features.
        
        Args:
            elapsed: Elapsed time since start in seconds
        """
        if not self.sentiment_history:
            logger.info("=" * 70)
            logger.info(f"🎯 Advanced Analytics (Bonus Features) - No Data Yet")
            logger.info(f"   Sentiment history: {len(self.sentiment_history)} records")
            logger.info("=" * 70)
            logger.info("")
            return
        
        logger.info("=" * 70)
        logger.info(f"🎯 Advanced Analytics (Bonus Features) - {len(self.sentiment_history)} records")
        logger.info("=" * 70)
        
        # Crowd Psychology Analysis
        if self.crowd_psychology and len(self.sentiment_history) >= 10:
            try:
                import pandas as pd
                recent_sentiment = pd.DataFrame(self.sentiment_history[-50:])
                crowd_metrics = self.crowd_psychology.analyze(recent_sentiment)
                
                logger.info(f"👥 Crowd Psychology:")
                logger.info(f"   FOMO Score: {crowd_metrics.fomo_score:.2f} {'🔥' if crowd_metrics.fomo_score > 0.7 else ''}")
                logger.info(f"   Panic Score: {crowd_metrics.panic_score:.2f} {'😱' if crowd_metrics.panic_score > 0.7 else ''}")
                logger.info(f"   Herd Behavior: {crowd_metrics.herd_behavior_score:.2f}")
                logger.info(f"   Euphoria Level: {crowd_metrics.euphoria_level:.2f}")
            except Exception as e:
                logger.info(f"⚠️  Crowd psychology analysis skipped: {e}")
        else:
            logger.info(f"⏸️  Crowd psychology: Need 10+ records (have {len(self.sentiment_history)})")
        
        # Market Regime Detection
        if self.regime_classifier and self.market_data_history:
            try:
                import pandas as pd
                recent_market = pd.DataFrame(self.market_data_history[-100:])
                if 'close' in recent_market.columns and len(recent_market) >= 20:
                    regime = self.regime_classifier.classify_current_regime(recent_market['close'].values)
                    regime_emoji = {'BULL': '🐂', 'BEAR': '🐻', 'SIDEWAYS': '↔️'}
                    logger.info(f"📊 Market Regime: {regime_emoji.get(regime, '❓')} {regime}")
                else:
                    logger.info(f"⏸️  Market regime: Need 20+ market records with 'close' column")
            except Exception as e:
                logger.info(f"⚠️  Regime classification skipped: {e}")
        else:
            logger.info(f"⏸️  Market regime: No market data history")
        
        # Cross-Market Correlation
        if self.cross_market and len(self.sentiment_history) >= 20:
            try:
                import pandas as pd
                sentiment_df = pd.DataFrame(self.sentiment_history[-100:])
                if 'asset' in sentiment_df.columns:
                    correlations = self.cross_market.analyze_sentiment_correlation(sentiment_df)
                    if correlations:
                        logger.info(f"🔗 Cross-Asset Correlation:")
                        for (asset1, asset2), corr in list(correlations.items())[:3]:
                            logger.info(f"   {asset1} ↔ {asset2}: {corr:.2f}")
                    else:
                        logger.info(f"⏸️  Cross-market: No correlations found")
                else:
                    logger.info(f"⏸️  Cross-market: No 'asset' column in sentiment data")
            except Exception as e:
                logger.info(f"⚠️  Cross-market analysis skipped: {e}")
        else:
            logger.info(f"⏸️  Cross-market: Need 20+ records (have {len(self.sentiment_history)})")
        
        logger.info("=" * 70)
        logger.info("")
    
    def stop(self) -> None:
        """Stop the sentiment analysis engine gracefully."""
        # Stop monitor first to show clean shutdown messages
        if self.monitor:
            self.monitor.stop()
            time.sleep(0.5)
        
        logger.info("")
        logger.info("🛑 Stopping sentiment analysis engine...")
        self.running = False
        self.stop_event.set()
        
        # Stop data ingestion streams
        logger.info("📡 Stopping data streams...")
        for stream in self.data_streams:
            try:
                if hasattr(stream, 'stop'):
                    stream.stop()
                else:
                    stream.running = False
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
        
        # Stop processing workers
        logger.info("⚙️  Stopping processing workers...")
        for worker in self.workers:
            try:
                worker.stop()
            except Exception as e:
                logger.error(f"Error stopping worker: {e}")
        
        # Stop signal generator
        if self.signal_generator:
            logger.info("📊 Stopping signal generator...")
            try:
                self.signal_generator.stop()
            except Exception as e:
                logger.error(f"Error stopping signal generator: {e}")
        
        # Stop signal consumer thread
        if self.signal_consumer_thread and self.signal_consumer_thread.is_alive():
            logger.info("🎯 Stopping signal consumer...")
            try:
                self.signal_consumer_thread.join(timeout=2.0)
            except Exception as e:
                logger.error(f"Error stopping signal consumer: {e}")
        
        # Wait for threads to finish
        time.sleep(2)
        
        # Display final statistics
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 Final Statistics")
        logger.info("=" * 70)
        total_processed = sum(w.processed_count for w in self.workers if hasattr(w, 'processed_count'))
        total_errors = sum(w.error_count for w in self.workers if hasattr(w, 'error_count'))
        logger.info(f"Total Processed: {total_processed} items")
        logger.info(f"Total Errors: {total_errors} errors")
        logger.info(f"Success Rate: {(total_processed - total_errors) / max(total_processed, 1) * 100:.2f}%")
        
        # Signal generation statistics
        if self.signal_generator:
            signal_summary = self.signal_generator.get_signal_summary()
            logger.info(f"Signals Generated: {signal_summary['total']}")
            if signal_summary.get('by_action'):
                logger.info(f"Signal Breakdown: {signal_summary['by_action']}")
            logger.info(f"Avg Confidence: {signal_summary.get('avg_confidence', 0):.2f}")
        
        logger.info("=" * 70)
        
        logger.success("✅ Sentiment analysis engine stopped cleanly")
    
    def start_data_ingestion(self) -> None:
        """Start data ingestion threads."""
        try:
            # Get configuration
            config = config_manager.get_section('sentiment_config', 'ingestion') or {}
            
            # Determine if using live APIs or simulation
            mode = self.mode if hasattr(self, 'mode') else 'simulation'
            
            # Twitter Stream (run in background thread)
            twitter_keywords = config.get('twitter_keywords', ['bitcoin', 'BTC', 'ethereum', 'ETH', 'crypto'])
            twitter_stream = TwitterStream(keywords=twitter_keywords, mode=mode)
            twitter_thread = threading.Thread(
                target=twitter_stream.start,
                args=(self.stop_event,),
                daemon=True,
                name="TwitterStream"
            )
            twitter_thread.start()
            self.data_streams.append(twitter_stream)
            logger.success(f"   ✓ Twitter stream started in {mode} mode (keywords: {len(twitter_keywords)})")
            
            # Reddit Stream (run in background thread)
            reddit_subreddits = config.get('reddit_subreddits', ['cryptocurrency', 'bitcoin', 'ethereum'])
            reddit_stream = RedditStream(subreddits=reddit_subreddits, mode=mode)
            reddit_thread = threading.Thread(
                target=reddit_stream.start,
                args=(self.stop_event,),
                daemon=True,
                name="RedditStream"
            )
            reddit_thread.start()
            self.data_streams.append(reddit_stream)
            logger.success(f"   ✓ Reddit stream started in {mode} mode (subreddits: {len(reddit_subreddits)})")
            
            # News Stream (run in background thread)
            news_keywords = config.get('news_keywords', ['bitcoin', 'ethereum', 'cryptocurrency'])
            news_stream = NewsStream(keywords=news_keywords, mode=mode)
            news_thread = threading.Thread(
                target=news_stream.start,
                args=(self.stop_event,),
                daemon=True,
                name="NewsStream"
            )
            news_thread.start()
            self.data_streams.append(news_stream)
            logger.success(f"   ✓ News stream started in {mode} mode (keywords: {len(news_keywords)})")
            
            # Market Data Feed (run in background thread)
            market_symbols = config.get('market_symbols', ['BTC-USD', 'ETH-USD'])
            market_feed = MarketDataFeed(symbols=market_symbols)
            market_thread = threading.Thread(
                target=market_feed.start,
                args=(self.stop_event,),
                daemon=True,
                name="MarketDataFeed"
            )
            market_thread.start()
            self.data_streams.append(market_feed)
            logger.success(f"   ✓ Market feed started (symbols: {len(market_symbols)})")
            
            logger.info(f"   Total: {len(self.data_streams)} data streams active")
            
        except Exception as e:
            logger.exception(f"Error starting data ingestion: {e}")
            raise
    
    def start_processing_workers(self) -> None:
        """Start processing worker threads."""
        try:
            config = config_manager.get_section('sentiment_config', 'processing') or {}
            batch_size = config.get('batch_size', 10)
            batch_timeout = config.get('batch_timeout', 0.5)
            
            for i in range(self.num_workers):
                worker = ProcessingWorker(
                    worker_id=i + 1,
                    batch_size=batch_size,
                    batch_timeout=batch_timeout
                )
                worker.start()
                self.workers.append(worker)
                logger.success(f"   ✓ Worker {i + 1} started")
            
            logger.info(f"   Total: {len(self.workers)} workers active")
            
        except Exception as e:
            logger.exception(f"Error starting processing workers: {e}")
            raise
    
    def start_signal_generator(self) -> None:
        """Start signal generator thread."""
        try:
            # Initialize signal generator
            self.signal_generator = SignalGenerator(config_path="config/signal_config.yaml")
            
            # Get queues from thread manager
            sentiment_queue = thread_manager.queues['sentiment_queue']
            signal_queue = thread_manager.queues['signal_queue']
            
            # Start signal generation
            self.signal_generator.start(sentiment_queue, signal_queue)
            logger.success(f"   ✓ Signal generator started")
            
        except Exception as e:
            logger.exception(f"Error starting signal generator: {e}")
            raise
    
    def start_signal_consumer(self) -> None:
        """Start signal consumer thread to display signals in real-time."""
        try:
            signal_queue = thread_manager.queues['signal_queue']
            sentiment_queue = thread_manager.queues['sentiment_queue']
            
            def consume_signals():
                """Consumer function that displays signals and collects analytics data"""
                logger.info("Signal consumer thread started")
                while self.running and not self.stop_event.is_set():
                    try:
                        # Process sentiment data for analytics (NON-BLOCKING)
                        try:
                            while not sentiment_queue.empty():
                                sentiment_data = sentiment_queue.get_nowait()
                                
                                # Apply advanced sentiment processing
                                enhanced_sentiment = self._enhance_sentiment_with_bonus_features(sentiment_data)
                                
                                # Store for analytics
                                self.sentiment_history.append(enhanced_sentiment)
                                if len(self.sentiment_history) > 1000:  # Keep last 1000
                                    self.sentiment_history = self.sentiment_history[-1000:]
                                
                                sentiment_queue.task_done()
                        except queue.Empty:
                            pass
                        
                        # Get signal from queue (with timeout)
                        signal = signal_queue.get(timeout=1.0)
                        
                        # Display signal
                        self._display_signal(signal)
                        
                        signal_queue.task_done()
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error consuming signal: {e}")
                        time.sleep(0.1)
                
                logger.info("Signal consumer thread stopped")
            
            # Start consumer thread
            self.signal_consumer_thread = threading.Thread(
                target=consume_signals,
                daemon=True,
                name="SignalConsumer"
            )
            self.signal_consumer_thread.start()
            logger.success(f"   ✓ Signal consumer started")
            
        except Exception as e:
            logger.exception(f"Error starting signal consumer: {e}")
            raise
    
    def _enhance_sentiment_with_bonus_features(self, sentiment_data: Dict) -> Dict:
        """
        Enhance sentiment data with bonus feature processing.
        
        Args:
            sentiment_data: Base sentiment data
            
        Returns:
            Enhanced sentiment data with bonus features
        """
        enhanced = sentiment_data.copy()
        text = sentiment_data.get('text', '')
        
        try:
            # Sarcasm Detection (LOW priority bonus)
            if self.sarcasm_detector and text:
                sarcasm_result = self.sarcasm_detector.detect(text)
                enhanced['is_sarcastic'] = sarcasm_result.is_sarcastic
                enhanced['sarcasm_confidence'] = sarcasm_result.confidence
                if sarcasm_result.is_sarcastic:
                    # Use adjusted sentiment if sarcastic
                    enhanced['original_sentiment'] = enhanced.get('sentiment_score', 0)
                    enhanced['sentiment_score'] = sarcasm_result.adjusted_sentiment
            
            # Ensemble Sentiment (MEDIUM priority bonus)
            if self.ensemble_sentiment and text:
                ensemble_result = self.ensemble_sentiment.analyze(text)
                enhanced['ensemble_sentiment'] = ensemble_result.weighted_sentiment
                enhanced['ensemble_confidence'] = ensemble_result.confidence
                enhanced['model_agreement'] = ensemble_result.model_agreement
                # Optionally override with ensemble
                if ensemble_result.confidence > 0.8:
                    enhanced['sentiment_score'] = ensemble_result.weighted_sentiment
            
            # Multi-language Support (HIGH priority bonus)
            if self.multilingual_analyzer and text:
                try:
                    ml_result = self.multilingual_analyzer.analyze(text)
                    enhanced['detected_language'] = ml_result.detected_language
                    if ml_result.detected_language != 'en':
                        enhanced['translated_text'] = ml_result.translated_text
                        enhanced['multilingual_sentiment'] = ml_result.sentiment_score
                        # Use multilingual sentiment for non-English
                        enhanced['sentiment_score'] = ml_result.sentiment_score
                except Exception as e:
                    logger.debug(f"Multilingual analysis failed: {e}")
            
        except Exception as e:
            logger.debug(f"Error enhancing sentiment: {e}")
        
        return enhanced
    
    def start_market_tracker(self) -> None:
        """Start market data tracker for regime analysis."""
        try:
            def track_market_data():
                """Track market data for regime classification"""
                logger.info("Market data tracker started")
                
                # Try to get market data queue if it exists
                raw_queue = thread_manager.queues.get('raw_data')
                if not raw_queue:
                    logger.warning("Raw data queue not found, market tracker disabled")
                    return
                
                while self.running and not self.stop_event.is_set():
                    try:
                        # Non-blocking check for market data
                        try:
                            data = raw_queue.get_nowait()
                            if data.get('type') == 'market_data':
                                # Store for regime analysis
                                self.market_data_history.append(data)
                                if len(self.market_data_history) > 500:  # Keep last 500
                                    self.market_data_history = self.market_data_history[-500:]
                            raw_queue.task_done()
                        except queue.Empty:
                            pass
                        
                        time.sleep(1)  # Check every second
                        
                    except Exception as e:
                        logger.debug(f"Market tracker error: {e}")
                        time.sleep(1)
                
                logger.info("Market data tracker stopped")
            
            market_thread = threading.Thread(
                target=track_market_data,
                daemon=True,
                name="MarketTracker"
            )
            market_thread.start()
            logger.success(f"   ✓ Market data tracker started")
            
        except Exception as e:
            logger.exception(f"Error starting market tracker: {e}")
            # Non-critical, continue without it
    
    def _display_signal(self, signal: Dict) -> None:
        """
        Display a trading signal in real-time.
        
        Args:
            signal: Signal dictionary from queue
        """
        action = signal.get('action', 'UNKNOWN')
        symbol = signal.get('symbol', 'N/A')
        confidence = signal.get('confidence', 0.0)
        position_size = signal.get('position_size', 0.0)
        fgi = signal.get('fear_greed_index', 0.0)
        reasoning = signal.get('reasoning', '')
        
        # Color coding for actions
        action_colors = {
            'STRONG_BUY': '🟢',
            'BUY': '🟩',
            'HOLD': '⚪',
            'SELL': '🟥',
            'STRONG_SELL': '🔴'
        }
        
        icon = action_colors.get(action, '⚫')
        
        # Display signal
        logger.info("")
        logger.info("🎯 " + "=" * 66)
        logger.info(f"{icon} TRADING SIGNAL: {action}")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Confidence: {confidence:.2%}")
        logger.info(f"   Position Size: {position_size:.1%}")
        logger.info(f"   Fear & Greed Index: {fgi:.1f}")
        logger.info(f"   Reasoning: {reasoning}")
        logger.info("=" * 70)
        logger.info("")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fear & Greed Sentiment Analysis Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in simulation mode with default settings
  python main.py
  
  # Run for 60 seconds with 4 workers
  python main.py --duration 60 --workers 4
  
  # Run in live mode (requires API keys)
  python main.py --mode live
  
  # Run with verbose logging
  python main.py --verbose
"""
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['simulation', 'live'],
        default='simulation',
        help='Operation mode: simulation (mock data) or live (real APIs)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of processing worker threads (default: 2)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Runtime duration in seconds (default: infinite)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Enable real-time monitoring dashboard'
    )
    
    return parser.parse_args()


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logger.info(f"\n⚠️  Received signal {signum}")
    sys.exit(0)


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start engine
    engine = SentimentEngine(
        mode=args.mode,
        num_workers=args.workers,
        duration=args.duration,
        enable_monitor=args.monitor
    )
    
    # Update logging level if verbose
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.debug("Verbose logging enabled")
    
    engine.start()


if __name__ == "__main__":
    main()
