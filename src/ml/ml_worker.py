"""
ML-enhanced sentiment worker using deep learning models.

Combines VADER (fast baseline) with transformer models (accurate) for best results.
"""

import queue
import threading
import logging
import time
from typing import Optional
from dataclasses import dataclass

from src.ml.sentiment_model import SentimentModel, ModelType, get_model
from src.sentiment.sentiment_analyzer import VADERSentimentAnalyzer
from src.utils.thread_manager import DataPacket

logger = logging.getLogger(__name__)


@dataclass
class MLConfig:
    """Configuration for ML worker."""
    use_ml: bool = True
    model_type: ModelType = ModelType.FINBERT
    use_ensemble: bool = False
    ml_weight: float = 0.7  # Weight for ML vs VADER (0.7 = 70% ML, 30% VADER)
    batch_size: int = 8
    device: str = "auto"


class MLSentimentWorker:
    """
    Enhanced sentiment worker using deep learning models.
    
    Uses hybrid approach:
    - VADER for fast baseline
    - Transformer model for accurate domain-specific sentiment
    - Weighted combination for final score
    """
    
    def __init__(
        self,
        worker_id: int,
        raw_queue: queue.Queue,
        sentiment_queue: queue.Queue,
        config: Optional[MLConfig] = None
    ):
        """
        Initialize ML worker.
        
        Args:
            worker_id: Worker ID
            raw_queue: Input queue with raw data
            sentiment_queue: Output queue with sentiment
            config: ML configuration
        """
        self.worker_id = worker_id
        self.raw_queue = raw_queue
        self.sentiment_queue = sentiment_queue
        self.config = config or MLConfig()
        
        # Initialize VADER (fast baseline)
        self.vader = VADERSentimentAnalyzer()
        
        # Initialize ML model (if enabled)
        self.ml_model = None
        if self.config.use_ml:
            try:
                logger.info(f"Worker {worker_id}: Loading ML model {self.config.model_type.name}...")
                self.ml_model = get_model(
                    model_type=self.config.model_type,
                    device=self.config.device
                )
                logger.info(f"Worker {worker_id}: ML model loaded successfully")
            except Exception as e:
                logger.error(f"Worker {worker_id}: Failed to load ML model: {e}")
                logger.info(f"Worker {worker_id}: Falling back to VADER only")
                self.config.use_ml = False
        
        # Statistics
        self.items_processed = 0
        self.errors = 0
        self.ml_predictions = 0
        self.vader_predictions = 0
        
        # Thread control
        self._running = False
        self._thread = None
        
        logger.info(f"MLSentimentWorker {worker_id} initialized "
                   f"(ML: {self.config.use_ml}, Model: {self.config.model_type.name})")
    
    def start(self):
        """Start worker thread."""
        if self._running:
            logger.warning(f"Worker {self.worker_id} already running")
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name=f"MLSentimentWorker-{self.worker_id}"
        )
        self._thread.start()
        
        logger.info(f"MLSentimentWorker {self.worker_id} started")
    
    def stop(self):
        """Stop worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        
        logger.info(f"MLSentimentWorker {self.worker_id} stopped "
                   f"(processed: {self.items_processed}, errors: {self.errors})")
    
    def _process_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                # Get data with timeout
                packet = self.raw_queue.get(timeout=1)
                
                # Process sentiment
                start_time = time.perf_counter()
                result = self._analyze_sentiment(packet)
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Add to sentiment queue
                if result:
                    result.data['processing_latency_ms'] = latency_ms
                    self.sentiment_queue.put(result)
                    self.items_processed += 1
                
                self.raw_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}", exc_info=True)
                self.errors += 1
    
    def _analyze_sentiment(self, packet: DataPacket) -> Optional[DataPacket]:
        """
        Analyze sentiment using hybrid approach.
        
        Args:
            packet: Input data packet (or raw object from streams)
        
        Returns:
            Data packet with sentiment scores
        """
        try:
            logger.debug(f"Worker {self.worker_id}: Processing {type(packet).__name__}")
            
            # Extract text from packet
            # Packets from streams are DataPacket objects with data field containing the actual object
            if isinstance(packet, DataPacket):
                # Check if data is a dict with 'text' key
                if isinstance(packet.data, dict):
                    text = packet.data.get('text', '')
                # Check if data is an object with .text or .content attribute
                elif hasattr(packet.data, 'text') and isinstance(packet.data.text, str):
                    text = packet.data.text
                elif hasattr(packet.data, 'content') and isinstance(packet.data.content, str):
                    text = packet.data.content
                else:
                    logger.debug(f"Could not extract text from DataPacket.data type: {type(packet.data).__name__}")
                    return None
            # Handle raw objects (not wrapped in DataPacket)
            elif hasattr(packet, 'text') and isinstance(packet.text, str):
                text = packet.text
            elif hasattr(packet, 'content') and isinstance(packet.content, str):
                text = packet.content
            else:
                logger.debug(f"Could not extract text from {type(packet).__name__}")
                return None
            
            logger.debug(f"Worker {self.worker_id}: Extracted text: {text[:50] if len(text) > 50 else text}...")
            
            if not text:
                return None
            
            # Get VADER sentiment (fast)
            vader_result = self.vader.analyze(text)
            # vader_result is a SentimentScore object, not a dict
            vader_score = vader_result.compound
            
            # Use the packet as-is (it's already a DataPacket from streams)
            result_packet = packet
            
            # Ensure packet.data is a dict for storing sentiment results
            if not isinstance(result_packet.data, dict):
                # Wrap the original data object in a dict
                original_data = result_packet.data
                result_packet.data = {
                    'text': text,
                    'original_object': original_data,
                    'source': type(original_data).__name__
                }
            
            # If ML is disabled, use VADER only
            if not self.config.use_ml or not self.ml_model:
                result_packet.data['sentiment_score'] = vader_score
                result_packet.data['sentiment_method'] = 'vader'
                result_packet.data['sentiment_confidence'] = abs(vader_score)
                self.vader_predictions += 1
                return result_packet
            
            # Get ML sentiment (accurate)
            try:
                ml_result = self.ml_model.analyze(text)
                ml_score = ml_result['score']
                ml_confidence = ml_result['confidence']
                ml_label = ml_result['label']
                
                # Combine scores using weighted average
                final_score = (
                    ml_score * self.config.ml_weight +
                    vader_score * (1 - self.config.ml_weight)
                )
                
                # Use ML confidence as primary confidence
                final_confidence = ml_confidence
                
                # Add all information to packet
                result_packet.data['sentiment_score'] = final_score
                result_packet.data['sentiment_confidence'] = final_confidence
                result_packet.data['sentiment_label'] = ml_label
                result_packet.data['sentiment_method'] = 'hybrid'
                result_packet.data['vader_score'] = vader_score
                result_packet.data['ml_score'] = ml_score
                result_packet.data['ml_model'] = self.config.model_type.name
                
                self.ml_predictions += 1
                
            except Exception as e:
                logger.warning(f"ML prediction failed, using VADER: {e}")
                result_packet.data['sentiment_score'] = vader_score
                result_packet.data['sentiment_method'] = 'vader_fallback'
                result_packet.data['sentiment_confidence'] = abs(vader_score)
                self.vader_predictions += 1
            
            return result_packet
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Get worker statistics."""
        return {
            'worker_id': self.worker_id,
            'items_processed': self.items_processed,
            'errors': self.errors,
            'ml_predictions': self.ml_predictions,
            'vader_predictions': self.vader_predictions,
            'ml_enabled': self.config.use_ml,
            'model': self.config.model_type.name if self.ml_model else 'none'
        }
