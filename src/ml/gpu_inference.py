"""
GPU Acceleration for ML Inference

Optimizes deep learning model inference using CUDA (if available).
Features:
- Automatic device selection (GPU/CPU)
- Batch processing optimization
- Memory management
- Multi-GPU support
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from loguru import logger
import time


@dataclass
class GPUConfig:
    """GPU configuration"""
    device: torch.device
    device_name: str
    is_cuda: bool
    cuda_version: Optional[str]
    memory_allocated_gb: float
    memory_reserved_gb: float
    num_gpus: int


class GPUInferenceEngine:
    """
    GPU-accelerated inference engine
    
    Features:
    - Automatic GPU detection and selection
    - Batch processing for throughput
    - Mixed precision (FP16) for speed
    - Memory optimization
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_fp16: bool = False,
        batch_size: int = 32
    ):
        """
        Initialize GPU engine
        
        Args:
            device: Device to use ('cuda', 'cpu', or 'cuda:0', 'cuda:1', etc.)
                   If None, automatically selects best device
            use_fp16: Use mixed precision (FP16) for faster inference
            batch_size: Default batch size for processing
        """
        self.device = self._select_device(device)
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        self.batch_size = batch_size
        
        self.config = self._get_gpu_config()
        
        logger.info(f"GPUInferenceEngine initialized:")
        logger.info(f"  Device: {self.config.device_name}")
        logger.info(f"  CUDA available: {self.config.is_cuda}")
        if self.config.is_cuda:
            logger.info(f"  CUDA version: {self.config.cuda_version}")
            logger.info(f"  GPU memory: {self.config.memory_allocated_gb:.2f} GB allocated")
            logger.info(f"  Num GPUs: {self.config.num_gpus}")
        logger.info(f"  FP16 mode: {self.use_fp16}")
        logger.info(f"  Batch size: {batch_size}")
    
    def _select_device(self, device: Optional[str]) -> torch.device:
        """Select best available device"""
        if device is not None:
            return torch.device(device)
        
        if torch.cuda.is_available():
            # Select GPU with most free memory
            max_free = 0
            best_gpu = 0
            
            for i in range(torch.cuda.device_count()):
                free = torch.cuda.get_device_properties(i).total_memory
                if free > max_free:
                    max_free = free
                    best_gpu = i
            
            return torch.device(f'cuda:{best_gpu}')
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device('cpu')
    
    def _get_gpu_config(self) -> GPUConfig:
        """Get GPU configuration details"""
        is_cuda = self.device.type == 'cuda'
        
        if is_cuda:
            device_name = torch.cuda.get_device_name(self.device)
            cuda_version = torch.version.cuda
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1e9
            num_gpus = torch.cuda.device_count()
        else:
            device_name = 'CPU'
            cuda_version = None
            memory_allocated = 0.0
            memory_reserved = 0.0
            num_gpus = 0
        
        return GPUConfig(
            device=self.device,
            device_name=device_name,
            is_cuda=is_cuda,
            cuda_version=cuda_version,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved,
            num_gpus=num_gpus
        )
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for inference
        
        Args:
            model: PyTorch model
        
        Returns:
            Optimized model
        """
        # Move to device
        model = model.to(self.device)
        
        # Set to eval mode
        model.eval()
        
        # Use FP16 if enabled
        if self.use_fp16:
            model = model.half()
            logger.info("Model converted to FP16")
        
        # Compile with TorchScript for faster execution (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile()")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
        
        return model
    
    def batch_inference(
        self,
        model: nn.Module,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Run batched inference for better throughput
        
        Args:
            model: PyTorch model
            inputs: Input tensor(s) or list of tensors
            batch_size: Batch size (uses default if None)
        
        Returns:
            Output tensor
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Handle list of inputs
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        
        # Ensure model is on correct device
        model = model.to(self.device)
        inputs = inputs.to(self.device)
        
        # Apply FP16 if enabled
        if self.use_fp16:
            inputs = inputs.half()
        
        # Process in batches
        num_samples = inputs.size(0)
        outputs = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = inputs[i:i + batch_size]
                
                # Forward pass
                batch_output = model(batch)
                outputs.append(batch_output)
        
        # Concatenate results
        return torch.cat(outputs, dim=0)
    
    def benchmark(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict:
        """
        Benchmark model inference speed
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input tensor (including batch dimension)
            num_iterations: Number of inference runs
            warmup_iterations: Warmup runs (not timed)
        
        Returns:
            Dict with benchmark results
        """
        logger.info(f"Benchmarking model on {self.device}...")
        
        # Prepare model
        model = self.optimize_model(model)
        
        # Generate random input
        dummy_input = torch.randn(*input_shape).to(self.device)
        if self.use_fp16:
            dummy_input = dummy_input.half()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)
        
        # Synchronize GPU
        if self.config.is_cuda:
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                
                _ = model(dummy_input)
                
                # Synchronize to get accurate timing
                if self.config.is_cuda:
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append(end - start)
        
        # Calculate statistics
        times = np.array(times) * 1000  # Convert to milliseconds
        
        results = {
            'device': str(self.device),
            'batch_size': input_shape[0],
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_samples_per_sec': input_shape[0] / (np.mean(times) / 1000)
        }
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Mean: {results['mean_ms']:.2f} ms")
        logger.info(f"  Throughput: {results['throughput_samples_per_sec']:.0f} samples/sec")
        
        return results
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        if self.config.is_cuda:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def get_memory_usage(self) -> Dict:
        """Get current GPU memory usage"""
        if not self.config.is_cuda:
            return {'device': 'cpu', 'available': True}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        
        return {
            'device': str(self.device),
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved,
            'utilization_pct': (reserved / total) * 100
        }


class MultiGPUEngine:
    """
    Multi-GPU inference engine using DataParallel
    """
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Initialize multi-GPU engine
        
        Args:
            device_ids: List of GPU IDs to use (None = all available)
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        self.device_ids = device_ids
        self.primary_device = torch.device(f'cuda:{device_ids[0]}')
        
        logger.info(f"MultiGPUEngine initialized with {len(device_ids)} GPUs: {device_ids}")
    
    def parallelize_model(self, model: nn.Module) -> nn.Module:
        """
        Wrap model with DataParallel for multi-GPU
        
        Args:
            model: PyTorch model
        
        Returns:
            Parallelized model
        """
        model = model.to(self.primary_device)
        model = nn.DataParallel(model, device_ids=self.device_ids)
        
        logger.info(f"Model parallelized across {len(self.device_ids)} GPUs")
        
        return model
    
    def inference(
        self,
        model: nn.Module,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Run inference with automatic data parallelization
        
        Args:
            model: Parallelized model
            inputs: Input tensor
        
        Returns:
            Output tensor
        """
        inputs = inputs.to(self.primary_device)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        return outputs


# Example usage
if __name__ == "__main__":
    # Initialize GPU engine
    engine = GPUInferenceEngine(use_fp16=True, batch_size=64)
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    
    # Optimize model
    model = engine.optimize_model(model)
    
    # Benchmark
    results = engine.benchmark(
        model,
        input_shape=(64, 768),  # Batch size 64, 768 features
        num_iterations=100
    )
    
    print(f"\nBenchmark Results:")
    print(f"Device: {results['device']}")
    print(f"Mean latency: {results['mean_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_samples_per_sec']:.0f} samples/sec")
    
    # Check memory
    memory = engine.get_memory_usage()
    print(f"\nGPU Memory:")
    print(f"Allocated: {memory.get('allocated_gb', 0):.2f} GB")
    print(f"Total: {memory.get('total_gb', 0):.2f} GB")
