"""
Performance Monitor for Out-of-Core Inference
Tracks memory usage, loading times, and generation speed.
"""
import time
import torch
from typing import Dict, Any
from hivemind.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """Monitor performance metrics for out-of-core inference."""
    
    def __init__(self):
        self.metrics = {
            'start_time': 0.0,
            'total_tokens': 0,
            'total_time': 0.0,
            'peak_ram_mb': 0.0,
            'peak_gpu_mb': 0.0,
            'page_loads': 0,
            'page_load_time': 0.0,
            'tensor_loads': 0,
            'tensor_load_time': 0.0,
        }
    
    def start(self):
        """Start timing"""
        self.metrics['start_time'] = time.time()
        self.metrics['total_tokens'] = 0
        self._check_memory()
    
    def stop(self):
        """Stop timing"""
        self.metrics['total_time'] = time.time() - self.metrics['start_time']
        self._check_memory()
    
    def record_token(self):
        """Record a generated token"""
        self.metrics['total_tokens'] += 1
    
    def record_page_load(self, load_time: float):
        """Record a page load"""
        self.metrics['page_loads'] += 1
        self.metrics['page_load_time'] += load_time
    
    def record_tensor_load(self, load_time: float):
        """Record a tensor load"""
        self.metrics['tensor_loads'] += 1
        self.metrics['tensor_load_time'] += load_time
    
    def _check_memory(self):
        """Check current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            ram_mb = process.memory_info().rss / (1024 * 1024)
            self.metrics['peak_ram_mb'] = max(self.metrics['peak_ram_mb'], ram_mb)
        except ImportError:
            pass
        
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            self.metrics['peak_gpu_mb'] = max(self.metrics['peak_gpu_mb'], gpu_mb)
    
    def get_tokens_per_second(self) -> float:
        """Get tokens per second"""
        if self.metrics['total_time'] == 0:
            return 0.0
        return self.metrics['total_tokens'] / self.metrics['total_time']
    
    def get_avg_page_load_time_ms(self) -> float:
        """Get average page load time in ms"""
        if self.metrics['page_loads'] == 0:
            return 0.0
        return (self.metrics['page_load_time'] / self.metrics['page_loads']) * 1000
    
    def get_avg_tensor_load_time_ms(self) -> float:
        """Get average tensor load time in ms"""
        if self.metrics['tensor_loads'] == 0:
            return 0.0
        return (self.metrics['tensor_load_time'] / self.metrics['tensor_loads']) * 1000
    
    def get_report(self) -> str:
        """Generate performance report"""
        tokens_per_sec = self.get_tokens_per_second()
        avg_page_load_ms = self.get_avg_page_load_time_ms()
        avg_tensor_load_ms = self.get_avg_tensor_load_time_ms()
        
        return f"""=== HiveMind Performance Report ===

Inference Time: {self.metrics['total_time']:.2f} seconds
Total Tokens: {self.metrics['total_tokens']}
Tokens/Second: {tokens_per_sec:.2f}

Memory Usage:
  Peak RAM: {self.metrics['peak_ram_mb']:.1f} MB
  Peak GPU: {self.metrics['peak_gpu_mb']:.1f} MB

Loading Statistics:
  Pages Loaded: {self.metrics['page_loads']}
  Avg Page Load Time: {avg_page_load_ms:.2f} ms
  Tensors Loaded: {self.metrics['tensor_loads']}
  Avg Tensor Load Time: {avg_tensor_load_ms:.2f} ms"""
    
    def reset(self):
        """Reset all metrics"""
        self.__init__()
