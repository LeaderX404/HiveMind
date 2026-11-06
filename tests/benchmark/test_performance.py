"""
Benchmark tests for performance measurement
"""
import pytest
import time
import torch
from unittest.mock import MagicMock, patch
from hivemind.core.model_loader import ModelLoader
from hivemind.core.sharder import ModelSharder
from hivemind.core.inference import LocalInferenceEngine, ShardedInferenceEngine


class TestModelLoadingBenchmark:
    """Benchmark tests for model loading"""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_model_loading_time(self):
        """Benchmark model loading time"""
        # This is a placeholder - in reality, we'd use a real small model
        # For benchmarking, we mock the loading process
        
        with patch('hivemind.core.model_loader.AutoTokenizer') as mock_tokenizer:
            with patch('hivemind.core.model_loader.AutoModelForCausalLM') as mock_model:
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                mock_model.from_pretrained.return_value = MagicMock()
                
                start_time = time.time()
                loader = ModelLoader(model_name="test-model", quantize=False)
                loader.load()
                loading_time = time.time() - start_time
                
                print(f"\nModel loading time: {loading_time:.4f} seconds")
                assert loading_time < 10.0  # Should be fast with mocks


class TestInferenceBenchmark:
    """Benchmark tests for inference speed"""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_local_inference_speed(self):
        """Benchmark local inference speed"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup mocks
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "test response"
        mock_outputs = MagicMock()
        mock_outputs.__getitem__ = MagicMock(return_value=torch.tensor([[1, 2, 3, 4]]))
        mock_model.generate.return_value = mock_outputs
        
        engine = LocalInferenceEngine(mock_model, mock_tokenizer, device="cpu")
        
        # Run multiple inferences
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            with patch('hivemind.core.inference.torch.no_grad'):
                engine.generate("test prompt", max_new_tokens=50)
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        
        print(f"\nLocal inference - Average time per request: {avg_time:.4f} seconds")
        print(f"Requests per second: {num_runs / total_time:.2f}")


class TestShardingBenchmark:
    """Benchmark tests for sharding performance"""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_sharding_time(self):
        """Benchmark model sharding time"""
        mock_model = MagicMock()
        
        # Create mock layers
        mock_layers = []
        for i in range(100):  # Simulate 100 layers
            mock_layer = MagicMock(spec=torch.nn.Linear)
            mock_layers.append((f"layer{i}", mock_layer))
        
        mock_model.named_modules.return_value = mock_layers
        
        sharder = ModelSharder(mock_model, num_shards=4)
        
        start_time = time.time()
        shards = sharder.split()
        sharding_time = time.time() - start_time
        
        print(f"\nSharding time for 100 layers into 4 shards: {sharding_time:.4f} seconds")
        assert len(shards) == 4


class TestMemoryBenchmark:
    """Benchmark tests for memory usage"""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_memory_usage(self):
        """Benchmark memory usage"""
        try:
            import psutil
            import os
            
            # Get current process
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create some tensors
            tensors = []
            for _ in range(10):
                tensors.append(torch.randn(1000, 1000))
            
            after_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = after_memory - initial_memory
            
            print(f"\nMemory usage for 10 tensors (1000x1000): {memory_used:.2f} MB")
            
            # Cleanup
            del tensors
            
        except ImportError:
            pytest.skip("psutil not installed")
