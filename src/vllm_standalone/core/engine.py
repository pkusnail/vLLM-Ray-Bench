"""Standalone vLLM engine optimized for single machine inference."""

import logging
import time
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class StandaloneVLLMEngine:
    """Single machine vLLM engine optimized for local inference."""
    
    def __init__(self, config):
        self.config = config
        self.llm: Optional[LLM] = None
        self._validate_single_machine_config()
    
    def _validate_single_machine_config(self):
        """Validate configuration for single machine usage."""
        if len(self.config.worker_nodes) > 0:
            logger.warning("Worker nodes specified but running in single machine mode - ignoring workers")
        
        if self.config.pipeline_parallel_size > 1:
            logger.warning("Pipeline parallelism > 1 not needed for single machine - setting to 1")
            self.config.pipeline_parallel_size = 1
    
    def initialize(self) -> None:
        """Initialize single machine vLLM engine."""
        logger.info("Initializing single machine vLLM engine...")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Tensor Parallel Size: {self.config.tensor_parallel_size}")
        
        # Create vLLM instance for single machine
        self.llm = LLM(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size or 1,
            # No pipeline_parallel_size for single machine
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            swap_space=self.config.swap_space,
            quantization=self.config.quantization,
            trust_remote_code=self.config.trust_remote_code,
            enforce_eager=True,  # More stable for single machine
        )
        
        logger.info("Single machine vLLM engine initialized successfully")
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Generate text for multiple prompts."""
        if not self.llm:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            stop=stop_sequences,
        )
        
        logger.info(f"Generating responses for {len(prompts)} prompts...")
        start_time = time.time()
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Format results
        results = []
        for output in outputs:
            results.append({
                "prompt": output.prompt,
                "text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
                "tokens": len(output.outputs[0].token_ids),
            })
        
        logger.info(f"Generation completed in {generation_time:.2f}s")
        logger.info(f"Average time per request: {generation_time / len(prompts):.2f}s")
        
        return results
    
    def generate_single(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict:
        """Generate text for a single prompt."""
        results = self.generate(
            [prompt], max_tokens, temperature, top_p, top_k, repetition_penalty, stop_sequences
        )
        return results[0]
    
    def benchmark(self, num_requests: int = 10, max_tokens: int = 100) -> Dict:
        """Run performance benchmark."""
        logger.info(f"Running benchmark with {num_requests} requests...")
        
        # Create test prompts
        test_prompts = [
            f"This is test request {i+1}. Please provide a detailed response about artificial intelligence and machine learning."
            for i in range(num_requests)
        ]
        
        start_time = time.time()
        results = self.generate(test_prompts, max_tokens=max_tokens, temperature=0.1)
        end_time = time.time()
        
        total_time = end_time - start_time
        total_tokens = sum(result["tokens"] for result in results)
        
        benchmark_results = {
            "total_requests": num_requests,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "requests_per_second": num_requests / total_time,
            "tokens_per_second": total_tokens / total_time,
            "average_time_per_request": total_time / num_requests,
            "average_tokens_per_request": total_tokens / num_requests,
        }
        
        logger.info(f"Benchmark Results:")
        logger.info(f"  Requests/sec: {benchmark_results['requests_per_second']:.2f}")
        logger.info(f"  Tokens/sec: {benchmark_results['tokens_per_second']:.2f}")
        logger.info(f"  Avg time/request: {benchmark_results['average_time_per_request']:.2f}s")
        
        return benchmark_results
    
    def get_model_info(self) -> Dict:
        """Get model and engine information."""
        return {
            "model_name": self.config.model_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "mode": "single_machine",
            "initialized": self.llm is not None,
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.llm:
            logger.info("Cleaning up single machine vLLM engine...")
            # vLLM doesn't have explicit cleanup, but we can clear references
            self.llm = None
            logger.info("Cleanup complete")