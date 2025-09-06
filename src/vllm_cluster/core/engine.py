"""Configuration-driven vLLM engine."""

import os
import logging
from typing import Dict, List, Optional, AsyncGenerator
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from ..config import ClusterConfig

logger = logging.getLogger(__name__)


class VLLMEngine:
    """Configuration-driven vLLM engine wrapper."""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.engine: Optional[AsyncLLMEngine] = None
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup environment variables based on configuration."""
        # Setup NCCL environment
        if self.config.use_default_nccl:
            # Clear any existing NCCL variables
            nccl_vars = [key for key in os.environ.keys() if key.startswith('NCCL_')]
            for var in nccl_vars:
                del os.environ[var]
            logger.info("Using default NCCL configuration")
        else:
            # Apply custom NCCL environment variables
            for key, value in self.config.nccl_custom_env.items():
                os.environ[key] = value
            logger.info(f"Applied {len(self.config.nccl_custom_env)} custom NCCL variables")
    
    async def initialize(self) -> None:
        """Initialize the vLLM engine."""
        logger.info("Initializing vLLM engine...")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"PP={self.config.pipeline_parallel_size}, TP={self.config.tensor_parallel_size}")
        
        # Create engine arguments
        engine_args = AsyncEngineArgs(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            swap_space=self.config.swap_space,
            max_num_seqs=self.config.max_num_seqs,
            quantization=self.config.quantization,
            trust_remote_code=self.config.trust_remote_code,
            enforce_eager=True,  # Stable execution
            distributed_executor_backend="ray",
            disable_log_stats=not self.config.enable_metrics,
        )
        
        # Initialize engine
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM engine initialized successfully")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict:
        """Generate text using the vLLM engine."""
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        try:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                stop=stop_sequences,
            )
            
            request_id = f"req_{id(prompt)}"
            
            # Generate response
            final_output = None
            async for request_output in self.engine.generate(
                prompt, sampling_params, request_id
            ):
                final_output = request_output
            
            if final_output is None:
                raise RuntimeError("Generation failed - no output received")
            
            generated_text = final_output.outputs[0].text
            finish_reason = final_output.outputs[0].finish_reason
            
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = len(final_output.outputs[0].token_ids)
            
            return {
                "text": generated_text,
                "prompt": prompt,
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            }
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return {
                "text": f"Generation failed: {str(e)}",
                "prompt": prompt,
                "finish_reason": "error",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    
    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict, None]:
        """Stream text generation using the vLLM engine."""
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        try:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                stop=stop_sequences,
            )
            
            request_id = f"stream_req_{id(prompt)}"
            
            # Stream response
            async for request_output in self.engine.generate(
                prompt, sampling_params, request_id
            ):
                if request_output.outputs:
                    output = request_output.outputs[0]
                    yield {
                        "text": output.text,
                        "finished": request_output.finished,
                        "finish_reason": output.finish_reason if request_output.finished else None,
                    }
        
        except Exception as e:
            logger.error(f"Streaming generation error: {str(e)}")
            yield {
                "text": f"Generation failed: {str(e)}",
                "finished": True,
                "finish_reason": "error",
            }
    
    async def health_check(self) -> Dict:
        """Perform health check on the engine."""
        if not self.engine:
            return {"status": "unhealthy", "reason": "Engine not initialized"}
        
        try:
            # Test with a simple generation
            test_output = await self.generate("Hello", max_tokens=1, temperature=0.0)
            
            if test_output["finish_reason"] == "error":
                return {"status": "unhealthy", "reason": "Generation test failed"}
            
            return {"status": "healthy", "engine_ready": True}
            
        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}
    
    def get_engine_info(self) -> Dict:
        """Get engine configuration information."""
        return {
            "model_name": self.config.model_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "pipeline_parallel_size": self.config.pipeline_parallel_size,
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_num_seqs": self.config.max_num_seqs,
            "initialized": self.engine is not None,
        }
    
    async def shutdown(self):
        """Shutdown the engine gracefully."""
        if self.engine:
            logger.info("Shutting down vLLM engine...")
            # vLLM doesn't provide explicit shutdown, but we can cleanup
            self.engine = None
            logger.info("vLLM engine shutdown complete")