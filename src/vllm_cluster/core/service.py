"""Ray Serve service wrapper for vLLM engine."""

import logging
from typing import Dict, List, Optional
from ray import serve
from pydantic import BaseModel
from .engine import VLLMEngine
from ..config import ClusterConfig

logger = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    text: str
    prompt: str
    finish_reason: str
    usage: Dict


@serve.deployment
class VLLMService:
    """Ray Serve deployment for vLLM engine."""
    
    def __init__(self, config_dict: Dict):
        """Initialize service with configuration dictionary."""
        # Convert dict back to ClusterConfig
        self.config = self._dict_to_config(config_dict)
        self.engine = VLLMEngine(self.config)
        logger.info(f"VLLMService initialized for model: {self.config.model_name}")
    
    def _dict_to_config(self, config_dict: Dict) -> ClusterConfig:
        """Convert dictionary to ClusterConfig object."""
        config = ClusterConfig()
        
        # Basic settings
        config.cluster_name = config_dict.get('cluster_name', config.cluster_name)
        config.model_name = config_dict.get('model_name', config.model_name)
        config.max_model_len = config_dict.get('max_model_len', config.max_model_len)
        config.gpu_memory_utilization = config_dict.get('gpu_memory_utilization', config.gpu_memory_utilization)
        config.trust_remote_code = config_dict.get('trust_remote_code', config.trust_remote_code)
        config.swap_space = config_dict.get('swap_space', config.swap_space)
        config.quantization = config_dict.get('quantization', config.quantization)
        
        # Distributed settings
        config.tensor_parallel_size = config_dict.get('tensor_parallel_size', config.tensor_parallel_size)
        config.pipeline_parallel_size = config_dict.get('pipeline_parallel_size', config.pipeline_parallel_size)
        
        # Service settings
        config.service_host = config_dict.get('service_host', config.service_host)
        config.service_port = config_dict.get('service_port', config.service_port)
        config.max_num_seqs = config_dict.get('max_num_seqs', config.max_num_seqs)
        
        # Network settings
        config.use_default_nccl = config_dict.get('use_default_nccl', config.use_default_nccl)
        config.nccl_custom_env = config_dict.get('nccl_custom_env', config.nccl_custom_env)
        
        # Monitoring
        config.enable_metrics = config_dict.get('enable_metrics', config.enable_metrics)
        config.log_level = config_dict.get('log_level', config.log_level)
        
        return config
    
    async def __call__(self, request):
        """Handle HTTP requests."""
        try:
            if not hasattr(self, '_initialized'):
                await self.engine.initialize()
                self._initialized = True
                logger.info("vLLM engine initialized in service")
            
            # Parse request
            if hasattr(request, 'json'):
                # HTTP request
                request_data = await request.json()
            else:
                # Direct dictionary
                request_data = request
            
            generation_request = GenerationRequest(**request_data)
            
            # Handle streaming vs non-streaming
            if generation_request.stream:
                return await self._handle_streaming_request(generation_request)
            else:
                return await self._handle_standard_request(generation_request)
        
        except Exception as e:
            logger.error(f"Service request error: {str(e)}")
            return {
                "error": str(e),
                "text": "",
                "prompt": request_data.get("prompt", "") if 'request_data' in locals() else "",
                "finish_reason": "error",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    
    async def _handle_standard_request(self, request: GenerationRequest) -> Dict:
        """Handle standard (non-streaming) generation request."""
        response = await self.engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            stop_sequences=request.stop_sequences,
        )
        return response
    
    async def _handle_streaming_request(self, request: GenerationRequest):
        """Handle streaming generation request."""
        # For simplicity, we'll return the final result
        # In a full implementation, you'd want to implement SSE or WebSocket streaming
        final_text = ""
        async for chunk in self.engine.stream_generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            stop_sequences=request.stop_sequences,
        ):
            final_text = chunk["text"]
            if chunk["finished"]:
                return {
                    "text": final_text,
                    "prompt": request.prompt,
                    "finish_reason": chunk["finish_reason"],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # TODO: implement token counting
                }
        
        return {
            "text": final_text,
            "prompt": request.prompt,
            "finish_reason": "length",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
    async def health(self, request=None):
        """Health check endpoint."""
        try:
            if not hasattr(self, '_initialized'):
                return {"status": "initializing"}
            
            health_result = await self.engine.health_check()
            return health_result
        
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def info(self, request=None):
        """Get service information."""
        try:
            engine_info = self.engine.get_engine_info()
            return {
                "service": "vllm-cluster",
                "cluster_name": self.config.cluster_name,
                "engine": engine_info,
            }
        
        except Exception as e:
            logger.error(f"Info request error: {str(e)}")
            return {"error": str(e)}


def create_vllm_service(config: ClusterConfig) -> serve.Deployment:
    """Create a vLLM service deployment with the given configuration."""
    
    # Convert config to dictionary for serialization
    config_dict = {
        'cluster_name': config.cluster_name,
        'model_name': config.model_name,
        'max_model_len': config.max_model_len,
        'gpu_memory_utilization': config.gpu_memory_utilization,
        'trust_remote_code': config.trust_remote_code,
        'swap_space': config.swap_space,
        'quantization': config.quantization,
        'tensor_parallel_size': config.tensor_parallel_size,
        'pipeline_parallel_size': config.pipeline_parallel_size,
        'service_host': config.service_host,
        'service_port': config.service_port,
        'max_num_seqs': config.max_num_seqs,
        'use_default_nccl': config.use_default_nccl,
        'nccl_custom_env': config.nccl_custom_env,
        'enable_metrics': config.enable_metrics,
        'log_level': config.log_level,
    }
    
    # Create deployment with appropriate resource requirements
    deployment = VLLMService.options(
        name=f"{config.cluster_name}-service",
        ray_actor_options={
            "num_gpus": 0,  # Let vLLM manage GPU allocation
            "num_cpus": 2,
        },
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 1,
            "target_num_ongoing_requests_per_replica": config.max_num_seqs // 2,
        }
    ).bind(config_dict)
    
    return deployment