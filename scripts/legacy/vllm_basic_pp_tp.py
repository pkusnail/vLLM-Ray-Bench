#!/usr/bin/env python3
"""
vLLM基础PP+TP配置 - 使用默认网络设置
跨机器Pipeline Parallel + 单机内Tensor Parallel
无NCCL自定义优化，让vLLM自动处理网络通信
"""

import ray
from ray import serve
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import asyncio
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging
import os

# 不设置任何NCCL环境变量，使用vLLM和NCCL的默认配置

# =============================================================================
# 基础PP+TP分布式配置
# =============================================================================
MODEL_NAME = "Qwen/Qwen3-32B"
MODEL_MAX_LEN = 32768

# PP+TP架构配置
TENSOR_PARALLEL_SIZE = 8        # 单机内TP：每个节点8个GPU
PIPELINE_PARALLEL_SIZE = 2      # 跨机PP：2个节点

# 基础性能配置
GPU_MEMORY_UTILIZATION = 0.80   
MAX_NUM_SEQS = 16               
SWAP_SPACE = 8                  

API_HOST = "127.0.0.1"
API_PORT = 8000

# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_sequences: Optional[List[str]] = None

class GenerationResponse(BaseModel):
    text: str
    prompt: str
    finish_reason: str
    usage: Dict

@serve.deployment(
    name="qwen-basic-pp-tp-deployment",
    ray_actor_options={
        "num_gpus": 0,  # 让vLLM自己管理GPU分配
        "num_cpus": 2,
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,
        "target_num_ongoing_requests_per_replica": 8,
    }
)
class VLLMBasicPPTPDeployment:
    def __init__(self):
        logger.info("初始化vLLM基础PP+TP引擎 - 默认网络配置")
        logger.info(f"配置: PP={PIPELINE_PARALLEL_SIZE}, TP={TENSOR_PARALLEL_SIZE}")
        
        # 使用最基础的引擎配置
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
            max_model_len=MODEL_MAX_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            swap_space=SWAP_SPACE,
            max_num_seqs=MAX_NUM_SEQS,
            quantization=None,
            trust_remote_code=True,
            # 基础配置 - 不设置可能导致问题的参数
            enforce_eager=True,                  # 保持稳定
            disable_log_stats=False,
            distributed_executor_backend="ray",  # 使用Ray执行器
            # 让vLLM自动决定是否使用custom allreduce
            # disable_custom_all_reduce=False (默认值)
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM基础PP+TP引擎初始化完成")

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                stop=request.stop_sequences,
            )
            
            request_id = f"req_{id(request)}"
            
            final_output = None
            async for request_output in self.engine.generate(
                request.prompt, 
                sampling_params, 
                request_id
            ):
                final_output = request_output
            
            if final_output is None:
                raise Exception("生成失败")
            
            generated_text = final_output.outputs[0].text
            finish_reason = final_output.outputs[0].finish_reason
            
            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = len(final_output.outputs[0].token_ids)
            
            return GenerationResponse(
                text=generated_text,
                prompt=request.prompt,
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"生成错误: {str(e)}")
            return GenerationResponse(
                text=f"生成失败: {str(e)}",
                prompt=request.prompt,
                finish_reason="error",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

    async def __call__(self, http_request) -> Dict:
        request_data = await http_request.json()
        request = GenerationRequest(**request_data)
        response = await self.generate(request)
        return response.dict()

def deploy_basic_pp_tp_service():
    logger.info("开始部署vLLM基础PP+TP服务")
    logger.info("使用默认网络配置，无NCCL自定义优化")
    
    serve.run(
        VLLMBasicPPTPDeployment.bind(),
        name="qwen3-basic-pp-tp-service",
        route_prefix="/v1/generate"
    )
    
    logger.info("vLLM基础PP+TP服务部署完成!")
    logger.info(f"API端点: http://{API_HOST}:{API_PORT}/v1/generate")

if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(address="auto")
    
    deploy_basic_pp_tp_service()