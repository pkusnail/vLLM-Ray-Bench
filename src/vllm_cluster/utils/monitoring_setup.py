#!/usr/bin/env python3
"""
vLLM + Ray 监控系统配置
包括Prometheus指标收集和性能监控
"""

import psutil
import time
import json
import logging
from typing import Dict, List
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import GPUtil
import ray
from ray import serve
import threading
import requests

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLLMMonitor:
    """vLLM性能监控器"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.setup_metrics()
        self.running = False
        
    def setup_metrics(self):
        """设置Prometheus指标"""
        # GPU指标
        self.gpu_memory_used = Gauge('vllm_gpu_memory_used_bytes', 'GPU内存使用量', ['gpu_id'])
        self.gpu_memory_total = Gauge('vllm_gpu_memory_total_bytes', 'GPU总内存', ['gpu_id'])
        self.gpu_utilization = Gauge('vllm_gpu_utilization_percent', 'GPU使用率', ['gpu_id'])
        self.gpu_temperature = Gauge('vllm_gpu_temperature_celsius', 'GPU温度', ['gpu_id'])
        
        # 系统指标
        self.cpu_usage = Gauge('vllm_cpu_usage_percent', 'CPU使用率')
        self.memory_usage = Gauge('vllm_memory_usage_bytes', 'RAM使用量')
        self.memory_total = Gauge('vllm_memory_total_bytes', 'RAM总量')
        
        # Ray集群指标
        self.ray_nodes_active = Gauge('vllm_ray_nodes_active', 'Ray活跃节点数')
        self.ray_cpus_total = Gauge('vllm_ray_cpus_total', 'Ray总CPU数')
        self.ray_cpus_used = Gauge('vllm_ray_cpus_used', 'Ray已用CPU数')
        self.ray_gpus_total = Gauge('vllm_ray_gpus_total', 'Ray总GPU数')
        self.ray_gpus_used = Gauge('vllm_ray_gpus_used', 'Ray已用GPU数')
        
        # 推理指标
        self.request_count = Counter('vllm_requests_total', '总请求数', ['status'])
        self.request_duration = Histogram('vllm_request_duration_seconds', '请求延迟')
        self.tokens_generated = Counter('vllm_tokens_generated_total', '生成的token数')
        self.active_requests = Gauge('vllm_active_requests', '当前活跃请求数')
    
    def collect_gpu_metrics(self):
        """收集GPU指标"""
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_id = str(gpu.id)
                self.gpu_memory_used.labels(gpu_id=gpu_id).set(gpu.memoryUsed * 1024 * 1024)  # MB to bytes
                self.gpu_memory_total.labels(gpu_id=gpu_id).set(gpu.memoryTotal * 1024 * 1024)
                self.gpu_utilization.labels(gpu_id=gpu_id).set(gpu.load * 100)
                self.gpu_temperature.labels(gpu_id=gpu_id).set(gpu.temperature)
        except Exception as e:
            logger.warning(f"GPU指标收集失败: {e}")
    
    def collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            self.memory_total.set(memory.total)
        except Exception as e:
            logger.warning(f"系统指标收集失败: {e}")
    
    def collect_ray_metrics(self):
        """收集Ray集群指标"""
        try:
            if ray.is_initialized():
                cluster_resources = ray.cluster_resources()
                cluster_status = ray.nodes()
                
                # 节点数量
                active_nodes = len([node for node in cluster_status if node['Alive']])
                self.ray_nodes_active.set(active_nodes)
                
                # CPU资源
                total_cpus = cluster_resources.get('CPU', 0)
                used_cpus = ray.available_resources().get('CPU', 0)
                self.ray_cpus_total.set(total_cpus)
                self.ray_cpus_used.set(total_cpus - used_cpus)
                
                # GPU资源
                total_gpus = cluster_resources.get('GPU', 0)
                used_gpus = ray.available_resources().get('GPU', 0)
                self.ray_gpus_total.set(total_gpus)
                self.ray_gpus_used.set(total_gpus - used_gpus)
        except Exception as e:
            logger.warning(f"Ray指标收集失败: {e}")
    
    def start_monitoring(self):
        """启动监控"""
        logger.info(f"启动Prometheus监控服务器，端口: {self.port}")
        start_http_server(self.port)
        self.running = True
        
        # 启动指标收集线程
        def collect_metrics():
            while self.running:
                self.collect_gpu_metrics()
                self.collect_system_metrics()
                self.collect_ray_metrics()
                time.sleep(10)  # 每10秒收集一次
        
        monitor_thread = threading.Thread(target=collect_metrics, daemon=True)
        monitor_thread.start()
        logger.info("监控系统启动完成")
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        logger.info("监控系统已停止")

class PerformanceOptimizer:
    """性能优化器"""
    
    @staticmethod
    def get_optimal_config(total_gpus: int, model_size: str) -> Dict:
        """根据硬件配置推荐最优参数"""
        configs = {
            "qwen-32b": {
                "tensor_parallel_size": min(total_gpus, 16),
                "max_model_len": 8192 if total_gpus >= 16 else 4096,
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 64 if total_gpus >= 16 else 32,
                "swap_space": 16,
            }
        }
        
        base_config = configs.get(model_size, configs["qwen-32b"])
        
        # 根据GPU数量调整
        if total_gpus < 8:
            base_config["gpu_memory_utilization"] = 0.85
            base_config["max_num_seqs"] = 16
        elif total_gpus >= 16:
            base_config["max_num_seqs"] = 128
            
        return base_config
    
    @staticmethod
    def generate_optimized_config(config: Dict) -> str:
        """生成优化的配置代码"""
        return f"""
# 优化的vLLM配置
engine_args = AsyncEngineArgs(
    model="Qwen/Qwen2.5-32B-Instruct",  # 注意：如需Qwen3请修改路径
    tensor_parallel_size={config['tensor_parallel_size']},
    pipeline_parallel_size=1,
    max_model_len={config['max_model_len']},
    gpu_memory_utilization={config['gpu_memory_utilization']},
    swap_space={config['swap_space']},
    max_num_seqs={config['max_num_seqs']},
    quantization=None,
    trust_remote_code=True,
    enforce_eager=False,
    disable_log_stats=False,
)
"""

def setup_monitoring():
    """设置监控系统"""
    monitor = VLLMMonitor(port=9090)
    monitor.start_monitoring()
    
    logger.info("监控系统配置完成")
    logger.info("Prometheus指标端点: http://localhost:9090/metrics")
    
    return monitor

def optimize_for_hardware():
    """基于硬件进行优化"""
    try:
        # 检测GPU数量
        gpus = GPUtil.getGPUs()
        total_gpus = len(gpus)
        
        logger.info(f"检测到 {total_gpus} 个GPU")
        
        # 获取优化配置
        optimizer = PerformanceOptimizer()
        optimal_config = optimizer.get_optimal_config(total_gpus, "qwen-32b")
        
        logger.info("推荐的性能配置:")
        for key, value in optimal_config.items():
            logger.info(f"  {key}: {value}")
        
        # 生成配置代码
        config_code = optimizer.generate_optimized_config(optimal_config)
        
        with open("/tmp/optimized_vllm_config.py", "w") as f:
            f.write(config_code)
        
        logger.info("优化配置已保存到: /tmp/optimized_vllm_config.py")
        
        return optimal_config
    except Exception as e:
        logger.error(f"硬件优化失败: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM监控和优化工具")
    parser.add_argument("--monitor", action="store_true", help="启动监控")
    parser.add_argument("--optimize", action="store_true", help="生成优化配置")
    parser.add_argument("--port", type=int, default=9090, help="监控端口")
    
    args = parser.parse_args()
    
    if args.optimize:
        optimize_for_hardware()
    
    if args.monitor:
        monitor = VLLMMonitor(port=args.port)
        monitor.start_monitoring()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            logger.info("监控系统已退出")