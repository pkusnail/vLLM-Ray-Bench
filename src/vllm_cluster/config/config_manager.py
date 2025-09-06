"""Configuration manager for vLLM distributed clusters."""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeConfig:
    """Node configuration."""
    ip: str
    gpus: int
    cpus: int
    memory: str
    port: Optional[int] = None


@dataclass
class ClusterConfig:
    """Complete cluster configuration."""
    # Basic cluster info
    cluster_name: str = "vllm-cluster"
    max_concurrent_requests: int = 32
    
    # Model configuration
    model_name: str = "Qwen/Qwen3-32B"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.80
    trust_remote_code: bool = True
    swap_space: int = 8
    quantization: Optional[str] = None
    
    # Distributed configuration
    tensor_parallel_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None
    distributed_strategy: str = "auto"
    
    # Nodes
    head_node: Optional[NodeConfig] = None
    worker_nodes: List[NodeConfig] = field(default_factory=list)
    
    # Service configuration
    service_host: str = "0.0.0.0"
    service_port: int = 8000
    max_num_seqs: int = 16
    
    # Environment
    python_path: str = "./vllm_ray_env/bin/python"
    ray_env_path: str = "./vllm_ray_env"
    working_directory: str = "/home/ubuntu/vllm-ray-bench"
    
    # Network
    use_default_nccl: bool = True
    nccl_custom_env: Dict[str, str] = field(default_factory=dict)
    
    # Security
    ssh_user: str = "ubuntu"
    ssh_key_path: str = "~/.ssh/id_rsa"
    api_key: Optional[str] = None
    
    # Monitoring
    enable_metrics: bool = True
    prometheus_port: int = 9090
    log_level: str = "INFO"


class ConfigManager:
    """Manages cluster configuration with dynamic node support."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config: Optional[ClusterConfig] = None
        
    def load_config(self, config_path: Optional[str] = None) -> ClusterConfig:
        """Load configuration from YAML file."""
        path = config_path or self.config_path
        if not path:
            raise ValueError("No configuration path provided")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self._config = self._parse_config(data)
        
        # Auto-calculate distributed parameters if needed
        if self._config.distributed_strategy == "auto":
            self._auto_configure_distributed()
        
        return self._config
    
    def _parse_config(self, data: Dict[str, Any]) -> ClusterConfig:
        """Parse YAML data into ClusterConfig."""
        config = ClusterConfig()
        
        # Cluster settings
        if 'cluster' in data:
            cluster = data['cluster']
            config.cluster_name = cluster.get('name', config.cluster_name)
            config.max_concurrent_requests = cluster.get('max_concurrent_requests', config.max_concurrent_requests)
        
        # Model settings
        if 'model' in data:
            model = data['model']
            config.model_name = model.get('name', config.model_name)
            config.max_model_len = model.get('max_model_len', config.max_model_len)
            config.gpu_memory_utilization = model.get('gpu_memory_utilization', config.gpu_memory_utilization)
            config.trust_remote_code = model.get('trust_remote_code', config.trust_remote_code)
            config.swap_space = model.get('swap_space', config.swap_space)
            config.quantization = model.get('quantization', config.quantization)
        
        # Distributed settings
        if 'distributed' in data:
            dist = data['distributed']
            config.distributed_strategy = dist.get('strategy', config.distributed_strategy)
            config.tensor_parallel_size = dist.get('tensor_parallel_size', config.tensor_parallel_size)
            config.pipeline_parallel_size = dist.get('pipeline_parallel_size', config.pipeline_parallel_size)
        
        # Nodes
        if 'nodes' in data:
            nodes = data['nodes']
            
            # Head node
            if 'head' in nodes:
                head = nodes['head']
                config.head_node = NodeConfig(
                    ip=head['ip'],
                    gpus=head.get('gpus', 8),
                    cpus=head.get('cpus', 32),
                    memory=head.get('memory', '320GB'),
                    port=head.get('port', 6379)
                )
            
            # Worker nodes
            if 'workers' in nodes:
                config.worker_nodes = []
                for worker in nodes['workers']:
                    config.worker_nodes.append(NodeConfig(
                        ip=worker['ip'],
                        gpus=worker.get('gpus', 8),
                        cpus=worker.get('cpus', 32),
                        memory=worker.get('memory', '320GB')
                    ))
        
        # Service settings
        if 'service' in data:
            service = data['service']
            config.service_host = service.get('host', config.service_host)
            config.service_port = service.get('port', config.service_port)
            config.max_num_seqs = service.get('max_num_seqs', config.max_num_seqs)
        
        # Network settings
        if 'network' in data and 'nccl' in data['network']:
            nccl = data['network']['nccl']
            config.use_default_nccl = nccl.get('use_default', config.use_default_nccl)
            config.nccl_custom_env = nccl.get('custom_env', config.nccl_custom_env)
        
        # Environment settings
        if 'environment' in data:
            env = data['environment']
            config.python_path = env.get('python_path', config.python_path)
            config.ray_env_path = env.get('ray_env_path', config.ray_env_path)
            config.working_directory = env.get('working_directory', config.working_directory)
        
        # Security settings
        if 'security' in data:
            security = data['security']
            config.ssh_user = security.get('ssh_user', config.ssh_user)
            config.ssh_key_path = security.get('ssh_key_path', config.ssh_key_path)
            config.api_key = security.get('api_key', config.api_key)
        
        # Monitoring settings
        if 'monitoring' in data:
            monitoring = data['monitoring']
            config.enable_metrics = monitoring.get('enable_metrics', config.enable_metrics)
            config.prometheus_port = monitoring.get('prometheus_port', config.prometheus_port)
            config.log_level = monitoring.get('log_level', config.log_level)
        
        return config
    
    def _auto_configure_distributed(self):
        """Automatically configure distributed parameters based on available nodes."""
        if not self._config or not self._config.head_node:
            return
        
        total_nodes = 1 + len(self._config.worker_nodes)  # head + workers
        total_gpus = self._config.head_node.gpus + sum(node.gpus for node in self._config.worker_nodes)
        
        logger.info(f"Auto-configuring for {total_nodes} nodes with {total_gpus} total GPUs")
        
        # Calculate optimal PP and TP
        pp_size, tp_size = self._calculate_optimal_parallelism(total_nodes, total_gpus)
        
        if self._config.tensor_parallel_size is None:
            self._config.tensor_parallel_size = tp_size
        
        if self._config.pipeline_parallel_size is None:
            self._config.pipeline_parallel_size = pp_size
        
        logger.info(f"Configured: PP={self._config.pipeline_parallel_size}, TP={self._config.tensor_parallel_size}")
    
    def _calculate_optimal_parallelism(self, total_nodes: int, total_gpus: int) -> Tuple[int, int]:
        """Calculate optimal pipeline and tensor parallelism sizes."""
        # Strategy: Use pipeline parallelism across nodes, tensor parallelism within nodes
        
        # Assume uniform GPU distribution
        gpus_per_node = total_gpus // total_nodes
        
        # PP size equals number of nodes for cross-node communication minimization
        pp_size = total_nodes
        
        # TP size equals GPUs per node for within-node parallelism
        tp_size = gpus_per_node
        
        # Validate that PP * TP equals total GPUs
        if pp_size * tp_size != total_gpus:
            logger.warning(f"Non-uniform GPU distribution detected. PP={pp_size}, TP={tp_size}, Total={total_gpus}")
            # Fallback to a configuration that uses all GPUs
            tp_size = total_gpus // pp_size
        
        return pp_size, tp_size
    
    def save_config(self, config: ClusterConfig, output_path: str):
        """Save configuration to YAML file."""
        data = self._config_to_dict(config)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def _config_to_dict(self, config: ClusterConfig) -> Dict[str, Any]:
        """Convert ClusterConfig to dictionary format."""
        data = {
            'cluster': {
                'name': config.cluster_name,
                'max_concurrent_requests': config.max_concurrent_requests,
            },
            'model': {
                'name': config.model_name,
                'max_model_len': config.max_model_len,
                'trust_remote_code': config.trust_remote_code,
                'gpu_memory_utilization': config.gpu_memory_utilization,
                'swap_space': config.swap_space,
                'quantization': config.quantization,
            },
            'distributed': {
                'strategy': config.distributed_strategy,
                'tensor_parallel_size': config.tensor_parallel_size,
                'pipeline_parallel_size': config.pipeline_parallel_size,
            },
            'service': {
                'host': config.service_host,
                'port': config.service_port,
                'max_num_seqs': config.max_num_seqs,
            },
            'network': {
                'nccl': {
                    'use_default': config.use_default_nccl,
                    'custom_env': config.nccl_custom_env,
                }
            },
            'environment': {
                'python_path': config.python_path,
                'ray_env_path': config.ray_env_path,
                'working_directory': config.working_directory,
            },
            'security': {
                'ssh_user': config.ssh_user,
                'ssh_key_path': config.ssh_key_path,
                'api_key': config.api_key,
            },
            'monitoring': {
                'enable_metrics': config.enable_metrics,
                'prometheus_port': config.prometheus_port,
                'log_level': config.log_level,
            }
        }
        
        # Add nodes
        if config.head_node:
            data['nodes'] = {'head': {
                'ip': config.head_node.ip,
                'port': config.head_node.port,
                'gpus': config.head_node.gpus,
                'cpus': config.head_node.cpus,
                'memory': config.head_node.memory,
            }}
            
            if config.worker_nodes:
                data['nodes']['workers'] = []
                for worker in config.worker_nodes:
                    data['nodes']['workers'].append({
                        'ip': worker.ip,
                        'gpus': worker.gpus,
                        'cpus': worker.cpus,
                        'memory': worker.memory,
                    })
        
        return data
    
    def get_node_list(self) -> List[str]:
        """Get list of all node IPs."""
        if not self._config:
            return []
        
        nodes = []
        if self._config.head_node:
            nodes.append(self._config.head_node.ip)
        nodes.extend([worker.ip for worker in self._config.worker_nodes])
        return nodes
    
    def get_total_gpus(self) -> int:
        """Get total number of GPUs across all nodes."""
        if not self._config:
            return 0
        
        total = 0
        if self._config.head_node:
            total += self._config.head_node.gpus
        total += sum(worker.gpus for worker in self._config.worker_nodes)
        return total
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        if not self._config:
            return ["Configuration not loaded"]
        
        issues = []
        
        # Check basic requirements
        if not self._config.head_node:
            issues.append("Head node configuration missing")
        
        if self._config.tensor_parallel_size and self._config.pipeline_parallel_size:
            expected_gpus = self._config.tensor_parallel_size * self._config.pipeline_parallel_size
            actual_gpus = self.get_total_gpus()
            if expected_gpus != actual_gpus:
                issues.append(f"GPU count mismatch: PP({self._config.pipeline_parallel_size}) * TP({self._config.tensor_parallel_size}) = {expected_gpus}, but found {actual_gpus} GPUs")
        
        # Check model requirements
        if not self._config.model_name:
            issues.append("Model name not specified")
        
        return issues
    
    @property 
    def config(self) -> Optional[ClusterConfig]:
        """Get current configuration."""
        return self._config