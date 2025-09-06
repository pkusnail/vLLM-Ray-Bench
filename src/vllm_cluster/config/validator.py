"""Configuration validator for vLLM cluster configurations."""

import re
import logging
from typing import List, Set
from .config_manager import ClusterConfig, NodeConfig

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates cluster configurations."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, config: ClusterConfig) -> bool:
        """Validate complete configuration. Returns True if valid."""
        self.errors = []
        self.warnings = []
        
        self._validate_cluster_config(config)
        self._validate_model_config(config)
        self._validate_distributed_config(config)
        self._validate_nodes_config(config)
        self._validate_service_config(config)
        self._validate_network_config(config)
        
        if self.errors:
            logger.error(f"Configuration validation failed with {len(self.errors)} errors")
            for error in self.errors:
                logger.error(f"  ERROR: {error}")
        
        if self.warnings:
            logger.warning(f"Configuration validation found {len(self.warnings)} warnings")
            for warning in self.warnings:
                logger.warning(f"  WARNING: {warning}")
        
        return len(self.errors) == 0
    
    def _validate_cluster_config(self, config: ClusterConfig):
        """Validate cluster-level configuration."""
        if not config.cluster_name:
            self.errors.append("Cluster name cannot be empty")
        elif not re.match(r'^[a-zA-Z0-9\-_]+$', config.cluster_name):
            self.errors.append("Cluster name can only contain alphanumeric characters, hyphens, and underscores")
        
        if config.max_concurrent_requests <= 0:
            self.errors.append("max_concurrent_requests must be positive")
        elif config.max_concurrent_requests > 1000:
            self.warnings.append("max_concurrent_requests > 1000 may impact performance")
    
    def _validate_model_config(self, config: ClusterConfig):
        """Validate model configuration."""
        if not config.model_name:
            self.errors.append("Model name cannot be empty")
        
        if config.max_model_len <= 0:
            self.errors.append("max_model_len must be positive")
        elif config.max_model_len > 100000:
            self.warnings.append("max_model_len > 100k may cause memory issues")
        
        if not (0.1 <= config.gpu_memory_utilization <= 0.95):
            self.errors.append("gpu_memory_utilization must be between 0.1 and 0.95")
        elif config.gpu_memory_utilization > 0.9:
            self.warnings.append("gpu_memory_utilization > 0.9 may cause out-of-memory errors")
        
        if config.swap_space < 0:
            self.errors.append("swap_space cannot be negative")
    
    def _validate_distributed_config(self, config: ClusterConfig):
        """Validate distributed computing configuration."""
        if config.distributed_strategy not in ['auto', 'manual']:
            self.errors.append("distributed_strategy must be 'auto' or 'manual'")
        
        if config.distributed_strategy == 'manual':
            if config.tensor_parallel_size is None or config.pipeline_parallel_size is None:
                self.errors.append("tensor_parallel_size and pipeline_parallel_size required for manual strategy")
        
        if config.tensor_parallel_size is not None:
            if config.tensor_parallel_size <= 0:
                self.errors.append("tensor_parallel_size must be positive")
            elif config.tensor_parallel_size > 8:
                self.warnings.append("tensor_parallel_size > 8 may not improve performance")
        
        if config.pipeline_parallel_size is not None:
            if config.pipeline_parallel_size <= 0:
                self.errors.append("pipeline_parallel_size must be positive")
    
    def _validate_nodes_config(self, config: ClusterConfig):
        """Validate node configurations."""
        if not config.head_node:
            self.errors.append("Head node configuration is required")
            return
        
        # Validate head node
        self._validate_node_config(config.head_node, "head")
        
        # Validate worker nodes
        if not config.worker_nodes:
            self.warnings.append("No worker nodes configured - running single-node cluster")
        else:
            for i, worker in enumerate(config.worker_nodes):
                self._validate_node_config(worker, f"worker-{i}")
        
        # Check for IP conflicts
        all_ips = [config.head_node.ip]
        all_ips.extend([worker.ip for worker in config.worker_nodes])
        
        if len(all_ips) != len(set(all_ips)):
            self.errors.append("Duplicate IP addresses found in node configurations")
        
        # Validate total GPU count
        total_gpus = config.head_node.gpus + sum(worker.gpus for worker in config.worker_nodes)
        if config.tensor_parallel_size and config.pipeline_parallel_size:
            expected_gpus = config.tensor_parallel_size * config.pipeline_parallel_size
            if total_gpus != expected_gpus:
                self.errors.append(f"Total GPUs ({total_gpus}) doesn't match PP({config.pipeline_parallel_size}) * TP({config.tensor_parallel_size}) = {expected_gpus}")
    
    def _validate_node_config(self, node: NodeConfig, node_name: str):
        """Validate individual node configuration."""
        # Validate IP address
        if not self._is_valid_ip(node.ip):
            self.errors.append(f"{node_name}: Invalid IP address '{node.ip}'")
        
        # Validate GPU count
        if node.gpus <= 0:
            self.errors.append(f"{node_name}: GPU count must be positive")
        elif node.gpus > 16:
            self.warnings.append(f"{node_name}: {node.gpus} GPUs is unusually high")
        
        # Validate CPU count
        if node.cpus <= 0:
            self.errors.append(f"{node_name}: CPU count must be positive")
        elif node.cpus > 128:
            self.warnings.append(f"{node_name}: {node.cpus} CPUs is unusually high")
        
        # Validate memory format
        if not re.match(r'^\d+GB$', node.memory):
            self.errors.append(f"{node_name}: Memory must be in format 'XXXGB' (e.g., '320GB')")
        else:
            memory_gb = int(node.memory[:-2])
            if memory_gb < 32:
                self.warnings.append(f"{node_name}: {memory_gb}GB memory may be insufficient for large models")
            elif memory_gb > 2048:
                self.warnings.append(f"{node_name}: {memory_gb}GB memory is unusually high")
        
        # Validate port (if specified)
        if node.port is not None:
            if not (1024 <= node.port <= 65535):
                self.errors.append(f"{node_name}: Port must be between 1024 and 65535")
    
    def _validate_service_config(self, config: ClusterConfig):
        """Validate service configuration."""
        if not (1024 <= config.service_port <= 65535):
            self.errors.append("service_port must be between 1024 and 65535")
        
        if config.max_num_seqs <= 0:
            self.errors.append("max_num_seqs must be positive")
        elif config.max_num_seqs > 256:
            self.warnings.append("max_num_seqs > 256 may impact memory usage")
    
    def _validate_network_config(self, config: ClusterConfig):
        """Validate network configuration."""
        if not config.use_default_nccl and not config.nccl_custom_env:
            self.warnings.append("Custom NCCL enabled but no custom environment variables provided")
        
        if config.nccl_custom_env:
            for key, value in config.nccl_custom_env.items():
                if not key.startswith('NCCL_'):
                    self.warnings.append(f"Non-NCCL environment variable '{key}' in nccl_custom_env")
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Check if IP address is valid."""
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        
        try:
            for part in parts:
                num = int(part)
                if not (0 <= num <= 255):
                    return False
            return True
        except ValueError:
            return False
    
    def get_validation_report(self) -> str:
        """Get formatted validation report."""
        report = []
        
        if self.errors:
            report.append("❌ ERRORS:")
            for error in self.errors:
                report.append(f"  • {error}")
        
        if self.warnings:
            report.append("⚠️  WARNINGS:")
            for warning in self.warnings:
                report.append(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            report.append("✅ Configuration validation passed")
        
        return "\n".join(report)