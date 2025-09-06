"""Cluster management for vLLM distributed deployments."""

import ray
from ray import serve
import logging
import time
import subprocess
import asyncio
from typing import Dict, List, Optional, Tuple
from ..config import ClusterConfig
from .service import create_vllm_service

logger = logging.getLogger(__name__)


class ClusterManager:
    """Manages vLLM distributed cluster deployment and lifecycle."""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.service_handle = None
        self._cluster_initialized = False
    
    def initialize_ray_cluster(self) -> bool:
        """Initialize Ray cluster with head and worker nodes."""
        try:
            logger.info("Initializing Ray cluster...")
            
            if not ray.is_initialized():
                # Connect to existing Ray cluster or start new one
                try:
                    ray.init(address="auto")
                    logger.info("Connected to existing Ray cluster")
                except Exception:
                    logger.info("Starting new Ray cluster...")
                    ray.init()
            
            # Wait for all nodes to be ready
            self._wait_for_nodes()
            
            # Verify cluster topology
            cluster_info = ray.cluster_resources()
            logger.info(f"Ray cluster initialized with resources: {cluster_info}")
            
            self._cluster_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray cluster: {str(e)}")
            return False
    
    def _wait_for_nodes(self, timeout: int = 300):
        """Wait for expected number of nodes to join the cluster."""
        expected_nodes = 1 + len(self.config.worker_nodes)  # head + workers
        expected_gpus = self.config.head_node.gpus + sum(w.gpus for w in self.config.worker_nodes)
        
        logger.info(f"Waiting for {expected_nodes} nodes with {expected_gpus} total GPUs...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            cluster_resources = ray.cluster_resources()
            
            current_nodes = len(ray.nodes())
            current_gpus = int(cluster_resources.get('GPU', 0))
            
            logger.info(f"Current: {current_nodes} nodes, {current_gpus} GPUs")
            
            if current_nodes >= expected_nodes and current_gpus >= expected_gpus:
                logger.info("All expected nodes and GPUs are available")
                return
            
            time.sleep(10)
        
        # Check final status
        cluster_resources = ray.cluster_resources()
        current_nodes = len(ray.nodes())
        current_gpus = int(cluster_resources.get('GPU', 0))
        
        if current_nodes < expected_nodes or current_gpus < expected_gpus:
            logger.warning(f"Timeout waiting for nodes. Expected: {expected_nodes} nodes, {expected_gpus} GPUs. "
                         f"Got: {current_nodes} nodes, {current_gpus} GPUs")
        else:
            logger.info("All nodes ready")
    
    def deploy_service(self) -> bool:
        """Deploy vLLM service to the cluster."""
        try:
            if not self._cluster_initialized:
                logger.error("Ray cluster not initialized. Call initialize_ray_cluster() first.")
                return False
            
            logger.info("Deploying vLLM API service...")
            
            # Build vLLM command with distributed parameters
            cmd = [
                self.config.python_path,
                "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.config.model_name,
                "--tensor-parallel-size", str(self.config.tensor_parallel_size),
                "--pipeline-parallel-size", str(self.config.pipeline_parallel_size),
                "--max-model-len", str(self.config.max_model_len),
                "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
                "--host", self.config.service_host,
                "--port", str(self.config.service_port),
            ]
            
            # Add optional parameters
            if self.config.trust_remote_code:
                cmd.extend(["--trust-remote-code"])
            
            if self.config.swap_space:
                cmd.extend(["--swap-space", str(self.config.swap_space)])
            
            if self.config.quantization:
                cmd.extend(["--quantization", self.config.quantization])
            
            logger.info(f"Starting vLLM service with command: {' '.join(cmd)}")
            
            # Start vLLM service as background process
            self.service_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.config.working_directory
            )
            
            # Wait for service to be ready
            self._wait_for_service_ready()
            
            logger.info("vLLM service deployed successfully")
            logger.info(f"Service available at: http://{self.config.service_host}:{self.config.service_port}/v1/")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy service: {str(e)}")
            return False
    
    def _wait_for_service_ready(self, timeout: int = 600):
        """Wait for vLLM service to be ready."""
        import requests
        logger.info("Waiting for vLLM service to initialize...")
        
        service_url = f"http://{self.config.service_host}:{self.config.service_port}"
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if service process is still running
                if hasattr(self, 'service_process') and self.service_process.poll() is not None:
                    logger.error("vLLM service process has terminated")
                    return False
                
                # Try to connect to service health endpoint
                response = requests.get(f"{service_url}/health", timeout=10)
                if response.status_code == 200:
                    logger.info("vLLM service health check passed")
                    return True
                
            except requests.exceptions.ConnectionError:
                # Service not ready yet
                logger.info("Service starting up...")
            except Exception as e:
                logger.debug(f"Service readiness check error: {str(e)}")
            
            time.sleep(30)
        
        logger.warning(f"Service readiness timeout after {timeout} seconds")
        return False
    
    def _health_check(self) -> bool:
        """Perform health check on the deployed service."""
        try:
            if self.service_handle:
                # Use Ray handle for health check
                result = ray.get(self.service_handle.health.remote())
                return result.get("status") == "healthy"
            return False
        except Exception as e:
            logger.debug(f"Health check error: {str(e)}")
            return False
    
    def get_cluster_status(self) -> Dict:
        """Get comprehensive cluster status."""
        try:
            status = {
                "cluster_name": self.config.cluster_name,
                "ray_initialized": ray.is_initialized(),
                "cluster_initialized": self._cluster_initialized,
            }
            
            if ray.is_initialized():
                # Ray cluster info
                cluster_resources = ray.cluster_resources()
                nodes = ray.nodes()
                
                status.update({
                    "ray_cluster": {
                        "total_nodes": len(nodes),
                        "total_cpus": int(cluster_resources.get('CPU', 0)),
                        "total_gpus": int(cluster_resources.get('GPU', 0)),
                        "total_memory": int(cluster_resources.get('memory', 0)),
                        "resources": cluster_resources,
                    }
                })
                
                # Service status
                try:
                    serve_status = serve.status()
                    deployment_name = f"{self.config.cluster_name}-service"
                    
                    if deployment_name in serve_status.deployment_statuses:
                        deployment_status = serve_status.deployment_statuses[deployment_name]
                        status["service"] = {
                            "deployed": True,
                            "status": deployment_status.status,
                            "replicas": len(deployment_status.replica_states),
                        }
                        
                        # Health check
                        status["service"]["healthy"] = self._health_check()
                    else:
                        status["service"] = {"deployed": False}
                        
                except Exception as e:
                    status["service"] = {"error": str(e)}
            
            return status
            
        except Exception as e:
            return {"error": f"Failed to get cluster status: {str(e)}"}
    
    def scale_cluster(self, new_worker_nodes: List[Dict]) -> bool:
        """Scale cluster by adding new worker nodes."""
        try:
            logger.info(f"Scaling cluster to add {len(new_worker_nodes)} worker nodes")
            
            # Add new nodes to configuration
            from ..config import NodeConfig
            for node_data in new_worker_nodes:
                new_node = NodeConfig(
                    ip=node_data["ip"],
                    gpus=node_data["gpus"],
                    cpus=node_data["cpus"],
                    memory=node_data["memory"]
                )
                self.config.worker_nodes.append(new_node)
            
            # Recalculate distributed parameters
            total_nodes = 1 + len(self.config.worker_nodes)
            total_gpus = self.config.head_node.gpus + sum(w.gpus for w in self.config.worker_nodes)
            
            if self.config.distributed_strategy == "auto":
                self.config.pipeline_parallel_size = total_nodes
                self.config.tensor_parallel_size = total_gpus // total_nodes
            
            logger.info(f"Updated cluster configuration: {total_nodes} nodes, {total_gpus} GPUs")
            logger.info(f"New parallelism: PP={self.config.pipeline_parallel_size}, TP={self.config.tensor_parallel_size}")
            
            # Wait for new nodes to join
            self._wait_for_nodes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale cluster: {str(e)}")
            return False
    
    def shutdown(self):
        """Shutdown cluster gracefully."""
        try:
            logger.info("Shutting down cluster...")
            
            # Shutdown Ray Serve
            if self.service_handle:
                serve.shutdown()
                logger.info("Ray Serve shutdown complete")
            
            # Shutdown Ray
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray cluster shutdown complete")
            
            self._cluster_initialized = False
            self.service_handle = None
            
        except Exception as e:
            logger.error(f"Error during cluster shutdown: {str(e)}")
    
    def get_service_metrics(self) -> Dict:
        """Get service performance metrics."""
        try:
            if not self.service_handle:
                return {"error": "Service not deployed"}
            
            # Get basic metrics from Ray
            serve_status = serve.status()
            deployment_name = f"{self.config.cluster_name}-service"
            
            if deployment_name in serve_status.deployment_statuses:
                deployment_status = serve_status.deployment_statuses[deployment_name]
                
                return {
                    "deployment_status": deployment_status.status,
                    "replica_count": len(deployment_status.replica_states),
                    "cluster_resources": ray.cluster_resources(),
                }
            
            return {"error": "Service deployment not found"}
            
        except Exception as e:
            return {"error": f"Failed to get metrics: {str(e)}"}
    
    async def test_generation(self, prompt: str = "Hello, world!") -> Dict:
        """Test text generation to verify service functionality."""
        try:
            if not self.service_handle:
                return {"error": "Service not deployed"}
            
            logger.info(f"Testing generation with prompt: '{prompt}'")
            
            # Make test request
            request = {"prompt": prompt, "max_tokens": 50, "temperature": 0.1}
            result = await self.service_handle.remote(request)
            
            logger.info("Generation test completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Generation test failed: {str(e)}")
            return {"error": f"Generation test failed: {str(e)}"}