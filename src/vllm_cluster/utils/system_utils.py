"""System utility functions."""

import subprocess
import psutil
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def get_gpu_info() -> List[Dict]:
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            logger.warning("nvidia-smi command failed")
            return []
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_total': int(parts[2]),
                        'memory_used': int(parts[3]),
                        'utilization': int(parts[4])
                    })
        
        return gpus
        
    except Exception as e:
        logger.error(f"Error getting GPU info: {str(e)}")
        return []


def get_cpu_info() -> Dict:
    """Get CPU information."""
    try:
        return {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'cpu_usage': psutil.cpu_percent(interval=1),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
    except Exception as e:
        logger.error(f"Error getting CPU info: {str(e)}")
        return {}


def get_memory_info() -> Dict:
    """Get memory information."""
    try:
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percentage': memory.percent
        }
    except Exception as e:
        logger.error(f"Error getting memory info: {str(e)}")
        return {}


def get_disk_info() -> List[Dict]:
    """Get disk information."""
    try:
        disks = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percentage': (usage.used / usage.total) * 100
                })
            except Exception:
                continue
        return disks
    except Exception as e:
        logger.error(f"Error getting disk info: {str(e)}")
        return []


def check_system_requirements(min_memory_gb: int = 32, min_gpus: int = 1) -> Dict:
    """Check if system meets minimum requirements."""
    requirements_met = True
    issues = []
    
    # Check memory
    memory_info = get_memory_info()
    if memory_info:
        total_memory_gb = memory_info['total'] / (1024**3)
        if total_memory_gb < min_memory_gb:
            requirements_met = False
            issues.append(f"Insufficient memory: {total_memory_gb:.1f}GB < {min_memory_gb}GB required")
    
    # Check GPUs
    gpu_info = get_gpu_info()
    if len(gpu_info) < min_gpus:
        requirements_met = False
        issues.append(f"Insufficient GPUs: {len(gpu_info)} < {min_gpus} required")
    
    # Check CUDA availability
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if result.returncode != 0:
            requirements_met = False
            issues.append("CUDA/nvidia-smi not available")
    except Exception:
        requirements_met = False
        issues.append("CUDA/nvidia-smi not available")
    
    return {
        'requirements_met': requirements_met,
        'issues': issues,
        'system_info': {
            'cpu': get_cpu_info(),
            'memory': memory_info,
            'gpus': gpu_info,
            'disks': get_disk_info()
        }
    }


def get_process_info(process_name: str) -> List[Dict]:
    """Get information about processes matching the given name."""
    processes = []
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            if process_name.lower() in proc.info['name'].lower():
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_mb': proc.info['memory_info'].rss / (1024*1024)
                })
    except Exception as e:
        logger.error(f"Error getting process info: {str(e)}")
    
    return processes


def kill_processes_by_name(process_name: str) -> int:
    """Kill all processes matching the given name. Returns count of killed processes."""
    killed_count = 0
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if process_name.lower() in proc.info['name'].lower():
                try:
                    proc.terminate()
                    killed_count += 1
                    logger.info(f"Terminated process {proc.info['pid']} ({proc.info['name']})")
                except Exception as e:
                    logger.warning(f"Failed to terminate process {proc.info['pid']}: {str(e)}")
    except Exception as e:
        logger.error(f"Error killing processes: {str(e)}")
    
    return killed_count


def check_port_usage(port: int) -> Optional[Dict]:
    """Check if a port is in use and by which process."""
    try:
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port:
                try:
                    proc = psutil.Process(conn.pid) if conn.pid else None
                    return {
                        'port': port,
                        'pid': conn.pid,
                        'process_name': proc.name() if proc else 'unknown',
                        'status': conn.status
                    }
                except Exception:
                    return {'port': port, 'pid': conn.pid, 'process_name': 'unknown', 'status': conn.status}
    except Exception as e:
        logger.error(f"Error checking port usage: {str(e)}")
    
    return None