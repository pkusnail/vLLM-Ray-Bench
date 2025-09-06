"""Network utility functions."""

import socket
import subprocess
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


def validate_ip_address(ip: str) -> bool:
    """Validate IP address format."""
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False


def check_connectivity(target_ip: str, port: int = 22, timeout: int = 5) -> bool:
    """Check if a host is reachable on specified port."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((target_ip, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.debug(f"Connectivity check failed for {target_ip}:{port} - {str(e)}")
        return False


def ping_host(host: str, count: int = 3) -> bool:
    """Ping a host to check basic connectivity."""
    try:
        result = subprocess.run(
            ['ping', '-c', str(count), host],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        logger.debug(f"Ping failed for {host} - {str(e)}")
        return False


def get_local_ip() -> str:
    """Get local IP address."""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def check_port_available(port: int, host: str = "localhost") -> bool:
    """Check if a port is available on the host."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result != 0  # Port is available if connection fails
    except Exception:
        return False


def scan_network_range(network: str, port: int = 22) -> List[str]:
    """Scan a network range for active hosts."""
    active_hosts = []
    
    # Simple implementation for class C networks (e.g., 192.168.1.0/24)
    if network.endswith('.0/24'):
        base_ip = network[:-5]  # Remove '/24'
        for i in range(1, 255):
            ip = f"{base_ip}.{i}"
            if check_connectivity(ip, port, timeout=1):
                active_hosts.append(ip)
    
    return active_hosts


def get_network_interfaces() -> Dict[str, str]:
    """Get network interface information."""
    interfaces = {}
    try:
        result = subprocess.run(['ip', 'addr', 'show'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            # Parse ip addr output (simplified)
            current_interface = None
            for line in result.stdout.split('\n'):
                if line.startswith((' ', '\t')):
                    if 'inet ' in line and current_interface:
                        ip = line.strip().split()[1].split('/')[0]
                        interfaces[current_interface] = ip
                else:
                    parts = line.split(':')
                    if len(parts) > 1:
                        current_interface = parts[1].strip()
    except Exception as e:
        logger.debug(f"Failed to get network interfaces: {str(e)}")
    
    return interfaces