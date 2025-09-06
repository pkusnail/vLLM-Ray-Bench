#!/usr/bin/env python3
"""
vLLM Cluster CLI - Unified cluster management tool
"""

import click
import json
import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vllm_cluster.config import ConfigManager, ConfigValidator
from vllm_cluster.core import ClusterManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', default='configs/cluster_config.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """vLLM Distributed Cluster Management CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.pass_context
def validate(ctx):
    """Validate cluster configuration"""
    config_path = ctx.obj['config_path']
    
    try:
        click.echo(f"🔍 Validating configuration: {config_path}")
        
        # Load and validate configuration
        manager = ConfigManager(config_path)
        config = manager.load_config()
        
        validator = ConfigValidator()
        is_valid = validator.validate(config)
        
        # Show validation report
        report = validator.get_validation_report()
        click.echo(report)
        
        if is_valid:
            click.echo("\n✅ Configuration is valid!")
            
            # Show computed settings
            click.echo(f"\n📊 Computed Configuration:")
            click.echo(f"  Cluster: {config.cluster_name}")
            click.echo(f"  Model: {config.model_name}")
            click.echo(f"  Total Nodes: {1 + len(config.worker_nodes)}")
            click.echo(f"  Total GPUs: {manager.get_total_gpus()}")
            click.echo(f"  Parallelism: PP={config.pipeline_parallel_size}, TP={config.tensor_parallel_size}")
            
            sys.exit(0)
        else:
            click.echo("\n❌ Configuration validation failed!")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error validating configuration: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--nodes', '-n', type=int, help='Number of worker nodes')
@click.option('--gpus-per-node', '-g', type=int, default=8, help='GPUs per node')
@click.option('--head-ip', '-h', help='Head node IP address')
@click.option('--output', '-o', help='Output configuration file path')
@click.pass_context
def init(ctx, nodes, gpus_per_node, head_ip, output):
    """Initialize a new cluster configuration"""
    try:
        click.echo("🚀 Initializing new cluster configuration...")
        
        if not nodes:
            nodes = click.prompt('Number of worker nodes', type=int, default=1)
        
        if not head_ip:
            head_ip = click.prompt('Head node IP address', default='192.168.1.100')
        
        if not output:
            output = click.prompt('Output configuration file', default='configs/my_cluster.yaml')
        
        # Create basic configuration
        config_data = {
            'cluster': {'name': 'my-vllm-cluster'},
            'model': {'name': 'Qwen/Qwen3-32B'},
            'distributed': {'strategy': 'auto'},
            'nodes': {
                'head': {
                    'ip': head_ip,
                    'gpus': gpus_per_node,
                    'cpus': 32,
                    'memory': '320GB'
                },
                'workers': []
            }
        }
        
        # Add worker nodes
        for i in range(nodes):
            worker_ip = click.prompt(f'Worker node {i+1} IP address', default=f'192.168.1.{101+i}')
            config_data['nodes']['workers'].append({
                'ip': worker_ip,
                'gpus': gpus_per_node,
                'cpus': 32,
                'memory': '320GB'
            })
        
        # Save configuration
        os.makedirs(os.path.dirname(output), exist_ok=True)
        import yaml
        with open(output, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        click.echo(f"✅ Configuration saved to: {output}")
        click.echo(f"📊 Created cluster with {nodes + 1} nodes and {(nodes + 1) * gpus_per_node} total GPUs")
        
    except Exception as e:
        click.echo(f"❌ Error initializing configuration: {str(e)}")
        sys.exit(1)


@cli.command()
@click.pass_context
def deploy(ctx):
    """Deploy vLLM cluster"""
    config_path = ctx.obj['config_path']
    
    try:
        click.echo(f"🚀 Deploying vLLM cluster...")
        click.echo(f"📋 Using configuration: {config_path}")
        
        # Load configuration
        manager = ConfigManager(config_path)
        config = manager.load_config()
        
        # Validate configuration
        validator = ConfigValidator()
        if not validator.validate(config):
            click.echo("❌ Configuration validation failed!")
            click.echo(validator.get_validation_report())
            sys.exit(1)
        
        # Initialize cluster
        cluster = ClusterManager(config)
        
        click.echo("🔄 Initializing Ray cluster...")
        if not cluster.initialize_ray_cluster():
            click.echo("❌ Failed to initialize Ray cluster")
            sys.exit(1)
        
        click.echo("🔄 Deploying vLLM service...")
        if not cluster.deploy_service():
            click.echo("❌ Failed to deploy vLLM service")
            sys.exit(1)
        
        click.echo("✅ Cluster deployed successfully!")
        click.echo(f"🌐 Service endpoint: http://{config.service_host}:{config.service_port}/v1/generate")
        
    except Exception as e:
        click.echo(f"❌ Deployment failed: {str(e)}")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show cluster status"""
    config_path = ctx.obj['config_path']
    
    try:
        # Load configuration
        manager = ConfigManager(config_path)
        config = manager.load_config()
        
        cluster = ClusterManager(config)
        status = cluster.get_cluster_status()
        
        click.echo("📊 Cluster Status")
        click.echo("=" * 50)
        
        # Basic info
        click.echo(f"Cluster Name: {status.get('cluster_name', 'N/A')}")
        click.echo(f"Ray Initialized: {status.get('ray_initialized', False)}")
        click.echo(f"Cluster Ready: {status.get('cluster_initialized', False)}")
        
        # Ray cluster info
        if 'ray_cluster' in status:
            ray_info = status['ray_cluster']
            click.echo(f"\n🔧 Ray Cluster:")
            click.echo(f"  Nodes: {ray_info.get('total_nodes', 0)}")
            click.echo(f"  CPUs: {ray_info.get('total_cpus', 0)}")
            click.echo(f"  GPUs: {ray_info.get('total_gpus', 0)}")
            click.echo(f"  Memory: {ray_info.get('total_memory', 0):.1f} GB")
        
        # Service info
        if 'service' in status:
            service_info = status['service']
            click.echo(f"\n🚀 vLLM Service:")
            click.echo(f"  Deployed: {service_info.get('deployed', False)}")
            if service_info.get('deployed'):
                click.echo(f"  Status: {service_info.get('status', 'Unknown')}")
                click.echo(f"  Replicas: {service_info.get('replicas', 0)}")
                click.echo(f"  Healthy: {service_info.get('healthy', False)}")
        
        # Error handling
        if 'error' in status:
            click.echo(f"\n❌ Error: {status['error']}")
        
    except Exception as e:
        click.echo(f"❌ Error getting status: {str(e)}")



@cli.command()
@click.option('--prompt', '-p', default='Hello, how are you?', help='Test prompt')
@click.pass_context
def test(ctx, prompt):
    """Test cluster with a generation request"""
    config_path = ctx.obj['config_path']
    
    async def run_test():
        try:
            # Load configuration
            manager = ConfigManager(config_path)
            config = manager.load_config()
            
            cluster = ClusterManager(config)
            
            click.echo(f"🧪 Testing cluster with prompt: '{prompt}'")
            
            result = await cluster.test_generation(prompt)
            
            if 'error' in result:
                click.echo(f"❌ Test failed: {result['error']}")
                return False
            
            click.echo("✅ Test successful!")
            click.echo(f"📝 Response: {result.get('text', '')}")
            click.echo(f"⏱️  Tokens: {result.get('usage', {}).get('total_tokens', 0)}")
            
            return True
            
        except Exception as e:
            click.echo(f"❌ Test error: {str(e)}")
            return False
    
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--worker-ips', '-w', multiple=True, help='Worker node IP addresses to add')
@click.option('--gpus', '-g', type=int, default=8, help='GPUs per new worker')
@click.pass_context
def scale(ctx, worker_ips, gpus):
    """Scale cluster by adding worker nodes"""
    config_path = ctx.obj['config_path']
    
    try:
        if not worker_ips:
            click.echo("❌ No worker IPs provided. Use --worker-ips option.")
            sys.exit(1)
        
        # Load configuration
        manager = ConfigManager(config_path)
        config = manager.load_config()
        
        cluster = ClusterManager(config)
        
        # Prepare new worker node configs
        new_workers = []
        for ip in worker_ips:
            new_workers.append({
                'ip': ip,
                'gpus': gpus,
                'cpus': 32,
                'memory': '320GB'
            })
        
        click.echo(f"📈 Scaling cluster by adding {len(new_workers)} worker nodes...")
        
        if cluster.scale_cluster(new_workers):
            click.echo("✅ Cluster scaling successful!")
            
            # Save updated configuration
            updated_config_path = config_path.replace('.yaml', '_scaled.yaml')
            manager.save_config(config, updated_config_path)
            click.echo(f"💾 Updated configuration saved to: {updated_config_path}")
        else:
            click.echo("❌ Cluster scaling failed!")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Scaling error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.pass_context
def shutdown(ctx):
    """Shutdown cluster"""
    config_path = ctx.obj['config_path']
    
    try:
        # Load configuration
        manager = ConfigManager(config_path)
        config = manager.load_config()
        
        cluster = ClusterManager(config)
        
        click.echo("🔄 Shutting down cluster...")
        cluster.shutdown()
        click.echo("✅ Cluster shutdown complete!")
        
    except Exception as e:
        click.echo(f"❌ Shutdown error: {str(e)}")


@cli.command()
@click.pass_context
def metrics(ctx):
    """Show cluster performance metrics"""
    config_path = ctx.obj['config_path']
    
    try:
        # Load configuration
        manager = ConfigManager(config_path)
        config = manager.load_config()
        
        cluster = ClusterManager(config)
        metrics = cluster.get_service_metrics()
        
        if 'error' in metrics:
            click.echo(f"❌ Error getting metrics: {metrics['error']}")
            return
        
        click.echo("📈 Service Metrics")
        click.echo("=" * 50)
        click.echo(json.dumps(metrics, indent=2))
        
    except Exception as e:
        click.echo(f"❌ Metrics error: {str(e)}")


def main():
    """Main CLI entry point"""
    cli()


if __name__ == '__main__':
    main()