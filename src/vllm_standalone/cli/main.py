#!/usr/bin/env python3
"""
vLLM Standalone CLI - Single machine high-performance inference
"""

import click
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vllm_cluster.config import ConfigManager, ConfigValidator
from vllm_standalone.core.engine import StandaloneVLLMEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', default='configs/examples/single_machine.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """vLLM Standalone - Single Machine High-Performance Inference"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--prompt', '-p', help='Single prompt to process')
@click.option('--max-tokens', '-t', type=int, default=200, help='Maximum tokens to generate')
@click.option('--temperature', type=float, default=0.7, help='Sampling temperature')
@click.pass_context
def generate(ctx, prompt, max_tokens, temperature):
    """Generate response for a single prompt"""
    config_path = ctx.obj['config_path']
    
    if not prompt:
        prompt = click.prompt("👤 Enter your prompt", type=str)
    
    try:
        engine = _initialize_engine(config_path)
        
        click.echo(f"💭 Processing: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        result = engine.generate_single(
            prompt, 
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        click.echo(f"\n📝 Response:\n{result['text']}")
        click.echo(f"\n🔢 Tokens: {result['tokens']} | Status: {result['finish_reason']}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.pass_context
def interactive(ctx):
    """Run interactive chat mode"""
    config_path = ctx.obj['config_path']
    
    try:
        engine = _initialize_engine(config_path)
        
        click.echo("\n🎯 Interactive Chat Mode")
        click.echo("Commands: 'quit', 'exit', 'q' to stop")
        click.echo("=" * 60)
        
        while True:
            try:
                prompt = click.prompt("\n👤 You", type=str)
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                click.echo("🤖 Generating...")
                result = engine.generate_single(prompt, max_tokens=300)
                
                click.echo(f"🤖 Assistant: {result['text']}")
                click.echo(f"💭 ({result['tokens']} tokens)")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                click.echo(f"❌ Error: {str(e)}")
        
        click.echo("\n👋 Goodbye!")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--num-requests', '-n', type=int, default=10, help='Number of test requests')
@click.option('--max-tokens', '-t', type=int, default=100, help='Max tokens per request')
@click.pass_context
def benchmark(ctx, num_requests, max_tokens):
    """Run performance benchmark"""
    config_path = ctx.obj['config_path']
    
    try:
        engine = _initialize_engine(config_path)
        
        click.echo(f"📊 Running benchmark: {num_requests} requests, {max_tokens} tokens each")
        
        results = engine.benchmark(num_requests=num_requests, max_tokens=max_tokens)
        
        click.echo("\n📈 Benchmark Results:")
        click.echo("=" * 50)
        click.echo(f"🔢 Total requests: {results['total_requests']}")
        click.echo(f"⏱️  Total time: {results['total_time']:.2f} seconds")
        click.echo(f"⚡ Requests/sec: {results['requests_per_second']:.2f}")
        click.echo(f"🚀 Tokens/sec: {results['tokens_per_second']:.2f}")
        click.echo(f"📊 Avg time per request: {results['average_time_per_request']:.3f}s")
        click.echo(f"📝 Avg tokens per request: {results['average_tokens_per_request']:.1f}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx):
    """Show model and system information"""
    config_path = ctx.obj['config_path']
    
    try:
        # Load config without initializing engine
        manager = ConfigManager(config_path)
        config = manager.load_config()
        
        click.echo("ℹ️  Model Information:")
        click.echo("=" * 50)
        click.echo(f"📋 Model: {config.model_name}")
        click.echo(f"🔧 Tensor Parallel: {config.tensor_parallel_size}")
        click.echo(f"💾 Max Model Length: {config.max_model_len}")
        click.echo(f"🎚️  GPU Memory Util: {config.gpu_memory_utilization}")
        click.echo(f"💿 Swap Space: {config.swap_space} GB")
        
        # System info
        import torch
        if torch.cuda.is_available():
            click.echo(f"\n🖥️  System Information:")
            click.echo(f"🎮 Available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                click.echo(f"  GPU {i}: {gpu_name}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.pass_context
def validate(ctx):
    """Validate configuration for single machine use"""
    config_path = ctx.obj['config_path']
    
    try:
        click.echo(f"🔍 Validating standalone configuration: {config_path}")
        
        manager = ConfigManager(config_path)
        config = manager.load_config()
        
        validator = ConfigValidator()
        is_valid = validator.validate(config)
        
        # Show validation report
        report = validator.get_validation_report()
        click.echo(report)
        
        # Additional single machine checks
        click.echo("\n🖥️  Standalone Mode Checks:")
        if len(config.worker_nodes) > 0:
            click.echo("⚠️  Worker nodes configured but will be ignored in standalone mode")
        
        if config.pipeline_parallel_size > 1:
            click.echo("⚠️  Pipeline parallelism > 1 not needed for standalone mode")
        
        if is_valid:
            click.echo("\n✅ Configuration is valid for standalone use!")
            
            # Show effective settings
            effective_tp = config.tensor_parallel_size or 1
            click.echo(f"\n📊 Effective Configuration:")
            click.echo(f"  Model: {config.model_name}")
            click.echo(f"  Tensor Parallel: {effective_tp}")
            click.echo(f"  Pipeline Parallel: 1 (standalone)")
            
            import torch
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                click.echo(f"  Available GPUs: {available_gpus}")
                if effective_tp > available_gpus:
                    click.echo(f"❌ Error: Tensor parallel size ({effective_tp}) > available GPUs ({available_gpus})")
                    sys.exit(1)
        else:
            click.echo("\n❌ Configuration validation failed!")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        sys.exit(1)


def _initialize_engine(config_path: str) -> StandaloneVLLMEngine:
    """Initialize standalone vLLM engine with validation."""
    # Load and validate configuration
    manager = ConfigManager(config_path)
    config = manager.load_config()
    
    validator = ConfigValidator()
    if not validator.validate(config):
        raise RuntimeError(f"Configuration validation failed:\n{validator.get_validation_report()}")
    
    # Initialize engine
    engine = StandaloneVLLMEngine(config)
    
    click.echo(f"🚀 Initializing vLLM engine...")
    click.echo(f"📋 Model: {config.model_name}")
    click.echo(f"🔧 Tensor Parallel: {config.tensor_parallel_size}")
    
    engine.initialize()
    click.echo("✅ Engine ready!")
    
    return engine


def main():
    """Main CLI entry point"""
    cli()


if __name__ == '__main__':
    main()