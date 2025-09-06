# Installation Guide

Complete setup guide for vLLM-Ray-Bench distributed inference cluster.

## 🚀 Ridiculously Simple Installation

**Just run what you want to do:**

```bash
# 1. Clone the repository
git clone https://github.com/your-org/vllm-ray-bench.git
cd vllm-ray-bench

# 2. Start your service (everything happens automatically!)
make su           # Single machine mode (or single-up)
# OR
make cup          # Distributed cluster mode (or cluster-up)

# 3. Test the setup
make s            # Check service status (or status)
make eval-quick   # Run quick evaluation
```

**That's it! 🎉** No separate setup commands needed.

### **🧠 What happens automatically:**
- ✅ **Environment setup** - Creates virtual environment and installs dependencies
- ✅ **Smart configuration** - Interactive setup on first run only
- ✅ **IP protection** - Real IPs never committed to Git
- ✅ **Auto-generation** - Creates all necessary config files
- ✅ **Service startup** - Launches your cluster immediately

### **🔄 Smart Behavior:**
- **First time**: Interactive setup asks for your server IPs
- **Every other time**: Uses existing configuration, starts directly
- **Reconfigure**: Run `make reconfig` to change settings

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8 - 3.11
- **CUDA**: 12.1+ (with compatible GPU drivers)
- **Memory**: 16GB+ RAM per GPU node
- **Storage**: 100GB+ free space for models

### Hardware Requirements

**Minimum:**
- 1 node with 8GB+ GPU (single machine setup)
- 32GB+ system RAM
- 50GB+ free disk space

**Recommended:**
- 2+ nodes with 8+ GPUs per node (A100/H100 recommended)
- 128GB+ system RAM per node
- 200GB+ free disk space
- 25Gbps+ network between nodes

### GPU Compatibility
- NVIDIA GPUs with CUDA Compute Capability 7.0+
- Tested on: A100, H100, V100, RTX 4090
- Multi-GPU setups supported via Tensor Parallel

## 🚀 Quick Start

## 🔒 IP Privacy Protection

### Environment Configuration System

This project protects your real IP addresses from being exposed in Git:

```bash
# 1. Set up your environment (one-time setup)
cp .env.example .env
nano .env  # Add your real server IPs

# 2. Generate local configurations
make setup-config

# 3. Check your setup
./scripts/setup_env.sh check
```

### Environment Variables

Edit your `.env` file with real values:
```bash
# Head node IP (your main server)
HEAD_NODE_IP=10.0.1.100

# Worker node IPs
WORKER_NODE_IP_1=10.0.1.101
WORKER_NODE_IP_2=10.0.1.102

# Model settings
MODEL_NAME=Qwen/Qwen3-32B
GPU_MEMORY_UTILIZATION=0.8
```

### Privacy Guarantees
- ✅ `.env` file is gitignored (never committed)
- ✅ `configs/local/` directory is gitignored
- ✅ Only template files with placeholders are in Git
- ✅ Safe to fork and share publicly

## 🔧 Usage

Once installation is complete, you can use these simple commands:

### Starting Services

```bash
# Single machine (simpler, good for development)
make su           # or make single-up

# Distributed cluster (production, multi-node)
make cup          # or make cluster-up

# Stop all services
make cd           # or make cluster-down (or sd)

# Check status
make s            # or make status
```

**Note:** These commands automatically use your real IP configurations from `configs/local/`.

### Monitoring & Testing

```bash
# Check service status
make status

# List available evaluations
make eval-list

# Run quick test
make eval-quick

# Run comprehensive evaluation
make eval-full
```

### Environment Management

```bash
# Check current environment setup
./scripts/setup_env.sh check

# Regenerate configurations
make setup-config

# Clean up local configurations
./scripts/setup_env.sh clean
```

### Development

```bash
# Install with development tools
make dev-install

# Run tests
make test

# Format code
make format
```

## 🛠️ Manual Installation (Advanced Users)

### Option 2: Manual Setup

If the automated setup doesn't work, follow these steps:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools
pip install -r requirements-eval.txt # Evaluation frameworks

# 5. Install package in development mode
pip install -e .

# 6. Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import vllm; print('vLLM installed successfully')"
python -c "import ray; print('Ray installed successfully')"
```

## 🔧 Configuration

### Single Machine Setup

For development or single-server deployment:

```bash
# Copy example configuration
cp configs/examples/single_machine.yaml configs/my_setup.yaml

# Edit configuration as needed
nano configs/my_setup.yaml

# Test single machine setup
./vllm-single start -c configs/my_setup.yaml
```

### Multi-Node Cluster Setup

For production distributed deployment:

```bash
# Initialize cluster configuration
./vllm-cluster init --nodes 2 --head-ip <YOUR_HEAD_IP>

# Edit generated configuration
nano configs/cluster_config.yaml

# Validate configuration
make validate-config

# Deploy cluster (run on head node)
./vllm-cluster deploy -c configs/cluster_config.yaml
```

## 🧪 Verification

### Test Installation

```bash
# 1. Check all components
make verify-env

# 2. Test single machine mode
./vllm-single generate --prompt "Hello, world!" -c configs/examples/single_machine.yaml

# 3. Test evaluation system
make eval-list
```

### Test Cluster (if multi-node)

```bash
# Check cluster status
make status

# Run quick benchmark
make eval-quick

# Test API endpoint
curl http://localhost:8000/v1/models
```

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Not Available**
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**2. Ray Connection Issues**
```bash
# Check Ray status
ray status

# Reset Ray cluster
ray stop
ray start --head
```

**3. Port Already in Use**
```bash
# Find process using port 8000
lsof -i :8000

# Kill process if needed
kill -9 <PID>
```

**4. Memory Issues**
```bash
# Check GPU memory
nvidia-smi

# Adjust GPU memory utilization in config
# Set gpu_memory_utilization: 0.8 or lower
```

### Environment Issues

**Python Version Conflicts:**
```bash
# Use specific Python version
python3.10 -m venv venv
```

**Permission Issues:**
```bash
# Fix permissions
chmod +x vllm-cluster vllm-single eval
```

**Network Configuration:**
- Ensure ports 8000, 6379, 10001 are open
- Check firewall settings for multi-node setups
- Verify SSH access between nodes

## 📁 Directory Structure

After installation, your directory should look like:

```
vllm-ray-bench/
├── venv/                   # Virtual environment
├── configs/                # Configuration files
├── src/                    # Source code
├── evaluations/           # Evaluation scripts
├── scripts/               # Deployment scripts
├── tests/                 # Test suite
├── Makefile              # Build automation
├── requirements*.txt     # Dependencies
└── README.md             # Main documentation
```

## 🔄 Updates

To update an existing installation:

```bash
# Activate environment
source venv/bin/activate

# Pull latest changes
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Reinstall package
pip install -e . --force-reinstall
```

## 🆘 Getting Help

- **Documentation**: Check `docs/` directory
- **Examples**: See `configs/examples/` for sample configurations
- **Issues**: Report problems on GitHub Issues
- **Community**: Join our Discord/Slack channel

## ⚡ Performance Tips

1. **GPU Memory**: Start with `gpu_memory_utilization: 0.8`
2. **Batch Size**: Adjust based on available memory
3. **Model Size**: Ensure sufficient GPU memory for your model
4. **Network**: Use high-speed interconnect for multi-node setups
5. **Storage**: Use SSD for model caching

---

For advanced configuration and deployment options, see the main [README.md](README.md).