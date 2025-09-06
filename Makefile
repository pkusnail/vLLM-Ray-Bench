# vLLM-Ray-Bench Makefile
# Simplified environment setup and common operations

.PHONY: help install-venv activate clean test lint format status

VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
ACTIVATE = source $(VENV_DIR)/bin/activate

help: ## Show this help message
	@echo "vLLM-Ray-Bench - Available Commands:"
	@echo "===================================="
	@echo ""
	@echo "🚀 Just Start What You Want (Auto-setup on first run):"
	@echo "  make su              Start single machine (or single-up)"
	@echo "  make cup             Start cluster (or cluster-up)"
	@echo "  make cd/sd           Stop all services (or cluster-down)"
	@echo "  make s               Check status (or status)"
	@echo ""
	@echo "📋 All Available Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

quick-start: ## 🚀 Complete setup in one command (RECOMMENDED)
	@echo "🌟 Starting vLLM-Ray-Bench Quick Setup..."
	@echo ""
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "📦 Installing environment first..."; \
		$(MAKE) install-venv; \
		echo ""; \
	fi
	@if [ ! -f ".env" ]; then \
		echo "🎯 First-time setup: Launching interactive configuration..."; \
		./scripts/quick_setup.sh; \
	else \
		echo "✅ Configuration exists, using existing .env file"; \
		$(MAKE) setup-config; \
	fi
	@echo ""
	@echo "🎉 Setup complete! You can now run:"
	@echo "   make single-up   # Single machine mode"
	@echo "   make cluster-up  # Distributed cluster mode"
	@echo "   make status      # Check cluster status"

install-venv: ## Create virtual environment and install dependencies
	@echo "🚀 Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -r requirements-eval.txt
	$(PIP) install -e .
	@echo "✅ Setup complete!"
	@echo "To activate: source $(VENV_DIR)/bin/activate"

activate: ## Show activation command
	@echo "Run: source $(VENV_DIR)/bin/activate"

verify-env: ## Verify installation
	@echo "🔍 Verifying installation..."
	$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	$(PYTHON) -c "import ray; print(f'Ray: {ray.__version__}')"
	$(PYTHON) -c "import vllm; print(f'vLLM: {vllm.__version__}')"
	$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	$(PYTHON) -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
	@echo "✅ Verification complete!"

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

lint: ## Run code linting
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100
	$(PYTHON) -m mypy src/

format: ## Format code
	$(PYTHON) -m black src/ tests/ --line-length=100
	$(PYTHON) -m isort src/ tests/

# Configuration setup
setup-config: ## Generate local configurations from .env file
	@echo "🔧 Setting up local configuration..."
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found! Run 'make quick-start' for first-time setup."; \
		exit 1; \
	fi
	./scripts/setup_env.sh setup
	@echo "✅ Local configuration ready!"

reconfig: ## Reconfigure settings (interactive)
	@echo "🔄 Reconfiguring vLLM-Ray-Bench..."
	./scripts/quick_setup.sh

# Smart setup helper - automatically handles first-time setup
check-and-setup:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "📦 First time setup: Installing environment..."; \
		$(MAKE) install-venv; \
		echo ""; \
	fi
	@if [ ! -f ".env" ]; then \
		echo "🎯 First time setup: Interactive configuration needed..."; \
		echo "This will only happen once!"; \
		echo ""; \
		./scripts/quick_setup.sh; \
	else \
		if [ ! -f "configs/local/single_machine.yaml" ] || [ ! -f "configs/local/2_node_cluster.yaml" ]; then \
			echo "🔧 Updating configurations..."; \
			$(MAKE) setup-config; \
		fi; \
	fi

# Cluster operations  
single-up: check-and-setup ## Start single machine vLLM (no Ray)
	@echo "🖥️  Starting single machine vLLM..."
	@bash -c "source $(VENV_DIR)/bin/activate && ./vllm-single -c configs/local/single_machine.yaml interactive"

cluster-up: check-and-setup ## Start distributed vLLM cluster
	@echo "🌐 Starting distributed vLLM cluster..."
	@bash -c "source $(VENV_DIR)/bin/activate && ./vllm-cluster deploy -c configs/local/2_node_cluster.yaml"

cluster-down: ## Stop all services
	@echo "🛑 Stopping all services..."
	@echo "Stopping vLLM processes..."
	@pkill -f "vllm" || true
	@echo "Stopping Ray cluster..."
	@if [ -d "$(VENV_DIR)" ]; then \
		bash -c "source $(VENV_DIR)/bin/activate && ray stop" || true; \
	else \
		ray stop 2>/dev/null || true; \
	fi
	@echo "Cleaning up any remaining processes..."
	@pkill -f "ray" || true
	@pkill -f "python.*serve" || true
	@echo "✅ All services stopped"

# Short aliases for convenience
su: single-up ## Start single machine (alias for single-up)  
cup: cluster-up ## Start cluster (alias for cluster-up)
cd: cluster-down ## Stop all services (alias for cluster-down)
sd: cluster-down ## Stop all services (alias for cluster-down, same as cd)
s: status ## Check status (alias for status)

init-config: ## Initialize cluster configuration
	$(PYTHON) -m vllm_cluster.cli.main init

validate-config: ## Validate cluster configuration
	@if [ -f configs/local/2_node_cluster.yaml ]; then \
		$(PYTHON) -m vllm_cluster.cli.main validate -c configs/local/2_node_cluster.yaml; \
	else \
		$(PYTHON) -m vllm_cluster.cli.main validate -c configs/examples/2_node_cluster.yaml; \
	fi

status: ## Check cluster status
	@echo "🔍 Checking service status..."
	@echo "Ray cluster:"
	@bash -c "source $(VENV_DIR)/bin/activate && ray status" || echo "❌ Ray cluster not running"
	@echo "\nvLLM API:"
	@curl -s http://localhost:8000/v1/models | head -3 || echo "❌ vLLM service not responding"
	@echo "\nGPU usage:"
	@nvidia-smi | grep -A 10 "Processes:" || echo "❌ No GPU processes"

# Evaluation
eval-list: ## List available benchmarks
	./eval standard --list-benchmarks

eval-quick: ## Run quick evaluation
	./eval standard --benchmarks mmlu --limit 5

eval-full: ## Run comprehensive evaluation
	./eval comprehensive

clean: ## Clean up environment
	rm -rf $(VENV_DIR)
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-logs: ## Clean up log files
	rm -rf /tmp/ray/
	rm -rf evaluations/results/
	rm -rf outputs/

# Development helpers
dev-install: install-venv verify-env ## Complete development setup

freeze: ## Generate requirements.txt from current environment
	$(PIP) freeze > requirements-frozen.txt
	@echo "📄 Frozen requirements saved to requirements-frozen.txt"

# Default target
all: install-venv verify-env ## Complete setup and verification