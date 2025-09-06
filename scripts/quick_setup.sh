#!/bin/bash
# Quick Setup Script - Simplified user experience
# One command to set up everything with minimal user input

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Function to print colored output
print_title() {
    echo -e "\n${BOLD}${BLUE}🚀 $1${NC}\n"
}

print_question() {
    echo -e "${YELLOW}❓ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Welcome message
clear
echo -e "${BOLD}${GREEN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║        🚀 vLLM-Ray-Bench Quick Setup                    ║
║                                                          ║
║        Get started in 3 simple steps!                   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# Step 1: Detect current setup
print_title "Step 1: Environment Detection"

print_info "Checking your system..."
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    print_success "Detected $GPU_COUNT GPUs"
else
    print_error "NVIDIA GPUs not detected. This may affect performance."
    GPU_COUNT=0
fi

# Get current IP
CURRENT_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "127.0.0.1")
print_info "Current IP: $CURRENT_IP"

# Step 2: Simple questions
print_title "Step 2: Quick Configuration (3 questions)"

# Question 1: Deployment mode
echo ""
print_question "1. How do you want to deploy?"
echo "   [1] 🖥️  Single machine (simple, good for development)"
echo "   [2] 🌐 Multi-machine cluster (production, multiple servers)"
echo ""
while true; do
    read -p "Choose [1/2]: " DEPLOY_MODE
    case $DEPLOY_MODE in
        1)
            MODE="single"
            print_success "Selected: Single machine mode"
            break
            ;;
        2)
            MODE="cluster"
            print_success "Selected: Multi-machine cluster mode"
            break
            ;;
        *)
            echo "Please choose 1 or 2"
            ;;
    esac
done

# Question 2: IP configuration
echo ""
if [ "$MODE" = "single" ]; then
    print_question "2. Use current IP ($CURRENT_IP) as head node?"
    while true; do
        read -p "Use this IP? [Y/n]: " USE_CURRENT_IP
        case $USE_CURRENT_IP in
            ""|y|Y|yes|Yes)
                HEAD_IP="$CURRENT_IP"
                print_success "Head node IP: $HEAD_IP"
                break
                ;;
            n|N|no|No)
                read -p "Enter head node IP: " HEAD_IP
                if [[ $HEAD_IP =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
                    print_success "Head node IP: $HEAD_IP"
                    break
                else
                    print_error "Invalid IP format. Please try again."
                fi
                ;;
            *)
                echo "Please answer Y or n"
                ;;
        esac
    done
else
    print_question "2. Cluster IP configuration"
    # Head node IP
    print_info "Head node (main server):"
    while true; do
        read -p "Head node IP [$CURRENT_IP]: " HEAD_IP
        HEAD_IP=${HEAD_IP:-$CURRENT_IP}
        if [[ $HEAD_IP =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
            print_success "Head node IP: $HEAD_IP"
            break
        else
            print_error "Invalid IP format. Please try again."
        fi
    done
    
    # Worker nodes
    print_info "Worker nodes (additional servers):"
    read -p "How many worker nodes? [1]: " WORKER_COUNT
    WORKER_COUNT=${WORKER_COUNT:-1}
    
    WORKER_IPS=""
    for ((i=1; i<=WORKER_COUNT; i++)); do
        while true; do
            read -p "Worker $i IP: " WORKER_IP
            if [[ $WORKER_IP =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
                if [ -z "$WORKER_IPS" ]; then
                    WORKER_IPS="$WORKER_IP"
                else
                    WORKER_IPS="$WORKER_IPS,$WORKER_IP"
                fi
                print_success "Worker $i IP: $WORKER_IP"
                break
            else
                print_error "Invalid IP format. Please try again."
            fi
        done
    done
fi

# Question 3: Model settings
echo ""
print_question "3. Use default model settings?"
print_info "Default: Qwen/Qwen3-32B with optimized GPU settings"
while true; do
    read -p "Use defaults? [Y/n]: " USE_DEFAULTS
    case $USE_DEFAULTS in
        ""|y|Y|yes|Yes)
            MODEL_NAME="Qwen/Qwen3-32B"
            MAX_MODEL_LEN="32768"
            GPU_MEMORY_UTILIZATION="0.8"
            print_success "Using default model settings"
            break
            ;;
        n|N|no|No)
            read -p "Model name [Qwen/Qwen3-32B]: " MODEL_NAME
            MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-32B"}
            read -p "Max model length [32768]: " MAX_MODEL_LEN
            MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
            read -p "GPU memory utilization [0.8]: " GPU_MEMORY_UTILIZATION
            GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
            print_success "Custom model settings saved"
            break
            ;;
        *)
            echo "Please answer Y or n"
            ;;
    esac
done

# Step 3: Generate configuration
print_title "Step 3: Generating Configuration"

# Create .env file
print_info "Creating environment configuration..."
cat > .env << EOF
# Auto-generated by vLLM-Ray-Bench Quick Setup
# $(date)

# Head node IP
HEAD_NODE_IP=$HEAD_IP

EOF

if [ "$MODE" = "cluster" ]; then
    echo "# Worker node IPs" >> .env
    IFS=',' read -ra WORKER_ARRAY <<< "$WORKER_IPS"
    for i in "${!WORKER_ARRAY[@]}"; do
        echo "WORKER_NODE_IP_$((i+1))=${WORKER_ARRAY[$i]}" >> .env
    done
    echo "WORKER_NODE_IPS=$WORKER_IPS" >> .env
    echo "" >> .env
fi

cat >> .env << EOF
# Model configuration
MODEL_NAME=$MODEL_NAME
MAX_MODEL_LEN=$MAX_MODEL_LEN
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION

# Ray configuration
RAY_DASHBOARD_PORT=8265
VLLM_API_PORT=8000

# Cluster settings
TENSOR_PARALLEL_SIZE=$GPU_COUNT
PIPELINE_PARALLEL_SIZE=$WORKER_COUNT
CLUSTER_NAME=vllm-cluster
EOF

print_success "Environment file created"

# Generate configurations
print_info "Generating configuration files..."
if [ ! -x "scripts/setup_env.sh" ]; then
    chmod +x scripts/setup_env.sh
fi
./scripts/setup_env.sh setup > /dev/null 2>&1

print_success "Configuration files generated"

# Final summary
print_title "🎉 Setup Complete!"

echo -e "${GREEN}Your vLLM cluster is ready to start!${NC}\n"

if [ "$MODE" = "single" ]; then
    echo -e "${BOLD}Next steps:${NC}"
    echo -e "  ${BLUE}make single-up${NC}    # Start single machine mode"
    echo -e "  ${BLUE}make status${NC}       # Check status"
    echo -e "  ${BLUE}make eval-quick${NC}   # Test with evaluation"
    echo -e "  ${BLUE}make cluster-down${NC} # Stop when done"
else
    echo -e "${BOLD}Next steps:${NC}"
    echo -e "  ${BLUE}make cluster-up${NC}   # Start distributed cluster"  
    echo -e "  ${BLUE}make status${NC}       # Check status"
    echo -e "  ${BLUE}make eval-quick${NC}   # Test with evaluation"
    echo -e "  ${BLUE}make cluster-down${NC} # Stop when done"
fi

echo ""
echo -e "${YELLOW}💡 Tip: Your real IPs are safely stored in .env (gitignored)${NC}"
echo -e "${YELLOW}💡 Configuration files are in configs/local/ (also gitignored)${NC}"

echo ""
echo -e "${GREEN}Happy clustering! 🚀${NC}"