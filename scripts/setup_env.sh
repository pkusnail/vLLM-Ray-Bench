#!/bin/bash
# Environment Configuration Script
# Helps users set up different environments without exposing real IPs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if .env exists
check_env_file() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found!"
        echo
        echo "Please create your .env file first:"
        echo "  cp .env.example .env"
        echo "  nano .env  # Edit with your real IPs"
        echo
        exit 1
    fi
}

# Load environment variables
load_env() {
    if [ -f ".env" ]; then
        print_info "Loading environment from .env file..."
        export $(grep -v '^#' .env | xargs)
        print_success "Environment loaded"
    fi
}

# Generate config from template
generate_config() {
    local template_file="$1"
    local output_file="$2"
    
    print_info "Generating $output_file from template..."
    
    # Use envsubst to replace environment variables
    envsubst < "$template_file" > "$output_file"
    print_success "Generated $output_file"
}

# Main function
main() {
    print_info "🔧 Setting up vLLM-Ray-Bench environment..."
    echo
    
    # Check if we're in the right directory
    if [ ! -f "Makefile" ] || [ ! -d "configs" ]; then
        print_error "Please run this script from the vLLM-Ray-Bench root directory"
        exit 1
    fi
    
    # Check for .env file
    check_env_file
    
    # Load environment variables
    load_env
    
    # Create configs directory if it doesn't exist
    mkdir -p configs/local
    
    # Generate configurations
    print_info "Generating configuration files..."
    
    # Generate single machine config
    if [ -f "configs/templates/single_machine.yaml.template" ]; then
        generate_config "configs/templates/single_machine.yaml.template" "configs/local/single_machine.yaml"
    fi
    
    # Generate 2-node cluster config
    if [ -f "configs/templates/2_node_cluster.yaml.template" ]; then
        generate_config "configs/templates/2_node_cluster.yaml.template" "configs/local/2_node_cluster.yaml"
    fi
    
    # Generate 4-node cluster config  
    if [ -f "configs/templates/4_node_cluster.yaml.template" ]; then
        generate_config "configs/templates/4_node_cluster.yaml.template" "configs/local/4_node_cluster.yaml"
    fi
    
    echo
    print_success "🎉 Environment setup complete!"
    echo
    print_info "Your configurations are ready in configs/local/"
    print_info "You can now use:"
    echo "  make single-up     # Uses configs/local/single_machine.yaml"
    echo "  make cluster-up    # Uses configs/local/2_node_cluster.yaml"
    echo
    print_warning "Remember: configs/local/ is gitignored for your privacy"
}

# Show help
show_help() {
    echo "Environment Configuration Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  setup, config    Set up configuration files from templates"
    echo "  check           Check current environment setup"
    echo "  clean           Remove generated configuration files"
    echo "  help            Show this help message"
    echo
    echo "Environment Variables (set in .env file):"
    echo "  HEAD_NODE_IP              Your head node IP address"
    echo "  WORKER_NODE_IPS           Comma-separated worker IPs"
    echo "  MODEL_NAME                Model to load (default: Qwen/Qwen3-32B)"
    echo "  MAX_MODEL_LEN             Maximum model length"
    echo "  GPU_MEMORY_UTILIZATION    GPU memory usage (0.0-1.0)"
}

# Check current setup
check_setup() {
    print_info "🔍 Checking environment setup..."
    echo
    
    # Check .env file
    if [ -f ".env" ]; then
        print_success ".env file exists"
        load_env
        echo "  HEAD_NODE_IP: ${HEAD_NODE_IP:-'not set'}"
        echo "  WORKER_NODE_IPS: ${WORKER_NODE_IPS:-'not set'}"
    else
        print_error ".env file missing"
        echo "  Run: cp .env.example .env && nano .env"
    fi
    
    echo
    
    # Check generated configs
    if [ -d "configs/local" ] && [ "$(ls -A configs/local)" ]; then
        print_success "Local configurations exist:"
        ls -la configs/local/
    else
        print_warning "No local configurations found"
        echo "  Run: ./scripts/setup_env.sh setup"
    fi
}

# Clean generated files
clean_setup() {
    print_info "🧹 Cleaning generated configuration files..."
    
    if [ -d "configs/local" ]; then
        rm -rf configs/local
        print_success "Removed configs/local directory"
    else
        print_info "configs/local directory doesn't exist"
    fi
}

# Parse command line arguments
case "${1:-setup}" in
    setup|config)
        main
        ;;
    check)
        check_setup
        ;;
    clean)
        clean_setup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_help
        exit 1
        ;;
esac