#!/bin/bash
# 完整的vLLM + Ray集群部署脚本
# 使用说明：在两台服务器上分别运行相应的命令

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <mode> [head_ip]"
    echo ""
    echo "模式:"
    echo "  setup     - 安装环境（两台服务器都要运行）"
    echo "  head      - 启动Head节点（第一台服务器）"
    echo "  worker    - 启动Worker节点（第二台服务器）"
    echo "  service   - 启动vLLM服务（在Head节点运行）"
    echo "  status    - 检查集群状态"
    echo "  stop      - 停止服务"
    echo ""
    echo "例子:"
    echo "  $0 setup"
    echo "  $0 head"
    echo "  $0 worker 192.168.1.100"
    echo "  $0 service"
    exit 1
fi

MODE=$1
HEAD_IP=${2:-""}

case $MODE in
    "setup")
        print_status "开始环境设置..."
        chmod +x setup_vllm_ray.sh
        ./setup_vllm_ray.sh
        print_status "环境设置完成！"
        ;;
        
    "head")
        print_status "启动Ray Head节点..."
        chmod +x start_ray_head.sh
        ./start_ray_head.sh
        ;;
        
    "worker")
        if [ -z "$HEAD_IP" ]; then
            print_error "Worker模式需要提供Head节点IP地址"
            exit 1
        fi
        print_status "连接到Head节点: $HEAD_IP"
        chmod +x start_ray_worker.sh
        ./start_ray_worker.sh $HEAD_IP
        ;;
        
    "service")
        print_status "启动vLLM推理服务..."
        
        # 检查Ray集群
        source vllm_ray_env/bin/activate
        if ! ray status > /dev/null 2>&1; then
            print_error "Ray集群未连接！请先启动Ray集群"
            exit 1
        fi
        
        chmod +x start_vllm_service.sh
        ./start_vllm_service.sh
        ;;
        
    "status")
        print_status "检查集群状态..."
        source vllm_ray_env/bin/activate
        
        echo "=== Ray集群状态 ==="
        ray status || print_warning "Ray集群未连接"
        
        echo ""
        echo "=== 系统资源 ==="
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
        
        echo ""
        echo "=== 服务状态 ==="
        curl -s http://localhost:8000/-/healthcheck && echo "vLLM服务正常" || echo "vLLM服务未启动"
        ;;
        
    "stop")
        print_status "停止所有服务..."
        
        # 停止vLLM服务
        pkill -f "vllm_serve_config.py" || true
        
        # 停止Ray
        source vllm_ray_env/bin/activate || true
        ray stop || true
        
        print_status "所有服务已停止"
        ;;
        
    *)
        print_error "未知模式: $MODE"
        exit 1
        ;;
esac