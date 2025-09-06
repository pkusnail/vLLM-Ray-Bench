#!/bin/bash
# 启动Ray Worker节点（在第二台服务器上运行）

set -e

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <HEAD_NODE_IP>"
    echo "例如: $0 192.168.1.100"
    exit 1
fi

HEAD_IP=$1

# 激活虚拟环境
source vllm_ray_env/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_DISABLE_IMPORT_WARNING=1

echo "连接到Ray Head节点: $HEAD_IP"

# 清理之前的Ray进程
ray stop
sleep 2

# 启动Ray Worker节点
echo "启动Ray Worker节点..."
ray start \
    --address=$HEAD_IP:6379 \
    --num-gpus=8 \
    --num-cpus=32 \
    --memory=300000000000 \
    --object-store-memory=50000000000 \
    --verbose

echo "Ray Worker节点启动完成！"
echo ""
echo "检查集群状态:"
echo "ray status --address=$HEAD_IP:6379"