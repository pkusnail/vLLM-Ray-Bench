#!/bin/bash
# 启动Ray Head节点（在第一台服务器上运行）

set -e

# 激活虚拟环境
source vllm_ray_env/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_DISABLE_IMPORT_WARNING=1

# 设置固定IP
HEAD_IP="192.168.6.70"
echo "Head节点IP: $HEAD_IP"

# 清理之前的Ray进程
ray stop
sleep 2

# 启动Ray Head节点
echo "启动Ray Head节点..."
ray start \
    --head \
    --port=6379 \
    --object-manager-port=8076 \
    --node-manager-port=8077 \
    --ray-client-server-port=10001 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=8 \
    --num-cpus=32 \
    --memory=300000000000 \
    --object-store-memory=50000000000 \
    --verbose

echo "Ray Head节点启动完成！"
echo "Ray Dashboard: http://$HEAD_IP:8265"
echo ""
echo "Worker节点连接命令:"
echo "ray start --address=$HEAD_IP:6379"
echo ""
echo "集群状态检查:"
echo "ray status"