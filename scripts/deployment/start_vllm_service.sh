#!/bin/bash
# 启动vLLM分布式推理服务

set -e

# 激活虚拟环境
source vllm_ray_env/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_DISABLE_IMPORT_WARNING=1
export VLLM_USE_MODELSCOPE=false
export HF_HUB_ENABLE_HF_TRANSFER=1

# 检查Ray集群状态
echo "检查Ray集群状态..."
ray status

if [ $? -ne 0 ]; then
    echo "错误: Ray集群未连接！"
    echo "请先启动Ray集群"
    exit 1
fi

# 检查模型路径（可选）
MODEL_PATH="/path/to/Qwen2.5-32B-Instruct"
if [ ! -d "$MODEL_PATH" ]; then
    echo "警告: 模型路径不存在，将从HuggingFace下载"
    echo "确保网络连接正常且有足够存储空间"
fi

# 启动vLLM Ray Serve服务
echo "启动vLLM分布式推理服务..."
python vllm_serve_config.py &

# 等待服务启动
sleep 10

# 检查服务状态
echo "检查服务状态..."
curl -s http://localhost:8000/-/healthcheck || echo "服务可能还在启动中..."

echo ""
echo "=== vLLM服务启动完成 ==="
echo "API端点: http://192.168.6.70:8000/v1/generate"
echo "Ray Dashboard: http://192.168.6.70:8265"
echo ""
echo "测试命令:"
echo "curl -X POST http://192.168.6.70:8000/v1/generate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"prompt\": \"你好，请介绍一下自己\", \"max_tokens\": 100}'"