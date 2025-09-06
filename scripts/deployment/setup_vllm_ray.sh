#!/bin/bash
# vLLM + Ray 环境安装脚本 - 使用虚拟环境隔离

set -e

echo "=== 创建vLLM + Ray虚拟环境 ==="

# 检查Python版本
python3 --version

# 创建专用虚拟环境
VENV_NAME="vllm_ray_env"
echo "创建虚拟环境: $VENV_NAME"
python3 -m venv $VENV_NAME

# 激活虚拟环境
echo "激活虚拟环境..."
source $VENV_NAME/bin/activate

# 确认我们在虚拟环境中
echo "当前Python路径: $(which python)"
echo "当前pip路径: $(which pip)"

# 升级pip
echo "升级pip..."
pip install --upgrade pip setuptools wheel

# 安装PyTorch（CUDA 12.1版本，兼容CUDA 12.4）
echo "安装PyTorch with CUDA支持..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证CUDA
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 安装Ray和Ray Serve
echo "安装Ray框架..."
pip install -U "ray[default,serve]" "ray[air]"

# 安装vLLM
echo "安装vLLM..."
pip install vllm

# 安装其他必要依赖
echo "安装其他依赖..."
pip install \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    fastapi \
    uvicorn \
    pydantic \
    numpy \
    aiofiles \
    psutil

# 创建requirements.txt备份
pip freeze > requirements.txt

echo "=== 安装验证 ==="
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "Ray版本: $(python -c 'import ray; print(ray.__version__)')"
echo "vLLM版本: $(python -c 'import vllm; print(vllm.__version__)')"
echo "CUDA设备数: $(python -c 'import torch; print(torch.cuda.device_count())')"

echo ""
echo "环境安装完成！"
echo "激活命令: source $VENV_NAME/bin/activate"
echo "依赖列表已保存到: requirements.txt"