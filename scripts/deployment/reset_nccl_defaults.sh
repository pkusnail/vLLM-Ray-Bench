#!/bin/bash
# 清除所有NCCL自定义配置，恢复默认网络设置

echo "=== 恢复NCCL默认配置 ==="

# 清除所有可能的NCCL环境变量
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_DISABLE
unset NCCL_NET_GDR_LEVEL
unset NCCL_P2P_DISABLE
unset NCCL_ALGO
unset NCCL_PROTO
unset NCCL_MIN_NCHANNELS
unset NCCL_MAX_NCHANNELS
unset NCCL_SOCKET_NTHREADS
unset NCCL_NSOCKS_PERTHREAD
unset NCCL_BUFFSIZE
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_TIMEOUT
unset NCCL_COMM_ID
unset NCCL_SOCKET_FAMILY

# 清除vLLM自定义环境变量
unset VLLM_USE_RAY_SPMD_WORKER
unset VLLM_DISABLE_CUSTOM_ALL_REDUCE

echo "已清除所有NCCL和vLLM自定义环境变量"
echo "恢复到默认网络配置"