# vLLM-Ray-Bench 评测系统

统一的模型评测框架，支持标准基准测试、集群性能测试和AIOps推理能力测试。

## 📁 目录结构

```
evaluations/
├── README.md                     # 本文档
├── standard_benchmarks/          # 标准基准测试
│   └── standard_benchmark_runner.py
├── cluster_performance/          # 集群性能测试
│   ├── cluster_performance_test.py
│   └── README.md
├── aiops_reasoning/              # AIOps推理能力测试  
│   ├── aiops_reasoning_test.py
│   └── README.md
└── results/                      # 测试结果存储目录
```

## 🚀 快速开始

### 使用统一评测入口

项目根目录下的 `eval` 脚本是所有评测的统一入口：

```bash
# 查看帮助
./eval --help

# 列出所有可用的标准基准测试
./eval standard --list-benchmarks

# 运行MMLU和BBH基准测试
./eval standard --benchmarks mmlu,bbh

# 运行所有标准基准测试(每个测试限制100个样本)
./eval standard --benchmarks all --limit 100

# 运行集群性能测试
./eval cluster-performance

# 运行AIOps推理能力测试
./eval aiops-reasoning

# 运行综合测试套件(包含所有测试类型)
./eval comprehensive
```

## 📊 支持的标准基准测试

| 基准测试 | 描述 | 任务数 |
|----------|------|--------|
| **MMLU** | Massive Multitask Language Understanding - 57个学科的多选题 | 57 |
| **BBH** | Big Bench Hard - 困难推理任务集合 | 27 |  
| **GPQA** | Graduate-level Problems in Question Answering - 研究生级别问题 | 1 |
| **MATH** | Mathematics Aptitude Test of Heuristics - 数学能力测试 | 7 |
| **GSM8K** | Grade School Math 8K - 小学数学问题 | 1 |
| **HellaSwag** | Common Sense Reasoning - 常识推理能力测试 | 1 |
| **WinoGrande** | Pronoun Resolution - 代词消解推理任务 | 1 |
| **TruthfulQA** | Truth and Falsehood Detection - 真实性判断能力 | 1 |

## ⚡ 集群性能测试

专注评估vLLM集群的系统性能指标：

- **延迟指标**: 平均响应时间、P95/P99延迟
- **吞吐量指标**: 请求吞吐量、Token吞吐量  
- **可靠性指标**: 成功率、稳定性
- **资源效率**: 集群效率、GPU利用率

## 🧠 AIOps推理能力测试

专注评估模型在运维场景下的推理质量：

- **故障诊断推理**: 基于监控数据进行根因分析
- **容量规划推理**: 基于历史数据预测未来需求  
- **安全事件推理**: 分析安全日志识别威胁
- **架构优化推理**: 分析系统瓶颈提出优化方案
- **监控告警推理**: 处理复杂告警事件

## 🔧 配置说明

### 环境要求

- Python 3.8+
- vLLM 集群运行在 http://localhost:8000 (可配置)
- lm-eval-harness 库 (用于标准基准测试)

### 模型配置

默认配置：
- **模型地址**: http://localhost:8000
- **模型名称**: Qwen/Qwen3-32B

可通过命令行参数修改：
```bash
./eval standard --model-url http://your-server:8000 --model-name your-model-name --benchmarks mmlu
```

## 📈 结果解读

### 标准基准测试结果

- 结果保存在 `evaluations/results/` 目录
- 包含详细的任务级别得分和总体统计
- JSON格式便于后续分析和可视化

### 性能测试结果

- **🌟 卓越** (>500 tok/s): 高性能集群
- **✅ 优秀** (300-500 tok/s): 生产级性能  
- **⚠️ 良好** (150-300 tok/s): 基本满足需求
- **❌ 待优化** (<150 tok/s): 性能不足

### AIOps推理测试结果

- **🌟 优秀** (3.5-4.0): 具备高级AIOps推理能力
- **✅ 良好** (2.5-3.5): 具备基本AIOps推理能力
- **⚠️ 一般** (1.5-2.5): AIOps推理能力待提升  
- **❌ 较差** (0-1.5): AIOps推理能力不足

## 🛠️ 高级用法

### 批量测试脚本

```bash
#!/bin/bash
# 批量运行不同配置的测试

# 快速验证测试 (限制样本数)
./eval standard --benchmarks mmlu,bbh,gsm8k --limit 50

# 完整基准测试
./eval standard --benchmarks all

# 性能基线测试
./eval cluster-performance

# 综合评测  
./eval comprehensive
```

### 自定义测试配置

可以修改 `evaluations/standard_benchmarks/standard_benchmark_runner.py` 中的配置来：
- 添加新的基准测试
- 调整测试参数 (few-shot数量、batch size等)
- 定制结果分析逻辑

## 🔍 故障排除

### 常见问题

1. **lm_eval 命令不存在**
   ```bash
   source vllm_ray_env/bin/activate
   pip install lm-eval[openai]
   ```

2. **vLLM服务连接失败**  
   - 检查vLLM集群是否正在运行
   - 确认服务地址和端口正确
   - 测试API连通性: `curl http://localhost:8000/v1/models`

3. **测试超时或内存不足**
   - 使用 `--limit` 参数限制样本数
   - 调整batch_size参数
   - 确保集群资源充足

### 日志和调试

- 标准基准测试日志保存在lm_eval输出中
- 集群性能测试有详细的进度显示
- AIOps推理测试包含推理过程输出

## 🤝 贡献指南

欢迎贡献新的评测任务和改进！

1. 在相应目录下添加新的测试脚本
2. 更新 `eval` 主程序以支持新的测试类型  
3. 添加相应的文档和使用示例
4. 确保测试脚本的错误处理和日志输出

## 📄 许可证

本项目遵循 MIT 许可证。