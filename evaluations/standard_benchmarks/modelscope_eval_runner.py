#!/usr/bin/env python3
"""
ModelScope EvalScope 集成运行器
使用ModelScope的EvalScope框架进行标准基准测试
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# 动态获取项目根路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
try:
    from evaluations.standard_benchmarks.comprehensive_benchmark_config import (
        COMPREHENSIVE_BENCHMARKS, 
        BENCHMARK_SUITES
    )
except ImportError:
    # 如果导入失败，创建默认配置
    from collections import namedtuple
    BenchmarkConfig = namedtuple('BenchmarkConfig', ['name', 'description', 'priority'])
    COMPREHENSIVE_BENCHMARKS = {
        'mmlu': BenchmarkConfig('MMLU', 'Massive Multitask Language Understanding', 1),
        'bbh': BenchmarkConfig('BBH', 'Big Bench Hard', 1),
        'gsm8k': BenchmarkConfig('GSM8K', '小学数学应用题', 1),
        'math': BenchmarkConfig('MATH', '高难度数学问题求解', 1),
        'hellaswag': BenchmarkConfig('HellaSwag', '常识推理 - 情境续写', 2),
        'winogrande': BenchmarkConfig('WinoGrande', '代词消解推理', 2),
        'cmmlu': BenchmarkConfig('CMMLU', 'Chinese MMLU', 2),
        'ceval': BenchmarkConfig('C-Eval', '中文综合能力评估', 2),
        'arc_challenge': BenchmarkConfig('ARC Challenge', '科学推理挑战题', 2),
        'arc_easy': BenchmarkConfig('ARC Easy', '基础科学推理题', 2)
    }
    BENCHMARK_SUITES = {}

class ModelScopeEvalRunner:
    """ModelScope EvalScope运行器"""
    
    def __init__(self, api_url: str = "http://localhost:8000/v1/chat/completions", api_key: str = "dummy"):
        self.api_url = api_url
        self.api_key = api_key
        # 使用相对于项目根目录的路径
        project_root = Path(__file__).parent.parent.parent
        self.results_dir = project_root / "evaluations" / "results" / "modelscope"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认生成配置
        self.default_generation_config = {
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 0.95
        }
        
        # ModelScope EvalScope支持的基准测试映射
        self.supported_benchmarks = {
            # 数学推理
            "gsm8k": "gsm8k",
            "math": "math", 
            
            # 常识推理
            "hellaswag": "hellaswag",
            "winogrande": "winogrande",
            
            # 多学科知识
            "mmlu": "mmlu",
            "cmmlu": "cmmlu",
            
            # 逻辑推理
            "bbh": "bbh", 
            
            # 阅读理解
            "arc_challenge": "arc_challenge",
            "arc_easy": "arc_easy",
            "arc": "arc_challenge",  # 别名
            
            # 中文评估
            "ceval": "ceval",
            
            # 其他基准测试
            "piqa": "piqa"
        }
    
    def list_supported_benchmarks(self):
        """列出ModelScope EvalScope支持的基准测试"""
        print("📋 ModelScope EvalScope支持的基准测试:")
        print("="*60)
        for key, modelscope_name in self.supported_benchmarks.items():
            if key in COMPREHENSIVE_BENCHMARKS:
                config = COMPREHENSIVE_BENCHMARKS[key]
                priority_emoji = "🌟" if config.priority == 1 else "✅" if config.priority == 2 else "📋"
                print(f"{priority_emoji} {key:<15} - {config.description}")
        print("="*60)
    
    def run_benchmark(self, benchmark_name: str, limit_samples: Optional[int] = None, 
                     generation_config: Optional[Dict] = None, seed: Optional[int] = None,
                     batch_size: int = 4) -> Dict:
        """运行单个基准测试"""
        if benchmark_name not in self.supported_benchmarks:
            print(f"❌ 基准测试 {benchmark_name} 不被ModelScope EvalScope支持")
            return {"status": "unsupported", "benchmark": benchmark_name}
        
        modelscope_name = self.supported_benchmarks[benchmark_name]
        config = COMPREHENSIVE_BENCHMARKS.get(benchmark_name)
        
        print(f"🚀 使用ModelScope EvalScope运行 {config.name if config else benchmark_name} 基准测试...")
        if config:
            print(f"📝 描述: {config.description}")
        
        # 合并生成配置
        final_generation_config = self.default_generation_config.copy()
        if generation_config:
            final_generation_config.update(generation_config)
        
        # 构建命令
        cmd = [
            "evalscope", "eval",
            "--eval-type", "openai_api",
            "--api-url", self.api_url.replace('/v1/chat/completions', '/v1'),  # EvalScope需要base URL
            "--api-key", self.api_key,
            "--model", "Qwen/Qwen3-32B",  # 指定模型名称
            "--datasets", modelscope_name,
            "--eval-batch-size", str(batch_size),
            "--generation-config", json.dumps(final_generation_config),
            "--work-dir", str(self.results_dir / f"{benchmark_name}_{int(time.time())}")
        ]
        
        # 添加seed参数
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        if limit_samples:
            cmd.extend(["--limit", str(limit_samples)])
        
        print(f"💻 执行命令: {' '.join(cmd)}")
        
        try:
            # 设置环境变量和工作目录
            env = os.environ.copy()
            project_root = Path(__file__).parent.parent.parent
            
            # 运行命令
            result = subprocess.run(
                cmd, 
                cwd=str(project_root),
                capture_output=True, 
                text=True, 
                timeout=1800,  # 30分钟超时
                env=env
            )
            
            if result.returncode != 0:
                print(f"❌ 测试失败: {result.stderr}")
                return {
                    "benchmark": benchmark_name,
                    "status": "failed",
                    "error": result.stderr,
                    "stdout": result.stdout
                }
            
            print(f"✅ {benchmark_name} 测试完成")
            
            return {
                "benchmark": benchmark_name,
                "status": "success",
                "config": config.__dict__ if config else {},
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "benchmark": benchmark_name,
                "status": "timeout",
                "error": "测试超时 (30分钟)"
            }
        except Exception as e:
            return {
                "benchmark": benchmark_name,
                "status": "error",
                "error": str(e)
            }
    
    def run_multiple_benchmarks(self, benchmark_names: List[str], limit_samples: Optional[int] = None) -> Dict:
        """运行多个基准测试"""
        print(f"🎯 开始使用ModelScope EvalScope运行 {len(benchmark_names)} 个基准测试...")
        
        all_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_url": self.api_url,
            "limit_samples": limit_samples,
            "framework": "ModelScope EvalScope",
            "benchmarks": {}
        }
        
        successful_tests = 0
        for i, benchmark_name in enumerate(benchmark_names, 1):
            print(f"\n{'='*80}")
            print(f"📊 测试进度 {i}/{len(benchmark_names)}: {benchmark_name.upper()}")
            print("="*80)
            
            result = self.run_benchmark(benchmark_name, limit_samples)
            all_results["benchmarks"][benchmark_name] = result
            
            if result["status"] == "success":
                successful_tests += 1
            
            # 测试间隔
            if i < len(benchmark_names):
                print("⏳ 等待5秒后进行下一个测试...")
                time.sleep(5)
        
        # 保存汇总结果
        summary_file = self.results_dir / f"modelscope_summary_{int(time.time())}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 ModelScope EvalScope评测完成!")
        print(f"📄 汇总结果已保存: {summary_file}")
        print(f"📊 成功测试: {successful_tests}/{len(benchmark_names)}")
        
        return all_results
    
    def run_advanced_batch(self, benchmark_names: List[str], limit_samples: Optional[int] = None,
                          generation_config: Optional[Dict] = None, batch_size: int = 16, 
                          seed: int = 123) -> Dict:
        """运行高级批量测试，支持自定义配置"""
        print(f"🚀 开始高级批量测试模式...")
        print(f"📊 测试列表: {benchmark_names}")
        print(f"🔧 批处理大小: {batch_size}")
        print(f"🎲 随机种子: {seed}")
        
        if generation_config:
            print(f"⚙️  生成配置: {json.dumps(generation_config, indent=2)}")
        
        if limit_samples:
            print(f"📈 样本限制: {limit_samples}")
        
        all_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_url": self.api_url,
            "limit_samples": limit_samples,
            "framework": "ModelScope EvalScope (Advanced)",
            "generation_config": generation_config or self.default_generation_config,
            "batch_size": batch_size,
            "seed": seed,
            "benchmarks": {}
        }
        
        successful_tests = 0
        for i, benchmark_name in enumerate(benchmark_names, 1):
            print(f"\n{'='*80}")
            print(f"📊 高级测试进度 {i}/{len(benchmark_names)}: {benchmark_name.upper()}")
            print("="*80)
            
            result = self.run_benchmark(
                benchmark_name=benchmark_name,
                limit_samples=limit_samples,
                generation_config=generation_config,
                seed=seed,
                batch_size=batch_size
            )
            all_results["benchmarks"][benchmark_name] = result
            
            if result["status"] == "success":
                successful_tests += 1
            
            # 测试间隔
            if i < len(benchmark_names):
                print("⏳ 等待5秒后进行下一个测试...")
                time.sleep(5)
        
        # 保存汇总结果
        summary_file = self.results_dir / f"advanced_batch_{int(time.time())}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 高级批量测试完成!")
        print(f"📄 汇总结果已保存: {summary_file}")
        print(f"📊 成功测试: {successful_tests}/{len(benchmark_names)}")
        
        return all_results

def main():
    """主函数用于测试"""
    runner = ModelScopeEvalRunner()
    
    print("🔍 测试ModelScope EvalScope与vLLM集群的连接...")
    
    # 测试支持的基准测试列表
    runner.list_supported_benchmarks()
    
    # 运行一个简单测试
    print("\n🧪 运行GSM8K标准测试...")
    result = runner.run_benchmark("gsm8k", limit_samples=2)
    print(f"标准测试结果: {result['status']}")
    
    # 运行高级批量测试示例
    print("\n🚀 测试高级批量功能...")
    advanced_config = {
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.95
    }
    
    batch_result = runner.run_advanced_batch(
        benchmark_names=["gsm8k", "hellaswag"],
        limit_samples=2,
        generation_config=advanced_config,
        batch_size=8,
        seed=123
    )
    print(f"高级批量测试结果: 完成 {len([r for r in batch_result['benchmarks'].values() if r['status'] == 'success'])}/{len(batch_result['benchmarks'])} 个测试")

if __name__ == "__main__":
    main()