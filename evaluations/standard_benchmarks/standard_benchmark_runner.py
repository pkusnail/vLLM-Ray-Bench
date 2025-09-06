#!/usr/bin/env python3
"""
标准基准测试运行器
支持全面的LLM基准测试集，包括 MMLU, BBH, GPQA, MATH 等30+主流测试集
"""

import subprocess
import json
import os
import time
from typing import Dict, List, Optional
from pathlib import Path

# 导入comprehensive benchmark configurations
from evaluations.standard_benchmarks.comprehensive_benchmark_config import (
    COMPREHENSIVE_BENCHMARKS, 
    BENCHMARK_SUITES,
    get_benchmark_by_category,
    get_benchmark_by_priority, 
    estimate_total_time,
    list_all_categories,
    get_suite_info
)

class StandardBenchmarkRunner:
    """标准基准测试运行器 - 支持30+主流基准测试"""
    
    def __init__(self, model_url: str = "http://localhost:8000", model_name: str = "Qwen/Qwen3-32B"):
        self.model_url = model_url
        self.model_name = model_name
        self.results_dir = Path("evaluations/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用comprehensive benchmark configurations
        self.benchmarks = COMPREHENSIVE_BENCHMARKS
        self.benchmark_suites = BENCHMARK_SUITES
        
        print(f"🎯 已加载 {len(self.benchmarks)} 个基准测试配置")
        print(f"📋 支持 {len(self.benchmark_suites)} 个测试套件: {list(self.benchmark_suites.keys())}")
    
    def get_available_categories(self) -> List[str]:
        """获取所有可用测试类别"""
        return list_all_categories()
    
    def get_benchmarks_by_category(self, category: str) -> List[str]:
        """根据类别获取基准测试列表"""
        return get_benchmark_by_category(category)
    
    def get_benchmarks_by_priority(self, priority: int) -> List[str]:
        """根据优先级获取基准测试列表"""
        return get_benchmark_by_priority(priority)
    
    def get_suite_benchmarks(self, suite_name: str) -> List[str]:
        """获取测试套件中的基准测试列表"""
        return self.benchmark_suites.get(suite_name, [])
    
    def run_benchmark(self, benchmark_name: str, limit_samples: Optional[int] = None) -> Dict:
        """运行单个基准测试"""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"未知的基准测试: {benchmark_name}. 可用的测试: {list(self.benchmarks.keys())}")
        
        config = self.benchmarks[benchmark_name]
        print(f"🚀 运行 {config.name} 基准测试...")
        print(f"📝 描述: {config.description}")
        print(f"🔢 任务数量: {len(config.tasks)}")
        
        # 构建lm_eval命令
        tasks_str = ",".join(config.tasks)
        model_args = f"base_url={self.model_url}/v1,model={self.model_name}"
        
        output_path = self.results_dir / f"{benchmark_name}_{int(time.time())}.json"
        
        # 使用local-chat-completions后端，这是为本地OpenAI API兼容服务器设计的
        cmd = [
            "lm_eval", 
            "--model", "local-chat-completions",
            "--model_args", model_args,
            "--tasks", tasks_str,
            "--num_fewshot", str(config.num_fewshot),
            "--batch_size", str(config.batch_size),
            "--output_path", str(output_path),
            "--write_out",
            "--apply_chat_template"
        ]
        
        if limit_samples:
            cmd.extend(["--limit", str(limit_samples)])
        
        print(f"💻 执行命令: {' '.join(cmd)}")
        
        try:
            # 设置环境变量，为local-chat-completions提供必要的配置
            env = os.environ.copy()
            env['OPENAI_API_KEY'] = 'dummy-key'  # local-chat-completions需要这个环境变量，但值可以是任意的
            
            # 运行lm_eval
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
            
            if result.returncode != 0:
                print(f"❌ 测试失败: {result.stderr}")
                return {
                    "benchmark": benchmark_name,
                    "status": "failed", 
                    "error": result.stderr,
                    "stdout": result.stdout
                }
            
            # 读取结果文件
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                print(f"✅ {config.name} 测试完成")
                self._print_results_summary(results, config.name)
                
                return {
                    "benchmark": benchmark_name,
                    "status": "success",
                    "config": config.__dict__,
                    "results": results,
                    "output_file": str(output_path)
                }
            else:
                return {
                    "benchmark": benchmark_name,
                    "status": "failed",
                    "error": "结果文件未生成",
                    "stdout": result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                "benchmark": benchmark_name,
                "status": "timeout",
                "error": "测试超时 (1小时)"
            }
        except Exception as e:
            return {
                "benchmark": benchmark_name,
                "status": "error",
                "error": str(e)
            }
    
    def run_multiple_benchmarks(self, benchmark_names: List[str], limit_samples: Optional[int] = None) -> Dict:
        """运行多个基准测试"""
        print(f"🎯 开始运行 {len(benchmark_names)} 个基准测试...")
        all_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_url": self.model_url,
            "model_name": self.model_name,
            "limit_samples": limit_samples,
            "benchmarks": {}
        }
        
        for i, benchmark_name in enumerate(benchmark_names, 1):
            print(f"\n{'='*80}")
            print(f"📊 测试进度 {i}/{len(benchmark_names)}: {benchmark_name.upper()}")
            print("="*80)
            
            result = self.run_benchmark(benchmark_name, limit_samples)
            all_results["benchmarks"][benchmark_name] = result
            
            # 测试间隔
            if i < len(benchmark_names):
                print("⏳ 等待10秒后进行下一个测试...")
                time.sleep(10)
        
        # 保存汇总结果
        summary_file = self.results_dir / f"benchmark_summary_{int(time.time())}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 所有基准测试完成!")
        print(f"📄 汇总结果已保存: {summary_file}")
        
        # 打印总结
        self._print_overall_summary(all_results)
        
        return all_results
    
    def _print_results_summary(self, results: Dict, benchmark_name: str):
        """打印单个测试结果摘要"""
        print(f"\n📊 {benchmark_name} 结果摘要:")
        print("-" * 50)
        
        if "results" in results:
            for task_name, task_result in results["results"].items():
                if isinstance(task_result, dict) and any(k.endswith("acc") or k.endswith("exact_match") for k in task_result.keys()):
                    # 找到准确率指标
                    acc_keys = [k for k in task_result.keys() if k.endswith("acc") or k.endswith("exact_match") or k == "acc_norm"]
                    for acc_key in acc_keys:
                        score = task_result[acc_key]
                        if isinstance(score, (int, float)):
                            print(f"   • {task_name} ({acc_key}): {score:.1%}")
                        break
    
    def _print_overall_summary(self, all_results: Dict):
        """打印总体测试摘要"""
        print(f"\n{'='*80}")
        print("🏆 基准测试总体结果")
        print("="*80)
        
        for benchmark_name, result in all_results["benchmarks"].items():
            if result["status"] == "success" and "results" in result:
                config = self.benchmarks.get(benchmark_name, BenchmarkConfig(benchmark_name, []))
                print(f"\n📈 {config.name}:")
                
                # 计算平均得分
                total_score = 0
                task_count = 0
                
                for task_name, task_result in result["results"]["results"].items():
                    if isinstance(task_result, dict):
                        acc_keys = [k for k in task_result.keys() if k.endswith("acc") or k.endswith("exact_match") or k == "acc_norm"]
                        for acc_key in acc_keys:
                            score = task_result[acc_key] 
                            if isinstance(score, (int, float)):
                                total_score += score
                                task_count += 1
                                break
                
                if task_count > 0:
                    avg_score = total_score / task_count
                    print(f"   平均得分: {avg_score:.1%}")
                else:
                    print("   状态: 完成")
            elif result["status"] == "failed":
                print(f"\n❌ {benchmark_name}: 测试失败")
            elif result["status"] == "timeout":
                print(f"\n⏰ {benchmark_name}: 测试超时")

    def list_available_benchmarks(self, show_details: bool = False):
        """列出所有可用的基准测试"""
        print(f"📋 可用的基准测试 ({len(self.benchmarks)} 个):")
        print("="*80)
        
        if show_details:
            # 按类别显示
            categories = self.get_available_categories()
            for category in categories:
                benchmarks = self.get_benchmarks_by_category(category)
                print(f"\n📂 {category.upper()} ({len(benchmarks)} 个):")
                for name in benchmarks:
                    config = self.benchmarks[name]
                    priority_emoji = "🌟" if config.priority == 1 else "✅" if config.priority == 2 else "📋"
                    print(f"   {priority_emoji} {name:<15} - {config.description}")
                    print(f"      任务数: {len(config.tasks)}, Few-shot: {config.num_fewshot}, 估计时间: {config.estimated_time_minutes}min")
        else:
            # 简化显示
            for name, config in self.benchmarks.items():
                priority_emoji = "🌟" if config.priority == 1 else "✅" if config.priority == 2 else "📋"
                print(f"{priority_emoji} {name:<15} - {config.description}")
        
        print("\n📦 预定义测试套件:")
        print("="*50)
        for suite_name, benchmarks in self.benchmark_suites.items():
            suite_info = get_suite_info(suite_name)
            time_str = f"{suite_info.get('estimated_time_hours', 0)}h" if 'estimated_time_hours' in suite_info else "未知"
            print(f"• {suite_name:<15} - {len(benchmarks)}个测试 (预计{time_str})")
            if show_details:
                print(f"   包含: {', '.join(benchmarks[:5])}{', ...' if len(benchmarks) > 5 else ''}")
        
        print("\n💡 使用方法:")
        print("  --benchmarks mmlu,bbh            # 运行指定测试")  
        print("  --benchmarks all                 # 运行所有测试")
        print("  --benchmarks suite:core          # 运行核心测试套件")
        print("  --benchmarks category:reasoning  # 运行推理类测试")
        print("  --benchmarks priority:1          # 运行优先级1的测试")
        print("="*80)
    
    def resolve_benchmark_names(self, benchmarks_arg: str) -> List[str]:
        """解析基准测试名称参数，支持套件、类别、优先级"""
        if benchmarks_arg == "all":
            return list(self.benchmarks.keys())
        
        if benchmarks_arg.startswith("suite:"):
            suite_name = benchmarks_arg[6:]
            return self.get_suite_benchmarks(suite_name)
        
        if benchmarks_arg.startswith("category:"):
            category = benchmarks_arg[9:]
            return self.get_benchmarks_by_category(category)
        
        if benchmarks_arg.startswith("priority:"):
            priority = int(benchmarks_arg[9:])
            return self.get_benchmarks_by_priority(priority)
        
        # 普通的逗号分隔列表
        return [b.strip() for b in benchmarks_arg.split(",")]