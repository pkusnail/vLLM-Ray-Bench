#!/usr/bin/env python3
"""
vLLM集群性能专项测试
专注测试集群的吞吐量、延迟、并发能力等系统性能指标，忽略回答内容质量
"""

import requests
import json
import time
import threading
import concurrent.futures
import statistics
from datetime import datetime

def simple_request(api_url, prompt, request_id):
    """简单请求，专注性能测试"""
    payload = {
        "model": "Qwen/Qwen3-32B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,  # 固定较小的token数，专注测试系统性能
        "temperature": 0.0  # 固定参数确保一致性
    }
    
    start_time = time.time()
    try:
        response = requests.post(api_url, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            response_time = end_time - start_time
            
            return {
                "request_id": request_id,
                "success": True,
                "response_time": response_time,
                "tokens": result['usage']['total_tokens'],
                "prompt_tokens": result['usage']['prompt_tokens'],
                "completion_tokens": result['usage']['completion_tokens'],
                "first_token_time": None  # 简化测试，不计算TTFT
            }
        else:
            return {
                "request_id": request_id,
                "success": False,
                "error": f"HTTP {response.status_code}",
                "response_time": end_time - start_time
            }
    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "error": str(e),
            "response_time": time.time() - start_time
        }

def run_performance_scenario(scenario_name, concurrent_requests, total_requests, prompt, api_url):
    """运行单个性能测试场景"""
    print(f"\n🚀 执行场景: {scenario_name}")
    print(f"   并发数: {concurrent_requests}")
    print(f"   总请求数: {total_requests}")
    print(f"   测试提示: {prompt[:50]}...")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = []
        for i in range(total_requests):
            future = executor.submit(simple_request, api_url, prompt, i+1)
            futures.append(future)
        
        results = []
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % max(1, total_requests // 10) == 0:  # 显示进度
                print(f"   进度: {completed}/{total_requests} ({completed/total_requests*100:.0f}%)")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 分析结果
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if successful_results:
        response_times = [r['response_time'] for r in successful_results]
        total_tokens = sum(r['tokens'] for r in successful_results)
        
        metrics = {
            "scenario_name": scenario_name,
            "test_config": {
                "concurrent_requests": concurrent_requests,
                "total_requests": total_requests,
                "prompt": prompt
            },
            "results": {
                "total_time": total_time,
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / total_requests * 100,
                
                # 延迟指标
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
                "p99_response_time": statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                
                # 吞吐量指标  
                "requests_per_second": len(successful_results) / total_time,
                "tokens_per_second": total_tokens / total_time,
                "avg_tokens_per_request": total_tokens / len(successful_results),
                
                # 资源效率
                "total_tokens_processed": total_tokens,
                "cluster_efficiency": len(successful_results) / (concurrent_requests * total_time) # 每并发每秒处理的请求数
            },
            "raw_data": results
        }
        
        print(f"   ✅ 完成时间: {total_time:.2f}s")
        print(f"   ✅ 成功率: {metrics['results']['success_rate']:.1f}%")
        print(f"   ⚡ 平均延迟: {metrics['results']['avg_response_time']:.3f}s")
        print(f"   ⚡ P95延迟: {metrics['results']['p95_response_time']:.3f}s")  
        print(f"   🚀 请求吞吐量: {metrics['results']['requests_per_second']:.2f} req/s")
        print(f"   🚀 Token吞吐量: {metrics['results']['tokens_per_second']:.1f} tok/s")
        
        return metrics
    else:
        print(f"   ❌ 所有请求都失败了!")
        return {
            "scenario_name": scenario_name,
            "error": "All requests failed",
            "results": results
        }

def test_cluster_performance():
    api_url = "http://localhost:8000/v1/chat/completions"
    
    print("=" * 80)
    print("⚡ vLLM集群性能专项测试")
    print("=" * 80)
    print("🎯 测试目标：评估集群的吞吐量、延迟、并发处理能力")
    print("📊 测试方式：多种负载模式，固定短回答，专注系统性能")
    print("🔧 架构：PP=2, TP=8 (2 nodes, 16 GPUs)")
    print()
    
    # 使用简单统一的测试提示，避免内容复杂度影响性能测试
    test_prompt = "What is the capital of China?"
    
    # 性能测试场景矩阵
    performance_scenarios = [
        # 基线延迟测试
        {"name": "基线延迟", "concurrent": 1, "total": 10, "description": "单线程基准延迟"},
        {"name": "低并发", "concurrent": 2, "total": 20, "description": "2并发稳定性"},
        {"name": "中并发", "concurrent": 5, "total": 50, "description": "5并发性能"},
        {"name": "高并发", "concurrent": 10, "total": 100, "description": "10并发压力测试"},
        {"name": "极限并发", "concurrent": 20, "total": 200, "description": "20并发极限测试"},
        
        # 长时间稳定性测试
        {"name": "持久性能", "concurrent": 8, "total": 160, "description": "持续负载稳定性"},
        
        # 突发流量测试  
        {"name": "突发流量", "concurrent": 15, "total": 75, "description": "模拟突发请求"}
    ]
    
    all_results = []
    
    for i, scenario in enumerate(performance_scenarios):
        print(f"\n{'='*80}")
        print(f"📊 测试 {i+1}/{len(performance_scenarios)}: {scenario['name']}")
        print(f"📝 说明: {scenario['description']}")
        print("="*80)
        
        # 执行测试场景
        result = run_performance_scenario(
            scenario['name'],
            scenario['concurrent'], 
            scenario['total'],
            test_prompt,
            api_url
        )
        
        if 'error' not in result:
            all_results.append(result)
        
        # 测试间隔，让集群稍作休息
        if i < len(performance_scenarios) - 1:
            print(f"   ⏳ 等待5秒后进行下一个测试...")
            time.sleep(5)
    
    # 性能分析汇总
    print(f"\n{'='*80}")
    print("📈 集群性能分析汇总")
    print("="*80)
    
    if all_results:
        # 提取关键性能指标
        throughput_data = [(r['test_config']['concurrent_requests'], r['results']['requests_per_second']) for r in all_results]
        latency_data = [(r['test_config']['concurrent_requests'], r['results']['avg_response_time']) for r in all_results]
        token_throughput_data = [(r['test_config']['concurrent_requests'], r['results']['tokens_per_second']) for r in all_results]
        
        print(f"\n📊 性能趋势分析:")
        print(f"{'并发数':<8} {'请求吞吐量':<12} {'Token吞吐量':<12} {'平均延迟':<10} {'P95延迟':<10}")
        print("-" * 60)
        
        for result in all_results:
            config = result['test_config']
            metrics = result['results']
            print(f"{config['concurrent_requests']:<8} {metrics['requests_per_second']:<12.2f} "
                  f"{metrics['tokens_per_second']:<12.1f} {metrics['avg_response_time']:<10.3f} "
                  f"{metrics['p95_response_time']:<10.3f}")
        
        # 找出最佳性能点
        max_throughput = max(all_results, key=lambda x: x['results']['requests_per_second'])
        max_token_throughput = max(all_results, key=lambda x: x['results']['tokens_per_second'])
        min_latency = min(all_results, key=lambda x: x['results']['avg_response_time'])
        
        print(f"\n🏆 性能峰值:")
        print(f"   • 最高请求吞吐量: {max_throughput['results']['requests_per_second']:.2f} req/s "
              f"(并发数: {max_throughput['test_config']['concurrent_requests']})")
        print(f"   • 最高Token吞吐量: {max_token_throughput['results']['tokens_per_second']:.1f} tok/s "
              f"(并发数: {max_token_throughput['test_config']['concurrent_requests']})")
        print(f"   • 最低平均延迟: {min_latency['results']['avg_response_time']:.3f}s "
              f"(并发数: {min_latency['test_config']['concurrent_requests']})")
        
        # 集群效率分析
        avg_success_rate = statistics.mean([r['results']['success_rate'] for r in all_results])
        peak_tokens_per_second = max_token_throughput['results']['tokens_per_second']
        
        print(f"\n📈 集群整体评估:")
        print(f"   • 平均成功率: {avg_success_rate:.1f}%")
        print(f"   • 峰值Token处理能力: {peak_tokens_per_second:.1f} tok/s")
        
        # 性能等级评估
        if peak_tokens_per_second > 500:
            performance_tier = "🌟 卓越 - 高性能集群"
        elif peak_tokens_per_second > 300:
            performance_tier = "✅ 优秀 - 生产级性能"
        elif peak_tokens_per_second > 150:
            performance_tier = "⚠️  良好 - 基本满足需求"
        else:
            performance_tier = "❌ 待优化 - 性能不足"
        
        print(f"   • 性能等级: {performance_tier}")
        
        # 并发扩展性分析
        low_concurrent_tps = [r['results']['tokens_per_second'] for r in all_results if r['test_config']['concurrent_requests'] <= 5]
        high_concurrent_tps = [r['results']['tokens_per_second'] for r in all_results if r['test_config']['concurrent_requests'] >= 10]
        
        if low_concurrent_tps and high_concurrent_tps:
            scalability_ratio = max(high_concurrent_tps) / max(low_concurrent_tps)
            print(f"   • 并发扩展性: {scalability_ratio:.2f}x (高并发相对低并发的性能提升)")
        
        # 保存详细结果
        summary_data = {
            "cluster_performance_evaluation": {
                "model": "Qwen/Qwen3-32B",
                "architecture": "PP=2, TP=8 (2 nodes, 16 GPUs)",
                "test_type": "集群性能专项测试",
                "test_focus": "吞吐量、延迟、并发处理能力等系统性能指标",
                "test_scenarios": all_results,
                "performance_summary": {
                    "max_request_throughput": max_throughput['results']['requests_per_second'],
                    "max_token_throughput": max_token_throughput['results']['tokens_per_second'], 
                    "min_avg_latency": min_latency['results']['avg_response_time'],
                    "avg_success_rate": avg_success_rate,
                    "performance_tier": performance_tier,
                    "optimal_concurrency": max_token_throughput['test_config']['concurrent_requests']
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        with open("cluster_performance_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细结果已保存: cluster_performance_evaluation.json")
        print(f"📄 该文件包含完整的性能指标和原始数据")
        
    else:
        print("❌ 没有成功完成任何性能测试")

if __name__ == "__main__":
    test_cluster_performance()