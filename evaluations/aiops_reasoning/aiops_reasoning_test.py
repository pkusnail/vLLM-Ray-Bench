#!/usr/bin/env python3
"""
AIOps推理能力专项测试
专注测试模型在运维场景下的推理质量，忽略性能指标
"""

import requests
import json
import time

def test_aiops_reasoning():
    api_url = "http://localhost:8000/v1/chat/completions"
    
    print("=" * 70)
    print("🧠 AIOps推理能力专项测试")
    print("=" * 70)
    print("📋 测试目标：评估模型在运维场景下的推理深度和准确性")
    print("⏱️  测试方式：单线程顺序执行，关注答案质量而非速度")
    print()
    
    # AIOps专项推理测试用例
    aiops_scenarios = [
        {
            "category": "故障诊断推理",
            "scenario": "CPU使用率异常分析",
            "context": """
            服务器监控数据显示：
            - CPU使用率从10%突增到95%
            - 内存使用率正常（60%）
            - 磁盘I/O正常
            - 网络流量正常
            - 时间：凌晨2:30
            - 无用户活动
            """,
            "question": "基于以上信息进行故障分析，请按以下步骤推理：1）列出可能的原因；2）分析每个原因的可能性；3）提出排查步骤；4）给出预防措施",
            "evaluation_criteria": {
                "逻辑性": "推理步骤是否清晰",
                "专业性": "是否展现运维专业知识",
                "完整性": "是否覆盖关键排查点",
                "实用性": "建议是否可行"
            }
        },
        {
            "category": "容量规划推理", 
            "scenario": "数据库性能预测",
            "context": """
            MySQL数据库当前状态：
            - 当前QPS: 1000
            - 当前连接数: 200/500
            - CPU使用率: 70%
            - 内存使用率: 80%
            - 业务增长率: 每月20%
            - 用户活跃时段: 9:00-21:00
            """,
            "question": "预测未来3个月的数据库性能瓶颈，请分析：1）哪个指标最先达到瓶颈？2）什么时间点需要扩容？3）推荐的扩容方案；4）风险评估",
            "evaluation_criteria": {
                "预测准确性": "计算是否合理",
                "风险意识": "是否考虑了各种风险",
                "方案可行性": "建议是否实际可行",
                "成本考虑": "是否平衡了性能和成本"
            }
        },
        {
            "category": "安全事件推理",
            "scenario": "异常登录行为分析", 
            "context": """
            安全日志显示：
            - 用户admin在过去1小时内登录50次
            - 登录来源IP: 来自5个不同国家
            - 登录时间: 均在业务时间外（凌晨1-4点）
            - 其中30次登录失败，20次成功
            - 成功登录后立即执行了系统配置修改
            """,
            "question": "这是安全事件吗？请进行深度分析：1）判断事件性质和严重程度；2）分析攻击手段和目的；3）评估已造成的影响；4）制定应急响应计划",
            "evaluation_criteria": {
                "威胁识别": "是否准确识别安全威胁",
                "影响评估": "是否全面评估影响范围", 
                "应急反应": "处置方案是否及时有效",
                "深度分析": "是否深入分析攻击意图"
            }
        },
        {
            "category": "架构优化推理",
            "scenario": "微服务性能优化",
            "context": """
            微服务架构现状：
            - 订单服务响应时间：2秒（目标<500ms）
            - 用户服务响应时间：200ms
            - 支付服务响应时间：1秒
            - Redis缓存命中率：60%
            - 数据库连接池：80%占用
            - 服务间调用：订单→用户→支付→订单
            """,
            "question": "如何优化整体架构性能？请分析：1）性能瓶颈根因分析；2）各种优化方案的优缺点；3）优化的优先级排序；4）实施风险和回滚策略",
            "evaluation_criteria": {
                "根因分析": "是否找到真正的性能瓶颈",
                "方案对比": "是否权衡了多种解决方案",
                "实施规划": "是否有清晰的实施路径",
                "风险控制": "是否考虑了实施风险"
            }
        },
        {
            "category": "监控告警推理",
            "scenario": "告警风暴处理",
            "context": """
            告警情况：
            - 过去10分钟收到500条告警
            - 涉及20个不同服务
            - 告警类型：磁盘空间、内存、响应时间、连接数
            - 告警时间集中在08:50-09:00
            - 正值业务高峰期
            - 部分服务开始返回5xx错误
            """,
            "question": "面对告警风暴如何有效处理？请分析：1）如何快速识别核心问题？2）告警优先级如何划分？3）处理顺序和策略；4）如何防止告警风暴再次发生",
            "evaluation_criteria": {
                "优先级判断": "是否能快速抓住核心问题",
                "处理策略": "应急处理步骤是否合理",
                "系统性思考": "是否从系统角度思考问题",
                "预防机制": "是否提出有效的预防措施"
            }
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(aiops_scenarios):
        print(f"\n{'='*70}")
        print(f"📊 测试 {i+1}/{len(aiops_scenarios)}: {test_case['category']}")
        print(f"🎯 场景: {test_case['scenario']}")
        print("="*70)
        
        # 构造完整的推理提示
        full_prompt = f"""
你是一名资深的AIOps工程师，请基于以下场景进行深度推理分析：

【场景背景】
{test_case['context']}

【分析任务】
{test_case['question']}

请进行深入的逻辑推理，展现你的运维专业知识和系统性思考能力。
"""
        
        payload = {
            "model": "Qwen/Qwen3-32B", 
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 800,  # 允许更长的推理回答
            "temperature": 0.1   # 低温度确保推理的准确性
        }
        
        print("🤔 模型推理中...")
        start_time = time.time()
        
        try:
            response = requests.post(api_url, json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                reasoning_response = result['choices'][0]['message']['content'].strip()
                response_time = end_time - start_time
                
                print(f"\n💭 推理结果:")
                print("-" * 70)
                print(reasoning_response)
                print("-" * 70)
                
                # 人工评估提示（实际使用时可以接入评估模型）
                print(f"\n📏 评估维度:")
                for criterion, description in test_case['evaluation_criteria'].items():
                    print(f"   • {criterion}: {description}")
                
                # 简单的质量指标
                word_count = len(reasoning_response.split())
                has_structured_thinking = any(marker in reasoning_response.lower() 
                                            for marker in ['1)', '2)', '3)', '4)', '一、', '二、', '首先', '其次'])
                has_professional_terms = any(term in reasoning_response.lower() 
                                           for term in ['cpu', '内存', '磁盘', 'io', '负载', '缓存', '数据库', 
                                                      '服务', '监控', '告警', '性能', '瓶颈'])
                
                quality_score = 0
                if word_count > 100: quality_score += 1  # 回答详细
                if has_structured_thinking: quality_score += 1  # 结构化思考  
                if has_professional_terms: quality_score += 1  # 专业性
                if '分析' in reasoning_response or '推理' in reasoning_response: quality_score += 1  # 分析性
                
                print(f"\n📊 自动质量评分: {quality_score}/4")
                print(f"   • 回答详细度: {'✓' if word_count > 100 else '✗'} ({word_count} words)")
                print(f"   • 结构化思考: {'✓' if has_structured_thinking else '✗'}")
                print(f"   • 专业术语使用: {'✓' if has_professional_terms else '✗'}")
                print(f"   • 分析深度: {'✓' if '分析' in reasoning_response or '推理' in reasoning_response else '✗'}")
                
                results.append({
                    "test_id": i + 1,
                    "category": test_case['category'],
                    "scenario": test_case['scenario'],
                    "context": test_case['context'],
                    "question": test_case['question'],
                    "response": reasoning_response,
                    "response_time": response_time,
                    "word_count": word_count,
                    "quality_score": quality_score,
                    "evaluation_criteria": test_case['evaluation_criteria'],
                    "auto_evaluation": {
                        "detailed_response": word_count > 100,
                        "structured_thinking": has_structured_thinking,
                        "professional_terms": has_professional_terms,
                        "analytical_depth": '分析' in reasoning_response or '推理' in reasoning_response
                    }
                })
                
                print(f"   • 推理用时: {response_time:.1f}s")
                
            else:
                print(f"❌ 请求失败: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            
        print(f"\n⏳ 等待2秒后进行下一个测试...")
        time.sleep(2)  # 给模型一些休息时间，确保推理质量
    
    # 汇总推理能力评估
    print(f"\n{'='*70}")
    print("🎯 AIOps推理能力评估汇总")
    print("="*70)
    
    if results:
        avg_quality_score = sum(r['quality_score'] for r in results) / len(results)
        avg_word_count = sum(r['word_count'] for r in results) / len(results)
        
        print(f"📊 测试完成: {len(results)}/{len(aiops_scenarios)} 个场景")
        print(f"📏 平均质量评分: {avg_quality_score:.1f}/4.0")
        print(f"📝 平均回答长度: {avg_word_count:.0f} words")
        
        # 分类统计
        categories = {}
        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['quality_score'])
        
        print(f"\n📋 分类评估:")
        for category, scores in categories.items():
            avg_score = sum(scores) / len(scores)
            print(f"   • {category}: {avg_score:.1f}/4.0")
        
        # 推理能力等级
        if avg_quality_score >= 3.5:
            reasoning_level = "🌟 优秀 - 具备高级AIOps推理能力"
        elif avg_quality_score >= 2.5:
            reasoning_level = "✅ 良好 - 具备基本AIOps推理能力"
        elif avg_quality_score >= 1.5:
            reasoning_level = "⚠️  一般 - AIOps推理能力待提升"
        else:
            reasoning_level = "❌ 较差 - AIOps推理能力不足"
        
        print(f"\n🎯 AIOps推理能力等级: {reasoning_level}")
        
        # 保存详细结果
        output_data = {
            "aiops_reasoning_evaluation": {
                "model": "Qwen/Qwen3-32B",
                "test_type": "推理能力专项测试",
                "test_focus": "AIOps场景下的逻辑推理、专业分析、系统性思考能力",
                "results": results,
                "summary": {
                    "total_tests": len(results),
                    "avg_quality_score": avg_quality_score,
                    "avg_word_count": avg_word_count,
                    "reasoning_level": reasoning_level,
                    "category_scores": {cat: sum(scores)/len(scores) for cat, scores in categories.items()}
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        with open("aiops_reasoning_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细结果已保存: aiops_reasoning_evaluation.json")
        print(f"📄 该文件包含完整的推理过程和质量评估")
        
    else:
        print("❌ 没有完成任何推理测试")

if __name__ == "__main__":
    test_aiops_reasoning()