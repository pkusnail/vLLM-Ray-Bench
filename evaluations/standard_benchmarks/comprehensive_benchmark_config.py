#!/usr/bin/env python3
"""
全面的LLM基准测试配置
包含当前主流的所有重要测试数据集，按能力类型分类
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    name: str
    tasks: List[str]
    num_fewshot: int = 5
    batch_size: int = 1
    max_samples: Optional[int] = None
    description: str = ""
    category: str = "general"
    priority: int = 1  # 1=核心, 2=重要, 3=扩展
    estimated_time_minutes: int = 60

# 全面的基准测试数据集配置
COMPREHENSIVE_BENCHMARKS: Dict[str, BenchmarkConfig] = {
    
    # ===================
    # 1. 核心通用能力测试
    # ===================
    
    # 多学科知识理解
    "mmlu": BenchmarkConfig(
        name="MMLU",
        tasks=["mmlu"],
        num_fewshot=5,
        description="Massive Multitask Language Understanding - 57个学科多选题",
        category="knowledge",
        priority=1,
        estimated_time_minutes=120
    ),
    
    "cmmlu": BenchmarkConfig(
        name="CMMLU",
        tasks=["cmmlu"],
        num_fewshot=5,
        description="Chinese Massive Multitask Language Understanding - 中文综合理解",
        category="knowledge", 
        priority=1,
        estimated_time_minutes=120
    ),
    
    "global_mmlu": BenchmarkConfig(
        name="Global-MMLU",
        tasks=["global_mmlu_en"],
        num_fewshot=5,
        description="多语言版本的MMLU测试",
        category="multilingual",
        priority=2,
        estimated_time_minutes=90
    ),
    
    # 推理能力测试
    "bbh": BenchmarkConfig(
        name="BBH",
        tasks=["bbh_cot_fewshot"],
        num_fewshot=3,
        description="Big Bench Hard - 27个困难推理任务",
        category="reasoning",
        priority=1,
        estimated_time_minutes=90
    ),
    
    "agieval": BenchmarkConfig(
        name="AGIEval",
        tasks=["agieval_en"],
        num_fewshot=3,
        description="AGI能力评估 - 标准化考试题目",
        category="reasoning",
        priority=1,
        estimated_time_minutes=60
    ),
    
    # 数学推理能力
    "math": BenchmarkConfig(
        name="MATH",
        tasks=["hendrycks_math"],
        num_fewshot=4,
        description="高难度数学问题求解",
        category="math",
        priority=1,
        estimated_time_minutes=120
    ),
    
    "gsm8k": BenchmarkConfig(
        name="GSM8K",
        tasks=["gsm8k"],
        num_fewshot=8,
        description="小学数学应用题",
        category="math",
        priority=1,
        estimated_time_minutes=30
    ),
    
    # 常识推理
    "hellaswag": BenchmarkConfig(
        name="HellaSwag",
        tasks=["hellaswag"],
        num_fewshot=10,
        description="常识推理 - 情境续写",
        category="commonsense",
        priority=1,
        estimated_time_minutes=45
    ),
    
    "winogrande": BenchmarkConfig(
        name="WinoGrande",
        tasks=["winogrande"],
        num_fewshot=5,
        description="代词消解推理",
        category="commonsense",
        priority=1,
        estimated_time_minutes=30
    ),
    
    # ===================
    # 2. 专业能力测试
    # ===================
    
    # 科学推理
    "gpqa": BenchmarkConfig(
        name="GPQA",
        tasks=["gpqa_main"],
        num_fewshot=5,
        description="研究生级别物理化学生物问题",
        category="science",
        priority=1,
        estimated_time_minutes=60
    ),
    
    "arc_challenge": BenchmarkConfig(
        name="ARC-Challenge",
        tasks=["arc_challenge"],
        num_fewshot=25,
        description="科学推理挑战题",
        category="science",
        priority=2,
        estimated_time_minutes=45
    ),
    
    "arc_easy": BenchmarkConfig(
        name="ARC-Easy", 
        tasks=["arc_easy"],
        num_fewshot=25,
        description="基础科学推理题",
        category="science",
        priority=2,
        estimated_time_minutes=30
    ),
    
    # 阅读理解
    "race": BenchmarkConfig(
        name="RACE",
        tasks=["race"],
        num_fewshot=0,
        description="阅读理解能力测试",
        category="reading",
        priority=2,
        estimated_time_minutes=60
    ),
    
    "drop": BenchmarkConfig(
        name="DROP",
        tasks=["drop"],
        num_fewshot=1,
        description="离散推理式阅读理解",
        category="reading",
        priority=2,
        estimated_time_minutes=90
    ),
    
    # 逻辑推理
    "logiqa": BenchmarkConfig(
        name="LogiQA",
        tasks=["logiqa"],
        num_fewshot=0,
        description="逻辑推理能力测试",
        category="logic",
        priority=2,
        estimated_time_minutes=45
    ),
    
    # 真实性检测
    "truthfulqa": BenchmarkConfig(
        name="TruthfulQA",
        tasks=["truthfulqa_mc"],
        num_fewshot=0,
        description="真实性与偏见检测",
        category="safety",
        priority=1,
        estimated_time_minutes=45
    ),
    
    # ===================
    # 3. 代码能力测试
    # ===================
    
    "humaneval": BenchmarkConfig(
        name="HumanEval",
        tasks=["openai_humaneval"],
        num_fewshot=0,
        description="Python代码生成能力",
        category="coding",
        priority=1,
        estimated_time_minutes=60
    ),
    
    "mbpp": BenchmarkConfig(
        name="MBPP",
        tasks=["mbpp"],
        num_fewshot=3,
        description="Python编程基准测试",
        category="coding",
        priority=1,
        estimated_time_minutes=45
    ),
    
    # ===================
    # 4. 多语言能力测试
    # ===================
    
    "mgsm": BenchmarkConfig(
        name="MGSM",
        tasks=["mgsm_en", "mgsm_zh", "mgsm_ja", "mgsm_fr", "mgsm_es"],
        num_fewshot=8,
        description="多语言小学数学问题",
        category="multilingual",
        priority=2,
        estimated_time_minutes=60
    ),
    
    "xnli": BenchmarkConfig(
        name="XNLI",
        tasks=["xnli_en"],
        num_fewshot=15,
        description="跨语言自然语言推理",
        category="multilingual",
        priority=2,
        estimated_time_minutes=45
    ),
    
    # ===================
    # 5. 专业领域测试
    # ===================
    
    # 法律
    "lawbench": BenchmarkConfig(
        name="LawBench",
        tasks=["lawbench"],
        num_fewshot=3,
        description="法律专业知识测试",
        category="professional",
        priority=3,
        estimated_time_minutes=90
    ),
    
    # 医学
    "medqa": BenchmarkConfig(
        name="MedQA",
        tasks=["medqa_usmle"],
        num_fewshot=5,
        description="医学执业资格考试题",
        category="professional",
        priority=3,
        estimated_time_minutes=60
    ),
    
    # 金融
    "flare_finqa": BenchmarkConfig(
        name="FinQA",
        tasks=["flare_finqa"],
        num_fewshot=0,
        description="金融问答推理",
        category="professional",
        priority=3,
        estimated_time_minutes=45
    ),
    
    # ===================
    # 6. 高级推理测试
    # ===================
    
    "bigbench_hard": BenchmarkConfig(
        name="BigBench-Hard",
        tasks=["bigbench"],
        num_fewshot=3,
        description="BigBench中的困难子集",
        category="reasoning",
        priority=2,
        estimated_time_minutes=120
    ),
    
    "super_glue": BenchmarkConfig(
        name="SuperGLUE",
        tasks=["super_glue"],
        num_fewshot=0,
        description="高难度语言理解基准",
        category="reasoning",
        priority=2,
        estimated_time_minutes=90
    ),
    
    # ===================
    # 7. 特殊能力测试
    # ===================
    
    # 指令遵循
    "ifeval": BenchmarkConfig(
        name="IFEval",
        tasks=["ifeval"],
        num_fewshot=0,
        description="指令遵循能力评估",
        category="instruction",
        priority=2,
        estimated_time_minutes=30
    ),
    
    # 数学推理链
    "mathqa": BenchmarkConfig(
        name="MathQA",
        tasks=["mathqa"],
        num_fewshot=4,
        description="数学推理过程评估",
        category="math",
        priority=2,
        estimated_time_minutes=60
    ),
    
    # 工具使用
    "tool_bench": BenchmarkConfig(
        name="ToolBench",
        tasks=["toolbench"],
        num_fewshot=0,
        description="工具使用能力评估",
        category="tool_use",
        priority=3,
        estimated_time_minutes=90
    ),
    
    # ===================
    # 8. 中文专项测试
    # ===================
    
    "ceval": BenchmarkConfig(
        name="C-Eval",
        tasks=["ceval-valid"],
        num_fewshot=5,
        description="中文综合能力评估",
        category="chinese",
        priority=1,
        estimated_time_minutes=90
    ),
    
    "gaokao_bench": BenchmarkConfig(
        name="Gaokao-Bench",
        tasks=["gaokao_bench"],
        num_fewshot=5,
        description="高考题目测试集",
        category="chinese",
        priority=2,
        estimated_time_minutes=120
    ),
    
    # ===================
    # 9. 安全性测试
    # ===================
    
    "jailbreak_bench": BenchmarkConfig(
        name="JailbreakBench",
        tasks=["jailbreakbench"],
        num_fewshot=0,
        description="越狱攻击防护能力",
        category="safety",
        priority=2,
        estimated_time_minutes=30
    ),
    
    "bias_bench": BenchmarkConfig(
        name="BiasBench",
        tasks=["biasbench"],
        num_fewshot=0,
        description="偏见检测测试",
        category="safety",
        priority=3,
        estimated_time_minutes=45
    ),
}

# 预定义的测试套件
BENCHMARK_SUITES = {
    "core": [
        "mmlu", "bbh", "math", "gsm8k", "hellaswag", 
        "truthfulqa", "humaneval", "ceval"
    ],
    "reasoning": [
        "bbh", "agieval", "math", "gsm8k", "gpqa", 
        "arc_challenge", "logiqa", "bigbench_hard"
    ],
    "knowledge": [
        "mmlu", "cmmlu", "arc_challenge", "arc_easy", 
        "race", "drop", "truthfulqa"
    ],
    "multilingual": [
        "global_mmlu", "mgsm", "xnli", "ceval", "cmmlu"
    ],
    "professional": [
        "lawbench", "medqa", "flare_finqa", "gpqa"
    ],
    "coding": [
        "humaneval", "mbpp", "tool_bench"
    ],
    "safety": [
        "truthfulqa", "jailbreak_bench", "bias_bench"
    ],
    "chinese": [
        "ceval", "cmmlu", "gaokao_bench"
    ],
    "math_intensive": [
        "math", "gsm8k", "mathqa", "mgsm"
    ],
    "comprehensive": [
        "mmlu", "bbh", "math", "gsm8k", "hellaswag", "truthfulqa",
        "humaneval", "ceval", "cmmlu", "gpqa", "agieval"
    ]
}

def get_benchmark_by_category(category: str) -> List[str]:
    """根据类别获取基准测试列表"""
    return [name for name, config in COMPREHENSIVE_BENCHMARKS.items() 
            if config.category == category]

def get_benchmark_by_priority(priority: int) -> List[str]:
    """根据优先级获取基准测试列表"""
    return [name for name, config in COMPREHENSIVE_BENCHMARKS.items() 
            if config.priority == priority]

def estimate_total_time(benchmark_names: List[str]) -> int:
    """估算总测试时间(分钟)"""
    return sum(COMPREHENSIVE_BENCHMARKS[name].estimated_time_minutes 
               for name in benchmark_names 
               if name in COMPREHENSIVE_BENCHMARKS)

def list_all_categories() -> List[str]:
    """列出所有测试类别"""
    categories = set()
    for config in COMPREHENSIVE_BENCHMARKS.values():
        categories.add(config.category)
    return sorted(list(categories))

def get_suite_info(suite_name: str) -> Dict:
    """获取测试套件信息"""
    if suite_name not in BENCHMARK_SUITES:
        return {}
    
    benchmarks = BENCHMARK_SUITES[suite_name]
    total_time = estimate_total_time(benchmarks)
    categories = set()
    for name in benchmarks:
        if name in COMPREHENSIVE_BENCHMARKS:
            categories.add(COMPREHENSIVE_BENCHMARKS[name].category)
    
    return {
        "benchmarks": benchmarks,
        "count": len(benchmarks),
        "estimated_time_hours": round(total_time / 60, 1),
        "categories": sorted(list(categories))
    }