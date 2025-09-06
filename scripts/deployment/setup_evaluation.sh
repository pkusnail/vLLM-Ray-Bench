#!/bin/bash
# Model Evaluation Environment Setup Script

echo "🧪 Setting up Model Evaluation Environment..."

# Activate virtual environment
source vllm_ray_env/bin/activate

# Install evaluation dependencies
echo "📦 Installing evaluation packages..."
pip install lm-eval[api] \
    human-eval \
    datasets==3.6.0 \
    evaluate \
    modelscope \
    fire \
    termcolor \
    tenacity \
    jsonlines \
    rouge-score \
    sacrebleu \
    scikit-learn \
    absl-py \
    nltk \
    tabulate \
    colorama \
    more_itertools \
    word2number

echo "✅ Evaluation environment setup complete!"

# Test API connection
echo "🔍 Testing API connection..."
python3 -c "
import requests
try:
    response = requests.get('http://localhost:8000/-/routes', timeout=5)
    if response.status_code == 200:
        print('✅ API connection successful')
        print('Available routes:', response.json())
    else:
        print('⚠️ API may not be running on port 8000')
except:
    print('⚠️ API connection failed - make sure vLLM service is running')
"

echo ""
echo "🎯 Available Evaluation Commands:"
echo "  python3 quick_benchmark.py          # Quick 4-test benchmark"
echo "  python3 aiops_evaluation.py        # AIOPS capability evaluation" 
echo "  python3 standard_code_eval.py      # HumanEval code generation"
echo ""
echo "📚 Documentation:"
echo "  eval.md                            # Complete evaluation guide"
echo "  AIOPS_BENCHMARK_GUIDE.md           # AIOPS-specific testing guide"
echo ""