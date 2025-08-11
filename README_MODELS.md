# ⚠️ CRITICAL: Supported Models for UMinFramework

## Important Notice

**The UMinFramework only works with specific pre-trained uncertainty head models.** Using unsupported models will cause the uncertainty quantification features to fail or provide inaccurate results.

## ✅ Supported Models (ONLY USE THESE)

### Primary Recommendation: Mistral-7B-Instruct-v0.2
```yaml
# In configuration files
baseline_model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"
augmented_model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"
```

### Alternative Options:
1. **Llama-3.1-8B-Instruct**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
2. **Gemma-2-9B-IT**: `google/gemma-2-9b-it`

## ❌ DO NOT USE These Models

- ~~`gpt2`~~ - No uncertainty head available
- ~~`microsoft/DialoGPT-medium`~~ - No uncertainty head available
- ~~`codegen-*`~~ - No uncertainty head available
- ~~`codet5-*`~~ - No uncertainty head available
- ~~Any other models~~ - Will cause framework failures

## Quick Setup

### 1. Update Configuration
All configuration files have been updated to use `mistralai/Mistral-7B-Instruct-v0.2` by default.

### 2. Run with Supported Models
```bash
# Correct usage
python scripts/run_benchmark.py \
    --baseline-model "mistralai/Mistral-7B-Instruct-v0.2" \
    --datasets humaneval \
    --max-problems 5

# Alternative supported model
python scripts/run_benchmark.py \
    --baseline-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --datasets humaneval \
    --max-problems 5
```

### 3. Hardware Requirements
These models require significant resources:
- **GPU**: 16GB+ VRAM recommended
- **RAM**: 32GB+ system memory
- **Storage**: ~15GB for model downloads

## For Complete Details
See: [docs/supported_models.md](docs/supported_models.md)

## Quick Verification
```python
from umin_framework import AugmentedLLMConfig

# This will work ✅
config = AugmentedLLMConfig(
    generation_model="mistralai/Mistral-7B-Instruct-v0.2"
)

# This will fail ❌
config = AugmentedLLMConfig(
    generation_model="gpt2"  # No uncertainty head available
)
```

---
**Remember: Only use models from the supported list above!**