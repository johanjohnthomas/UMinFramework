# Supported Models for UMinFramework

## Overview

The UMinFramework's uncertainty quantification capabilities rely on pre-trained uncertainty heads from the [llm-uncertainty-head](https://huggingface.co/llm-uncertainty-head) project by IINemo. These uncertainty heads are specifically trained for certain language models and provide superior uncertainty estimation compared to generic methods.

## ⚠️ Important: Model Compatibility

**The uncertainty quantification features in UMinFramework only work with the specific models listed below.** Using other models will either fail or provide inaccurate uncertainty estimates, as the uncertainty heads are trained specifically for these model architectures and parameters.

## Supported Models List

### 1. Mistral Models

#### Mistral-7B-Instruct-v0.2
- **Hugging Face ID**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Available Uncertainty Heads**:
  - `llm-uncertainty-head/uhead_claim_att_exp5_128heads_Mistral-7B-Instruct-v0.2`
  - `llm-uncertainty-head/uhead_claim_att_exp3_Mistral-7B-Instruct-v0.2`
  - `llm-uncertainty-head/uhead_claim_exp6_Mistral-7B-Instruct-v0.2`
  - `llm-uncertainty-head/uhead_claim_exp5_Mistral-7B-Instruct-v0.2`
  - `llm-uncertainty-head/uhead_claim_new_Mistral-7B-Instruct-v0.2`
- **Parameters**: 7B
- **Type**: Instruction-tuned
- **Recommended**: ✅ Primary supported model with multiple uncertainty head variants

### 2. Llama Models

#### Llama-3.1-8B-Instruct
- **Hugging Face ID**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Available Uncertainty Heads**:
  - `llm-uncertainty-head/uhead_claim_Llama-3.1-8B-Instruct`
- **Parameters**: 8B
- **Type**: Instruction-tuned
- **Recommended**: ✅ Well-supported with dedicated uncertainty head

### 3. Gemma Models

#### Gemma-2-9B-IT
- **Hugging Face ID**: `google/gemma-2-9b-it`
- **Available Uncertainty Heads**:
  - `llm-uncertainty-head/uhead_claim_exp2_gemma-2-9b-it`
  - `llm-uncertainty-head/uhead_gemma-2-9b-it`
  - `llm-uncertainty-head/saplma_gemma-2-9b-it`
- **Parameters**: 9B
- **Type**: Instruction-tuned
- **Recommended**: ✅ Multiple uncertainty head variants available

### 4. Qwen Models

#### Qwen-3-0.6B (Auxiliary Model)
- **Hugging Face ID**: Not specified (used for claim extraction)
- **Available Components**:
  - `llm-uncertainty-head/Qwen3-0.6B-claim-extractor`
- **Parameters**: 0.6B
- **Type**: Specialized for claim extraction
- **Usage**: Auxiliary model for enhanced uncertainty estimation

## Configuration Examples

### Default Configuration (Mistral-7B)
```yaml
# config/default.yaml
baseline_model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"
  device: null
  trust_remote_code: false

augmented_model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"
  device: null
  trust_remote_code: false
```

### Llama Configuration
```yaml
# config/llama_config.yaml
baseline_model:
  name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  device: null
  trust_remote_code: false

augmented_model:
  name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  device: null
  trust_remote_code: false
```

### Gemma Configuration
```yaml
# config/gemma_config.yaml
baseline_model:
  name: "google/gemma-2-9b-it"
  device: null
  trust_remote_code: false

augmented_model:
  name: "google/gemma-2-9b-it"
  device: null
  trust_remote_code: false
```

## Usage in Code

### AugmentedLLM Configuration
```python
from umin_framework import AugmentedLLM, AugmentedLLMConfig

# Mistral example
config = AugmentedLLMConfig(
    generation_model="mistralai/Mistral-7B-Instruct-v0.2",
    uncertainty_threshold=0.7,
    backtrack_window=3
)

# Llama example  
config = AugmentedLLMConfig(
    generation_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    uncertainty_threshold=0.7,
    backtrack_window=3
)

# Gemma example
config = AugmentedLLMConfig(
    generation_model="google/gemma-2-9b-it",
    uncertainty_threshold=0.7,
    backtrack_window=3
)

augmented_llm = AugmentedLLM(config=config)
```

### Benchmarking with Supported Models
```bash
# Mistral benchmarking
python scripts/run_benchmark.py \
    --baseline-model "mistralai/Mistral-7B-Instruct-v0.2" \
    --datasets humaneval \
    --max-problems 10

# Llama benchmarking
python scripts/run_benchmark.py \
    --baseline-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --datasets humaneval \
    --max-problems 10

# Gemma benchmarking
python scripts/run_benchmark.py \
    --baseline-model "google/gemma-2-9b-it" \
    --datasets humaneval \
    --max-problems 10
```

## Model Requirements

### Hardware Requirements
- **GPU**: Recommended for 7B+ models (at least 16GB VRAM for 7B models)
- **RAM**: Minimum 32GB system RAM for larger models
- **Storage**: Models are downloaded from Hugging Face Hub (5-20GB per model)

### Software Requirements
- **Python**: 3.8+
- **PyTorch**: Compatible with CUDA for GPU acceleration
- **Transformers**: 4.0+ for model loading
- **Hugging Face Account**: Required for accessing some gated models (especially Llama)

### Access Requirements
Some models may require:
- **Hugging Face Account**: Free registration at huggingface.co
- **Model Access Request**: Some models (like Llama) require requesting access
- **Authentication**: Use `huggingface-cli login` for gated models

## Uncertainty Head Integration

The uncertainty heads are automatically loaded when using the supported models. The framework:

1. **Detects Model**: Identifies which uncertainty head to use based on the base model
2. **Downloads Head**: Automatically downloads the appropriate uncertainty head from Hugging Face
3. **Integrates Seamlessly**: Provides uncertainty scores without additional configuration

### Available Uncertainty Methods
For supported models, you can use these uncertainty quantification methods:
- **Entropy**: Shannon entropy of token probability distributions
- **Max Probability**: 1 - maximum probability (higher = more uncertain)
- **Margin**: 1 - margin between top 2 predictions
- **Variance**: Variance of probability distribution
- **Pre-trained Heads**: Advanced uncertainty estimation from trained heads

## Model Selection Guidelines

### For Research and Experimentation
- **Mistral-7B-Instruct-v0.2**: Best overall choice with multiple uncertainty head variants
- **Gemma-2-9B-IT**: Good performance with multiple uncertainty head options

### For Production Use
- **Llama-3.1-8B-Instruct**: Robust and well-tested instruction-following
- **Mistral-7B-Instruct-v0.2**: Efficient and reliable

### For Resource-Constrained Environments
- **Mistral-7B-Instruct-v0.2**: Smallest supported model at 7B parameters
- Consider using quantization or smaller batch sizes

## Troubleshooting

### Common Issues

#### "Model not supported" Error
```
UnsupportedModelError: Model 'gpt2' is not supported for uncertainty quantification
```
**Solution**: Use one of the supported models listed above.

#### Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Use CPU instead: `device="cpu"`
- Enable model quantization: `load_in_8bit=True`
- Reduce batch size or sequence length
- Use smaller models (7B vs 9B)

#### Access Denied Errors
```
Repository not found or access denied
```
**Solutions**:
- Login to Hugging Face: `huggingface-cli login`
- Request access to gated models
- Check model name spelling

### Performance Optimization

#### GPU Usage
```python
config = AugmentedLLMConfig(
    generation_model="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda",  # Use GPU
    torch_dtype="float16"  # Use half precision
)
```

#### Memory Optimization
```python
config = AugmentedLLMConfig(
    generation_model="mistralai/Mistral-7B-Instruct-v0.2",
    load_in_8bit=True,  # 8-bit quantization
    max_length=256,     # Shorter sequences
)
```

## Future Model Support

The UMinFramework will support additional models as uncertainty heads become available. Check the [llm-uncertainty-head Hugging Face organization](https://huggingface.co/llm-uncertainty-head) for updates on newly supported models.

To request support for a specific model, consider:
1. Opening an issue in the llm-uncertainty-head repository
2. Contributing uncertainty head training for your desired model
3. Using the framework's baseline uncertainty methods (less accurate) for unsupported models

## References

- **llm-uncertainty-head**: https://huggingface.co/llm-uncertainty-head
- **Original Paper**: "A Head to Predict and a Head to Question: Pre-trained Uncertainty Quantification Heads for Hallucination Detection in LLM Outputs" (arXiv:2505.08200)
- **GitHub Repository**: https://github.com/IINemo/llm-uncertainty-head