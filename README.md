# UMinFramework: Uncertainty-Guided Code Generation

A comprehensive framework for uncertainty-aware language model code generation with backtracking, prompt refinement, and systematic evaluation capabilities.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and setup
git clone git@github.com:johanjohnthomas/UMinFramework.git
cd UMinFramework

# Create virtual environment and install dependencies
make setup
# OR manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Datasets
```bash
# Download and preprocess benchmark datasets
make download
# OR manually:
python scripts/download_datasets.py
```

### 3. Run Your First Benchmark
```bash
# Quick test with baseline model only
python scripts/run_benchmark.py \
    --baseline-model "mistralai/Mistral-7B-Instruct-v0.2" \
    --datasets humaneval \
    --max-problems 5 \
    --no-augmented
```

## ⚠️ Model Compatibility

**CRITICAL**: UMinFramework only works with specific models that have pre-trained uncertainty heads. Using unsupported models will cause failures.

### ✅ Supported Models
- **`mistralai/Mistral-7B-Instruct-v0.2`** (Primary recommendation)
- **`meta-llama/Meta-Llama-3.1-8B-Instruct`**
- **`google/gemma-2-9b-it`**

See [README_MODELS.md](README_MODELS.md) for complete details.

## 📁 Project Structure

```
UMinFramework/
├── src/umin_framework/          # Core framework code
│   ├── augmented_llm.py         # Main AugmentedLLM class
│   ├── uncertainty_head.py      # Uncertainty quantification
│   ├── prompt_refiner.py        # Prompt refinement models
│   ├── generation_loop.py       # Generation with backtracking
│   └── code_executor.py         # Safe code execution & evaluation
├── scripts/                     # Execution scripts
│   ├── run_benchmark.py         # Main benchmarking script
│   ├── finetune_prompt_refiner.py # Fine-tune prompt refiner
│   ├── download_datasets.py     # Download benchmark data
│   └── preprocess_askcq.py      # Preprocess AskCQ dataset
├── config/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   ├── default.yaml             # Default settings
│   └── benchmark_example.yaml   # Example benchmark config
├── data/                        # Benchmark datasets (created by download)
├── models/                      # Trained models (created by fine-tuning)
├── results/                     # Benchmark results
├── examples/                    # Usage examples
├── docs/                        # Documentation
├── tests/                       # Unit tests
└── colab/                       # Google Colab demo
```

## 🔧 Core Components

### 1. AugmentedLLM
Enhanced language model with uncertainty-guided generation:
```python
from umin_framework import AugmentedLLM, AugmentedLLMConfig

config = AugmentedLLMConfig(
    generation_model="mistralai/Mistral-7B-Instruct-v0.2",
    uncertainty_threshold=0.7,
    backtrack_window=3,
    enable_prompt_refinement=True
)

llm = AugmentedLLM(config=config)
result = llm.generate("Write a function to calculate fibonacci numbers")
```

### 2. Uncertainty Quantification
Built-in uncertainty heads for supported models:
- Pre-trained uncertainty heads from [llm-uncertainty-head](https://huggingface.co/llm-uncertainty-head)
- Multiple uncertainty methods (entropy, max probability, variance)
- Token-level uncertainty scoring

### 3. Backtracking System
Intelligent backtracking when uncertainty is high:
- Configurable uncertainty thresholds
- Multiple backtracking strategies
- Chain-of-thought integration

### 4. Prompt Refinement
Optional T5-based prompt refinement:
```bash
# Fine-tune prompt refiner on AskCQ dataset
python scripts/finetune_prompt_refiner.py \
    --data-dir data/processed_askcq \
    --output-dir models/prompt_refiner \
    --num-epochs 3
```

## 📊 Benchmarking & Evaluation

### Standard Benchmarking
```bash
# Full benchmark with both baseline and augmented models
python scripts/run_benchmark.py \
    --baseline-model "mistralai/Mistral-7B-Instruct-v0.2" \
    --datasets humaneval mbpp \
    --output-dir results/my_experiment \
    --max-problems 50
```

### Advanced Configuration
```bash
# Use configuration file for complex setups
python scripts/run_benchmark.py \
    --config config/benchmark_example.yaml \
    --baseline-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --output-dir results/llama_experiment
```

### Results & Analysis
Benchmarks generate comprehensive results:
- **JSON**: Detailed results with metadata (`benchmark_results.json`)
- **CSV**: Easy analysis format (`benchmark_results.csv`)
- **Qualitative Export**: Human evaluation format (`qualitative_evaluation.csv`)

## 🔬 Fine-tuning Components

### 1. Prompt Refiner Fine-tuning
Train a T5 model for prompt clarification:
```bash
# Preprocess AskCQ dataset
python scripts/preprocess_askcq.py \
    --input data/askcq.jsonl \
    --output data/processed_askcq

# Fine-tune T5 prompt refiner
python scripts/finetune_prompt_refiner.py \
    --data-dir data/processed_askcq \
    --output-dir models/prompt_refiner \
    --learning-rate 5e-4 \
    --num-epochs 3 \
    --batch-size 8
```

### 2. Training Configuration
Customize fine-tuning with extensive options:
- Learning rate scheduling
- Early stopping
- Mixed precision training
- Gradient accumulation
- Custom validation metrics

## 🛠️ Configuration System

### YAML Configuration
```yaml
# config/my_config.yaml
augmented_model:
  name: "mistralai/Mistral-7B-Instruct-v0.2"
  device: "cuda"
  load_in_8bit: false

uncertainty:
  enabled: true
  method: "entropy"
  threshold: 0.7

backtracking:
  enabled: true
  max_backtracks_per_generation: 5
  window_size: 3
  
prompt_refiner:
  enabled: true
  model_path: "models/prompt_refiner"

generation:
  max_length: 256
  temperature: 0.8
  top_p: 0.9
```

### Python Configuration
```python
from umin_framework.config import get_config, ConfigManager

# Load from file
config_manager = ConfigManager()
config = config_manager.load_config("config/my_config.yaml")

# Or use defaults
config = get_config()
```

## 📈 Usage Examples

### Basic Code Generation
```python
from umin_framework import AugmentedLLM, AugmentedLLMConfig

# Configure the model
config = AugmentedLLMConfig(
    generation_model="mistralai/Mistral-7B-Instral-v0.2"
)

# Initialize and generate
llm = AugmentedLLM(config=config)
result = llm.generate(
    "Write a Python function to find prime numbers",
    return_metadata=True
)

print(f"Generated code:\n{result['text']}")
print(f"Uncertainty score: {result['avg_uncertainty']:.3f}")
print(f"Backtrack events: {result['backtrack_events']}")
```

### Benchmark Comparison
```python
# See examples/run_benchmark_example.py for complete example
python examples/run_benchmark_example.py
```

### Qualitative Evaluation
```python
# See examples/qualitative_evaluation_demo.py
python examples/qualitative_evaluation_demo.py
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python run_tests.py

# Run specific test module
python -m pytest tests/test_augmented_llm.py -v
```

## 📚 Documentation

- **[Model Support](README_MODELS.md)**: Supported models and compatibility
- **[Detailed Model Guide](docs/supported_models.md)**: Complete model information
- **[Qualitative Evaluation](docs/qualitative_evaluation_guide.md)**: Human evaluation guidelines

## 💻 Hardware Requirements

### Minimum Requirements
- **GPU**: 16GB VRAM (for 7B models)
- **RAM**: 32GB system memory
- **Storage**: ~20GB for models and data

### Recommended Setup
- **GPU**: 24GB+ VRAM (RTX 4090, A100)
- **RAM**: 64GB+ system memory
- **Storage**: SSD with 50GB+ free space

### CPU-Only Mode
```python
config = AugmentedLLMConfig(
    generation_model="mistralai/Mistral-7B-Instruct-v0.2",
    device="cpu",
    load_in_8bit=True  # Reduce memory usage
)
```

## 🚨 Troubleshooting

### Common Issues

#### Model Not Supported
```
UnsupportedModelError: Model 'gpt2' is not supported
```
**Solution**: Use only supported models from [README_MODELS.md](README_MODELS.md)

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Use `load_in_8bit=True`
- Reduce `max_length`
- Use CPU: `device="cpu"`
- Use smaller models

#### Missing Dependencies
```
ImportError: No module named 'transformers'
```
**Solution**:
```bash
pip install -r requirements.txt
```

#### Access Denied for Models
```
Repository not found or access denied
```
**Solution** for gated models (like Llama):
```bash
huggingface-cli login
# Then request access on the model's Hugging Face page
```

## 🎯 Performance Tips

### GPU Optimization
```python
config = AugmentedLLMConfig(
    generation_model="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda",
    torch_dtype="float16",  # Half precision
    load_in_8bit=False      # Full precision for accuracy
)
```

### Memory Optimization
```python
config = AugmentedLLMConfig(
    generation_model="mistralai/Mistral-7B-Instruct-v0.2",
    load_in_8bit=True,      # 8-bit quantization
    max_length=128,         # Shorter sequences
    batch_size=1            # Smaller batches
)
```

### Speed Optimization
```bash
# Use fewer problems for quick testing
python scripts/run_benchmark.py \
    --baseline-model "mistralai/Mistral-7B-Instruct-v0.2" \
    --max-problems 10 \
    --no-augmented \
    --timeout 10
```

## 🔄 Workflow Examples

### Research Workflow
1. **Setup**: `make download`
2. **Quick Test**: Run with `--max-problems 5`
3. **Full Benchmark**: Run on complete datasets
4. **Analysis**: Review CSV exports
5. **Fine-tune**: Train prompt refiner if needed

### Development Workflow
1. **Environment**: Use virtual environment
2. **Testing**: Run `python run_tests.py`
3. **Debugging**: Use `--verbose` flag
4. **Validation**: Test with known working models

### Production Workflow
1. **Configuration**: Use YAML config files
2. **Monitoring**: Check logs in `logs/`
3. **Results**: Automated result storage
4. **Scaling**: Use GPU optimization

## 📝 Citation

If you use UMinFramework in your research, please cite:

```bibtex
@software{uminframework2024,
  title={UMinFramework: Uncertainty-Guided Code Generation},
  author={johanjohnthomas},
  year={2025},
  url={https://github.com/johanjohnthomas/UMinFramework}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [llm-uncertainty-head](https://huggingface.co/llm-uncertainty-head) for pre-trained uncertainty heads
- HumanEval and MBPP datasets for code generation benchmarks
- AskCQ dataset for prompt refinement training

---

**Quick Start Reminder**: Only use supported models listed in [README_MODELS.md](README_MODELS.md)!