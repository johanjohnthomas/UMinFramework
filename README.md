# UMinFramework
A framework for minimizing uncertainty and hallucination in LLM generated outputs using uncertainty heads, backtracking, and Chain-of-Thought reasoning.

## Overview

UMinFramework provides a sophisticated approach to improving language model outputs by:

1. **Uncertainty Detection**: Uses uncertainty heads to identify when the model is uncertain about generated tokens
2. **Backtracking**: Removes uncertain tokens and backtracks to a more confident state
3. **Chain-of-Thought Triggering**: Automatically triggers CoT reasoning when uncertainty is detected
4. **Quality Improvement**: Iteratively improves output quality through multiple attempts

## Features

- 🎯 **Uncertainty-aware Generation**: Detects and handles uncertain tokens in real-time
- 🔄 **Intelligent Backtracking**: Removes uncertain content and tries alternative approaches
- 🧠 **CoT Integration**: Automatically triggers reasoning when uncertainty is detected
- ⚙️ **Configurable Framework**: Multiple pre-configured settings for different use cases
- 📊 **Comprehensive Evaluation**: Built-in tools for analyzing and comparing performance
- 🔍 **Detailed Analytics**: Track generation history and uncertainty patterns

## Requirements

- Python 3.11 required
- CUDA-capable GPU (recommended)
- PyTorch >= 1.9.0
- Transformers >= 4.20.0

## Installation

1. Activate the virtual environment:
   ```bash
   source bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Test the installation:
   ```bash
   python test_framework.py
   ```

   For a full test including model loading:
   ```bash
   python test_framework.py --full
   ```

## Quick Start

### Basic Usage

```python
from UMinFramework import UncertaintyMinimizer, get_config

# Load a pre-configured setup
config = get_config("balanced")

# Initialize the minimizer
minimizer = UncertaintyMinimizer(
    model_name=config.model_name,
    uhead_name=config.uhead_name,
    uncertainty_threshold=config.uncertainty_threshold,
    max_backtrack_attempts=config.max_backtrack_attempts,
    cot_trigger_token=config.cot_trigger_token
)

# Generate with uncertainty minimization
question = {"role": "user", "content": "How many fingers are on a koala's foot?"}
result = minimizer.generate_with_uncertainty_minimization(
    messages=[question],
    max_length=512,
    temperature=0.7
)

print(f"Final answer: {result['final_text']}")
print(f"Attempts needed: {result['total_attempts']}")
print(f"Uncertainty detected: {result['uncertainty_minimized']}")
```

### Configuration Options

The framework includes several pre-configured setups:

- **`conservative`**: Low uncertainty threshold, more CoT reasoning
- **`balanced`**: Balanced settings for general use
- **`aggressive`**: High uncertainty threshold, faster generation
- **`research`**: Detailed logging and analysis

```python
# Use a specific configuration
config = get_config("conservative")

# Or create a custom configuration
from UMinFramework import create_custom_config
custom_config = create_custom_config(
    uncertainty_threshold=0.3,
    max_backtrack_attempts=5,
    temperature=0.5,
    cot_trigger_token="<reasoning>"
)
```

## Advanced Usage

### Batch Processing

```python
# Process multiple questions
questions = [
    "What is the meaning of life?",
    "How does photosynthesis work?",
    "What would happen if you traveled faster than light?"
]

results = []
for question in questions:
    msg = {"role": "user", "content": question}
    result = minimizer.generate_with_uncertainty_minimization(messages=[msg])
    results.append(result)

# Analyze batch results
history = minimizer.analyze_generation_history()
print(f"Uncertainty minimization rate: {history['uncertainty_minimization_rate']:.2%}")
```

### Evaluation and Comparison

```python
from UMinFramework import UncertaintyEvaluator

# Compare different configurations
configs = {
    "conservative": get_config("conservative"),
    "balanced": get_config("balanced"),
    "aggressive": get_config("aggressive")
}

evaluator = UncertaintyEvaluator()
results = evaluator.compare_configurations(configs)

# Generate comparison report
report = evaluator.generate_comparison_report(results)
print(report)

# Create visualizations
evaluator.visualize_results(results, "comparison.png")
```

## Framework Architecture

### Core Components

1. **UncertaintyMinimizer**: Main class that orchestrates the uncertainty minimization process
2. **UncertaintyConfig**: Configuration management system
3. **UncertaintyEvaluator**: Evaluation and benchmarking tools
4. **Utilities**: Helper functions for analysis and visualization

### Process Flow

1. **Input Processing**: Convert user input to tokens
2. **Generation**: Generate tokens with uncertainty estimation
3. **Uncertainty Detection**: Identify uncertain tokens using the uncertainty head
4. **Backtracking**: Remove uncertain tokens and backtrack to a confident state
5. **CoT Triggering**: Add reasoning trigger tokens to prompt better thinking
6. **Iteration**: Repeat until confident generation or max attempts reached

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `uncertainty_threshold` | Threshold for detecting uncertain tokens | 0.5 |
| `max_backtrack_attempts` | Maximum number of backtracking attempts | 3 |
| `cot_trigger_token` | Token to trigger Chain-of-Thought reasoning | `<think>` |
| `temperature` | Generation temperature | 0.7 |
| `max_length` | Maximum generation length | 512 |

## Examples

### Running Examples

```bash
# Run basic examples
python example_usage.py

# Run comprehensive evaluation
python evaluation.py
```

### Custom Chain-of-Thought

```python
# Use custom CoT trigger
minimizer = UncertaintyMinimizer(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    uhead_name="llm-uncertainty-head/uhead_Mistral-7B-Instruct-v0.2",
    cot_trigger_token="<reasoning>",
    uncertainty_threshold=0.3
)
```

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- **Uncertainty Detection Rate**: Percentage of generations with detected uncertainty
- **Average Attempts**: Average number of attempts per generation
- **Success Rate**: Percentage of generations completed without uncertainty
- **CoT Usage Rate**: Percentage of generations using Chain-of-Thought
- **Generation Quality**: Length, coherence, and repetition metrics

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `max_length` or use smaller models
2. **Slow Generation**: Increase `uncertainty_threshold` or reduce `max_backtrack_attempts`
3. **No Uncertainty Detection**: Lower `uncertainty_threshold` or check model compatibility

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed logs of the uncertainty minimization process
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use UMinFramework in your research, please cite:

```bibtex
@software{uminframework,
  title={UMinFramework: A Framework for Minimizing Uncertainty in Language Model Outputs},
  author={UMinFramework Team},
  year={2024},
  url={https://github.com/your-username/UMinFramework}
}
```