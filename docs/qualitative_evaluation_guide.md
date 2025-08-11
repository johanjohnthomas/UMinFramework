# Qualitative Evaluation Export Guide

## Overview

The UMinFramework includes a comprehensive qualitative evaluation export system that creates detailed CSV files for human review of model comparisons. This feature enables researchers to perform systematic qualitative analysis alongside quantitative metrics.

## Key Features

### Automated Export Generation
- **Paired Code Comparison**: Side-by-side baseline and augmented model outputs
- **Execution Results**: Detailed test execution outputs and error messages  
- **Performance Metrics**: Generation time, token counts, and success rates
- **Augmented Model Insights**: Uncertainty scores, backtrack events, and prompt refinement details

### Structured Human Evaluation Fields
- **Scoring Fields**: Numerical rating scales for code quality assessment
- **Preference Selection**: Systematic preference indication between models
- **Free-form Notes**: Detailed qualitative observations and reasoning
- **Review Status Tracking**: Workflow management for evaluation teams

### Automatic Code Quality Assessment
- **Structural Analysis**: Function definitions, loops, conditionals, error handling
- **Execution Analysis**: Success/failure status and error categorization
- **Comparative Analysis**: Performance differences and approach variations
- **Categorization**: Automatic problem categorization for evaluation focus

## Usage Examples

### Basic Usage with Benchmarking Script

```bash
# Enable qualitative export (default behavior)
python scripts/run_benchmark.py \
    --baseline-model mistralai/Mistral-7B-Instruct-v0.2 \
    --datasets humaneval \
    --max-problems 10 \
    --qualitative-export

# Disable qualitative export
python scripts/run_benchmark.py \
    --baseline-model mistralai/Mistral-7B-Instruct-v0.2 \
    --datasets humaneval \
    --no-qualitative-export
```

### Configuration File Usage

```yaml
# config/benchmark_example.yaml
benchmark:
  datasets: ["humaneval", "mbpp"]
  max_problems: 50
  qualitative_export: true
  export_qualitative_csv: true
  output_dir: "results/evaluation_study"
```

```bash
# Run with configuration file
python scripts/run_benchmark.py \
    --config config/benchmark_example.yaml \
    --baseline-model mistralai/Mistral-7B-Instruct-v0.2
```

### Programmatic Usage

```python
from scripts.run_benchmark import BenchmarkRunner

# Create runner with qualitative export enabled
runner = BenchmarkRunner(
    baseline_model="mistralai/Mistral-7B-Instruct-v0.2",
    dataset_path="data",
    output_dir="results",
    # ... other parameters
)

# Run benchmark and save results
stats = runner.run_benchmark(["humaneval"])
runner.save_results(qualitative_export=True)

# Export only qualitative data
runner.save_qualitative_evaluation_export("detailed_review.csv")
```

## CSV File Structure

### Problem Identification Columns
- `problem_id`: Unique identifier for the coding problem
- `dataset`: Source dataset (humaneval, mbpp, etc.)
- `prompt_text`: Full problem statement and requirements
- `expected_solution`: Reference solution for comparison
- `test_case`: Test cases used for validation
- `evaluation_category`: Automatic categorization (both_passed, augmented_better, etc.)

### Baseline Model Results
- `baseline_model`: Name of the baseline model
- `baseline_generated_code`: Generated code solution
- `baseline_passed`: Boolean test result
- `baseline_generation_time`: Time to generate solution
- `baseline_tokens_generated`: Number of tokens generated
- `baseline_execution_output`: Full execution output
- `baseline_execution_error`: Error messages if any
- `baseline_code_quality_notes`: Automatic quality assessment

### Augmented Model Results
- `augmented_generated_code`: Augmented model's code solution
- `augmented_passed`: Boolean test result
- `augmented_generation_time`: Time to generate solution
- `augmented_tokens_generated`: Number of tokens generated
- `augmented_execution_output`: Full execution output
- `augmented_execution_error`: Error messages if any
- `augmented_code_quality_notes`: Automatic quality assessment
- `augmented_uncertainty_score`: Average uncertainty during generation
- `augmented_backtrack_events`: Number of backtracking events
- `augmented_prompt_refined`: Whether prompt refinement was used

### Comparative Analysis
- `performance_comparison`: Automated performance comparison
- `code_quality_comparison`: Quality comparison notes
- `correctness_comparison`: Correctness assessment
- `approach_differences`: Identified differences in approach

### Human Evaluation Fields (Empty for Manual Completion)
- `human_baseline_score`: Numerical score for baseline (1-10)
- `human_augmented_score`: Numerical score for augmented (1-10)
- `human_preference`: Preference selection (baseline/augmented/tie)
- `human_notes`: Detailed qualitative observations
- `review_status`: Tracking field (pending/in_progress/completed)

## Human Evaluation Workflow

### 1. Setup and Preparation
1. Run benchmarks with qualitative export enabled
2. Open the generated CSV file in a spreadsheet application (Excel, Google Sheets, etc.)
3. Set up evaluation criteria and scoring rubrics
4. Assign problems to reviewers if working with a team

### 2. Code Review Process
1. **Problem Understanding**: Review `prompt_text` and `expected_solution`
2. **Code Analysis**: Compare `baseline_generated_code` and `augmented_generated_code`
3. **Execution Review**: Examine execution outputs and error messages
4. **Quality Assessment**: Consider factors like:
   - Correctness and functionality
   - Code clarity and readability  
   - Efficiency and approach
   - Error handling and edge cases
   - Use of appropriate algorithms

### 3. Scoring and Documentation
1. **Numerical Scores**: Rate each solution on a 1-10 scale
2. **Preference Selection**: Choose overall preference
3. **Detailed Notes**: Document specific observations:
   - Strengths and weaknesses of each approach
   - Specific bugs or issues identified
   - Quality differences in implementation
   - Insights about model behavior

### 4. Review Completion
1. Update `review_status` from "pending" to "completed"
2. Ensure all evaluation fields are filled
3. Validate consistency across similar problems
4. Export completed evaluations for analysis

## Evaluation Categories

The system automatically categorizes problems based on test results:

- **both_passed**: Both models generate correct solutions
- **augmented_better**: Only augmented model passes tests
- **baseline_better**: Only baseline model passes tests
- **both_failed**: Neither model generates correct solution
- **mixed_results**: Inconsistent or unclear results
- **both_failed_to_generate**: Neither model produced code

## Quality Assessment Features

### Automatic Code Analysis
The system provides automatic assessments including:
- **Structure Analysis**: Functions, classes, imports, loops, conditionals
- **Execution Analysis**: Success/failure status and error details
- **Length Analysis**: Code complexity indicators
- **Pattern Recognition**: Common coding patterns and practices

### Performance Comparison
Automated comparison of:
- **Generation Time**: Speed differences between models
- **Token Efficiency**: Relative verbosity of solutions
- **Success Rates**: Correctness comparison
- **Uncertainty Metrics**: Confidence indicators for augmented model

## Integration with Research Workflows

### Team-based Evaluation
- Distribute CSV files to multiple reviewers
- Use review_status for workflow tracking
- Aggregate scores and preferences for analysis
- Document inter-rater reliability metrics

### Data Analysis Integration
```python
import pandas as pd

# Load completed evaluations
df = pd.read_csv("results/qualitative_evaluation.csv")

# Filter completed reviews
completed = df[df['review_status'] == 'completed']

# Analyze preferences
preference_counts = completed['human_preference'].value_counts()
print(f"Augmented preferred: {preference_counts.get('augmented', 0)} times")

# Compare scores
score_diff = completed['human_augmented_score'] - completed['human_baseline_score']
print(f"Average score difference: {score_diff.mean():.2f}")

# Category analysis
category_preferences = completed.groupby('evaluation_category')['human_preference'].value_counts()
```

### Statistical Analysis
- Compare human preferences with automated metrics
- Analyze correlation between uncertainty and human scores
- Identify patterns in problem categories
- Generate insights for model improvement

## Best Practices

### For Individual Reviewers
1. **Consistent Criteria**: Establish clear scoring standards
2. **Blind Review**: Focus on code quality over model identity when possible
3. **Detailed Notes**: Provide specific, actionable feedback
4. **Regular Calibration**: Periodically review scoring consistency

### For Research Teams
1. **Inter-rater Reliability**: Have multiple reviewers evaluate subsets
2. **Calibration Sessions**: Align on scoring criteria across reviewers
3. **Systematic Sampling**: Ensure representative problem coverage
4. **Documentation**: Maintain detailed evaluation protocols

### For Data Quality
1. **Complete Reviews**: Ensure all fields are properly filled
2. **Validation Checks**: Verify consistency between scores and preferences
3. **Edge Case Documentation**: Note unusual or interesting cases
4. **Regular Backups**: Maintain version control for evaluation data

## Demo and Examples

Run the demonstration script to see the qualitative evaluation system in action:

```bash
python examples/qualitative_evaluation_demo.py
```

This will:
- Create mock benchmark results
- Generate a sample qualitative evaluation CSV
- Show the file structure and analysis capabilities
- Provide guidance for human evaluation workflows

The demo creates realistic examples of different evaluation scenarios to help you understand the system's capabilities and plan your evaluation strategy.