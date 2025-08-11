#!/usr/bin/env python3
"""
Qualitative Evaluation Export Demonstration for UMinFramework.

This script demonstrates the qualitative evaluation export functionality
that creates detailed CSV files for human review of model comparisons.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from umin_framework.code_executor import ExecutionResult
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    sys.exit(1)

# Import the BenchmarkRunner from the scripts directory
scripts_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, scripts_path)

try:
    from run_benchmark import BenchmarkRunner, BenchmarkResult
    print("✅ Successfully imported BenchmarkRunner")
except ImportError as e:
    print(f"❌ Failed to import BenchmarkRunner: {e}")
    print("Make sure run_benchmark.py exists in the scripts directory")
    sys.exit(1)


def create_mock_benchmark_results() -> List[BenchmarkResult]:
    """
    Create mock benchmark results to demonstrate qualitative evaluation export.
    
    Returns:
        List of mock BenchmarkResult objects
    """
    results = []
    
    # Example 1: Both models pass
    result1 = BenchmarkResult(
        problem_id="HumanEval/0",
        dataset="humaneval",
        prompt='def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        expected_solution='    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n',
        test_case='assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True',
        baseline_model="gpt2",
        baseline_generated='    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False',
        baseline_passed=True,
        baseline_generation_time=1.2,
        baseline_tokens_generated=35,
        baseline_execution_result=ExecutionResult(
            success=True,
            output="All tests passed",
            error=None,
            execution_time=0.05
        ),
        augmented_generated='    # Let me think step by step about this problem\n    for i, num1 in enumerate(numbers):\n        for j, num2 in enumerate(numbers):\n            if i != j and abs(num1 - num2) < threshold:\n                return True\n    return False',
        augmented_passed=True,
        augmented_generation_time=2.1,
        augmented_tokens_generated=42,
        augmented_uncertainty_score=0.35,
        augmented_backtrack_events=1,
        augmented_prompt_refined=True,
        augmented_execution_result=ExecutionResult(
            success=True,
            output="All tests passed",
            error=None,
            execution_time=0.04
        ),
        timestamp="2024-01-15 10:30:00"
    )
    results.append(result1)
    
    # Example 2: Augmented better than baseline
    result2 = BenchmarkResult(
        problem_id="HumanEval/1",
        dataset="humaneval",
        prompt='def separate_paren_groups(paren_string: str) -> List[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those group and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups("( ) (( )) (( )( ))")\n    ["()", "(())", "(()())"]\n    """\n',
        expected_solution='    result = []\n    current_string = ""\n    current_depth = 0\n\n    for c in paren_string:\n        if c == "(":\n            current_depth += 1\n            current_string += c\n        elif c == ")":\n            current_depth -= 1\n            current_string += c\n\n            if current_depth == 0:\n                result.append(current_string)\n                current_string = ""\n\n    return result',
        test_case='assert separate_paren_groups("( ) (( )) (( )( ))") == ["()", "(())", "(()())"]',
        baseline_model="gpt2",
        baseline_generated='    groups = []\n    group = ""\n    depth = 0\n    for char in paren_string:\n        if char == "(":\n            group += char\n            depth += 1\n        elif char == ")":\n            depth -= 1\n            group += char\n    return groups  # Missing logic to add groups',
        baseline_passed=False,
        baseline_generation_time=0.8,
        baseline_tokens_generated=28,
        baseline_execution_result=ExecutionResult(
            success=False,
            output="",
            error="AssertionError: Expected [\"()\", \"(())\", \"(()())\"] but got []",
            execution_time=0.02
        ),
        augmented_generated='    # I need to track parentheses depth and group them\n    result = []\n    current_group = ""\n    depth = 0\n    \n    for char in paren_string:\n        if char == "(":\n            current_group += char\n            depth += 1\n        elif char == ")":\n            current_group += char\n            depth -= 1\n            if depth == 0:\n                result.append(current_group)\n                current_group = ""\n    \n    return result',
        augmented_passed=True,
        augmented_generation_time=2.5,
        augmented_tokens_generated=58,
        augmented_uncertainty_score=0.72,
        augmented_backtrack_events=2,
        augmented_prompt_refined=False,
        augmented_execution_result=ExecutionResult(
            success=True,
            output="All tests passed",
            error=None,
            execution_time=0.03
        ),
        timestamp="2024-01-15 10:31:30"
    )
    results.append(result2)
    
    # Example 3: Both models fail
    result3 = BenchmarkResult(
        problem_id="MBPP/1",
        dataset="mbpp",
        prompt='def triangle_area(a, b, c):\n    """\n    Calculate the area of a triangle given three side lengths using Heron\'s formula.\n    >>> triangle_area(3, 4, 5)\n    6.0\n    """\n',
        expected_solution='    s = (a + b + c) / 2\n    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5\n    return area',
        test_case='assert abs(triangle_area(3, 4, 5) - 6.0) < 0.001',
        baseline_model="gpt2",
        baseline_generated='    # Calculate area\n    return a * b / 2  # Wrong formula',
        baseline_passed=False,
        baseline_generation_time=0.6,
        baseline_tokens_generated=12,
        baseline_execution_result=ExecutionResult(
            success=False,
            output="",
            error="AssertionError: Expected ~6.0 but got 6.0",
            execution_time=0.01
        ),
        augmented_generated='    # Using triangle area formula\n    # Wait, let me reconsider this\n    # Actually, let me think about this more carefully\n    area = (a + b + c) / 2  # Still wrong - this is perimeter/2\n    return area',
        augmented_passed=False,
        augmented_generation_time=3.2,
        augmented_tokens_generated=35,
        augmented_uncertainty_score=0.85,
        augmented_backtrack_events=3,
        augmented_prompt_refined=True,
        augmented_execution_result=ExecutionResult(
            success=False,
            output="",
            error="AssertionError: Expected ~6.0 but got 6.0",
            execution_time=0.01
        ),
        timestamp="2024-01-15 10:32:45"
    )
    results.append(result3)
    
    return results


def demonstrate_qualitative_export():
    """Demonstrate the qualitative evaluation export functionality."""
    print("\n" + "="*60)
    print("QUALITATIVE EVALUATION EXPORT DEMONSTRATION")
    print("="*60)
    
    # Create output directory
    output_dir = Path("demo_qualitative_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Demo output directory: {output_dir.absolute()}")
    
    # Create a mock benchmark runner with results
    runner = BenchmarkRunner.__new__(BenchmarkRunner)  # Create without calling __init__
    runner.baseline_model = "gpt2"
    runner.output_dir = output_dir
    runner.results = create_mock_benchmark_results()
    
    # Set up logger manually
    import logging
    runner.logger = logging.getLogger('qualitative_demo')
    runner.logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    runner.logger.addHandler(handler)
    
    print(f"\nCreated {len(runner.results)} mock benchmark results:")
    for i, result in enumerate(runner.results, 1):
        baseline_status = "✓" if result.baseline_passed else "✗"
        augmented_status = "✓" if result.augmented_passed else "✗"
        print(f"  {i}. {result.problem_id} - Baseline: {baseline_status}, Augmented: {augmented_status}")
    
    # Generate qualitative evaluation export
    print("\n📊 Generating qualitative evaluation export...")
    runner.save_qualitative_evaluation_export()
    
    # Show what was created
    qualitative_file = output_dir / "qualitative_evaluation.csv"
    if qualitative_file.exists():
        file_size = qualitative_file.stat().st_size
        print(f"✅ Qualitative evaluation CSV created: {qualitative_file}")
        print(f"   File size: {file_size:,} bytes")
        
        # Show CSV structure
        print("\n📋 CSV File Structure:")
        with open(qualitative_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            columns = header.split(',')
            
            print(f"   Total columns: {len(columns)}")
            print("   Key columns:")
            
            key_columns = [
                'problem_id', 'evaluation_category', 
                'baseline_generated_code', 'augmented_generated_code',
                'performance_comparison', 'correctness_comparison',
                'human_preference', 'human_notes'
            ]
            
            for col in key_columns:
                if col in columns:
                    idx = columns.index(col) + 1
                    print(f"     • {col} (column {idx})")
    
    print(f"\n🎯 Analysis Categories Generated:")
    categories = set()
    for result in runner.results:
        if result.baseline_passed and result.augmented_passed:
            categories.add("both_passed")
        elif result.baseline_passed and not result.augmented_passed:
            categories.add("baseline_better")
        elif not result.baseline_passed and result.augmented_passed:
            categories.add("augmented_better")
        elif not result.baseline_passed and not result.augmented_passed:
            categories.add("both_failed")
    
    for category in sorted(categories):
        print(f"   • {category}")
    
    print(f"\n📝 Human Evaluation Instructions:")
    print("   1. Open the CSV file in a spreadsheet application")
    print("   2. Review the 'baseline_generated_code' and 'augmented_generated_code' columns")
    print("   3. Fill in the human evaluation columns:")
    print("      - human_baseline_score: Rate baseline code (1-10)")
    print("      - human_augmented_score: Rate augmented code (1-10)")  
    print("      - human_preference: 'baseline', 'augmented', or 'tie'")
    print("      - human_notes: Detailed observations and reasoning")
    print("      - review_status: Change from 'pending' to 'completed'")
    print("   4. Use the metadata columns for additional context")
    
    print(f"\n🔍 Quality Assessment Features:")
    print("   • Automatic code quality analysis")
    print("   • Performance comparison metrics")
    print("   • Execution output and error details")
    print("   • Uncertainty and backtracking information")
    print("   • Structured fields for human annotation")
    
    return output_dir


def analyze_export_structure(csv_file: Path):
    """Analyze and display the structure of the generated CSV."""
    print(f"\n📊 Analyzing CSV structure: {csv_file.name}")
    
    try:
        import csv
        with open(csv_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            if not fieldnames:
                print("   ❌ No fieldnames found")
                return
            
            # Group fields by category
            categories = {
                "Problem Info": ['problem_id', 'dataset', 'prompt_text', 'expected_solution', 'test_case'],
                "Baseline Results": [f for f in fieldnames if f.startswith('baseline_')],
                "Augmented Results": [f for f in fieldnames if f.startswith('augmented_')],
                "Comparative Analysis": ['performance_comparison', 'code_quality_comparison', 'correctness_comparison', 'approach_differences'],
                "Human Evaluation": ['human_baseline_score', 'human_augmented_score', 'human_preference', 'human_notes', 'review_status'],
                "Metadata": ['evaluation_category', 'timestamp', 'error_message']
            }
            
            print(f"   Total fields: {len(fieldnames)}")
            
            for category, fields in categories.items():
                matching_fields = [f for f in fields if f in fieldnames]
                if matching_fields:
                    print(f"\n   {category} ({len(matching_fields)} fields):")
                    for field in matching_fields:
                        print(f"     • {field}")
            
            # Count rows
            row_count = sum(1 for _ in reader)
            print(f"\n   Data rows: {row_count}")
            
    except Exception as e:
        print(f"   ❌ Error analyzing CSV: {e}")


def main():
    """Run the qualitative evaluation export demonstration."""
    print("UMinFramework Qualitative Evaluation Export Demo")
    print("=" * 50)
    
    try:
        output_dir = demonstrate_qualitative_export()
        
        # Analyze the generated CSV
        csv_file = output_dir / "qualitative_evaluation.csv"
        if csv_file.exists():
            analyze_export_structure(csv_file)
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Examine the generated files in: {output_dir.absolute()}")
        print(f"   2. Open qualitative_evaluation.csv in a spreadsheet application")
        print(f"   3. Practice filling in the human evaluation columns")
        print(f"   4. Use this format for real benchmark evaluations")
        
        print(f"\n🚀 Integration with Benchmarking:")
        print(f"   • Run benchmarks with --qualitative-export (default)")
        print(f"   • Use configuration files to control export behavior")
        print(f"   • Combine quantitative metrics with human evaluation")
        print(f"   • Scale qualitative review across research teams")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())