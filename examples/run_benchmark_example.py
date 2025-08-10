#!/usr/bin/env python3
"""
Example of running the benchmark script with sample data.

This script demonstrates how to use the benchmarking suite to compare
a baseline model against an AugmentedLLM on coding tasks.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from umin_framework import AugmentedLLMConfig
    print("✓ Successfully imported UMinFramework")
except ImportError as e:
    print(f"❌ Failed to import UMinFramework: {e}")
    print("Make sure transformers and torch are installed")
    sys.exit(1)


def run_simple_benchmark():
    """Run a simple benchmark test."""
    print("="*60)
    print("BENCHMARK EXAMPLE - Baseline Only")
    print("="*60)
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Run benchmark with baseline only (no AugmentedLLM)
    import subprocess
    
    cmd = [
        "python3",
        str(project_root / "scripts" / "run_benchmark.py"),
        "--baseline-model", "gpt2",
        "--data-path", str(project_root / "data"),
        "--output-dir", str(project_root / "results" / "example"),
        "--no-augmented",  # Skip AugmentedLLM to make it faster
        "--max-length", "100",
        "--temperature", "0.1",
        "--timeout", "10",
        "--verbose"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Benchmark completed successfully!")
        else:
            print(f"❌ Benchmark failed with return code: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Failed to run benchmark: {e}")
        return False


def run_augmented_benchmark():
    """Run a benchmark with AugmentedLLM if prompt refiner is available."""
    print("\n" + "="*60)
    print("BENCHMARK EXAMPLE - With AugmentedLLM")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    refiner_path = project_root / "models" / "prompt_refiner"
    
    if not refiner_path.exists():
        print(f"❌ Prompt refiner not found at {refiner_path}")
        print("Skipping AugmentedLLM benchmark")
        print("To enable: run `python scripts/finetune_prompt_refiner.py` first")
        return False
    
    print(f"✓ Found prompt refiner at {refiner_path}")
    
    cmd = [
        "python3",
        str(project_root / "scripts" / "run_benchmark.py"),
        "--baseline-model", "gpt2",
        "--augmented-model", "gpt2",
        "--prompt-refiner-model", str(refiner_path),
        "--data-path", str(project_root / "data"),
        "--output-dir", str(project_root / "results" / "example_augmented"),
        "--uncertainty-threshold", "0.8",
        "--backtrack-window", "2",
        "--max-length", "100",
        "--temperature", "0.1",
        "--timeout", "10",
        "--verbose"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Augmented benchmark completed successfully!")
        else:
            print(f"❌ Augmented benchmark failed with return code: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Failed to run augmented benchmark: {e}")
        return False


def analyze_results():
    """Analyze the generated benchmark results."""
    print("\n" + "="*60)
    print("ANALYZING RESULTS")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / "example"
    
    results_file = results_dir / "benchmark_results.json"
    csv_file = results_dir / "benchmark_results.csv"
    
    if results_file.exists():
        print(f"✓ Found results file: {results_file}")
        
        try:
            import json
            with open(results_file) as f:
                data = json.load(f)
            
            print("\nResults Summary:")
            print(f"  Total problems: {data['metadata']['total_results']}")
            print(f"  Baseline model: {data['metadata']['baseline_model']}")
            
            if 'statistics' in data and 'baseline' in data['statistics']:
                stats = data['statistics']['baseline']
                print(f"  Baseline pass rate: {stats.get('pass_rate', 0):.2%}")
                print(f"  Average generation time: {stats.get('avg_generation_time', 0):.2f}s")
            
            if 'pass_at_k' in data and 'baseline' in data['pass_at_k']:
                pass_k = data['pass_at_k']['baseline']['overall']
                print(f"  Pass@k metrics:")
                for k, score in pass_k.items():
                    print(f"    Pass@{k}: {score:.3f}")
            
        except Exception as e:
            print(f"❌ Error reading results: {e}")
    else:
        print(f"❌ Results file not found: {results_file}")
    
    if csv_file.exists():
        print(f"✓ Found CSV file: {csv_file}")
    else:
        print(f"❌ CSV file not found: {csv_file}")


if __name__ == "__main__":
    print("Benchmark Example Script")
    print("=======================")
    
    # Run basic benchmark
    success1 = run_simple_benchmark()
    
    # Run augmented benchmark if available
    success2 = run_augmented_benchmark()
    
    # Analyze results
    if success1 or success2:
        analyze_results()
    
    print("\n" + "="*60)
    if success1:
        print("✅ Basic benchmark example completed successfully!")
    if success2:
        print("✅ Augmented benchmark example completed successfully!")
    
    if not success1 and not success2:
        print("❌ No benchmarks completed successfully")
        print("This might be due to missing dependencies or model issues")
        print("Try installing: pip install transformers torch")
    
    print("="*60)