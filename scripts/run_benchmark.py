#!/usr/bin/env python3
"""
Core benchmarking script for comparing baseline LLM vs AugmentedLLM performance.

This script evaluates both standard language models and AugmentedLLM instances
on coding benchmarks (HumanEval, MBPP), measuring pass@k metrics and collecting
detailed performance statistics.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from umin_framework import AugmentedLLM, AugmentedLLMConfig
    from umin_framework.code_executor import SafeCodeExecutor, PassAtKCalculator, ExecutionResult
    from umin_framework.config import UMinConfig, ConfigManager, LoggingSetup, get_config
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_REQUIRED_DEPS = True
except ImportError as e:
    logging.error(f"Required dependencies not available: {e}")
    HAS_REQUIRED_DEPS = False


@dataclass
class BenchmarkResult:
    """Data structure for storing benchmark results."""
    problem_id: str
    dataset: str
    prompt: str
    expected_solution: str
    test_case: str
    
    # Baseline model results
    baseline_model: str
    baseline_generated: str
    baseline_execution_result: Optional[ExecutionResult] = None
    baseline_passed: Optional[bool] = None
    baseline_generation_time: Optional[float] = None
    baseline_tokens_generated: Optional[int] = None
    
    # Augmented model results  
    augmented_generated: Optional[str] = None
    augmented_execution_result: Optional[ExecutionResult] = None
    augmented_passed: Optional[bool] = None
    augmented_generation_time: Optional[float] = None
    augmented_tokens_generated: Optional[int] = None
    augmented_uncertainty_score: Optional[float] = None
    augmented_backtrack_events: Optional[int] = None
    augmented_prompt_refined: Optional[bool] = None
    
    # Metadata
    timestamp: Optional[str] = None
    error_message: Optional[str] = None


class BenchmarkRunner:
    """Main class for running LLM benchmarks."""
    
    def __init__(
        self,
        baseline_model: str,
        dataset_path: str,
        output_dir: str,
        augmented_config: Optional[AugmentedLLMConfig] = None,
        max_problems: Optional[int] = None,
        max_length: int = 256,
        temperature: float = 0.2,
        timeout: float = 30.0
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            baseline_model: Name/path of baseline model to compare against
            dataset_path: Path to the dataset directory
            output_dir: Directory to save results
            augmented_config: Configuration for AugmentedLLM
            max_problems: Maximum number of problems to evaluate (None for all)
            max_length: Maximum generation length
            temperature: Generation temperature
            timeout: Timeout for code execution
        """
        self.baseline_model = baseline_model
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.max_problems = max_problems
        self.max_length = max_length
        self.temperature = temperature
        self.timeout = timeout
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize models
        self.baseline_tokenizer = None
        self.baseline_model_instance = None
        self.augmented_llm = None
        
        # Load baseline model
        self._load_baseline_model()
        
        # Load augmented model if config provided
        if augmented_config:
            self._load_augmented_model(augmented_config)
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        
        # Initialize code executor
        self.code_executor = SafeCodeExecutor(timeout=timeout)
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('benchmark')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / 'benchmark.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_baseline_model(self):
        """Load the baseline model."""
        try:
            self.logger.info(f"Loading baseline model: {self.baseline_model}")
            
            self.baseline_tokenizer = AutoTokenizer.from_pretrained(self.baseline_model)
            if self.baseline_tokenizer.pad_token is None:
                self.baseline_tokenizer.pad_token = self.baseline_tokenizer.eos_token
            
            self.baseline_model_instance = AutoModelForCausalLM.from_pretrained(
                self.baseline_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            self.logger.info("✓ Baseline model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load baseline model: {e}")
            raise
    
    def _load_augmented_model(self, config: AugmentedLLMConfig):
        """Load the augmented model."""
        try:
            self.logger.info("Loading AugmentedLLM...")
            
            # Update config with benchmark settings
            config.max_length = self.max_length
            config.temperature = self.temperature
            
            self.augmented_llm = AugmentedLLM(config=config)
            
            self.logger.info("✓ AugmentedLLM loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load AugmentedLLM: {e}")
            raise
    
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Load a dataset from JSONL file.
        
        Args:
            dataset_name: Name of the dataset ('humaneval' or 'mbpp')
            
        Returns:
            List of dataset problems
        """
        dataset_file = self.dataset_path / f"{dataset_name}.jsonl"
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        problems = []
        with open(dataset_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    problem = json.loads(line.strip())
                    problems.append(problem)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Skipping line {line_num} in {dataset_file}: {e}")
        
        self.logger.info(f"Loaded {len(problems)} problems from {dataset_name}")
        
        # Limit number of problems if specified
        if self.max_problems and len(problems) > self.max_problems:
            problems = problems[:self.max_problems]
            self.logger.info(f"Limited to first {self.max_problems} problems")
        
        return problems
    
    def generate_baseline_solution(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a solution using the baseline model.
        
        Args:
            prompt: The problem prompt
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.baseline_tokenizer.encode(
                prompt, 
                return_tensors="pt"
            ).to(self.baseline_model_instance.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.baseline_model_instance.generate(
                    inputs,
                    max_length=inputs.size(1) + self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.baseline_tokenizer.eos_token_id,
                    eos_token_id=self.baseline_tokenizer.eos_token_id,
                )
            
            # Decode generated text
            generated_text = self.baseline_tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Remove the original prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            generation_time = time.time() - start_time
            
            metadata = {
                'generation_time': generation_time,
                'tokens_generated': len(outputs[0]) - len(inputs[0])
            }
            
            return generated_text, metadata
            
        except Exception as e:
            self.logger.error(f"Baseline generation failed: {e}")
            return "", {
                'generation_time': time.time() - start_time,
                'error': str(e)
            }
    
    def generate_augmented_solution(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a solution using the AugmentedLLM.
        
        Args:
            prompt: The problem prompt
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        if self.augmented_llm is None:
            return "", {'error': 'AugmentedLLM not loaded'}
        
        start_time = time.time()
        
        try:
            result = self.augmented_llm.generate(
                prompt,
                return_metadata=True,
                max_length=self.max_length,
                temperature=self.temperature
            )
            
            generation_time = time.time() - start_time
            
            metadata = {
                'generation_time': generation_time,
                'tokens_generated': result.get('generated_tokens', 0),
                'uncertainty_score': result.get('avg_uncertainty', 0.0),
                'backtrack_events': result.get('backtrack_events', 0),
                'prompt_refined': result.get('prompt_was_refined', False)
            }
            
            return result['text'], metadata
            
        except Exception as e:
            self.logger.error(f"Augmented generation failed: {e}")
            return "", {
                'generation_time': time.time() - start_time,
                'error': str(e)
            }
    
    def execute_code(self, code: str, test_case: str = "") -> ExecutionResult:
        """
        Execute generated code and test it using safe sandbox.
        
        Args:
            code: Generated code to execute
            test_case: Test case to run
            
        Returns:
            ExecutionResult with detailed execution information
        """
        return self.code_executor.execute(code, test_case)
    
    def evaluate_problem(self, problem: Dict[str, Any]) -> BenchmarkResult:
        """
        Evaluate a single problem with both models.
        
        Args:
            problem: Problem dictionary from dataset
            
        Returns:
            BenchmarkResult with evaluation results
        """
        problem_id = problem.get('id', problem.get('task_id', 'unknown'))
        dataset = problem.get('dataset', 'unknown')
        prompt = problem.get('prompt', '')
        expected_solution = problem.get('canonical_solution', problem.get('code', ''))
        test_case = problem.get('test', '')
        
        self.logger.info(f"Evaluating problem: {problem_id}")
        
        # Initialize result
        result = BenchmarkResult(
            problem_id=problem_id,
            dataset=dataset,
            prompt=prompt,
            expected_solution=expected_solution,
            test_case=test_case,
            baseline_model=self.baseline_model,
            baseline_generated="",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        try:
            # Generate baseline solution
            self.logger.debug(f"Generating baseline solution for {problem_id}")
            baseline_code, baseline_meta = self.generate_baseline_solution(prompt)
            
            result.baseline_generated = baseline_code
            result.baseline_generation_time = baseline_meta.get('generation_time')
            result.baseline_tokens_generated = baseline_meta.get('tokens_generated')
            
            # Test baseline solution
            if baseline_code.strip():
                exec_result = self.execute_code(baseline_code, test_case)
                result.baseline_passed = exec_result.success
                result.baseline_execution_result = exec_result
            else:
                result.baseline_passed = False
                result.baseline_execution_result = ExecutionResult(
                    success=False, 
                    error="No code generated"
                )
            
            # Generate augmented solution if available
            if self.augmented_llm:
                self.logger.debug(f"Generating augmented solution for {problem_id}")
                augmented_code, augmented_meta = self.generate_augmented_solution(prompt)
                
                result.augmented_generated = augmented_code
                result.augmented_generation_time = augmented_meta.get('generation_time')
                result.augmented_tokens_generated = augmented_meta.get('tokens_generated')
                result.augmented_uncertainty_score = augmented_meta.get('uncertainty_score')
                result.augmented_backtrack_events = augmented_meta.get('backtrack_events')
                result.augmented_prompt_refined = augmented_meta.get('prompt_refined')
                
                # Test augmented solution
                if augmented_code.strip():
                    exec_result = self.execute_code(augmented_code, test_case)
                    result.augmented_passed = exec_result.success
                    result.augmented_execution_result = exec_result
                else:
                    result.augmented_passed = False
                    result.augmented_execution_result = ExecutionResult(
                        success=False,
                        error="No code generated"
                    )
            
        except Exception as e:
            self.logger.error(f"Error evaluating problem {problem_id}: {e}")
            result.error_message = str(e)
        
        return result
    
    def run_benchmark(self, dataset_names: List[str]) -> Dict[str, Any]:
        """
        Run the complete benchmark evaluation.
        
        Args:
            dataset_names: List of dataset names to evaluate
            
        Returns:
            Dictionary with benchmark statistics
        """
        self.logger.info("Starting benchmark evaluation")
        start_time = time.time()
        
        total_problems = 0
        
        for dataset_name in dataset_names:
            self.logger.info(f"Processing dataset: {dataset_name}")
            
            try:
                problems = self.load_dataset(dataset_name)
                total_problems += len(problems)
                
                for i, problem in enumerate(problems, 1):
                    self.logger.info(f"Problem {i}/{len(problems)} in {dataset_name}")
                    result = self.evaluate_problem(problem)
                    self.results.append(result)
                    
                    # Log progress
                    if result.baseline_passed is not None:
                        baseline_status = "✓" if result.baseline_passed else "✗"
                    else:
                        baseline_status = "?"
                        
                    if result.augmented_passed is not None:
                        augmented_status = "✓" if result.augmented_passed else "✗"
                    else:
                        augmented_status = "N/A"
                    
                    self.logger.info(f"  Baseline: {baseline_status}, Augmented: {augmented_status}")
                    
            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset_name}: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        stats = self._calculate_statistics()
        stats.update({
            'total_problems': total_problems,
            'total_time': total_time,
            'problems_per_second': total_problems / total_time if total_time > 0 else 0
        })
        
        self.logger.info("Benchmark evaluation completed")
        return stats
    
    def calculate_pass_at_k(self, k_values: List[int] = None) -> Dict[str, Any]:
        """
        Calculate pass@k metrics for the benchmark results.
        
        Args:
            k_values: List of k values to calculate (default: [1, 3, 5])
            
        Returns:
            Dictionary with pass@k statistics
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        # Group results by problem and dataset
        baseline_by_problem = {}
        augmented_by_problem = {}
        
        for result in self.results:
            problem_key = f"{result.dataset}_{result.problem_id}"
            
            # Baseline results
            if result.baseline_passed is not None:
                if problem_key not in baseline_by_problem:
                    baseline_by_problem[problem_key] = []
                baseline_by_problem[problem_key].append(result.baseline_passed)
            
            # Augmented results  
            if result.augmented_passed is not None:
                if problem_key not in augmented_by_problem:
                    augmented_by_problem[problem_key] = []
                augmented_by_problem[problem_key].append(result.augmented_passed)
        
        # Calculate pass@k for baseline
        baseline_pass_at_k = {}
        if baseline_by_problem:
            # Aggregate all results across problems
            all_baseline_results = []
            for problem_results in baseline_by_problem.values():
                all_baseline_results.extend(problem_results)
            
            baseline_pass_at_k = PassAtKCalculator.calculate_multiple_k(
                all_baseline_results, k_values
            )
        
        # Calculate pass@k for augmented
        augmented_pass_at_k = {}
        if augmented_by_problem:
            # Aggregate all results across problems
            all_augmented_results = []
            for problem_results in augmented_by_problem.values():
                all_augmented_results.extend(problem_results)
            
            augmented_pass_at_k = PassAtKCalculator.calculate_multiple_k(
                all_augmented_results, k_values
            )
        
        # Calculate by-problem pass@k
        baseline_by_problem_pass_at_k = PassAtKCalculator.calculate_by_problem(
            baseline_by_problem, k_values
        )
        augmented_by_problem_pass_at_k = PassAtKCalculator.calculate_by_problem(
            augmented_by_problem, k_values
        )
        
        return {
            'k_values': k_values,
            'baseline': {
                'overall': baseline_pass_at_k,
                'by_problem': baseline_by_problem_pass_at_k
            },
            'augmented': {
                'overall': augmented_pass_at_k,
                'by_problem': augmented_by_problem_pass_at_k
            },
            'problem_counts': {
                'baseline': len(baseline_by_problem),
                'augmented': len(augmented_by_problem)
            }
        }
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate benchmark statistics."""
        if not self.results:
            return {}
        
        # Filter valid results
        valid_baseline = [r for r in self.results if r.baseline_passed is not None]
        valid_augmented = [r for r in self.results if r.augmented_passed is not None]
        
        # Baseline statistics
        baseline_stats = {}
        if valid_baseline:
            baseline_passed = sum(1 for r in valid_baseline if r.baseline_passed)
            baseline_stats = {
                'total_evaluated': len(valid_baseline),
                'passed': baseline_passed,
                'pass_rate': baseline_passed / len(valid_baseline),
                'avg_generation_time': sum(r.baseline_generation_time or 0 for r in valid_baseline) / len(valid_baseline),
                'avg_tokens': sum(r.baseline_tokens_generated or 0 for r in valid_baseline) / len(valid_baseline)
            }
        
        # Augmented statistics
        augmented_stats = {}
        if valid_augmented:
            augmented_passed = sum(1 for r in valid_augmented if r.augmented_passed)
            augmented_stats = {
                'total_evaluated': len(valid_augmented),
                'passed': augmented_passed,
                'pass_rate': augmented_passed / len(valid_augmented),
                'avg_generation_time': sum(r.augmented_generation_time or 0 for r in valid_augmented) / len(valid_augmented),
                'avg_tokens': sum(r.augmented_tokens_generated or 0 for r in valid_augmented) / len(valid_augmented),
                'avg_uncertainty': sum(r.augmented_uncertainty_score or 0 for r in valid_augmented) / len(valid_augmented),
                'avg_backtrack_events': sum(r.augmented_backtrack_events or 0 for r in valid_augmented) / len(valid_augmented),
                'problems_with_refinement': sum(1 for r in valid_augmented if r.augmented_prompt_refined)
            }
        
        return {
            'baseline': baseline_stats,
            'augmented': augmented_stats,
            'total_results': len(self.results)
        }
    
    def save_results(self, filename: str = "benchmark_results.json", qualitative_export: bool = True):
        """Save benchmark results to file."""
        results_file = self.output_dir / filename
        
        # Convert results to dict format, handling ExecutionResult objects
        results_list = []
        for result in self.results:
            result_dict = asdict(result)
            
            # Convert ExecutionResult objects to dictionaries
            if isinstance(result.baseline_execution_result, ExecutionResult):
                result_dict['baseline_execution_result'] = result.baseline_execution_result.to_dict()
            
            if isinstance(result.augmented_execution_result, ExecutionResult):
                result_dict['augmented_execution_result'] = result.augmented_execution_result.to_dict()
            
            results_list.append(result_dict)
        
        # Calculate pass@k metrics
        pass_at_k_stats = self.calculate_pass_at_k()
        
        results_data = {
            'metadata': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'baseline_model': self.baseline_model,
                'max_length': self.max_length,
                'temperature': self.temperature,
                'timeout': self.timeout,
                'total_results': len(self.results)
            },
            'statistics': self._calculate_statistics(),
            'pass_at_k': pass_at_k_stats,
            'results': results_list
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_file}")
        
        # Also save a CSV for easier analysis
        self._save_csv_results()
        
        # Save qualitative evaluation export if requested
        if qualitative_export:
            self.save_qualitative_evaluation_export()
    
    def _save_csv_results(self):
        """Save results in CSV format for easier analysis."""
        import csv
        
        csv_file = self.output_dir / "benchmark_results.csv"
        
        if not self.results:
            return
        
        fieldnames = [
            'problem_id', 'dataset', 'baseline_model', 
            'baseline_passed', 'baseline_generation_time', 'baseline_tokens_generated',
            'baseline_execution_time', 'baseline_execution_error',
            'augmented_passed', 'augmented_generation_time', 'augmented_tokens_generated',
            'augmented_execution_time', 'augmented_execution_error',
            'augmented_uncertainty_score', 'augmented_backtrack_events', 'augmented_prompt_refined',
            'timestamp', 'error_message'
        ]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                # Extract execution details
                baseline_exec_time = None
                baseline_exec_error = None
                if isinstance(result.baseline_execution_result, ExecutionResult):
                    baseline_exec_time = result.baseline_execution_result.execution_time
                    baseline_exec_error = result.baseline_execution_result.error
                
                augmented_exec_time = None
                augmented_exec_error = None
                if isinstance(result.augmented_execution_result, ExecutionResult):
                    augmented_exec_time = result.augmented_execution_result.execution_time
                    augmented_exec_error = result.augmented_execution_result.error
                
                row = {
                    'problem_id': result.problem_id,
                    'dataset': result.dataset,
                    'baseline_model': result.baseline_model,
                    'baseline_passed': result.baseline_passed,
                    'baseline_generation_time': result.baseline_generation_time,
                    'baseline_tokens_generated': result.baseline_tokens_generated,
                    'baseline_execution_time': baseline_exec_time,
                    'baseline_execution_error': baseline_exec_error,
                    'augmented_passed': result.augmented_passed,
                    'augmented_generation_time': result.augmented_generation_time,
                    'augmented_tokens_generated': result.augmented_tokens_generated,
                    'augmented_execution_time': augmented_exec_time,
                    'augmented_execution_error': augmented_exec_error,
                    'augmented_uncertainty_score': result.augmented_uncertainty_score,
                    'augmented_backtrack_events': result.augmented_backtrack_events,
                    'augmented_prompt_refined': result.augmented_prompt_refined,
                    'timestamp': result.timestamp,
                    'error_message': result.error_message
                }
                writer.writerow(row)
        
        self.logger.info(f"CSV results saved to: {csv_file}")
    
    def save_qualitative_evaluation_export(self, filename: str = "qualitative_evaluation.csv"):
        """
        Save detailed qualitative evaluation data for human review.
        
        This includes paired code snippets, metadata, and structured data
        for manual evaluation of model outputs.
        
        Args:
            filename: Name of the CSV file to save
        """
        import csv
        
        csv_file = self.output_dir / filename
        
        if not self.results:
            self.logger.warning("No results to export for qualitative evaluation")
            return
        
        # Define comprehensive fieldnames for qualitative review
        fieldnames = [
            # Problem identification
            'problem_id', 'dataset', 'prompt_text', 'expected_solution',
            'test_case', 'evaluation_category',
            
            # Baseline model results
            'baseline_model', 'baseline_generated_code', 'baseline_passed', 
            'baseline_generation_time', 'baseline_tokens_generated',
            'baseline_execution_output', 'baseline_execution_error',
            'baseline_execution_time', 'baseline_code_quality_notes',
            
            # Augmented model results
            'augmented_generated_code', 'augmented_passed', 
            'augmented_generation_time', 'augmented_tokens_generated',
            'augmented_execution_output', 'augmented_execution_error',
            'augmented_execution_time', 'augmented_code_quality_notes',
            
            # Augmented model specific metrics
            'augmented_uncertainty_score', 'augmented_backtrack_events',
            'augmented_prompt_refined', 'augmented_refinement_details',
            
            # Comparative analysis
            'performance_comparison', 'code_quality_comparison',
            'correctness_comparison', 'approach_differences',
            
            # Human evaluation fields (empty for manual filling)
            'human_baseline_score', 'human_augmented_score',
            'human_preference', 'human_notes', 'review_status',
            
            # Metadata
            'timestamp', 'error_message'
        ]
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                # Prepare code quality assessment
                baseline_code_notes = self._assess_code_quality(
                    result.baseline_generated, 
                    result.baseline_execution_result
                )
                
                augmented_code_notes = ""
                augmented_refinement_details = ""
                if result.augmented_generated:
                    augmented_code_notes = self._assess_code_quality(
                        result.augmented_generated,
                        result.augmented_execution_result
                    )
                    
                    # Add refinement details if available
                    if result.augmented_prompt_refined:
                        augmented_refinement_details = "Prompt was refined; see generation logs for details"
                
                # Determine evaluation category
                eval_category = self._determine_evaluation_category(result)
                
                # Perform comparative analysis
                performance_comp, quality_comp, correctness_comp, approach_diff = self._compare_results(result)
                
                # Extract execution details with full output
                baseline_output, baseline_error = self._extract_execution_details(result.baseline_execution_result)
                augmented_output, augmented_error = self._extract_execution_details(result.augmented_execution_result)
                
                row = {
                    # Problem identification
                    'problem_id': result.problem_id,
                    'dataset': result.dataset,
                    'prompt_text': self._clean_text_for_csv(result.prompt),
                    'expected_solution': self._clean_text_for_csv(result.expected_solution),
                    'test_case': self._clean_text_for_csv(result.test_case),
                    'evaluation_category': eval_category,
                    
                    # Baseline results
                    'baseline_model': result.baseline_model,
                    'baseline_generated_code': self._clean_text_for_csv(result.baseline_generated),
                    'baseline_passed': result.baseline_passed,
                    'baseline_generation_time': result.baseline_generation_time,
                    'baseline_tokens_generated': result.baseline_tokens_generated,
                    'baseline_execution_output': baseline_output,
                    'baseline_execution_error': baseline_error,
                    'baseline_execution_time': getattr(result.baseline_execution_result, 'execution_time', None),
                    'baseline_code_quality_notes': baseline_code_notes,
                    
                    # Augmented results
                    'augmented_generated_code': self._clean_text_for_csv(result.augmented_generated or ""),
                    'augmented_passed': result.augmented_passed,
                    'augmented_generation_time': result.augmented_generation_time,
                    'augmented_tokens_generated': result.augmented_tokens_generated,
                    'augmented_execution_output': augmented_output,
                    'augmented_execution_error': augmented_error,
                    'augmented_execution_time': getattr(result.augmented_execution_result, 'execution_time', None) if result.augmented_execution_result else None,
                    'augmented_code_quality_notes': augmented_code_notes,
                    
                    # Augmented model specific metrics
                    'augmented_uncertainty_score': result.augmented_uncertainty_score,
                    'augmented_backtrack_events': result.augmented_backtrack_events,
                    'augmented_prompt_refined': result.augmented_prompt_refined,
                    'augmented_refinement_details': augmented_refinement_details,
                    
                    # Comparative analysis
                    'performance_comparison': performance_comp,
                    'code_quality_comparison': quality_comp,
                    'correctness_comparison': correctness_comp,
                    'approach_differences': approach_diff,
                    
                    # Human evaluation fields (empty for manual completion)
                    'human_baseline_score': '',
                    'human_augmented_score': '',
                    'human_preference': '',
                    'human_notes': '',
                    'review_status': 'pending',
                    
                    # Metadata
                    'timestamp': result.timestamp,
                    'error_message': result.error_message or ''
                }
                
                writer.writerow(row)
        
        self.logger.info(f"Qualitative evaluation export saved to: {csv_file}")
        self.logger.info("Human reviewers can now evaluate the paired code snippets and fill in evaluation fields")
    
    def _assess_code_quality(self, code: str, execution_result: Optional[ExecutionResult]) -> str:
        """
        Assess code quality and provide notes for human review.
        
        Args:
            code: Generated code to assess
            execution_result: Result of code execution
            
        Returns:
            String with quality assessment notes
        """
        if not code or not code.strip():
            return "No code generated"
        
        notes = []
        
        # Basic code structure assessment
        if len(code.splitlines()) == 1:
            notes.append("Single line solution")
        elif len(code.splitlines()) > 50:
            notes.append("Very long solution")
        
        # Check for common patterns
        if "def " in code:
            notes.append("Defines function(s)")
        if "class " in code:
            notes.append("Defines class(es)")
        if "import " in code or "from " in code:
            notes.append("Uses imports")
        if "for " in code or "while " in code:
            notes.append("Contains loops")
        if "if " in code:
            notes.append("Contains conditionals")
        if "try:" in code or "except" in code:
            notes.append("Has error handling")
        
        # Execution assessment
        if execution_result:
            if execution_result.success:
                notes.append("Executes successfully")
            else:
                notes.append(f"Execution failed: {execution_result.error}")
                
        return "; ".join(notes) if notes else "Basic code structure"
    
    def _determine_evaluation_category(self, result: BenchmarkResult) -> str:
        """
        Categorize the problem for evaluation purposes.
        
        Args:
            result: Benchmark result to categorize
            
        Returns:
            Category string for human evaluation
        """
        baseline_passed = result.baseline_passed
        augmented_passed = result.augmented_passed
        
        if baseline_passed is None and augmented_passed is None:
            return "both_failed_to_generate"
        elif baseline_passed and augmented_passed:
            return "both_passed"
        elif baseline_passed and not augmented_passed:
            return "baseline_better"
        elif not baseline_passed and augmented_passed:
            return "augmented_better"
        elif baseline_passed is False and augmented_passed is False:
            return "both_failed"
        else:
            return "mixed_results"
    
    def _compare_results(self, result: BenchmarkResult) -> Tuple[str, str, str, str]:
        """
        Compare baseline and augmented results for qualitative analysis.
        
        Args:
            result: Benchmark result to compare
            
        Returns:
            Tuple of (performance_comparison, quality_comparison, correctness_comparison, approach_differences)
        """
        # Performance comparison
        perf_comp = "N/A"
        if result.baseline_generation_time and result.augmented_generation_time:
            baseline_time = result.baseline_generation_time
            augmented_time = result.augmented_generation_time
            
            time_diff = ((augmented_time - baseline_time) / baseline_time) * 100
            if abs(time_diff) < 10:
                perf_comp = "Similar generation time"
            elif time_diff > 10:
                perf_comp = f"Augmented {time_diff:.1f}% slower"
            else:
                perf_comp = f"Augmented {abs(time_diff):.1f}% faster"
        
        # Quality comparison (basic heuristics)
        quality_comp = "Requires human evaluation"
        
        # Correctness comparison
        correctness_comp = "N/A"
        if result.baseline_passed is not None and result.augmented_passed is not None:
            if result.baseline_passed and result.augmented_passed:
                correctness_comp = "Both correct"
            elif result.baseline_passed and not result.augmented_passed:
                correctness_comp = "Baseline correct, augmented incorrect"
            elif not result.baseline_passed and result.augmented_passed:
                correctness_comp = "Augmented correct, baseline incorrect"
            else:
                correctness_comp = "Both incorrect"
        
        # Approach differences
        approach_diff = "Code comparison required"
        if result.baseline_generated and result.augmented_generated:
            baseline_len = len(result.baseline_generated.splitlines())
            augmented_len = len(result.augmented_generated.splitlines())
            
            if abs(baseline_len - augmented_len) > 5:
                approach_diff = f"Different code lengths (baseline: {baseline_len}, augmented: {augmented_len} lines)"
        
        return perf_comp, quality_comp, correctness_comp, approach_diff
    
    def _extract_execution_details(self, execution_result: Optional[ExecutionResult]) -> Tuple[str, str]:
        """
        Extract execution output and error details.
        
        Args:
            execution_result: Execution result to extract from
            
        Returns:
            Tuple of (output, error) strings
        """
        if not execution_result:
            return "", ""
        
        output = getattr(execution_result, 'output', '') or ''
        error = getattr(execution_result, 'error', '') or ''
        
        # Clean and truncate for CSV
        output = self._clean_text_for_csv(output)
        error = self._clean_text_for_csv(error)
        
        return output, error
    
    def _clean_text_for_csv(self, text: str) -> str:
        """
        Clean text for CSV export by handling newlines and special characters.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text suitable for CSV
        """
        if not text:
            return ""
        
        # Replace problematic characters
        cleaned = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        
        # Truncate very long text
        if len(cleaned) > 2000:
            cleaned = cleaned[:1997] + "..."
        
        return cleaned


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark baseline LLM vs AugmentedLLM on coding tasks"
    )
    
    # Required arguments
    parser.add_argument(
        "--baseline-model", 
        required=True,
        help="Name or path of the baseline model (e.g., 'mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Meta-Llama-3.1-8B-Instruct')"
    )
    
    parser.add_argument(
        "--data-path", 
        default="data",
        help="Path to the dataset directory (default: data)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="results",
        help="Output directory for results (default: results)"
    )
    
    # Dataset selection
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["humaneval", "mbpp"],
        help="Datasets to evaluate on (default: humaneval mbpp)"
    )
    
    # Configuration file support
    parser.add_argument(
        "--config",
        help="Path to YAML/JSON configuration file"
    )
    
    # AugmentedLLM configuration
    parser.add_argument(
        "--augmented-model",
        help="Base model for AugmentedLLM (if different from baseline)"
    )
    
    parser.add_argument(
        "--prompt-refiner-model",
        help="Path to prompt refiner model"
    )
    
    parser.add_argument(
        "--no-augmented", 
        action="store_true",
        help="Skip AugmentedLLM evaluation (baseline only)"
    )
    
    parser.add_argument(
        "--uncertainty-threshold", 
        type=float, 
        default=0.7,
        help="Uncertainty threshold for backtracking (default: 0.7)"
    )
    
    parser.add_argument(
        "--backtrack-window", 
        type=int, 
        default=3,
        help="Number of tokens to backtrack (default: 3)"
    )
    
    # Generation settings
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=256,
        help="Maximum generation length (default: 256)"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.2,
        help="Generation temperature (default: 0.2)"
    )
    
    parser.add_argument(
        "--max-problems", 
        type=int,
        help="Maximum number of problems to evaluate per dataset"
    )
    
    # Execution settings
    parser.add_argument(
        "--timeout", 
        type=float, 
        default=30.0,
        help="Code execution timeout in seconds (default: 30)"
    )
    
    # Qualitative evaluation
    parser.add_argument(
        "--qualitative-export",
        action="store_true",
        default=True,
        help="Export detailed qualitative evaluation CSV for human review (default: True)"
    )
    
    parser.add_argument(
        "--no-qualitative-export",
        action="store_true",
        help="Skip qualitative evaluation export"
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not HAS_REQUIRED_DEPS:
        print("❌ Required dependencies not available. Please install:")
        print("pip install transformers torch")
        return 1
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration if provided
        config_manager = None
        framework_config = None
        
        if args.config:
            config_manager = ConfigManager()
            framework_config = config_manager.load_config(args.config)
            
            # Set up logging from configuration
            LoggingSetup.setup_logging(framework_config.logging)
            logger = logging.getLogger('benchmark')
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            # Use default configuration
            framework_config = get_config()
            logger = logging.getLogger('benchmark')
        
        # Create AugmentedLLM configuration if not skipped
        augmented_config = None
        if not args.no_augmented:
            if framework_config:
                # Use framework configuration with CLI overrides
                augmented_config = AugmentedLLMConfig(
                    generation_model=args.augmented_model or args.baseline_model or framework_config.augmented_model.name,
                    prompt_refiner_model=args.prompt_refiner_model or framework_config.prompt_refiner.model_path,
                    enable_prompt_refinement=bool(args.prompt_refiner_model) or framework_config.prompt_refiner.enabled,
                    uncertainty_threshold=args.uncertainty_threshold or framework_config.uncertainty.threshold,
                    backtrack_window=args.backtrack_window or framework_config.backtracking.window_size,
                    max_length=args.max_length or framework_config.generation.max_length,
                    temperature=args.temperature or framework_config.generation.temperature
                )
            else:
                # Fallback to CLI arguments only
                augmented_config = AugmentedLLMConfig(
                    generation_model=args.augmented_model or args.baseline_model,
                    prompt_refiner_model=args.prompt_refiner_model,
                    enable_prompt_refinement=bool(args.prompt_refiner_model),
                    uncertainty_threshold=args.uncertainty_threshold,
                    backtrack_window=args.backtrack_window,
                    max_length=args.max_length,
                    temperature=args.temperature
                )
        
        # Initialize benchmark runner
        runner = BenchmarkRunner(
            baseline_model=args.baseline_model,
            dataset_path=args.data_path,
            output_dir=args.output_dir,
            augmented_config=augmented_config,
            max_problems=args.max_problems,
            max_length=args.max_length,
            temperature=args.temperature,
            timeout=args.timeout
        )
        
        # Run benchmark
        stats = runner.run_benchmark(args.datasets)
        
        # Save results (with qualitative export based on CLI args and config)
        qualitative_export = args.qualitative_export and not args.no_qualitative_export
        
        # Override with framework config if available
        if framework_config and hasattr(framework_config, 'benchmark'):
            config_export = getattr(framework_config.benchmark, 'export_qualitative_csv', True)
            qualitative_export = qualitative_export and config_export
        
        runner.save_results(qualitative_export=qualitative_export)
        
        # Calculate pass@k metrics
        pass_at_k_stats = runner.calculate_pass_at_k()
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        if 'baseline' in stats and stats['baseline']:
            baseline_stats = stats['baseline']
            print(f"Baseline Model ({args.baseline_model}):")
            print(f"  Problems evaluated: {baseline_stats['total_evaluated']}")
            print(f"  Passed: {baseline_stats['passed']}")
            print(f"  Pass rate: {baseline_stats['pass_rate']:.2%}")
            print(f"  Avg generation time: {baseline_stats['avg_generation_time']:.2f}s")
            print(f"  Avg tokens generated: {baseline_stats['avg_tokens']:.1f}")
        
        if 'augmented' in stats and stats['augmented']:
            augmented_stats = stats['augmented']
            print(f"\nAugmented Model:")
            print(f"  Problems evaluated: {augmented_stats['total_evaluated']}")
            print(f"  Passed: {augmented_stats['passed']}")
            print(f"  Pass rate: {augmented_stats['pass_rate']:.2%}")
            print(f"  Avg generation time: {augmented_stats['avg_generation_time']:.2f}s")
            print(f"  Avg tokens generated: {augmented_stats['avg_tokens']:.1f}")
            print(f"  Avg uncertainty score: {augmented_stats['avg_uncertainty']:.3f}")
            print(f"  Avg backtrack events: {augmented_stats['avg_backtrack_events']:.1f}")
            print(f"  Problems with prompt refinement: {augmented_stats.get('problems_with_refinement', 0)}")
        
        # Print pass@k metrics
        if pass_at_k_stats['baseline']['overall']:
            print(f"\nBaseline Pass@k Metrics:")
            for k, score in pass_at_k_stats['baseline']['overall'].items():
                print(f"  Pass@{k}: {score:.3f}")
        
        if pass_at_k_stats['augmented']['overall']:
            print(f"\nAugmented Pass@k Metrics:")
            for k, score in pass_at_k_stats['augmented']['overall'].items():
                print(f"  Pass@{k}: {score:.3f}")
        
        print(f"\nTotal evaluation time: {stats.get('total_time', 0):.1f}s")
        print(f"Results saved to: {Path(args.output_dir).absolute()}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())