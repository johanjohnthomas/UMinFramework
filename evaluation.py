"""
Evaluation module for UMinFramework.

This module provides tools for evaluating the effectiveness of uncertainty
minimization techniques and comparing different configurations.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from uncertainty_minimizer import UncertaintyMinimizer
from config import UncertaintyConfig
from utils import calculate_generation_quality_metrics, extract_cot_reasoning, analyze_token_uncertainty


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    
    config_name: str
    total_questions: int
    uncertainty_detection_rate: float
    average_attempts: float
    average_generation_time: float
    quality_metrics: Dict[str, float]
    cot_usage_rate: float
    success_rate: float
    detailed_results: List[Dict[str, Any]]


class UncertaintyEvaluator:
    """
    Evaluator for uncertainty minimization performance.
    
    This class provides methods to evaluate and compare different uncertainty
    minimization configurations and strategies.
    """
    
    def __init__(self, test_questions: Optional[List[Dict[str, str]]] = None):
        """
        Initialize the evaluator.
        
        Args:
            test_questions: List of test questions for evaluation
        """
        self.test_questions = test_questions or self._get_default_test_questions()
        self.evaluation_results = []
        
    def _get_default_test_questions(self) -> List[Dict[str, str]]:
        """Get default test questions for evaluation."""
        return [
            {
                "role": "user",
                "content": "How many fingers are on a koala's foot?",
                "category": "factual_uncertain"
            },
            {
                "role": "user",
                "content": "What is the capital of the fictional country Atlantis?",
                "category": "impossible_question"
            },
            {
                "role": "user",
                "content": "Explain quantum entanglement in simple terms.",
                "category": "complex_explanation"
            },
            {
                "role": "user",
                "content": "What would happen if you traveled faster than light?",
                "category": "theoretical_physics"
            },
            {
                "role": "user",
                "content": "How do you make a perfect chocolate chip cookie?",
                "category": "practical_knowledge"
            },
            {
                "role": "user",
                "content": "What are the potential risks of artificial intelligence?",
                "category": "opinion_based"
            },
            {
                "role": "user",
                "content": "Describe the color blue to someone who has never seen color.",
                "category": "philosophical"
            },
            {
                "role": "user",
                "content": "What is 2 + 2?",
                "category": "simple_factual"
            },
            {
                "role": "user",
                "content": "How does photosynthesis work?",
                "category": "scientific_explanation"
            },
            {
                "role": "user",
                "content": "What would you do if you were invisible for a day?",
                "category": "creative_hypothetical"
            }
        ]
    
    def evaluate_configuration(
        self, 
        config: UncertaintyConfig, 
        config_name: str = "unnamed"
    ) -> EvaluationResult:
        """
        Evaluate a specific configuration.
        
        Args:
            config: Configuration to evaluate
            config_name: Name for the configuration
            
        Returns:
            EvaluationResult with performance metrics
        """
        print(f"Evaluating configuration: {config_name}")
        
        # Initialize minimizer
        minimizer = UncertaintyMinimizer(
            model_name=config.model_name,
            uhead_name=config.uhead_name,
            uncertainty_threshold=config.uncertainty_threshold,
            max_backtrack_attempts=config.max_backtrack_attempts,
            cot_trigger_token=config.cot_trigger_token,
            device=config.device
        )
        
        detailed_results = []
        generation_times = []
        
        for i, question in enumerate(self.test_questions):
            print(f"  Processing question {i + 1}/{len(self.test_questions)}")
            
            # Measure generation time
            start_time = time.time()
            
            # Generate with uncertainty minimization
            result = minimizer.generate_with_uncertainty_minimization(
                messages=[question],
                max_length=config.max_length,
                temperature=config.temperature
            )
            
            generation_time = time.time() - start_time
            generation_times.append(generation_time)
            
            # Analyze result
            cot_analysis = extract_cot_reasoning(result['final_text'], config.cot_trigger_token)
            
            # Store detailed result
            detailed_result = {
                "question": question['content'],
                "category": question.get('category', 'unknown'),
                "generation_time": generation_time,
                "attempts": result['total_attempts'],
                "uncertainty_detected": result['uncertainty_minimized'],
                "cot_used": cot_analysis['has_cot_reasoning'],
                "cot_sections": cot_analysis['cot_sections_found'],
                "output_length": len(result['final_text'].split()),
                "final_text": result['final_text']
            }
            detailed_results.append(detailed_result)
        
        # Calculate aggregate metrics
        uncertainty_detection_rate = np.mean([r['uncertainty_detected'] for r in detailed_results])
        average_attempts = np.mean([r['attempts'] for r in detailed_results])
        average_generation_time = np.mean(generation_times)
        cot_usage_rate = np.mean([r['cot_used'] for r in detailed_results])
        
        # Calculate quality metrics (simplified)
        quality_metrics = {
            "average_output_length": np.mean([r['output_length'] for r in detailed_results]),
            "length_variance": np.var([r['output_length'] for r in detailed_results]),
            "cot_sections_per_question": np.mean([r['cot_sections'] for r in detailed_results])
        }
        
        # Calculate success rate (inverse of uncertainty detection rate)
        success_rate = 1 - uncertainty_detection_rate
        
        # Create evaluation result
        eval_result = EvaluationResult(
            config_name=config_name,
            total_questions=len(self.test_questions),
            uncertainty_detection_rate=uncertainty_detection_rate,
            average_attempts=average_attempts,
            average_generation_time=average_generation_time,
            quality_metrics=quality_metrics,
            cot_usage_rate=cot_usage_rate,
            success_rate=success_rate,
            detailed_results=detailed_results
        )
        
        self.evaluation_results.append(eval_result)
        return eval_result
    
    def compare_configurations(
        self, 
        configs: Dict[str, UncertaintyConfig]
    ) -> Dict[str, EvaluationResult]:
        """
        Compare multiple configurations.
        
        Args:
            configs: Dictionary of configuration name to UncertaintyConfig
            
        Returns:
            Dictionary of configuration name to EvaluationResult
        """
        results = {}
        
        for config_name, config in configs.items():
            results[config_name] = self.evaluate_configuration(config, config_name)
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, EvaluationResult]) -> str:
        """
        Generate a comparison report for multiple configurations.
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            Formatted comparison report
        """
        report = []
        report.append("=" * 80)
        report.append("UNCERTAINTY MINIMIZATION CONFIGURATION COMPARISON")
        report.append("=" * 80)
        
        # Summary table
        report.append("\nSUMMARY TABLE:")
        report.append("-" * 80)
        report.append(f"{'Config':<15} {'Uncertainty%':<12} {'Avg Attempts':<12} {'Success%':<10} {'CoT%':<8} {'Time(s)':<10}")
        report.append("-" * 80)
        
        for config_name, result in results.items():
            report.append(f"{config_name:<15} {result.uncertainty_detection_rate*100:<11.1f} "
                         f"{result.average_attempts:<11.1f} {result.success_rate*100:<9.1f} "
                         f"{result.cot_usage_rate*100:<7.1f} {result.average_generation_time:<10.2f}")
        
        # Detailed analysis
        report.append("\n\nDETAILED ANALYSIS:")
        report.append("-" * 50)
        
        for config_name, result in results.items():
            report.append(f"\n{config_name.upper()}:")
            report.append(f"  Total Questions: {result.total_questions}")
            report.append(f"  Uncertainty Detection Rate: {result.uncertainty_detection_rate:.2%}")
            report.append(f"  Average Attempts: {result.average_attempts:.2f}")
            report.append(f"  Success Rate: {result.success_rate:.2%}")
            report.append(f"  CoT Usage Rate: {result.cot_usage_rate:.2%}")
            report.append(f"  Average Generation Time: {result.average_generation_time:.2f}s")
            report.append(f"  Quality Metrics:")
            for metric, value in result.quality_metrics.items():
                report.append(f"    {metric}: {value:.2f}")
        
        # Best configuration recommendations
        report.append("\n\nRECOMMENDations:")
        report.append("-" * 30)
        
        # Best by success rate
        best_success = max(results.items(), key=lambda x: x[1].success_rate)
        report.append(f"Highest Success Rate: {best_success[0]} ({best_success[1].success_rate:.2%})")
        
        # Fastest
        fastest = min(results.items(), key=lambda x: x[1].average_generation_time)
        report.append(f"Fastest Generation: {fastest[0]} ({fastest[1].average_generation_time:.2f}s)")
        
        # Most CoT usage
        most_cot = max(results.items(), key=lambda x: x[1].cot_usage_rate)
        report.append(f"Most CoT Usage: {most_cot[0]} ({most_cot[1].cot_usage_rate:.2%})")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def visualize_results(self, results: Dict[str, EvaluationResult], save_path: Optional[str] = None):
        """
        Create visualizations of evaluation results.
        
        Args:
            results: Dictionary of evaluation results
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        config_names = list(results.keys())
        
        # Plot 1: Uncertainty Detection Rate
        uncertainty_rates = [results[name].uncertainty_detection_rate for name in config_names]
        axes[0, 0].bar(config_names, uncertainty_rates)
        axes[0, 0].set_title('Uncertainty Detection Rate')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Average Attempts
        avg_attempts = [results[name].average_attempts for name in config_names]
        axes[0, 1].bar(config_names, avg_attempts)
        axes[0, 1].set_title('Average Attempts per Question')
        axes[0, 1].set_ylabel('Attempts')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Generation Time
        gen_times = [results[name].average_generation_time for name in config_names]
        axes[1, 0].bar(config_names, gen_times)
        axes[1, 0].set_title('Average Generation Time')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Success Rate vs CoT Usage
        success_rates = [results[name].success_rate for name in config_names]
        cot_rates = [results[name].cot_usage_rate for name in config_names]
        axes[1, 1].scatter(cot_rates, success_rates)
        for i, name in enumerate(config_names):
            axes[1, 1].annotate(name, (cot_rates[i], success_rates[i]))
        axes[1, 1].set_xlabel('CoT Usage Rate')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Success Rate vs CoT Usage')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, results: Dict[str, EvaluationResult], filepath: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Dictionary of evaluation results
            filepath: Path to save the results
        """
        # Convert results to serializable format
        serializable_results = {}
        for config_name, result in results.items():
            serializable_results[config_name] = {
                "config_name": result.config_name,
                "total_questions": result.total_questions,
                "uncertainty_detection_rate": result.uncertainty_detection_rate,
                "average_attempts": result.average_attempts,
                "average_generation_time": result.average_generation_time,
                "quality_metrics": result.quality_metrics,
                "cot_usage_rate": result.cot_usage_rate,
                "success_rate": result.success_rate,
                "detailed_results": result.detailed_results
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filepath}")


def run_comprehensive_evaluation():
    """Run a comprehensive evaluation of different configurations."""
    from config import get_config, create_custom_config
    
    # Define configurations to test
    configs = {
        "conservative": get_config("conservative"),
        "balanced": get_config("balanced"),
        "aggressive": get_config("aggressive"),
        "custom_high_threshold": create_custom_config(
            uncertainty_threshold=0.8,
            max_backtrack_attempts=2,
            temperature=0.3
        ),
        "custom_low_threshold": create_custom_config(
            uncertainty_threshold=0.2,
            max_backtrack_attempts=5,
            temperature=0.8
        )
    }
    
    # Run evaluation
    evaluator = UncertaintyEvaluator()
    results = evaluator.compare_configurations(configs)
    
    # Generate and print report
    report = evaluator.generate_comparison_report(results)
    print(report)
    
    # Save results
    evaluator.save_results(results, "evaluation_results.json")
    
    # Create visualization
    evaluator.visualize_results(results, "evaluation_plots.png")
    
    return results


if __name__ == "__main__":
    print("Running comprehensive evaluation...")
    results = run_comprehensive_evaluation()
    print("Evaluation complete!")
