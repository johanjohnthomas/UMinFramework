"""
UMinFramework: A framework for minimizing uncertainty in language model outputs.

This package provides tools for detecting uncertain tokens in language model
generations and using backtracking with Chain-of-Thought reasoning to improve
output quality and confidence.

Main Components:
- UncertaintyMinimizer: Core class for uncertainty minimization
- UncertaintyConfig: Configuration management
- UncertaintyEvaluator: Evaluation and comparison tools
- Utilities: Helper functions for analysis and visualization

Example Usage:
    from UMinFramework import UncertaintyMinimizer, get_config
    
    config = get_config("balanced")
    minimizer = UncertaintyMinimizer(
        model_name=config.model_name,
        uhead_name=config.uhead_name,
        uncertainty_threshold=config.uncertainty_threshold
    )
    
    result = minimizer.generate_with_uncertainty_minimization(
        messages=[{"role": "user", "content": "Your question here"}]
    )
"""

from .uncertainty_minimizer import UncertaintyMinimizer
from .config import UncertaintyConfig, get_config, create_custom_config, CONFIGS
from .evaluation import UncertaintyEvaluator, EvaluationResult, run_comprehensive_evaluation
from .utils import (
    calculate_uncertainty_metrics,
    analyze_token_uncertainty,
    detect_uncertainty_patterns,
    extract_cot_reasoning,
    calculate_generation_quality_metrics,
    format_generation_report,
    visualize_uncertainty_distribution
)

__version__ = "0.1.0"
__author__ = "UMinFramework Team"
__description__ = "A framework for minimizing uncertainty in language model outputs"

__all__ = [
    # Core classes
    "UncertaintyMinimizer",
    "UncertaintyConfig",
    "UncertaintyEvaluator",
    "EvaluationResult",
    
    # Configuration functions
    "get_config",
    "create_custom_config",
    "CONFIGS",
    
    # Evaluation functions
    "run_comprehensive_evaluation",
    
    # Utility functions
    "calculate_uncertainty_metrics",
    "analyze_token_uncertainty",
    "detect_uncertainty_patterns",
    "extract_cot_reasoning",
    "calculate_generation_quality_metrics",
    "format_generation_report",
    "visualize_uncertainty_distribution",
]

# Package metadata
__package_info__ = {
    "name": "UMinFramework",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "requires": [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "luh",  # llm-uncertainty-head
    ],
    "python_requires": ">=3.8",
}
