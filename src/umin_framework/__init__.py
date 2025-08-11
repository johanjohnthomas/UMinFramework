"""
UMinFramework - A library for comparing standard LLMs against augmented LLMs.

The UMinFramework integrates prompt refinement, uncertainty quantification, 
and continuous chain-of-thought backtracking to reduce hallucinations and 
improve reasoning in Large Language Models.
"""

__version__ = "0.1.0"

from .prompt_refiner import PromptRefiner
from .uncertainty_head import UncertaintyHead
from .generation_loop import GenerationLoop, BacktrackConfig, GenerationState
from .augmented_llm import AugmentedLLM, AugmentedLLMConfig
from .code_executor import SafeCodeExecutor, PassAtKCalculator, ExecutionResult, CodeValidator
from .config import (
    UMinConfig, ConfigManager, LoggingSetup, 
    get_config, set_config, load_config_from_file, save_current_config,
    log_uncertainty_event, log_backtrack_event, log_generation_progress
)

__all__ = [
    "PromptRefiner", 
    "UncertaintyHead", 
    "GenerationLoop", 
    "BacktrackConfig", 
    "GenerationState",
    "AugmentedLLM",
    "AugmentedLLMConfig",
    "SafeCodeExecutor",
    "PassAtKCalculator", 
    "ExecutionResult",
    "CodeValidator",
    "UMinConfig",
    "ConfigManager",
    "LoggingSetup",
    "get_config",
    "set_config",
    "load_config_from_file",
    "save_current_config",
    "log_uncertainty_event",
    "log_backtrack_event",
    "log_generation_progress"
]