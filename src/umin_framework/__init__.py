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

__all__ = ["PromptRefiner", "UncertaintyHead", "GenerationLoop", "BacktrackConfig", "GenerationState"]