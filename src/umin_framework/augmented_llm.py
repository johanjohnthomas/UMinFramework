"""
AugmentedLLM class that combines PromptRefiner, UncertaintyHead, and GenerationLoop.

This module provides the main AugmentedLLM class that encapsulates the entire pipeline
for uncertainty-aware text generation with prompt refinement and backtracking.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Tuple
from dataclasses import dataclass, field
import copy

# Try to import required components
try:
    from .prompt_refiner import PromptRefiner
    from .uncertainty_head import UncertaintyHead
    from .generation_loop import GenerationLoop, BacktrackConfig
    from .config import UMinConfig, get_config, log_generation_progress
    HAS_COMPONENTS = True
except ImportError as e:
    logging.warning(f"UMinFramework components not available: {e}")
    logging.warning("AugmentedLLM will not be functional without these components")
    HAS_COMPONENTS = False
    PromptRefiner = None
    UncertaintyHead = None
    GenerationLoop = None
    BacktrackConfig = None
    UMinConfig = None

# Try to import transformers
try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError as e:
    logging.warning(f"Transformers/PyTorch not available: {e}")
    HAS_TRANSFORMERS = False
    PreTrainedModel = None
    PreTrainedTokenizer = None


@dataclass
class AugmentedLLMConfig:
    """Configuration for the AugmentedLLM pipeline."""
    
    # Prompt refinement settings
    enable_prompt_refinement: bool = True
    prompt_refiner_model: Optional[str] = None
    
    # Generation model settings
    generation_model: str = "gpt2"  # Default model
    device: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None
    trust_remote_code: bool = False
    
    # Uncertainty settings
    enable_uncertainty_monitoring: bool = True
    uncertainty_threshold: float = 0.7
    uncertainty_method: str = "entropy"
    
    # Backtracking settings
    enable_backtracking: bool = True
    backtrack_window: int = 3
    max_backtracks_per_generation: int = 5
    max_backtracks_per_position: int = 2
    cot_templates: Optional[List[str]] = None
    
    # Generation parameters
    max_length: int = 100
    min_length: int = 1
    do_sample: bool = True
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    num_return_sequences: int = 1
    
    def __post_init__(self):
        """Post-initialization validation and defaults."""
        if self.cot_templates is None:
            self.cot_templates = [
                " Let me think step by step.",
                " Let me reconsider this.",
                " Actually, let me think about this more carefully.",
                " Wait, let me approach this differently.",
                " Let me break this down:"
            ]
        
        if self.prompt_refiner_model is None and self.enable_prompt_refinement:
            # Try to find a default prompt refiner model
            default_path = Path("models/prompt_refiner")
            if default_path.exists():
                self.prompt_refiner_model = str(default_path)


class AugmentedLLM:
    """
    A unified wrapper that combines prompt refinement, uncertainty quantification,
    and backtracking-enabled generation into a single pipeline.
    
    The AugmentedLLM class provides a drop-in replacement for standard language models
    with enhanced capabilities for handling ambiguous prompts and uncertain generations.
    
    Key Features:
    - Prompt refinement using fine-tuned models
    - Real-time uncertainty quantification during generation
    - Automatic backtracking with Chain-of-Thought injection
    - Configurable pipeline components via YAML/JSON files
    - Structured logging for observability
    - Compatible with Hugging Face transformers
    
    Example:
        >>> config = AugmentedLLMConfig(
        ...     generation_model="microsoft/DialoGPT-medium",
        ...     uncertainty_threshold=0.8
        ... )
        >>> augmented_llm = AugmentedLLM(config)
        >>> result = augmented_llm.generate("What is the best programming language?")
        >>> print(f"Generated: {result['text']}")
    """
    
    @classmethod
    def from_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        global_config: Optional[UMinConfig] = None
    ) -> 'AugmentedLLM':
        """
        Create AugmentedLLM instance from configuration file or global config.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            global_config: Pre-loaded UMinConfig instance
            
        Returns:
            AugmentedLLM instance configured from file
        """
        if not HAS_COMPONENTS or UMinConfig is None:
            raise ImportError("UMinFramework components not available")
        
        if global_config is not None:
            config = global_config
        elif config_path is not None:
            from .config import load_config_from_file
            config = load_config_from_file(config_path)
        else:
            config = get_config()
        
        # Convert UMinConfig to AugmentedLLMConfig for compatibility
        augmented_config = AugmentedLLMConfig(
            generation_model=config.augmented_model.name,
            device=config.augmented_model.device,
            torch_dtype=getattr(torch, config.augmented_model.torch_dtype) if config.augmented_model.torch_dtype else None,
            trust_remote_code=config.augmented_model.trust_remote_code,
            
            enable_prompt_refinement=config.prompt_refiner.enabled,
            prompt_refiner_model=config.prompt_refiner.model_path,
            
            enable_uncertainty_monitoring=config.uncertainty.enabled,
            uncertainty_threshold=config.uncertainty.threshold,
            uncertainty_method=config.uncertainty.method,
            
            enable_backtracking=config.backtracking.enabled,
            backtrack_window=config.backtracking.window_size,
            max_backtracks_per_generation=config.backtracking.max_backtracks_per_generation,
            max_backtracks_per_position=config.backtracking.max_backtracks_per_position,
            cot_templates=config.backtracking.cot_templates,
            
            max_length=config.generation.max_length,
            min_length=config.generation.min_length,
            do_sample=config.generation.do_sample,
            temperature=config.generation.temperature,
            top_k=config.generation.top_k,
            top_p=config.generation.top_p,
            repetition_penalty=config.generation.repetition_penalty,
            num_return_sequences=config.generation.num_return_sequences
        )
        
        return cls(config=augmented_config)
    
    def __init__(
        self,
        config: Optional[AugmentedLLMConfig] = None,
        prompt_refiner: Optional[PromptRefiner] = None,
        generation_loop: Optional[GenerationLoop] = None,
        uncertainty_head: Optional[UncertaintyHead] = None,
    ):
        """
        Initialize the AugmentedLLM with configuration and optional pre-loaded components.
        
        Args:
            config: Configuration object for the pipeline
            prompt_refiner: Pre-loaded PromptRefiner instance
            generation_loop: Pre-loaded GenerationLoop instance
            uncertainty_head: Pre-loaded UncertaintyHead instance
            
        Raises:
            ImportError: If required components are not available
            ValueError: If configuration is invalid or components fail to load
        """
        if not HAS_COMPONENTS:
            raise ImportError(
                "UMinFramework components are required for AugmentedLLM. "
                "Ensure prompt_refiner.py, uncertainty_head.py, and generation_loop.py are available."
            )
        
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers and PyTorch are required for AugmentedLLM. "
                "Please install with: pip install transformers torch"
            )
        
        # Set configuration
        self.config = config or AugmentedLLMConfig()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AugmentedLLM...")
        
        # Initialize components
        self.prompt_refiner = None
        self.generation_loop = None
        self.uncertainty_head = None
        self.tokenizer = None
        self.model = None
        
        # Load components
        self._load_components(prompt_refiner, generation_loop, uncertainty_head)
        
        # Validate pipeline
        self._validate_pipeline()
        
        self.logger.info("✓ AugmentedLLM initialized successfully")
        self.logger.info(f"Components: "
                        f"PromptRefiner={self.prompt_refiner is not None}, "
                        f"UncertaintyHead={self.uncertainty_head is not None}, "
                        f"GenerationLoop={self.generation_loop is not None}")
    
    def _load_components(
        self,
        prompt_refiner: Optional[PromptRefiner],
        generation_loop: Optional[GenerationLoop],
        uncertainty_head: Optional[UncertaintyHead]
    ):
        """Load and initialize all pipeline components."""
        try:
            # Load PromptRefiner
            if self.config.enable_prompt_refinement:
                if prompt_refiner is not None:
                    self.prompt_refiner = prompt_refiner
                elif self.config.prompt_refiner_model:
                    self.logger.info(f"Loading PromptRefiner from: {self.config.prompt_refiner_model}")
                    self.prompt_refiner = PromptRefiner(
                        model_path=self.config.prompt_refiner_model,
                        device=self.config.device
                    )
                else:
                    self.logger.warning("Prompt refinement enabled but no model specified. Disabling.")
                    self.config.enable_prompt_refinement = False
            
            # Load GenerationLoop (which includes UncertaintyHead)
            if generation_loop is not None:
                self.generation_loop = generation_loop
                # Extract uncertainty head and tokenizer from generation loop
                self.uncertainty_head = self.generation_loop.uncertainty_head
                self.tokenizer = self.generation_loop.tokenizer
                self.model = self.generation_loop.model
            else:
                self.logger.info(f"Loading GenerationLoop with model: {self.config.generation_model}")
                
                # First load uncertainty head if provided separately
                if uncertainty_head is not None:
                    self.uncertainty_head = uncertainty_head
                    uh_for_gen_loop = uncertainty_head
                else:
                    uh_for_gen_loop = None
                
                # Create generation loop
                self.generation_loop = GenerationLoop(
                    model=self.config.generation_model,
                    tokenizer=None,  # Will be auto-loaded
                    uncertainty_head=uh_for_gen_loop,
                    device=self.config.device,
                    torch_dtype=self.config.torch_dtype
                )
                
                # Extract components
                self.uncertainty_head = self.generation_loop.uncertainty_head
                self.tokenizer = self.generation_loop.tokenizer
                self.model = self.generation_loop.model
            
        except Exception as e:
            raise ValueError(f"Failed to load pipeline components: {e}")
    
    def _validate_pipeline(self):
        """Validate that the pipeline is properly configured."""
        if self.generation_loop is None:
            raise ValueError("GenerationLoop is required but not loaded")
        
        if self.config.enable_uncertainty_monitoring and self.uncertainty_head is None:
            raise ValueError("UncertaintyHead is required when uncertainty monitoring is enabled")
        
        if self.config.enable_prompt_refinement and self.prompt_refiner is None:
            self.logger.warning("Prompt refinement enabled but PromptRefiner not available")
            self.config.enable_prompt_refinement = False
        
        self.logger.debug("Pipeline validation completed")
    
    def generate(
        self,
        prompt: str,
        return_metadata: bool = True,
        override_config: Optional[Dict[str, Any]] = None,
        **generation_kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate text using the full augmented pipeline.
        
        This method orchestrates the complete pipeline:
        1. Optionally refine the input prompt
        2. Generate text with uncertainty monitoring
        3. Perform backtracking with CoT injection as needed
        
        Args:
            prompt: Input prompt to generate from
            return_metadata: Whether to return detailed metadata along with text
            override_config: Temporary configuration overrides for this generation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text (if return_metadata=False) or detailed result dictionary
            
        Raises:
            ValueError: If prompt is invalid
            RuntimeError: If generation fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            # Apply configuration overrides
            effective_config = self._apply_config_overrides(override_config)
            
            self.logger.info("Starting augmented generation")
            self.logger.info(f"Input prompt length: {len(prompt)} characters")
            
            # Step 1: Optionally refine the prompt
            if effective_config.enable_prompt_refinement and self.prompt_refiner:
                self.logger.info("Refining prompt using trained model...")
                refined_prompt = self.prompt_refiner.refine(prompt)
                
                if refined_prompt != prompt:
                    self.logger.info("Prompt successfully refined")
                    self.logger.debug(f"Original: {prompt}")
                    self.logger.debug(f"Refined: {refined_prompt}")
                else:
                    self.logger.info("No refinement needed")
            else:
                refined_prompt = prompt
                if effective_config.enable_prompt_refinement:
                    self.logger.warning("Prompt refinement enabled but no refiner available")
            
            # Step 2: Prepare backtracking configuration
            backtrack_config = BacktrackConfig(
                uncertainty_threshold=effective_config.uncertainty_threshold,
                backtrack_window=effective_config.backtrack_window,
                max_backtracks_per_generation=effective_config.max_backtracks_per_generation,
                max_backtracks_per_position=effective_config.max_backtracks_per_position,
                cot_templates=effective_config.cot_templates,
                uncertainty_method=effective_config.uncertainty_method
            ) if effective_config.enable_backtracking else None
            
            # Step 3: Generate with uncertainty monitoring and backtracking
            generation_params = self._prepare_generation_params(effective_config, generation_kwargs)
            
            self.logger.info(f"Generation parameters: max_length={generation_params.get('max_length')}, "
                           f"temperature={generation_params.get('temperature')}, "
                           f"uncertainty_threshold={effective_config.uncertainty_threshold}")
            
            if backtrack_config:
                self.logger.info("Using uncertainty-aware generation with backtracking")
                result = self.generation_loop.generate(
                    refined_prompt,
                    config=backtrack_config,
                    **generation_params
                )
                
                # Log generation statistics
                if result.get('backtrack_events', 0) > 0:
                    self.logger.info(f"Generation completed with {result['backtrack_events']} backtrack events")
                else:
                    self.logger.info("Generation completed without backtracking")
                
                self.logger.info(f"Average uncertainty score: {result.get('avg_uncertainty', 0):.3f}")
            else:
                self.logger.info("Using standard generation (no backtracking)")
                result = self._generate_without_backtracking(refined_prompt, generation_params)
            
            # Log final generation statistics
            log_generation_progress(self.logger, result.get('generated_tokens', 0), generation_params.get('max_length', 0))
            
            # Step 4: Prepare final result
            if return_metadata:
                # Add pipeline metadata
                result.update({
                    "original_prompt": prompt,
                    "refined_prompt": refined_prompt if effective_config.enable_prompt_refinement else None,
                    "prompt_was_refined": (refined_prompt != prompt) if effective_config.enable_prompt_refinement else False,
                    "pipeline_config": effective_config,
                    "uncertainty_monitoring_enabled": effective_config.enable_uncertainty_monitoring,
                    "backtracking_enabled": effective_config.enable_backtracking,
                })
                return result
            else:
                return result.get("text", "")
                
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    def _apply_config_overrides(self, overrides: Optional[Dict[str, Any]]) -> AugmentedLLMConfig:
        """Apply temporary configuration overrides."""
        if not overrides:
            return self.config
        
        # Create a copy of the current config
        config_dict = copy.deepcopy(self.config.__dict__)
        
        # Apply overrides
        config_dict.update(overrides)
        
        # Create new config object
        effective_config = AugmentedLLMConfig(**config_dict)
        
        return effective_config
    
    def _prepare_generation_params(
        self, 
        config: AugmentedLLMConfig, 
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare generation parameters from config and kwargs."""
        params = {
            "max_length": config.max_length,
            "min_length": config.min_length,
            "do_sample": config.do_sample,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "num_return_sequences": config.num_return_sequences,
            "return_full_state": True  # Always return full state for metadata
        }
        
        # Override with user-provided kwargs
        params.update(kwargs)
        
        return params
    
    def _generate_without_backtracking(
        self, 
        prompt: str, 
        generation_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate text without backtracking (fallback method)."""
        # Use the model directly for simple generation
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=generation_params.get("max_length", 100),
                min_length=generation_params.get("min_length", 1),
                do_sample=generation_params.get("do_sample", True),
                temperature=generation_params.get("temperature", 0.8),
                top_k=generation_params.get("top_k", 50),
                top_p=generation_params.get("top_p", 0.9),
                repetition_penalty=generation_params.get("repetition_penalty", 1.1),
                num_return_sequences=generation_params.get("num_return_sequences", 1),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return {
            "text": generated_text,
            "prompt": prompt,
            "full_text": prompt + generated_text,
            "generated_tokens": len(outputs[0]) - len(inputs[0]),
            "uncertainty_scores": [],
            "avg_uncertainty": 0.0,
            "backtrack_events": 0,
            "backtrack_details": [],
            "total_steps": 1,
            "config": None
        }
    
    def generate_batch(
        self, 
        prompts: List[str], 
        return_metadata: bool = True,
        **generation_kwargs
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            return_metadata: Whether to return metadata for each generation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated texts or result dictionaries
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        results = []
        for prompt in prompts:
            try:
                result = self.generate(
                    prompt=prompt,
                    return_metadata=return_metadata,
                    **generation_kwargs
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to generate for prompt '{prompt[:50]}...': {e}")
                if return_metadata:
                    results.append({
                        "text": "",
                        "error": str(e),
                        "original_prompt": prompt
                    })
                else:
                    results.append("")
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the loaded pipeline components."""
        info = {
            "config": self.config.__dict__,
            "components": {
                "prompt_refiner": self.prompt_refiner.get_model_info() if self.prompt_refiner else None,
                "uncertainty_head": self.uncertainty_head.get_model_info() if self.uncertainty_head else None,
                "generation_model": {
                    "name": getattr(self.model, 'name_or_path', 'unknown'),
                    "device": str(self.model.device) if self.model else None,
                    "parameters": self.model.num_parameters() if self.model else 0,
                }
            },
            "pipeline_enabled": {
                "prompt_refinement": self.config.enable_prompt_refinement and self.prompt_refiner is not None,
                "uncertainty_monitoring": self.config.enable_uncertainty_monitoring and self.uncertainty_head is not None,
                "backtracking": self.config.enable_backtracking,
            }
        }
        
        return info
    
    def update_config(self, **kwargs):
        """Update the pipeline configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config.{key} = {value}")
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
        
        # Re-validate pipeline after config changes
        self._validate_pipeline()
    
    def benchmark_generation(
        self, 
        prompt: str, 
        num_runs: int = 5,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Benchmark generation performance and consistency.
        
        Args:
            prompt: Test prompt to use for benchmarking
            num_runs: Number of generation runs to perform
            return_details: Whether to return detailed results from each run
            
        Returns:
            Dictionary containing benchmark statistics
        """
        import time
        
        results = []
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            try:
                result = self.generate(prompt, return_metadata=True)
                generation_time = time.time() - start_time
                
                results.append(result)
                times.append(generation_time)
            except Exception as e:
                self.logger.error(f"Benchmark run {i+1} failed: {e}")
                times.append(float('inf'))
        
        # Calculate statistics
        successful_runs = [r for r in results if 'error' not in r]
        valid_times = [t for t in times if t != float('inf')]
        
        stats = {
            "total_runs": num_runs,
            "successful_runs": len(successful_runs),
            "success_rate": len(successful_runs) / num_runs,
            "avg_generation_time": sum(valid_times) / len(valid_times) if valid_times else 0,
            "avg_tokens_generated": sum(r.get('generated_tokens', 0) for r in successful_runs) / len(successful_runs) if successful_runs else 0,
            "avg_uncertainty": sum(r.get('avg_uncertainty', 0) for r in successful_runs) / len(successful_runs) if successful_runs else 0,
            "total_backtrack_events": sum(r.get('backtrack_events', 0) for r in successful_runs),
            "avg_backtrack_events": sum(r.get('backtrack_events', 0) for r in successful_runs) / len(successful_runs) if successful_runs else 0,
        }
        
        if return_details:
            stats["detailed_results"] = results
            stats["generation_times"] = times
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of AugmentedLLM."""
        components = []
        if self.config.enable_prompt_refinement and self.prompt_refiner:
            components.append("PromptRefiner")
        if self.config.enable_uncertainty_monitoring and self.uncertainty_head:
            components.append("UncertaintyHead")
        if self.config.enable_backtracking:
            components.append("Backtracking")
        
        return (
            f"AugmentedLLM(model='{getattr(self.model, 'name_or_path', 'unknown')}', "
            f"components=[{', '.join(components)}])"
        )