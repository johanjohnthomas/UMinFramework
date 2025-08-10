"""
PromptRefiner class for clarifying ambiguous prompts using a fine-tuned T5 model.

This module provides the PromptRefiner class that loads a fine-tuned T5 model
and provides inference capabilities to clarify and refine ambiguous user prompts.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Union

# Try to import transformers, but make it optional for basic functionality
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError as e:
    logging.warning(f"Transformers not available: {e}")
    logging.warning("PromptRefiner will not be functional without transformers library")
    HAS_TRANSFORMERS = False
    # Create dummy classes for type hints
    T5ForConditionalGeneration = None
    T5Tokenizer = None


class PromptRefiner:
    """
    A class for refining ambiguous prompts using a fine-tuned T5 model.
    
    The PromptRefiner loads a fine-tuned T5 model checkpoint and provides
    an inference method to generate clarifying questions or refined versions
    of ambiguous input prompts.
    
    Example:
        >>> refiner = PromptRefiner("models/prompt_refiner")
        >>> clarification = refiner.refine("What is the best programming language?")
        >>> print(clarification)
        "For what purpose? Web development, data science, or mobile apps?"
    """
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        device: Optional[str] = None,
        max_length: int = 256,
        num_beams: int = 4,
        temperature: float = 0.7,
        do_sample: bool = True
    ):
        """
        Initialize the PromptRefiner with a fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model directory containing
                       model.safetensors, config.json, and tokenizer files
            device: Device to load model on ('cpu', 'cuda', or None for auto)
            max_length: Maximum length of generated clarifications
            num_beams: Number of beams for beam search
            temperature: Sampling temperature (only used if do_sample=True)
            do_sample: Whether to use sampling for generation
            
        Raises:
            ImportError: If transformers library is not available
            FileNotFoundError: If model path does not exist
            ValueError: If model files are missing or invalid
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers library is required for PromptRefiner. "
                "Please install with: pip install transformers torch"
            )
        
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.num_beams = num_beams
        self.temperature = temperature
        self.do_sample = do_sample
        
        # Set up device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load the fine-tuned T5 model and tokenizer from the checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
            
        # Check for required files
        required_files = ["config.json", "tokenizer_config.json"]
        model_file = self.model_path / "model.safetensors"
        pytorch_file = self.model_path / "pytorch_model.bin"
        
        if not model_file.exists() and not pytorch_file.exists():
            raise ValueError(f"No model file found in {self.model_path}")
            
        for req_file in required_files:
            if not (self.model_path / req_file).exists():
                raise ValueError(f"Required file missing: {self.model_path / req_file}")
        
        try:
            # Load tokenizer
            self.logger.info(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = T5Tokenizer.from_pretrained(str(self.model_path))
            
            # Load model
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = T5ForConditionalGeneration.from_pretrained(str(self.model_path))
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.logger.info(f"✓ PromptRefiner loaded successfully on {self.device}")
            self.logger.info(f"Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            raise ValueError(f"Failed to load model from {self.model_path}: {e}")
    
    def refine(self, prompt: str, **generation_kwargs) -> str:
        """
        Refine an ambiguous prompt by generating a clarifying question or refined version.
        
        Args:
            prompt: The input prompt to refine
            **generation_kwargs: Additional arguments for generation (override defaults)
            
        Returns:
            str: The refined/clarified prompt
            
        Raises:
            ValueError: If prompt is empty or model not loaded
            RuntimeError: If generation fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded")
        
        # Prepare input with the clarify prefix used during training
        input_text = f"clarify: {prompt.strip()}"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Match training max_input_length
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prepare generation parameters
            gen_params = {
                "max_length": self.max_length,
                "num_beams": self.num_beams,
                "temperature": self.temperature,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Override with any user-provided parameters
            gen_params.update(generation_kwargs)
            
            # Generate
            self.logger.debug(f"Generating with parameters: {gen_params}")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_params
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            self.logger.debug(f"Input: {input_text}")
            self.logger.debug(f"Generated: {generated_text}")
            
            return generated_text.strip()
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
    
    def refine_batch(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """
        Refine multiple prompts in a batch for improved efficiency.
        
        Args:
            prompts: List of input prompts to refine
            **generation_kwargs: Additional arguments for generation
            
        Returns:
            List[str]: List of refined/clarified prompts
            
        Raises:
            ValueError: If prompts list is empty or contains invalid prompts
            RuntimeError: If batch generation fails
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
            
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded")
        
        # Prepare inputs with clarify prefix
        input_texts = [f"clarify: {prompt.strip()}" for prompt in prompts if prompt.strip()]
        
        if not input_texts:
            raise ValueError("All prompts are empty after cleaning")
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prepare generation parameters
            gen_params = {
                "max_length": self.max_length,
                "num_beams": self.num_beams,
                "temperature": self.temperature,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            gen_params.update(generation_kwargs)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_params
                )
            
            # Decode outputs
            results = []
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                results.append(generated_text.strip())
            
            self.logger.debug(f"Batch refined {len(results)} prompts")
            return results
            
        except Exception as e:
            raise RuntimeError(f"Batch generation failed: {e}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including path, device, parameters
        """
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "parameters": self.model.num_parameters() if self.model else 0,
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }
    
    def __repr__(self) -> str:
        """String representation of the PromptRefiner."""
        return (
            f"PromptRefiner(model_path='{self.model_path}', "
            f"device='{self.device}', loaded={self.model is not None})"
        )