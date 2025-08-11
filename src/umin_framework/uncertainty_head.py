"""
UncertaintyHead class for token-level uncertainty quantification in language models.

This module provides uncertainty quantification methods that can be integrated
with Hugging Face transformers models to estimate confidence in generated tokens.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Tuple
import numpy as np

# Try to import structured logging utilities
try:
    from .config import log_uncertainty_event
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Try to import required libraries
try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    HAS_TRANSFORMERS = True
except ImportError as e:
    logging.warning(f"Transformers/PyTorch not available: {e}")
    logging.warning("UncertaintyHead will not be functional without these dependencies")
    HAS_TRANSFORMERS = False
    # Create dummy classes for type hints
    PreTrainedModel = None
    PreTrainedTokenizer = None


class UncertaintyHead:
    """
    A class for quantifying token-level uncertainty in language models.
    
    The UncertaintyHead wraps a Hugging Face transformers model and provides
    methods to calculate uncertainty scores (such as entropy) for generated tokens.
    This can be used to identify when the model is uncertain about its predictions,
    which may indicate potential hallucinations or low-confidence outputs.
    
    Example:
        >>> model_path = "gpt2"
        >>> ue_head = UncertaintyHead(model_path)
        >>> scores = ue_head.score("The capital of France is")
        >>> print(f"Uncertainty scores: {scores}")
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel, Path],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the UncertaintyHead with a language model.
        
        Args:
            model: Either a model path/name (str), a loaded model, or Path object
            tokenizer: Pre-loaded tokenizer (optional, will auto-load if None)
            device: Device to load model on ('cpu', 'cuda', or None for auto)
            trust_remote_code: Whether to trust remote code when loading models
            torch_dtype: PyTorch data type for the model (e.g., torch.float16)
            
        Raises:
            ImportError: If transformers/torch not available
            ValueError: If model loading fails
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers and PyTorch are required for UncertaintyHead. "
                "Please install with: pip install transformers torch"
            )
        
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self.model_name = None
        
        self._load_model_and_tokenizer(model, tokenizer)
        
    def _setup_device(self, device: Optional[str]) -> str:
        """Set up the computation device."""
        if device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model_and_tokenizer(
        self, 
        model: Union[str, PreTrainedModel, Path], 
        tokenizer: Optional[PreTrainedTokenizer]
    ):
        """Load the model and tokenizer."""
        try:
            if isinstance(model, (str, Path)):
                # Load from path/name
                model_path = str(model)
                self.model_name = model_path
                
                self.logger.info(f"Loading model from: {model_path}")
                
                # Load tokenizer
                if tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=self.trust_remote_code
                    )
                    # Add padding token if missing
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer = tokenizer
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device if self.device != "cpu" else None
                )
                
                # Move to device if needed
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                
            elif isinstance(model, PreTrainedModel):
                # Use pre-loaded model
                self.model = model
                self.model_name = getattr(model, 'name_or_path', 'unknown')
                
                if tokenizer is None:
                    raise ValueError("Tokenizer must be provided when using pre-loaded model")
                self.tokenizer = tokenizer
                
            else:
                raise ValueError(f"Invalid model type: {type(model)}")
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.logger.info(f"✓ UncertaintyHead loaded successfully")
            self.logger.info(f"Model: {self.model_name}")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            raise ValueError(f"Failed to load model and tokenizer: {e}")
    
    def score(
        self, 
        sequence: Union[str, List[str]], 
        method: str = "entropy",
        return_tokens: bool = False,
        max_length: Optional[int] = None
    ) -> Union[List[float], Tuple[List[float], List[str]]]:
        """
        Calculate token-level uncertainty scores for a sequence.
        
        Args:
            sequence: Input text sequence(s) to analyze
            method: Uncertainty quantification method ("entropy", "max_prob", "margin")
            return_tokens: Whether to return tokens along with scores
            max_length: Maximum sequence length (truncate if longer)
            
        Returns:
            List of uncertainty scores per token, optionally with tokens
            
        Raises:
            ValueError: If sequence is empty or method is invalid
            RuntimeError: If scoring fails
        """
        if isinstance(sequence, str):
            sequences = [sequence]
            single_input = True
        else:
            sequences = sequence
            single_input = False
            
        if not sequences or any(not s.strip() for s in sequences):
            raise ValueError("Sequence(s) cannot be empty")
            
        valid_methods = ["entropy", "max_prob", "margin", "variance"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got: {method}")
        
        try:
            all_scores = []
            all_tokens = []
            
            for seq in sequences:
                scores, tokens = self._calculate_uncertainty(seq, method, max_length)
                all_scores.extend(scores) if single_input else all_scores.append(scores)
                all_tokens.extend(tokens) if single_input else all_tokens.append(tokens)
            
            if return_tokens:
                return (all_scores, all_tokens)
            return all_scores
            
        except Exception as e:
            raise RuntimeError(f"Uncertainty scoring failed: {e}")
    
    def _calculate_uncertainty(
        self, 
        sequence: str, 
        method: str, 
        max_length: Optional[int]
    ) -> Tuple[List[float], List[str]]:
        """Calculate uncertainty scores for a single sequence."""
        # Tokenize input
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
        
        # Remove batch dimension (we process one sequence at a time)
        probs = probs.squeeze(0)  # Shape: (seq_len, vocab_size)
        
        # Calculate uncertainty scores based on method
        if method == "entropy":
            # Shannon entropy: -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10)  # Add small epsilon for numerical stability
            entropy = -torch.sum(probs * log_probs, dim=-1)
            scores = entropy.cpu().numpy().tolist()
            
        elif method == "max_prob":
            # Maximum probability (lower = more uncertain)
            max_probs = torch.max(probs, dim=-1)[0]
            # Convert to uncertainty (1 - max_prob, so higher = more uncertain)
            scores = (1.0 - max_probs).cpu().numpy().tolist()
            
        elif method == "margin":
            # Margin between top 2 predictions (lower = more uncertain)
            sorted_probs = torch.sort(probs, dim=-1, descending=True)[0]
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            # Convert to uncertainty (1 - margin)
            scores = (1.0 - margins).cpu().numpy().tolist()
            
        elif method == "variance":
            # Variance of the probability distribution
            variances = torch.var(probs, dim=-1)
            scores = variances.cpu().numpy().tolist()
        
        # Get token strings
        input_ids = inputs["input_ids"].squeeze(0)
        tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids]
        
        # Remove scores for special tokens at the end if needed
        scores = scores[:len(tokens)]
        
        # Log uncertainty scores if structured logging is available
        if HAS_CONFIG:
            threshold = getattr(self, 'uncertainty_threshold', 0.5)
            for token, score in zip(tokens, scores):
                log_uncertainty_event(self.logger, token, score, threshold)
        
        return scores, tokens
    
    def score_generation(
        self, 
        prompt: str, 
        generation: str, 
        method: str = "entropy"
    ) -> Tuple[List[float], List[str]]:
        """
        Score uncertainty for a generated sequence given its prompt.
        
        This method scores only the generated tokens, not the prompt tokens.
        
        Args:
            prompt: The input prompt
            generation: The generated text
            method: Uncertainty quantification method
            
        Returns:
            Tuple of (uncertainty_scores, tokens) for generated tokens only
        """
        full_sequence = prompt + generation
        full_scores, full_tokens = self._calculate_uncertainty(full_sequence, method, None)
        
        # Find where generation starts
        prompt_tokens = self.tokenizer.tokenize(prompt)
        generation_start_idx = len(prompt_tokens)
        
        # Return only generation scores and tokens
        gen_scores = full_scores[generation_start_idx:]
        gen_tokens = full_tokens[generation_start_idx:]
        
        return gen_scores, gen_tokens
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": self.model.num_parameters() if self.model else 0,
            "torch_dtype": str(self.torch_dtype),
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0
        }
    
    def set_uncertainty_threshold(self, threshold: float) -> None:
        """Set threshold for high uncertainty detection."""
        self.uncertainty_threshold = threshold
        self.logger.info(f"Set uncertainty threshold to: {threshold}")
    
    def is_high_uncertainty(self, scores: List[float], threshold: Optional[float] = None) -> List[bool]:
        """
        Determine which tokens have high uncertainty.
        
        Args:
            scores: List of uncertainty scores
            threshold: Threshold value (uses instance threshold if None)
            
        Returns:
            List of boolean values indicating high uncertainty tokens
        """
        if threshold is None:
            threshold = getattr(self, 'uncertainty_threshold', 0.5)
        
        return [score > threshold for score in scores]
    
    def __repr__(self) -> str:
        """String representation of UncertaintyHead."""
        return (
            f"UncertaintyHead(model='{self.model_name}', "
            f"device='{self.device}', loaded={self.model is not None})"
        )