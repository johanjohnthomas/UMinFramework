"""
GenerationLoop class for uncertainty-aware text generation with backtracking.

This module implements the core generation loop that orchestrates token-by-token
generation while monitoring uncertainty and performing backtracking with 
Chain-of-Thought (CoT) injection when uncertainty exceeds thresholds.
"""

import logging
import warnings
from typing import Optional, List, Union, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import copy

# Try to import required libraries
try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
        GenerationConfig,
        LogitsProcessor,
        LogitsProcessorList,
    )
    HAS_TRANSFORMERS = True
except ImportError as e:
    logging.warning(f"Transformers/PyTorch not available: {e}")
    logging.warning("GenerationLoop will not be functional without these dependencies")
    HAS_TRANSFORMERS = False
    # Create dummy classes for type hints
    PreTrainedModel = None
    PreTrainedTokenizer = None
    LogitsProcessor = None

# Import our UncertaintyHead
try:
    from .uncertainty_head import UncertaintyHead
    HAS_UNCERTAINTY_HEAD = True
except ImportError as e:
    logging.warning(f"UncertaintyHead not available: {e}")
    HAS_UNCERTAINTY_HEAD = False
    UncertaintyHead = None


class BacktrackingEvent(Enum):
    """Types of events that can trigger backtracking."""
    HIGH_UNCERTAINTY = "high_uncertainty"
    GENERATION_ERROR = "generation_error"
    MAX_ATTEMPTS = "max_attempts"
    CUSTOM_CONDITION = "custom_condition"


@dataclass
class GenerationState:
    """Represents the current state of generation."""
    input_ids: torch.Tensor
    generated_tokens: List[int]
    generated_text: str
    uncertainty_scores: List[float]
    backtrack_events: List[Dict[str, Any]]
    step_count: int
    total_tokens_generated: int
    past_key_values: Optional[Tuple] = None
    attention_mask: Optional[torch.Tensor] = None


@dataclass
class BacktrackConfig:
    """Configuration for backtracking behavior."""
    uncertainty_threshold: float = 0.7
    backtrack_window: int = 3
    max_backtracks_per_generation: int = 5
    max_backtracks_per_position: int = 2
    cot_templates: List[str] = None
    uncertainty_method: str = "entropy"
    
    def __post_init__(self):
        if self.cot_templates is None:
            self.cot_templates = [
                " Let me think step by step.",
                " Let me reconsider this.",
                " Actually, let me think about this more carefully.",
                " Wait, let me approach this differently.",
                " Let me break this down:"
            ]


class GenerationLoop:
    """
    Advanced generation loop with uncertainty monitoring and backtracking.
    
    This class implements a sophisticated generation strategy that monitors
    token-level uncertainty during generation and performs backtracking with
    Chain-of-Thought injection when uncertainty exceeds configurable thresholds.
    
    Key features:
    - Token-by-token generation with uncertainty monitoring
    - Configurable backtracking when uncertainty is high
    - Chain-of-Thought prompt injection during backtracking
    - State management for generation buffer and KV cache
    - Support for custom uncertainty methods and thresholds
    
    Example:
        >>> model_path = "gpt2"
        >>> gen_loop = GenerationLoop(model_path)
        >>> config = BacktrackConfig(uncertainty_threshold=0.8)
        >>> result = gen_loop.generate("The capital of France is", config=config)
        >>> print(f"Generated: {result['text']}")
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        uncertainty_head: Optional[UncertaintyHead] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the GenerationLoop.
        
        Args:
            model: Model path/name or pre-loaded model
            tokenizer: Pre-loaded tokenizer (optional)
            uncertainty_head: Pre-configured UncertaintyHead (optional)
            device: Device to use ('cpu', 'cuda', or None for auto)
            torch_dtype: PyTorch data type for the model
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers and PyTorch are required for GenerationLoop. "
                "Please install with: pip install transformers torch"
            )
        
        if not HAS_UNCERTAINTY_HEAD:
            raise ImportError(
                "UncertaintyHead is required for GenerationLoop. "
                "Ensure uncertainty_head.py is available."
            )
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self._load_model_and_tokenizer(model, tokenizer)
        
        # Set up uncertainty head
        if uncertainty_head is not None:
            self.uncertainty_head = uncertainty_head
        else:
            self.logger.info("Creating UncertaintyHead from model")
            self.uncertainty_head = UncertaintyHead(
                self.model, 
                self.tokenizer,
                device=self.device
            )
        
        # Generation state
        self._reset_state()
        
        self.logger.info(f"✓ GenerationLoop initialized successfully")
    
    def _load_model_and_tokenizer(
        self, 
        model: Union[str, PreTrainedModel], 
        tokenizer: Optional[PreTrainedTokenizer]
    ):
        """Load model and tokenizer."""
        try:
            if isinstance(model, str):
                self.model_name = model
                self.logger.info(f"Loading model: {model}")
                
                # Load tokenizer
                if tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(model)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer = tokenizer
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device if self.device != "cpu" else None
                )
                
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                    
            elif isinstance(model, PreTrainedModel):
                self.model = model
                self.model_name = getattr(model, 'name_or_path', 'unknown')
                
                if tokenizer is None:
                    raise ValueError("Tokenizer must be provided with pre-loaded model")
                self.tokenizer = tokenizer
            else:
                raise ValueError(f"Invalid model type: {type(model)}")
            
            # Set to eval mode
            self.model.eval()
            
            self.logger.info(f"Model loaded: {self.model_name}")
            self.logger.info(f"Parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            raise ValueError(f"Failed to load model and tokenizer: {e}")
    
    def _reset_state(self):
        """Reset the generation state."""
        self.current_state = None
        self._generation_history = []
    
    def generate(
        self,
        prompt: str,
        config: Optional[BacktrackConfig] = None,
        max_length: int = 100,
        min_length: int = 1,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        num_return_sequences: int = 1,
        return_full_state: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate text with uncertainty monitoring and backtracking.
        
        Args:
            prompt: Input prompt to generate from
            config: Backtracking configuration
            max_length: Maximum total sequence length
            min_length: Minimum sequence length to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty factor
            num_return_sequences: Number of sequences to generate
            return_full_state: Whether to return full generation state
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary or list of dictionaries containing generated text and metadata
        """
        if config is None:
            config = BacktrackConfig()
        
        if num_return_sequences == 1:
            return self._generate_single(
                prompt, config, max_length, min_length, do_sample,
                temperature, top_k, top_p, repetition_penalty,
                return_full_state, **kwargs
            )
        else:
            results = []
            for i in range(num_return_sequences):
                result = self._generate_single(
                    prompt, config, max_length, min_length, do_sample,
                    temperature, top_k, top_p, repetition_penalty,
                    return_full_state, **kwargs
                )
                results.append(result)
            return results
    
    def _generate_single(
        self,
        prompt: str,
        config: BacktrackConfig,
        max_length: int,
        min_length: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        return_full_state: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a single sequence with backtracking."""
        # Initialize state
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        initial_length = input_ids.size(1)
        
        state = GenerationState(
            input_ids=input_ids,
            generated_tokens=[],
            generated_text="",
            uncertainty_scores=[],
            backtrack_events=[],
            step_count=0,
            total_tokens_generated=0,
            past_key_values=None,
            attention_mask=torch.ones_like(input_ids).to(self.device)
        )
        
        self.current_state = state
        backtrack_positions = {}  # Track backtracks per position
        
        try:
            while (state.input_ids.size(1) < max_length and 
                   len(state.generated_tokens) < max_length - initial_length):
                
                # Generate next token
                next_token_id, uncertainty_score, past_key_values = self._generate_next_token(
                    state, do_sample, temperature, top_k, top_p, repetition_penalty
                )
                
                # Check for EOS token
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                # Store the new token and uncertainty
                state.generated_tokens.append(next_token_id)
                state.uncertainty_scores.append(uncertainty_score)
                state.past_key_values = past_key_values
                state.step_count += 1
                state.total_tokens_generated += 1
                
                # Add token to input_ids for next iteration
                new_token_tensor = torch.tensor([[next_token_id]], device=self.device)
                state.input_ids = torch.cat([state.input_ids, new_token_tensor], dim=1)
                state.attention_mask = torch.cat([
                    state.attention_mask, 
                    torch.ones((1, 1), device=self.device)
                ], dim=1)
                
                # Check if backtracking is needed
                current_position = len(state.generated_tokens)
                should_backtrack, backtrack_reason = self._should_backtrack(
                    uncertainty_score, config, state, backtrack_positions
                )
                
                if should_backtrack:
                    # Record backtrack event
                    backtrack_event = {
                        "position": current_position,
                        "reason": backtrack_reason,
                        "uncertainty_score": uncertainty_score,
                        "tokens_before": state.generated_tokens.copy()
                    }
                    
                    # Perform backtracking
                    success = self._perform_backtrack(state, config, backtrack_positions)
                    
                    backtrack_event["success"] = success
                    backtrack_event["tokens_after"] = state.generated_tokens.copy()
                    state.backtrack_events.append(backtrack_event)
                    
                    if success:
                        self.logger.debug(f"Backtracked at position {current_position}")
                    else:
                        self.logger.warning(f"Backtrack failed at position {current_position}")
                
                # Check minimum length
                if (len(state.generated_tokens) >= min_length - initial_length and
                    next_token_id == self.tokenizer.eos_token_id):
                    break
            
            # Decode generated text
            if state.generated_tokens:
                generated_ids = torch.tensor([state.generated_tokens], device=self.device)
                state.generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Prepare result
            result = {
                "text": state.generated_text,
                "prompt": prompt,
                "full_text": prompt + state.generated_text,
                "generated_tokens": len(state.generated_tokens),
                "uncertainty_scores": state.uncertainty_scores,
                "avg_uncertainty": sum(state.uncertainty_scores) / len(state.uncertainty_scores) if state.uncertainty_scores else 0.0,
                "backtrack_events": len(state.backtrack_events),
                "backtrack_details": state.backtrack_events,
                "total_steps": state.step_count,
                "config": config
            }
            
            if return_full_state:
                result["full_state"] = state
            
            return result
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    def _generate_next_token(
        self,
        state: GenerationState,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float
    ) -> Tuple[int, float, Optional[Tuple]]:
        """Generate the next token and calculate its uncertainty."""
        with torch.no_grad():
            # Prepare inputs for forward pass
            if state.past_key_values is not None:
                # When using cache, only pass the last token
                model_input_ids = state.input_ids[:, -1:]
                model_attention_mask = state.attention_mask
            else:
                # First generation or after backtrack, use full sequence
                model_input_ids = state.input_ids
                model_attention_mask = state.attention_mask
            
            # Forward pass
            outputs = self.model(
                input_ids=model_input_ids,
                attention_mask=model_attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True
            )
            
            # Get logits for the last position
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(state.input_ids[0].tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty
            
            # Sample next token
            if do_sample:
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy decoding
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            # Calculate uncertainty for this token
            probs = F.softmax(next_token_logits, dim=-1)
            uncertainty_score = self._calculate_token_uncertainty(probs, method="entropy")
            
            return next_token_id, uncertainty_score, outputs.past_key_values
    
    def _calculate_token_uncertainty(self, probs: torch.Tensor, method: str = "entropy") -> float:
        """Calculate uncertainty for a single token's probability distribution."""
        if method == "entropy":
            # Shannon entropy
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            return entropy.item()
        elif method == "max_prob":
            # 1 - max probability (higher = more uncertain)
            max_prob = torch.max(probs, dim=-1)[0]
            return (1.0 - max_prob).item()
        elif method == "margin":
            # 1 - margin between top 2 (higher = more uncertain)
            sorted_probs = torch.sort(probs, dim=-1, descending=True)[0]
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            return (1.0 - margin).item()
        else:
            # Default to entropy
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            return entropy.item()
    
    def _should_backtrack(
        self,
        uncertainty_score: float,
        config: BacktrackConfig,
        state: GenerationState,
        backtrack_positions: Dict[int, int]
    ) -> Tuple[bool, str]:
        """Determine if backtracking should be triggered."""
        # Check uncertainty threshold
        if uncertainty_score > config.uncertainty_threshold:
            # Check if we've exceeded max backtracks for this generation
            if len(state.backtrack_events) >= config.max_backtracks_per_generation:
                return False, "max_generation_backtracks_exceeded"
            
            # Check if we've exceeded max backtracks for this position
            current_pos = len(state.generated_tokens)
            pos_backtracks = backtrack_positions.get(current_pos, 0)
            if pos_backtracks >= config.max_backtracks_per_position:
                return False, "max_position_backtracks_exceeded"
            
            return True, BacktrackingEvent.HIGH_UNCERTAINTY.value
        
        return False, "no_backtrack_needed"
    
    def _perform_backtrack(
        self,
        state: GenerationState,
        config: BacktrackConfig,
        backtrack_positions: Dict[int, int]
    ) -> bool:
        """Perform backtracking with CoT injection."""
        try:
            # Record backtrack at current position
            current_pos = len(state.generated_tokens)
            backtrack_positions[current_pos] = backtrack_positions.get(current_pos, 0) + 1
            
            # Calculate how many tokens to remove
            backtrack_window = min(config.backtrack_window, len(state.generated_tokens))
            if backtrack_window == 0:
                return False
            
            # Remove tokens from state
            state.generated_tokens = state.generated_tokens[:-backtrack_window]
            state.uncertainty_scores = state.uncertainty_scores[:-backtrack_window]
            
            # Rebuild input_ids by removing the backtracked tokens
            # Keep only: original prompt + remaining generated tokens
            new_length = state.input_ids.size(1) - backtrack_window
            state.input_ids = state.input_ids[:, :new_length]
            
            # Update attention mask to match
            state.attention_mask = torch.ones_like(state.input_ids).to(self.device)
            
            # Inject CoT prompt
            self._inject_cot_prompt(state, config)
            
            # Reset past_key_values (cache invalidated by backtracking)
            state.past_key_values = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backtracking failed: {e}")
            return False
    
    def _inject_cot_prompt(self, state: GenerationState, config: BacktrackConfig):
        """Inject a Chain-of-Thought prompt at the current position."""
        # Choose a CoT template (could be random or based on context)
        import random
        cot_text = random.choice(config.cot_templates)
        
        # Tokenize and add to input
        cot_tokens = self.tokenizer.encode(cot_text, add_special_tokens=False)
        cot_tensor = torch.tensor([cot_tokens], device=self.device)
        
        # Add to state
        state.input_ids = torch.cat([state.input_ids, cot_tensor], dim=1)
        state.attention_mask = torch.cat([
            state.attention_mask,
            torch.ones((1, len(cot_tokens)), device=self.device)
        ], dim=1)
        
        self.logger.debug(f"Injected CoT: '{cot_text}'")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current generation."""
        if self.current_state is None:
            return {"error": "No active generation"}
        
        state = self.current_state
        return {
            "total_tokens": len(state.generated_tokens),
            "avg_uncertainty": sum(state.uncertainty_scores) / len(state.uncertainty_scores) if state.uncertainty_scores else 0,
            "max_uncertainty": max(state.uncertainty_scores) if state.uncertainty_scores else 0,
            "min_uncertainty": min(state.uncertainty_scores) if state.uncertainty_scores else 0,
            "backtrack_events": len(state.backtrack_events),
            "steps_taken": state.step_count,
            "high_uncertainty_tokens": sum(1 for score in state.uncertainty_scores if score > 0.7)
        }
    
    def __repr__(self) -> str:
        """String representation of GenerationLoop."""
        return (
            f"GenerationLoop(model='{self.model_name}', "
            f"device='{self.device}', loaded={self.model is not None})"
        )