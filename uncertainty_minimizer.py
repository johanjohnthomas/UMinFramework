"""
UMinFramework: Uncertainty Minimization Framework

This module provides a framework for minimizing uncertainty in language model outputs
by detecting uncertain tokens and triggering Chain-of-Thought (CoT) reasoning.
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from luh import AutoUncertaintyHead, CausalLMWithUncertainty
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintyMinimizer:
    """
    A framework for minimizing uncertainty in language model outputs.
    
    This class uses an uncertainty head to detect uncertain tokens and implements
    a backtracking mechanism with Chain-of-Thought (CoT) reasoning to improve
    model confidence.
    """
    
    def __init__(
        self,
        model_name: str,
        uhead_name: str,
        uncertainty_threshold: float = 0.5,
        max_backtrack_attempts: int = 3,
        cot_trigger_token: str = "<think>",
        device: str = "cuda"
    ):
        """
        Initialize the UncertaintyMinimizer.
        
        Args:
            model_name: Name of the base language model
            uhead_name: Name of the uncertainty head model
            uncertainty_threshold: Threshold for determining uncertain tokens
            max_backtrack_attempts: Maximum number of backtracking attempts
            cot_trigger_token: Token to trigger Chain-of-Thought reasoning
            device: Device to run models on
        """
        self.model_name = model_name
        self.uhead_name = uhead_name
        self.uncertainty_threshold = uncertainty_threshold
        self.max_backtrack_attempts = max_backtrack_attempts
        self.cot_trigger_token = cot_trigger_token
        self.device = device
        
        # Initialize models
        self._initialize_models()
        
        # Track generation history
        self.generation_history = []
        
    def _initialize_models(self):
        """Initialize the language model, tokenizer, and uncertainty head."""
        logger.info(f"Initializing models: {self.model_name}")
        
        # Load base model
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map=self.device,
            torch_dtype=torch.float16
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load uncertainty head
        self.uhead = AutoUncertaintyHead.from_pretrained(
            self.uhead_name, 
            base_model=self.llm
        )
        
        # Create uncertainty-aware adapter
        self.llm_adapter = CausalLMWithUncertainty(
            self.llm, 
            self.uhead, 
            tokenizer=self.tokenizer
        )
        
        logger.info("Models initialized successfully")
    
    def _detect_uncertain_tokens(self, uncertainty_logits: torch.Tensor) -> List[int]:
        """
        Detect uncertain tokens based on uncertainty logits.
        
        Args:
            uncertainty_logits: Tensor of uncertainty scores for each token
            
        Returns:
            List of indices of uncertain tokens
        """
        # Convert uncertainty logits to probabilities
        uncertainty_probs = torch.softmax(uncertainty_logits, dim=-1)
        
        # Get uncertainty scores (assuming higher score means more uncertain)
        uncertainty_scores = uncertainty_probs[..., 1] if uncertainty_probs.shape[-1] > 1 else uncertainty_probs.squeeze()
        
        # Find tokens above threshold
        uncertain_indices = torch.where(uncertainty_scores > self.uncertainty_threshold)[0].tolist()
        
        return uncertain_indices
    
    def _backtrack_to_uncertain_token(self, tokens: List[int], uncertain_index: int) -> List[int]:
        """
        Backtrack to remove uncertain tokens and everything after them.
        
        Args:
            tokens: List of token ids
            uncertain_index: Index of the first uncertain token
            
        Returns:
            Truncated list of tokens
        """
        if uncertain_index == 0:
            return tokens[:1]  # Keep at least the first token
        
        return tokens[:uncertain_index]
    
    def _trigger_cot_reasoning(self, prompt_tokens: List[int]) -> List[int]:
        """
        Trigger Chain-of-Thought reasoning by adding the think token.
        
        Args:
            prompt_tokens: Original prompt tokens
            
        Returns:
            Modified prompt tokens with CoT trigger
        """
        # Encode the CoT trigger token
        cot_tokens = self.tokenizer.encode(
            self.cot_trigger_token, 
            add_special_tokens=False
        )
        
        # Add CoT trigger to the end of the prompt
        return prompt_tokens + cot_tokens
    
    def _generate_with_uncertainty(
        self, 
        input_tokens: List[int], 
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text with uncertainty estimation.
        
        Args:
            input_tokens: Input token ids
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary containing generated tokens and uncertainty information
        """
        # Prepare inputs
        inputs = torch.tensor([input_tokens]).to(self.device)
        
        # Generate with uncertainty
        with torch.no_grad():
            output = self.llm_adapter.generate(
                {"input_ids": inputs},
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        return {
            "generated_ids": output["sequences"][0].tolist(),
            "uncertainty_logits": output.get("uncertainty_logits", None),
            "input_length": len(input_tokens)
        }
    
    def generate_with_uncertainty_minimization(
        self, 
        messages: List[Dict[str, str]], 
        max_length: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text with uncertainty minimization using backtracking and CoT.
        
        Args:
            messages: List of chat messages
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Dictionary containing final generation and process information
        """
        # Apply chat template
        if isinstance(messages[0], dict):
            chat_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_bos_token=False
            )
        else:
            chat_text = messages[0]
        
        # Tokenize input
        input_tokens = self.tokenizer.encode(chat_text, add_special_tokens=False)
        original_length = len(input_tokens)
        
        # Track generation attempts
        attempts = []
        current_tokens = input_tokens.copy()
        
        for attempt in range(self.max_backtrack_attempts):
            logger.info(f"Generation attempt {attempt + 1}/{self.max_backtrack_attempts}")
            
            # Generate with uncertainty
            generation_result = self._generate_with_uncertainty(
                current_tokens, 
                max_length=max_length,
                temperature=temperature
            )
            
            generated_ids = generation_result["generated_ids"]
            uncertainty_logits = generation_result["uncertainty_logits"]
            
            # Extract only the newly generated tokens
            new_tokens = generated_ids[len(current_tokens):]
            
            # Record this attempt
            attempt_info = {
                "attempt": attempt + 1,
                "input_tokens": current_tokens.copy(),
                "generated_tokens": new_tokens,
                "full_output": generated_ids,
                "uncertainty_detected": False,
                "uncertain_indices": []
            }
            
            # Check for uncertainty if we have uncertainty logits
            if uncertainty_logits is not None:
                # Only check uncertainty in the newly generated tokens
                if len(new_tokens) > 0:
                    # Get uncertainty for new tokens only
                    new_token_uncertainties = uncertainty_logits[len(current_tokens):]
                    uncertain_indices = self._detect_uncertain_tokens(new_token_uncertainties)
                    
                    if uncertain_indices:
                        attempt_info["uncertainty_detected"] = True
                        attempt_info["uncertain_indices"] = uncertain_indices
                        
                        # Backtrack to the first uncertain token
                        first_uncertain_idx = uncertain_indices[0]
                        backtrack_point = len(current_tokens) + first_uncertain_idx
                        
                        # Truncate generation at uncertain token
                        truncated_tokens = generated_ids[:backtrack_point]
                        
                        # Trigger CoT reasoning
                        current_tokens = self._trigger_cot_reasoning(truncated_tokens)
                        
                        logger.info(f"Uncertainty detected at position {backtrack_point}. "
                                  f"Triggering CoT reasoning.")
                        
                        attempts.append(attempt_info)
                        continue
            
            # If no uncertainty or last attempt, finalize
            attempts.append(attempt_info)
            final_tokens = generated_ids
            break
        
        # Decode final output
        final_text = self.tokenizer.decode(final_tokens, skip_special_tokens=True)
        
        # Prepare result
        result = {
            "final_text": final_text,
            "final_tokens": final_tokens,
            "original_input_length": original_length,
            "total_attempts": len(attempts),
            "attempts": attempts,
            "uncertainty_minimized": any(attempt["uncertainty_detected"] for attempt in attempts)
        }
        
        # Store in history
        self.generation_history.append(result)
        
        return result
    
    def analyze_generation_history(self) -> Dict[str, Any]:
        """
        Analyze the generation history to provide insights.
        
        Returns:
            Dictionary with analysis of generation patterns
        """
        if not self.generation_history:
            return {"message": "No generation history available"}
        
        total_generations = len(self.generation_history)
        uncertainty_minimized_count = sum(
            1 for gen in self.generation_history 
            if gen["uncertainty_minimized"]
        )
        
        avg_attempts = sum(gen["total_attempts"] for gen in self.generation_history) / total_generations
        
        return {
            "total_generations": total_generations,
            "uncertainty_minimized_count": uncertainty_minimized_count,
            "uncertainty_minimization_rate": uncertainty_minimized_count / total_generations,
            "average_attempts_per_generation": avg_attempts,
            "most_recent_generation": self.generation_history[-1] if self.generation_history else None
        }
    
    def clear_history(self):
        """Clear the generation history."""
        self.generation_history = []
        logger.info("Generation history cleared")
