"""
Utility functions for UMinFramework.

This module contains helper functions for uncertainty analysis,
text processing, and evaluation metrics.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re


def calculate_uncertainty_metrics(uncertainty_logits: torch.Tensor) -> Dict[str, float]:
    """
    Calculate various uncertainty metrics from logits.
    
    Args:
        uncertainty_logits: Tensor of uncertainty scores
        
    Returns:
        Dictionary of uncertainty metrics
    """
    if uncertainty_logits is None:
        return {}
    
    # Convert to probabilities
    probs = torch.softmax(uncertainty_logits, dim=-1)
    
    # Calculate entropy (higher = more uncertain)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    
    # Calculate variance
    variance = torch.var(probs, dim=-1)
    
    # Calculate max probability (higher = more certain)
    max_prob = torch.max(probs, dim=-1)[0]
    
    # Calculate predictive uncertainty (1 - max_prob)
    predictive_uncertainty = 1 - max_prob
    
    return {
        "mean_entropy": entropy.mean().item(),
        "max_entropy": entropy.max().item(),
        "min_entropy": entropy.min().item(),
        "mean_variance": variance.mean().item(),
        "mean_max_prob": max_prob.mean().item(),
        "mean_predictive_uncertainty": predictive_uncertainty.mean().item()
    }


def analyze_token_uncertainty(
    tokens: List[int], 
    uncertainty_scores: List[float], 
    tokenizer, 
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Analyze uncertainty at the token level.
    
    Args:
        tokens: List of token IDs
        uncertainty_scores: List of uncertainty scores for each token
        tokenizer: Tokenizer to decode tokens
        top_k: Number of most uncertain tokens to return
        
    Returns:
        Dictionary with token-level analysis
    """
    if len(tokens) != len(uncertainty_scores):
        raise ValueError("Number of tokens and uncertainty scores must match")
    
    # Decode tokens
    decoded_tokens = [tokenizer.decode([token]) for token in tokens]
    
    # Create token-uncertainty pairs
    token_uncertainty_pairs = list(zip(decoded_tokens, uncertainty_scores, tokens))
    
    # Sort by uncertainty (descending)
    sorted_pairs = sorted(token_uncertainty_pairs, key=lambda x: x[1], reverse=True)
    
    # Get top uncertain tokens
    top_uncertain = sorted_pairs[:top_k]
    
    # Calculate statistics
    stats = {
        "total_tokens": len(tokens),
        "mean_uncertainty": np.mean(uncertainty_scores),
        "std_uncertainty": np.std(uncertainty_scores),
        "max_uncertainty": max(uncertainty_scores),
        "min_uncertainty": min(uncertainty_scores),
        "top_uncertain_tokens": [
            {
                "token": token,
                "uncertainty": score,
                "token_id": token_id,
                "rank": i + 1
            }
            for i, (token, score, token_id) in enumerate(top_uncertain)
        ]
    }
    
    return stats


def detect_uncertainty_patterns(generation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect patterns in uncertainty across multiple generations.
    
    Args:
        generation_history: List of generation results
        
    Returns:
        Dictionary with pattern analysis
    """
    if not generation_history:
        return {"message": "No generation history available"}
    
    # Extract uncertainty statistics
    uncertainty_rates = []
    attempt_counts = []
    backtrack_positions = []
    
    for gen in generation_history:
        if gen["uncertainty_minimized"]:
            uncertainty_rates.append(1)
            attempt_counts.append(gen["total_attempts"])
            
            # Find backtrack positions
            for attempt in gen["attempts"]:
                if attempt["uncertainty_detected"]:
                    backtrack_positions.extend(attempt["uncertain_indices"])
        else:
            uncertainty_rates.append(0)
            attempt_counts.append(gen["total_attempts"])
    
    # Calculate pattern statistics
    patterns = {
        "uncertainty_frequency": np.mean(uncertainty_rates),
        "average_attempts": np.mean(attempt_counts),
        "backtrack_position_distribution": Counter(backtrack_positions),
        "total_backtracks": len(backtrack_positions),
        "success_rate": 1 - np.mean(uncertainty_rates)  # Lower uncertainty = higher success
    }
    
    return patterns


def visualize_uncertainty_distribution(
    uncertainty_scores: List[float], 
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the distribution of uncertainty scores.
    
    Args:
        uncertainty_scores: List of uncertainty scores
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.subplot(1, 2, 1)
    plt.hist(uncertainty_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Uncertainty Scores')
    plt.grid(True, alpha=0.3)
    
    # Create box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(uncertainty_scores, vert=True)
    plt.ylabel('Uncertainty Score')
    plt.title('Uncertainty Score Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def extract_cot_reasoning(text: str, cot_trigger: str = "<think>") -> Dict[str, str]:
    """
    Extract Chain-of-Thought reasoning from generated text.
    
    Args:
        text: Generated text containing CoT reasoning
        cot_trigger: Token that triggers CoT reasoning
        
    Returns:
        Dictionary with extracted reasoning and final answer
    """
    # Find CoT sections
    cot_pattern = re.escape(cot_trigger) + r'(.*?)(?=' + re.escape(cot_trigger) + r'|$)'
    cot_matches = re.findall(cot_pattern, text, re.DOTALL)
    
    # Extract reasoning steps
    reasoning_steps = []
    for match in cot_matches:
        steps = [step.strip() for step in match.split('\n') if step.strip()]
        reasoning_steps.extend(steps)
    
    # Extract final answer (text after last CoT section)
    if cot_matches:
        last_cot_end = text.rfind(cot_trigger)
        if last_cot_end != -1:
            # Find the end of the last CoT section
            remaining_text = text[last_cot_end + len(cot_trigger):]
            final_answer = remaining_text.strip()
        else:
            final_answer = text
    else:
        final_answer = text
    
    return {
        "reasoning_steps": reasoning_steps,
        "final_answer": final_answer,
        "cot_sections_found": len(cot_matches),
        "has_cot_reasoning": len(cot_matches) > 0
    }


def calculate_generation_quality_metrics(
    original_text: str, 
    uncertainty_minimized_text: str
) -> Dict[str, float]:
    """
    Calculate quality metrics comparing original and uncertainty-minimized generations.
    
    Args:
        original_text: Original generated text
        uncertainty_minimized_text: Text after uncertainty minimization
        
    Returns:
        Dictionary with quality metrics
    """
    # Length metrics
    original_length = len(original_text.split())
    minimized_length = len(uncertainty_minimized_text.split())
    
    # Repetition metrics
    original_words = original_text.split()
    minimized_words = uncertainty_minimized_text.split()
    
    original_unique_ratio = len(set(original_words)) / len(original_words) if original_words else 0
    minimized_unique_ratio = len(set(minimized_words)) / len(minimized_words) if minimized_words else 0
    
    # Coherence metrics (simple heuristics)
    original_sentences = original_text.split('.')
    minimized_sentences = uncertainty_minimized_text.split('.')
    
    avg_original_sentence_length = np.mean([len(s.split()) for s in original_sentences if s.strip()])
    avg_minimized_sentence_length = np.mean([len(s.split()) for s in minimized_sentences if s.strip()])
    
    return {
        "length_ratio": minimized_length / original_length if original_length > 0 else 0,
        "unique_word_ratio_improvement": minimized_unique_ratio - original_unique_ratio,
        "sentence_length_ratio": avg_minimized_sentence_length / avg_original_sentence_length if avg_original_sentence_length > 0 else 0,
        "original_length": original_length,
        "minimized_length": minimized_length
    }


def format_generation_report(generation_result: Dict[str, Any]) -> str:
    """
    Format a generation result into a human-readable report.
    
    Args:
        generation_result: Result from uncertainty minimization
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 50)
    report.append("UNCERTAINTY MINIMIZATION REPORT")
    report.append("=" * 50)
    
    # Summary
    report.append(f"Total Attempts: {generation_result['total_attempts']}")
    report.append(f"Uncertainty Detected: {'Yes' if generation_result['uncertainty_minimized'] else 'No'}")
    report.append(f"Input Length: {generation_result['original_input_length']} tokens")
    report.append(f"Output Length: {len(generation_result['final_tokens'])} tokens")
    
    report.append("\n" + "-" * 30)
    report.append("FINAL OUTPUT:")
    report.append("-" * 30)
    report.append(generation_result['final_text'])
    
    # Attempt details
    if generation_result['attempts']:
        report.append("\n" + "-" * 30)
        report.append("ATTEMPT DETAILS:")
        report.append("-" * 30)
        
        for i, attempt in enumerate(generation_result['attempts']):
            report.append(f"\nAttempt {i + 1}:")
            report.append(f"  Uncertainty Detected: {attempt['uncertainty_detected']}")
            if attempt['uncertainty_detected']:
                report.append(f"  Uncertain Positions: {attempt['uncertain_indices']}")
            report.append(f"  Generated Tokens: {len(attempt['generated_tokens'])}")
    
    report.append("\n" + "=" * 50)
    
    return "\n".join(report)
