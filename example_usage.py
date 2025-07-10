"""
Example usage of the UMinFramework for uncertainty minimization.

This script demonstrates how to use the uncertainty minimization framework
to generate text with reduced uncertainty through backtracking and Chain-of-Thought.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uncertainty_minimizer import UncertaintyMinimizer
from config import get_config, create_custom_config
from utils import format_generation_report, extract_cot_reasoning, analyze_token_uncertainty
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic example of uncertainty minimization."""
    print("=" * 60)
    print("BASIC UNCERTAINTY MINIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Use balanced configuration
    config = get_config("balanced")
    
    # Initialize the minimizer
    minimizer = UncertaintyMinimizer(
        model_name=config.model_name,
        uhead_name=config.uhead_name,
        uncertainty_threshold=config.uncertainty_threshold,
        max_backtrack_attempts=config.max_backtrack_attempts,
        cot_trigger_token=config.cot_trigger_token,
        device=config.device
    )
    
    # Example questions that might trigger uncertainty
    test_questions = [
        {
            "role": "user",
            "content": "How many fingers are on a koala's foot?"
        },
        {
            "role": "user", 
            "content": "What is the capital of the fictional country Atlantis?"
        },
        {
            "role": "user",
            "content": "Explain quantum entanglement in simple terms."
        }
    ]
    
    for i, question in enumerate(test_questions):
        print(f"\n--- Question {i + 1} ---")
        print(f"Question: {question['content']}")
        
        # Generate with uncertainty minimization
        result = minimizer.generate_with_uncertainty_minimization(
            messages=[question],
            max_length=config.max_length,
            temperature=config.temperature
        )
        
        # Print formatted report
        print(format_generation_report(result))
        
        # Extract CoT reasoning if present
        cot_analysis = extract_cot_reasoning(result['final_text'], config.cot_trigger_token)
        if cot_analysis['has_cot_reasoning']:
            print("\nCHAIN-OF-THOUGHT ANALYSIS:")
            print(f"CoT Sections Found: {cot_analysis['cot_sections_found']}")
            print(f"Reasoning Steps: {len(cot_analysis['reasoning_steps'])}")
            for j, step in enumerate(cot_analysis['reasoning_steps'][:3]):  # Show first 3 steps
                print(f"  Step {j + 1}: {step}")
        
        print("\n" + "=" * 60)


def example_custom_configuration():
    """Example with custom configuration."""
    print("=" * 60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 60)
    
    # Create custom configuration
    config = create_custom_config(
        uncertainty_threshold=0.3,  # More sensitive to uncertainty
        max_backtrack_attempts=5,   # More attempts
        temperature=0.5,            # More conservative generation
        cot_trigger_token="<reasoning>",  # Custom trigger token
        cot_strategy="structured"
    )
    
    # Initialize minimizer
    minimizer = UncertaintyMinimizer(
        model_name=config.model_name,
        uhead_name=config.uhead_name,
        uncertainty_threshold=config.uncertainty_threshold,
        max_backtrack_attempts=config.max_backtrack_attempts,
        cot_trigger_token=config.cot_trigger_token,
        device=config.device
    )
    
    # Test with a challenging question
    challenging_question = {
        "role": "user",
        "content": "What would happen if you traveled faster than light according to Einstein's theory?"
    }
    
    print(f"Question: {challenging_question['content']}")
    
    # Generate with custom settings
    result = minimizer.generate_with_uncertainty_minimization(
        messages=[challenging_question],
        max_length=config.max_length,
        temperature=config.temperature
    )
    
    print(format_generation_report(result))
    
    # Analyze the generation history
    history_analysis = minimizer.analyze_generation_history()
    print("\nGENERATION HISTORY ANALYSIS:")
    print(f"Total generations: {history_analysis['total_generations']}")
    print(f"Uncertainty minimization rate: {history_analysis['uncertainty_minimization_rate']:.2%}")
    print(f"Average attempts per generation: {history_analysis['average_attempts_per_generation']:.2f}")


def example_batch_processing():
    """Example of processing multiple questions in batch."""
    print("=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    # Use research configuration for detailed analysis
    config = get_config("research")
    
    minimizer = UncertaintyMinimizer(
        model_name=config.model_name,
        uhead_name=config.uhead_name,
        uncertainty_threshold=config.uncertainty_threshold,
        max_backtrack_attempts=config.max_backtrack_attempts,
        cot_trigger_token=config.cot_trigger_token,
        device=config.device
    )
    
    # Batch of questions
    questions = [
        "What is the meaning of life?",
        "How do you make a perfect chocolate chip cookie?",
        "Explain the difference between AI and machine learning.",
        "What would you do if you were invisible for a day?",
        "How does photosynthesis work?"
    ]
    
    results = []
    for question in questions:
        message = {"role": "user", "content": question}
        result = minimizer.generate_with_uncertainty_minimization(
            messages=[message],
            max_length=config.max_length,
            temperature=config.temperature
        )
        results.append(result)
        print(f"Processed: {question[:50]}...")
    
    # Analyze batch results
    history_analysis = minimizer.analyze_generation_history()
    
    print("\nBATCH ANALYSIS SUMMARY:")
    print(f"Questions processed: {len(questions)}")
    print(f"Questions with uncertainty: {history_analysis['uncertainty_minimized_count']}")
    print(f"Uncertainty rate: {history_analysis['uncertainty_minimization_rate']:.2%}")
    print(f"Average attempts: {history_analysis['average_attempts_per_generation']:.2f}")
    
    # Show most challenging question
    most_attempts = max(results, key=lambda x: x['total_attempts'])
    print(f"\nMost challenging question required {most_attempts['total_attempts']} attempts")
    print("Generated text preview:", most_attempts['final_text'][:200] + "...")


def example_comparison_mode():
    """Example comparing regular generation vs uncertainty minimization."""
    print("=" * 60)
    print("COMPARISON MODE EXAMPLE")
    print("=" * 60)
    
    config = get_config("balanced")
    
    # Initialize minimizer
    minimizer = UncertaintyMinimizer(
        model_name=config.model_name,
        uhead_name=config.uhead_name,
        uncertainty_threshold=config.uncertainty_threshold,
        max_backtrack_attempts=config.max_backtrack_attempts,
        cot_trigger_token=config.cot_trigger_token,
        device=config.device
    )
    
    # Test question
    test_question = {
        "role": "user",
        "content": "What are the potential risks and benefits of genetic engineering?"
    }
    
    print(f"Question: {test_question['content']}")
    
    # Generate with uncertainty minimization
    result_with_minimization = minimizer.generate_with_uncertainty_minimization(
        messages=[test_question],
        max_length=config.max_length,
        temperature=config.temperature
    )
    
    # For comparison, we'd need to generate without uncertainty minimization
    # This would require modifying the basic generation to skip uncertainty checking
    print("\nWITH UNCERTAINTY MINIMIZATION:")
    print("=" * 40)
    print(result_with_minimization['final_text'])
    print(f"\nAttempts required: {result_with_minimization['total_attempts']}")
    print(f"Uncertainty detected: {result_with_minimization['uncertainty_minimized']}")


if __name__ == "__main__":
    print("UMinFramework - Uncertainty Minimization Examples")
    print("=" * 60)
    
    # Check if models are available
    try:
        # Run examples
        example_basic_usage()
        
        print("\n" + "=" * 60)
        input("Press Enter to continue to custom configuration example...")
        
        example_custom_configuration()
        
        print("\n" + "=" * 60)
        input("Press Enter to continue to batch processing example...")
        
        example_batch_processing()
        
        print("\n" + "=" * 60)
        input("Press Enter to continue to comparison example...")
        
        example_comparison_mode()
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        print(f"Error: {e}")
        print("Make sure you have the required models and dependencies installed.")
        print("Check the README.md for installation instructions.")
