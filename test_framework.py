"""
Quick test script to verify the UMinFramework works correctly.

This script provides a lightweight test to ensure the uncertainty minimization
framework is working properly with the existing UQ head setup.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uncertainty_minimizer import UncertaintyMinimizer
from config import get_config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_functionality():
    """Test basic functionality of the framework."""
    print("Testing UMinFramework basic functionality...")
    
    try:
        # Use balanced configuration
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
        
        print("✓ Successfully initialized UncertaintyMinimizer")
        
        # Test with the same question from basicUsage.py
        test_question = {
            "role": "user",
            "content": "How many fingers are on a koala's foot?"
        }
        
        print(f"Testing with question: {test_question['content']}")
        
        # Generate with uncertainty minimization
        result = minimizer.generate_with_uncertainty_minimization(
            messages=[test_question],
            max_length=200,  # Shorter for testing
            temperature=0.7
        )
        
        print("✓ Successfully generated response with uncertainty minimization")
        
        # Display results
        print("\nRESULTS:")
        print(f"Total attempts: {result['total_attempts']}")
        print(f"Uncertainty detected: {result['uncertainty_minimized']}")
        print(f"Final response: {result['final_text']}")
        
        # Test history analysis
        history = minimizer.analyze_generation_history()
        print(f"\nHistory analysis:")
        print(f"Total generations: {history['total_generations']}")
        print(f"Uncertainty minimization rate: {history['uncertainty_minimization_rate']:.2%}")
        
        print("\n✓ All tests passed! Framework is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


def test_configuration_system():
    """Test the configuration system."""
    print("\nTesting configuration system...")
    
    try:
        # Test predefined configs
        configs = ["conservative", "balanced", "aggressive", "research"]
        
        for config_name in configs:
            config = get_config(config_name)
            print(f"✓ Successfully loaded {config_name} configuration")
            print(f"  Uncertainty threshold: {config.uncertainty_threshold}")
            print(f"  Max attempts: {config.max_backtrack_attempts}")
            print(f"  CoT trigger: {config.cot_trigger_token}")
        
        # Test custom config
        from config import create_custom_config
        custom_config = create_custom_config(
            uncertainty_threshold=0.6,
            max_backtrack_attempts=4,
            temperature=0.8
        )
        print("✓ Successfully created custom configuration")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from utils import extract_cot_reasoning, format_generation_report
        
        # Test CoT extraction
        sample_text = "Let me think about this. <think> This is a reasoning step. </think> Here's my answer."
        cot_result = extract_cot_reasoning(sample_text)
        print(f"✓ CoT extraction works: {cot_result['has_cot_reasoning']}")
        
        # Test report formatting
        sample_result = {
            'total_attempts': 2,
            'uncertainty_minimized': True,
            'original_input_length': 10,
            'final_tokens': [1, 2, 3, 4, 5],
            'final_text': "Test response",
            'attempts': [
                {'uncertainty_detected': True, 'uncertain_indices': [2, 3], 'generated_tokens': [1, 2, 3]}
            ]
        }
        
        report = format_generation_report(sample_result)
        print("✓ Report formatting works")
        
        return True
        
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("UMinFramework Quick Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_configuration_system()
    all_passed &= test_utilities()
    
    # Only run the model test if requested (it requires GPU and models)
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        all_passed &= test_basic_functionality()
    else:
        print("\nSkipping model test (use --full to run complete test)")
        print("Note: Model test requires GPU and pre-trained models")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! Framework appears to be working correctly.")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        sys.exit(1)
