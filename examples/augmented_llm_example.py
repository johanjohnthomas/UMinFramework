#!/usr/bin/env python3
"""
Example usage of the AugmentedLLM class.

This script demonstrates how to use the AugmentedLLM wrapper class to combine
prompt refinement, uncertainty quantification, and backtracking generation.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from umin_framework import AugmentedLLM, AugmentedLLMConfig
    print("✓ Successfully imported UMinFramework components")
except ImportError as e:
    print(f"❌ Failed to import UMinFramework: {e}")
    print("Make sure transformers and torch are installed: pip install transformers torch")
    sys.exit(1)


def example_basic_usage():
    """Example of basic AugmentedLLM usage with default configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic AugmentedLLM Usage")
    print("="*60)
    
    try:
        # Create configuration
        config = AugmentedLLMConfig(
            generation_model="mistralai/Mistral-7B-Instruct-v0.2",  # Primary supported model
            enable_prompt_refinement=False,  # Disable since we don't have a trained refiner
            enable_uncertainty_monitoring=True,
            enable_backtracking=True,
            uncertainty_threshold=0.8,
            max_length=50
        )
        
        print(f"Configuration:")
        print(f"  - Model: {config.generation_model}")
        print(f"  - Prompt refinement: {config.enable_prompt_refinement}")
        print(f"  - Uncertainty monitoring: {config.enable_uncertainty_monitoring}")
        print(f"  - Backtracking: {config.enable_backtracking}")
        print(f"  - Uncertainty threshold: {config.uncertainty_threshold}")
        
        # Initialize AugmentedLLM
        print("\nInitializing AugmentedLLM...")
        augmented_llm = AugmentedLLM(config=config)
        print("✓ AugmentedLLM initialized successfully!")
        
        # Test prompts
        test_prompts = [
            "The capital of France is",
            "To solve this math problem, I need to",
            "The best programming language for beginners is"
        ]
        
        print(f"\nGenerating responses for {len(test_prompts)} prompts...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Prompt {i} ---")
            print(f"Input: {prompt}")
            
            try:
                result = augmented_llm.generate(prompt, return_metadata=True)
                
                print(f"Output: {result['text']}")
                print(f"Generated tokens: {result['generated_tokens']}")
                print(f"Average uncertainty: {result['avg_uncertainty']:.3f}")
                print(f"Backtrack events: {result['backtrack_events']}")
                
                if result['backtrack_events'] > 0:
                    print("⚠️ Backtracking occurred during generation")
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
    
    except Exception as e:
        print(f"❌ Example failed: {e}")


def example_with_prompt_refinement():
    """Example with prompt refinement (if available)."""
    print("\n" + "="*60)
    print("EXAMPLE 2: AugmentedLLM with Prompt Refinement")
    print("="*60)
    
    # Check if prompt refiner model exists
    refiner_path = Path("models/prompt_refiner")
    if not refiner_path.exists():
        print(f"❌ Prompt refiner model not found at {refiner_path}")
        print("To use prompt refinement:")
        print("1. Run the fine-tuning script: python scripts/finetune_prompt_refiner.py")
        print("2. Or disable prompt refinement in the config")
        return
    
    try:
        config = AugmentedLLMConfig(
            generation_model="mistralai/Mistral-7B-Instruct-v0.2",
            prompt_refiner_model=str(refiner_path),
            enable_prompt_refinement=True,
            enable_uncertainty_monitoring=True,
            enable_backtracking=True,
            max_length=50
        )
        
        print(f"Configuration with prompt refinement enabled:")
        print(f"  - Refiner model: {config.prompt_refiner_model}")
        
        augmented_llm = AugmentedLLM(config=config)
        print("✓ AugmentedLLM with prompt refinement initialized!")
        
        # Test with ambiguous prompt
        ambiguous_prompt = "What is the best solution?"
        print(f"\nTesting with ambiguous prompt: '{ambiguous_prompt}'")
        
        result = augmented_llm.generate(ambiguous_prompt, return_metadata=True)
        
        print(f"Original prompt: {result['original_prompt']}")
        print(f"Refined prompt: {result['refined_prompt']}")
        print(f"Prompt was refined: {result['prompt_was_refined']}")
        print(f"Generated text: {result['text']}")
        
    except Exception as e:
        print(f"❌ Example with prompt refinement failed: {e}")


def example_configuration_options():
    """Example showing different configuration options."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Configuration Options")
    print("="*60)
    
    configurations = [
        {
            "name": "Conservative (High Uncertainty Threshold)",
            "config": AugmentedLLMConfig(
                generation_model="mistralai/Mistral-7B-Instruct-v0.2",
                enable_prompt_refinement=False,
                uncertainty_threshold=0.9,  # Very conservative
                backtrack_window=2,
                max_length=30
            )
        },
        {
            "name": "Aggressive (Low Uncertainty Threshold)",
            "config": AugmentedLLMConfig(
                generation_model="mistralai/Mistral-7B-Instruct-v0.2",
                enable_prompt_refinement=False,
                uncertainty_threshold=0.3,  # Very aggressive
                backtrack_window=5,
                max_length=30
            )
        },
        {
            "name": "No Backtracking",
            "config": AugmentedLLMConfig(
                generation_model="mistralai/Mistral-7B-Instruct-v0.2",
                enable_prompt_refinement=False,
                enable_backtracking=False,
                max_length=30
            )
        }
    ]
    
    test_prompt = "The solution to this problem is"
    
    for config_info in configurations:
        print(f"\n--- {config_info['name']} ---")
        config = config_info["config"]
        
        try:
            augmented_llm = AugmentedLLM(config=config)
            result = augmented_llm.generate(test_prompt, return_metadata=True)
            
            print(f"Generated: {result['text']}")
            print(f"Avg uncertainty: {result['avg_uncertainty']:.3f}")
            print(f"Backtrack events: {result['backtrack_events']}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")


def example_batch_generation():
    """Example of batch generation."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Generation")
    print("="*60)
    
    try:
        config = AugmentedLLMConfig(
            generation_model="mistralai/Mistral-7B-Instruct-v0.2",
            enable_prompt_refinement=False,
            max_length=40
        )
        
        augmented_llm = AugmentedLLM(config=config)
        
        prompts = [
            "Python is a programming language that",
            "Machine learning helps us",
            "The future of AI will be",
            "Data science involves",
            "Neural networks are"
        ]
        
        print(f"Generating responses for {len(prompts)} prompts in batch...")
        
        results = augmented_llm.generate_batch(prompts, return_metadata=False)
        
        for i, (prompt, response) in enumerate(zip(prompts, results), 1):
            print(f"\n{i}. Prompt: {prompt}")
            print(f"   Response: {response}")
            
    except Exception as e:
        print(f"❌ Batch generation example failed: {e}")


def example_pipeline_info():
    """Example showing how to get pipeline information."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Pipeline Information")
    print("="*60)
    
    try:
        config = AugmentedLLMConfig(generation_model="mistralai/Mistral-7B-Instruct-v0.2", enable_prompt_refinement=False)
        augmented_llm = AugmentedLLM(config=config)
        
        info = augmented_llm.get_pipeline_info()
        
        print("Pipeline Information:")
        print(f"  Model: {info['components']['generation_model']['name']}")
        print(f"  Parameters: {info['components']['generation_model']['parameters']:,}")
        print(f"  Device: {info['components']['generation_model']['device']}")
        
        print("\nEnabled Components:")
        for component, enabled in info['pipeline_enabled'].items():
            status = "✓" if enabled else "✗"
            print(f"  {status} {component.replace('_', ' ').title()}")
        
        print(f"\nConfiguration:")
        print(f"  Uncertainty threshold: {info['config']['uncertainty_threshold']}")
        print(f"  Backtrack window: {info['config']['backtrack_window']}")
        print(f"  Max length: {info['config']['max_length']}")
        
    except Exception as e:
        print(f"❌ Pipeline info example failed: {e}")


if __name__ == "__main__":
    print("AugmentedLLM Examples")
    print("===================")
    
    try:
        # Run examples
        example_basic_usage()
        example_with_prompt_refinement()
        example_configuration_options()
        example_batch_generation()
        example_pipeline_info()
        
        print("\n" + "="*60)
        print("✅ All examples completed!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n❌ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed with error: {e}")
        import traceback
        traceback.print_exc()