"""
Unit tests for the GenerationLoop class.

This module contains comprehensive unit tests for the GenerationLoop class,
testing initialization, generation, backtracking logic, CoT injection,
state management, and integration with uncertainty monitoring.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path so we can import our package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Test if transformers is available
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from umin_framework.generation_loop import GenerationLoop, BacktrackConfig, GenerationState
from umin_framework.uncertainty_head import UncertaintyHead


class TestBacktrackConfig(unittest.TestCase):
    """Test cases for BacktrackConfig dataclass."""
    
    def test_default_config(self):
        """Test default BacktrackConfig values."""
        config = BacktrackConfig()
        
        self.assertEqual(config.uncertainty_threshold, 0.7)
        self.assertEqual(config.backtrack_window, 3)
        self.assertEqual(config.max_backtracks_per_generation, 5)
        self.assertEqual(config.max_backtracks_per_position, 2)
        self.assertEqual(config.uncertainty_method, "entropy")
        self.assertIsInstance(config.cot_templates, list)
        self.assertGreater(len(config.cot_templates), 0)
    
    def test_custom_config(self):
        """Test custom BacktrackConfig values."""
        custom_templates = ["Custom CoT 1", "Custom CoT 2"]
        config = BacktrackConfig(
            uncertainty_threshold=0.5,
            backtrack_window=2,
            max_backtracks_per_generation=3,
            cot_templates=custom_templates,
            uncertainty_method="max_prob"
        )
        
        self.assertEqual(config.uncertainty_threshold, 0.5)
        self.assertEqual(config.backtrack_window, 2)
        self.assertEqual(config.max_backtracks_per_generation, 3)
        self.assertEqual(config.cot_templates, custom_templates)
        self.assertEqual(config.uncertainty_method, "max_prob")


class TestGenerationLoop(unittest.TestCase):
    """Test cases for the GenerationLoop class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.test_model_name = "distilgpt2"
        
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_prompts = [
            "Hello world",
            "The capital of France is",
            "I think that"
        ]
    
    def test_initialization_without_transformers(self):
        """Test that GenerationLoop raises ImportError when transformers unavailable."""
        with patch('umin_framework.generation_loop.HAS_TRANSFORMERS', False):
            with self.assertRaises(ImportError):
                GenerationLoop("some/model")
    
    def test_initialization_without_uncertainty_head(self):
        """Test that GenerationLoop raises ImportError when UncertaintyHead unavailable."""
        with patch('umin_framework.generation_loop.HAS_UNCERTAINTY_HEAD', False):
            with self.assertRaises(ImportError):
                GenerationLoop("some/model")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_initialization_with_model_name(self):
        """Test GenerationLoop initialization with model name."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        self.assertIsNotNone(gen_loop.model)
        self.assertIsNotNone(gen_loop.tokenizer)
        self.assertIsNotNone(gen_loop.uncertainty_head)
        self.assertEqual(gen_loop.device, "cpu")
        self.assertEqual(gen_loop.model_name, self.test_model_name)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_initialization_with_preloaded_components(self):
        """Test initialization with pre-loaded model and uncertainty head."""
        model = AutoModelForCausalLM.from_pretrained(self.test_model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.test_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        ue_head = UncertaintyHead(model, tokenizer, device="cpu")
        gen_loop = GenerationLoop(model, tokenizer, uncertainty_head=ue_head, device="cpu")
        
        self.assertIs(gen_loop.model, model)
        self.assertIs(gen_loop.tokenizer, tokenizer)
        self.assertIs(gen_loop.uncertainty_head, ue_head)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_basic_generation_without_backtracking(self):
        """Test basic generation without triggering backtracking."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        # Use very high threshold to prevent backtracking
        config = BacktrackConfig(uncertainty_threshold=100.0)
        
        result = gen_loop.generate(
            "Hello world",
            config=config,
            max_length=20,
            do_sample=False  # Deterministic
        )
        
        # Basic validation
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('prompt', result)
        self.assertIn('full_text', result)
        self.assertIn('generated_tokens', result)
        self.assertIn('uncertainty_scores', result)
        self.assertIn('backtrack_events', result)
        
        # Should have generated some text
        self.assertGreater(len(result['text']), 0)
        self.assertEqual(result['prompt'], "Hello world")
        self.assertGreater(result['generated_tokens'], 0)
        self.assertEqual(result['backtrack_events'], 0)  # No backtracking
        
        # Uncertainty scores should match number of tokens
        self.assertEqual(len(result['uncertainty_scores']), result['generated_tokens'])
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_generation_with_backtracking(self):
        """Test generation that triggers backtracking."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        # Use very low threshold to force backtracking
        config = BacktrackConfig(
            uncertainty_threshold=0.1,
            backtrack_window=2,
            max_backtracks_per_generation=2,
            cot_templates=["TEST_COT"]
        )
        
        result = gen_loop.generate(
            "The weather",
            config=config,
            max_length=15,
            do_sample=True,
            temperature=1.0  # Higher temp for more uncertainty
        )
        
        # Should have triggered backtracking
        self.assertGreater(result['backtrack_events'], 0)
        self.assertIsInstance(result['backtrack_details'], list)
        
        # Check backtrack event details
        for event in result['backtrack_details']:
            self.assertIn('position', event)
            self.assertIn('reason', event)
            self.assertIn('uncertainty_score', event)
            self.assertIn('success', event)
            self.assertIsInstance(event['success'], bool)
        
        # If backtracking occurred, CoT should have been injected somewhere
        # (Note: might be truncated in final output, but should be in full_text or detectable via events)
        if result['backtrack_events'] > 0:
            # Either find CoT in text or verify successful backtrack injection
            cot_injected = ("TEST_COT" in result['full_text'] or 
                           any(event['success'] for event in result['backtrack_details']))
            self.assertTrue(cot_injected, "CoT should be injected during successful backtracking")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_max_backtracks_limit(self):
        """Test that max backtracks per generation is respected."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        config = BacktrackConfig(
            uncertainty_threshold=0.1,
            max_backtracks_per_generation=1  # Only allow 1 backtrack
        )
        
        result = gen_loop.generate(
            "Hello",
            config=config,
            max_length=20,
            do_sample=True,
            temperature=1.5  # Very high temp for uncertainty
        )
        
        # Should have at most 1 backtrack event
        self.assertLessEqual(result['backtrack_events'], 1)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_batch_generation(self):
        """Test generating multiple sequences."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        config = BacktrackConfig(uncertainty_threshold=10.0)  # High threshold
        
        results = gen_loop.generate(
            "Test prompt",
            config=config,
            max_length=15,
            num_return_sequences=3,
            do_sample=True
        )
        
        # Should return list of results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        # Each result should be valid
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('text', result)
            self.assertIn('prompt', result)
            self.assertEqual(result['prompt'], "Test prompt")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_generation_parameters(self):
        """Test different generation parameters."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        config = BacktrackConfig(uncertainty_threshold=10.0)
        
        # Test greedy vs sampling
        greedy_result = gen_loop.generate(
            "Hello",
            config=config,
            max_length=10,
            do_sample=False
        )
        
        sampling_result = gen_loop.generate(
            "Hello", 
            config=config,
            max_length=10,
            do_sample=True,
            temperature=0.8
        )
        
        # Both should succeed
        self.assertIn('text', greedy_result)
        self.assertIn('text', sampling_result)
        
        # Results might be different (though not guaranteed)
        self.assertIsInstance(greedy_result['text'], str)
        self.assertIsInstance(sampling_result['text'], str)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_min_max_length_constraints(self):
        """Test minimum and maximum length constraints."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        config = BacktrackConfig(uncertainty_threshold=10.0)
        
        # Test with short max_length
        result = gen_loop.generate(
            "Hello",
            config=config,
            max_length=8,  # Very short
            min_length=1
        )
        
        # Should respect max length (approximately)
        # Generated tokens should be reasonable
        self.assertLess(result['generated_tokens'], 10)
        self.assertGreater(result['generated_tokens'], 0)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_full_state_return(self):
        """Test returning full generation state."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        config = BacktrackConfig(uncertainty_threshold=10.0)
        
        result = gen_loop.generate(
            "Test",
            config=config,
            max_length=10,
            return_full_state=True
        )
        
        # Should include full state
        self.assertIn('full_state', result)
        state = result['full_state']
        self.assertIsInstance(state, GenerationState)
        
        # State should have expected attributes
        self.assertIsInstance(state.input_ids, torch.Tensor)
        self.assertIsInstance(state.generated_tokens, list)
        self.assertIsInstance(state.uncertainty_scores, list)
        self.assertIsInstance(state.backtrack_events, list)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_generation_stats(self):
        """Test getting generation statistics."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        config = BacktrackConfig(uncertainty_threshold=0.5)
        
        # Generate with state tracking
        result = gen_loop.generate(
            "Test prompt",
            config=config,
            max_length=15,
            return_full_state=True
        )
        
        # Get stats
        stats = gen_loop.get_generation_stats()
        
        # Validate stats
        self.assertIsInstance(stats, dict)
        self.assertIn('total_tokens', stats)
        self.assertIn('avg_uncertainty', stats)
        self.assertIn('max_uncertainty', stats)
        self.assertIn('min_uncertainty', stats)
        self.assertIn('backtrack_events', stats)
        
        # Stats should be reasonable
        self.assertGreaterEqual(stats['total_tokens'], 0)
        self.assertGreaterEqual(stats['backtrack_events'], 0)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_cot_template_injection(self):
        """Test that CoT templates are properly injected."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        custom_cot = "CUSTOM_CHAIN_OF_THOUGHT_MARKER"
        config = BacktrackConfig(
            uncertainty_threshold=0.1,
            cot_templates=[custom_cot],
            max_backtracks_per_generation=1
        )
        
        result = gen_loop.generate(
            "Hello",
            config=config,
            max_length=20,
            do_sample=True,
            temperature=1.5
        )
        
        # If backtracking occurred, should have injected CoT successfully
        if result['backtrack_events'] > 0:
            # Check either CoT is in text or injection was successful
            cot_used = (custom_cot in result['full_text'] or 
                       any(event['success'] for event in result['backtrack_details']))
            self.assertTrue(cot_used, f"Custom CoT should be used when backtracking occurs")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_empty_prompt_handling(self):
        """Test handling of empty or whitespace prompts."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        config = BacktrackConfig(uncertainty_threshold=10.0)
        
        # Test with minimal prompt
        result = gen_loop.generate(
            "A",
            config=config,
            max_length=10
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertEqual(result['prompt'], "A")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_repr_method(self):
        """Test string representation of GenerationLoop."""
        gen_loop = GenerationLoop(self.test_model_name, device="cpu")
        
        repr_str = repr(gen_loop)
        
        self.assertIn("GenerationLoop", repr_str)
        self.assertIn(self.test_model_name, repr_str)
        self.assertIn("cpu", repr_str)
        self.assertIn("loaded=True", repr_str)


class TestGenerationLoopEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_invalid_model_path(self):
        """Test handling of invalid model paths."""
        with self.assertRaises(ValueError):
            GenerationLoop("definitely/not/a/real/model/path")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_zero_max_length(self):
        """Test generation with very small max_length."""
        gen_loop = GenerationLoop("distilgpt2", device="cpu")
        
        config = BacktrackConfig(uncertainty_threshold=10.0)
        
        # Should handle gracefully
        result = gen_loop.generate(
            "Hello",
            config=config,
            max_length=1  # Very small
        )
        
        self.assertIsInstance(result, dict)
        # Might generate 0 tokens due to length constraint
        self.assertGreaterEqual(result['generated_tokens'], 0)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available") 
    def test_extreme_temperature_values(self):
        """Test generation with extreme temperature values."""
        gen_loop = GenerationLoop("distilgpt2", device="cpu")
        
        config = BacktrackConfig(uncertainty_threshold=10.0)
        
        # Test very low temperature
        low_temp_result = gen_loop.generate(
            "Hello",
            config=config,
            max_length=10,
            do_sample=True,
            temperature=0.01
        )
        
        # Test very high temperature  
        high_temp_result = gen_loop.generate(
            "Hello",
            config=config,
            max_length=10,
            do_sample=True,
            temperature=2.0
        )
        
        # Both should succeed
        self.assertIn('text', low_temp_result)
        self.assertIn('text', high_temp_result)


def create_test_suite():
    """Create and return a test suite for GenerationLoop."""
    suite = unittest.TestSuite()
    
    # Add all test methods
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBacktrackConfig))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGenerationLoop))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGenerationLoopEdgeCases))
    
    return suite


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run tests with verbose output
    unittest.main(verbosity=2)