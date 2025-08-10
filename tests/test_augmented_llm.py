"""
Tests for the AugmentedLLM wrapper class.

This module contains comprehensive tests for the AugmentedLLM class,
including initialization, configuration, generation pipeline, and integration tests.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None

from umin_framework.augmented_llm import AugmentedLLM, AugmentedLLMConfig


class TestAugmentedLLMConfig(unittest.TestCase):
    """Test cases for AugmentedLLMConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AugmentedLLMConfig()
        
        self.assertTrue(config.enable_prompt_refinement)
        self.assertTrue(config.enable_uncertainty_monitoring)
        self.assertTrue(config.enable_backtracking)
        self.assertEqual(config.generation_model, "gpt2")
        self.assertEqual(config.uncertainty_threshold, 0.7)
        self.assertEqual(config.backtrack_window, 3)
        self.assertEqual(config.max_backtracks_per_generation, 5)
        self.assertEqual(config.uncertainty_method, "entropy")
        
        # Check that cot_templates are populated
        self.assertIsNotNone(config.cot_templates)
        self.assertGreater(len(config.cot_templates), 0)
        self.assertIn(" Let me think step by step.", config.cot_templates)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_templates = [" Custom CoT template."]
        
        config = AugmentedLLMConfig(
            enable_prompt_refinement=False,
            generation_model="microsoft/DialoGPT-small",
            uncertainty_threshold=0.8,
            backtrack_window=5,
            cot_templates=custom_templates
        )
        
        self.assertFalse(config.enable_prompt_refinement)
        self.assertEqual(config.generation_model, "microsoft/DialoGPT-small")
        self.assertEqual(config.uncertainty_threshold, 0.8)
        self.assertEqual(config.backtrack_window, 5)
        self.assertEqual(config.cot_templates, custom_templates)
    
    def test_default_prompt_refiner_path(self):
        """Test automatic detection of default prompt refiner path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a fake models directory
            models_dir = Path(tmp_dir) / "models" / "prompt_refiner"
            models_dir.mkdir(parents=True)
            
            # Change to temp directory and create config
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                config = AugmentedLLMConfig()
                self.assertEqual(config.prompt_refiner_model, "models/prompt_refiner")
            finally:
                os.chdir(original_cwd)


@unittest.skipUnless(HAS_TRANSFORMERS, "Transformers not available")
class TestAugmentedLLM(unittest.TestCase):
    """Test cases for AugmentedLLM class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_prompt = "What is the capital of France?"
        
        # Create mock components
        self.mock_prompt_refiner = Mock()
        self.mock_prompt_refiner.refine.return_value = "What is the capital of France? Please specify the historical period."
        self.mock_prompt_refiner.get_model_info.return_value = {"model_path": "mock", "loaded": True}
        
        self.mock_uncertainty_head = Mock()
        self.mock_uncertainty_head.get_model_info.return_value = {"model_name": "mock", "loaded": True}
        
        self.mock_generation_loop = Mock()
        self.mock_generation_loop.uncertainty_head = self.mock_uncertainty_head
        self.mock_generation_loop.tokenizer = Mock()
        self.mock_generation_loop.model = Mock()
        self.mock_generation_loop.model.name_or_path = "mock-model"
        self.mock_generation_loop.model.num_parameters.return_value = 1000000
        self.mock_generation_loop.model.device = "cpu"
        
        # Mock generation result - create a function that returns different results based on prompt
        def mock_generate(prompt, **kwargs):
            return {
                "text": f"Generated response for: {prompt}",
                "prompt": prompt,
                "full_text": prompt + f"Generated response for: {prompt}",
                "generated_tokens": 7,
                "uncertainty_scores": [0.1, 0.2, 0.15, 0.3, 0.25, 0.1, 0.05],
                "avg_uncertainty": 0.16,
                "backtrack_events": 1,
                "backtrack_details": [],
                "total_steps": 8,
                "config": None
            }
        
        self.mock_generation_loop.generate.side_effect = mock_generate
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', True)
    def test_initialization_with_preloaded_components(self):
        """Test initialization with pre-loaded components."""
        config = AugmentedLLMConfig()
        
        augmented_llm = AugmentedLLM(
            config=config,
            prompt_refiner=self.mock_prompt_refiner,
            generation_loop=self.mock_generation_loop,
            uncertainty_head=self.mock_uncertainty_head
        )
        
        self.assertIsNotNone(augmented_llm)
        self.assertEqual(augmented_llm.prompt_refiner, self.mock_prompt_refiner)
        self.assertEqual(augmented_llm.generation_loop, self.mock_generation_loop)
        self.assertEqual(augmented_llm.uncertainty_head, self.mock_uncertainty_head)
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', False)
    def test_initialization_fails_without_components(self):
        """Test that initialization fails when components are not available."""
        config = AugmentedLLMConfig()
        
        with self.assertRaises(ImportError) as context:
            AugmentedLLM(config=config)
        
        self.assertIn("UMinFramework components are required", str(context.exception))
    
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', False)
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    def test_initialization_fails_without_transformers(self):
        """Test that initialization fails when transformers is not available."""
        config = AugmentedLLMConfig()
        
        with self.assertRaises(ImportError) as context:
            AugmentedLLM(config=config)
        
        self.assertIn("Transformers and PyTorch are required", str(context.exception))
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', True)
    def test_generate_with_full_pipeline(self):
        """Test generation with full pipeline enabled."""
        config = AugmentedLLMConfig(
            enable_prompt_refinement=True,
            enable_uncertainty_monitoring=True,
            enable_backtracking=True
        )
        
        augmented_llm = AugmentedLLM(
            config=config,
            prompt_refiner=self.mock_prompt_refiner,
            generation_loop=self.mock_generation_loop,
            uncertainty_head=self.mock_uncertainty_head
        )
        
        # Test generation
        result = augmented_llm.generate(self.test_prompt, return_metadata=True)
        
        # Verify prompt was refined
        self.mock_prompt_refiner.refine.assert_called_once_with(self.test_prompt)
        
        # Verify generation was called with refined prompt
        self.mock_generation_loop.generate.assert_called_once()
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertIn("original_prompt", result)
        self.assertIn("refined_prompt", result)
        self.assertIn("prompt_was_refined", result)
        self.assertIn("pipeline_config", result)
        
        self.assertEqual(result["original_prompt"], self.test_prompt)
        self.assertNotEqual(result["refined_prompt"], self.test_prompt)
        self.assertTrue(result["prompt_was_refined"])
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', True)
    def test_generate_without_prompt_refinement(self):
        """Test generation with prompt refinement disabled."""
        config = AugmentedLLMConfig(enable_prompt_refinement=False)
        
        augmented_llm = AugmentedLLM(
            config=config,
            prompt_refiner=self.mock_prompt_refiner,
            generation_loop=self.mock_generation_loop,
            uncertainty_head=self.mock_uncertainty_head
        )
        
        result = augmented_llm.generate(self.test_prompt, return_metadata=True)
        
        # Verify prompt refinement was not called
        self.mock_prompt_refiner.refine.assert_not_called()
        
        # Verify result
        self.assertEqual(result["original_prompt"], self.test_prompt)
        self.assertIsNone(result["refined_prompt"])
        self.assertFalse(result["prompt_was_refined"])
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', True)
    def test_generate_text_only(self):
        """Test generation returning only text."""
        config = AugmentedLLMConfig()
        
        augmented_llm = AugmentedLLM(
            config=config,
            prompt_refiner=self.mock_prompt_refiner,
            generation_loop=self.mock_generation_loop,
            uncertainty_head=self.mock_uncertainty_head
        )
        
        result = augmented_llm.generate(self.test_prompt, return_metadata=False)
        
        # Should return only the text string
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Generated response for:"))
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', True)
    def test_generate_with_config_overrides(self):
        """Test generation with temporary config overrides."""
        config = AugmentedLLMConfig(uncertainty_threshold=0.7)
        
        augmented_llm = AugmentedLLM(
            config=config,
            prompt_refiner=self.mock_prompt_refiner,
            generation_loop=self.mock_generation_loop,
            uncertainty_head=self.mock_uncertainty_head
        )
        
        # Override config for this generation
        result = augmented_llm.generate(
            self.test_prompt, 
            return_metadata=True,
            override_config={"uncertainty_threshold": 0.9, "backtrack_window": 5}
        )
        
        # Verify original config is unchanged
        self.assertEqual(augmented_llm.config.uncertainty_threshold, 0.7)
        
        # Verify effective config in result
        self.assertEqual(result["pipeline_config"].uncertainty_threshold, 0.9)
        self.assertEqual(result["pipeline_config"].backtrack_window, 5)
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', True)
    def test_generate_batch(self):
        """Test batch generation."""
        config = AugmentedLLMConfig()
        
        augmented_llm = AugmentedLLM(
            config=config,
            prompt_refiner=self.mock_prompt_refiner,
            generation_loop=self.mock_generation_loop,
            uncertainty_head=self.mock_uncertainty_head
        )
        
        prompts = [
            "What is the capital of France?",
            "What is the largest planet?",
            "Who wrote Romeo and Juliet?"
        ]
        
        results = augmented_llm.generate_batch(prompts, return_metadata=True)
        
        self.assertEqual(len(results), len(prompts))
        self.assertEqual(self.mock_generation_loop.generate.call_count, len(prompts))
        
        # Check each result
        for i, result in enumerate(results):
            self.assertIsInstance(result, dict)
            self.assertIn("text", result)
            self.assertEqual(result["original_prompt"], prompts[i])
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', True)
    def test_empty_prompt_validation(self):
        """Test validation of empty prompts."""
        config = AugmentedLLMConfig()
        
        augmented_llm = AugmentedLLM(
            config=config,
            prompt_refiner=self.mock_prompt_refiner,
            generation_loop=self.mock_generation_loop,
            uncertainty_head=self.mock_uncertainty_head
        )
        
        with self.assertRaises(ValueError) as context:
            augmented_llm.generate("")
        
        self.assertIn("Prompt cannot be empty", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            augmented_llm.generate("   ")
        
        self.assertIn("Prompt cannot be empty", str(context.exception))
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', True)
    def test_get_pipeline_info(self):
        """Test getting pipeline information."""
        config = AugmentedLLMConfig()
        
        augmented_llm = AugmentedLLM(
            config=config,
            prompt_refiner=self.mock_prompt_refiner,
            generation_loop=self.mock_generation_loop,
            uncertainty_head=self.mock_uncertainty_head
        )
        
        info = augmented_llm.get_pipeline_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("config", info)
        self.assertIn("components", info)
        self.assertIn("pipeline_enabled", info)
        
        # Check components info
        self.assertIn("prompt_refiner", info["components"])
        self.assertIn("uncertainty_head", info["components"])
        self.assertIn("generation_model", info["components"])
        
        # Check pipeline enabled flags
        self.assertTrue(info["pipeline_enabled"]["prompt_refinement"])
        self.assertTrue(info["pipeline_enabled"]["uncertainty_monitoring"])
        self.assertTrue(info["pipeline_enabled"]["backtracking"])
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', True)
    def test_update_config(self):
        """Test updating configuration after initialization."""
        config = AugmentedLLMConfig(uncertainty_threshold=0.7)
        
        augmented_llm = AugmentedLLM(
            config=config,
            prompt_refiner=self.mock_prompt_refiner,
            generation_loop=self.mock_generation_loop,
            uncertainty_head=self.mock_uncertainty_head
        )
        
        # Update config
        augmented_llm.update_config(uncertainty_threshold=0.9, backtrack_window=5)
        
        self.assertEqual(augmented_llm.config.uncertainty_threshold, 0.9)
        self.assertEqual(augmented_llm.config.backtrack_window, 5)
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', True)
    def test_benchmark_generation(self):
        """Test generation benchmarking."""
        config = AugmentedLLMConfig()
        
        augmented_llm = AugmentedLLM(
            config=config,
            prompt_refiner=self.mock_prompt_refiner,
            generation_loop=self.mock_generation_loop,
            uncertainty_head=self.mock_uncertainty_head
        )
        
        stats = augmented_llm.benchmark_generation(self.test_prompt, num_runs=3)
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_runs", stats)
        self.assertIn("successful_runs", stats)
        self.assertIn("success_rate", stats)
        self.assertIn("avg_generation_time", stats)
        
        self.assertEqual(stats["total_runs"], 3)
        self.assertEqual(stats["successful_runs"], 3)
        self.assertEqual(stats["success_rate"], 1.0)
        self.assertEqual(self.mock_generation_loop.generate.call_count, 3)
    
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.HAS_TRANSFORMERS', True)
    def test_repr(self):
        """Test string representation."""
        config = AugmentedLLMConfig()
        
        augmented_llm = AugmentedLLM(
            config=config,
            prompt_refiner=self.mock_prompt_refiner,
            generation_loop=self.mock_generation_loop,
            uncertainty_head=self.mock_uncertainty_head
        )
        
        repr_str = repr(augmented_llm)
        
        self.assertIsInstance(repr_str, str)
        self.assertIn("AugmentedLLM", repr_str)
        self.assertIn("model=", repr_str)
        self.assertIn("components=", repr_str)


class TestAugmentedLLMIntegration(unittest.TestCase):
    """Integration tests for AugmentedLLM with real components."""
    
    @unittest.skipUnless(HAS_TRANSFORMERS, "Transformers not available")
    @patch('umin_framework.augmented_llm.HAS_COMPONENTS', True)
    @patch('umin_framework.augmented_llm.PromptRefiner')
    @patch('umin_framework.augmented_llm.GenerationLoop')
    def test_real_component_loading(self, mock_generation_loop_class, mock_prompt_refiner_class):
        """Test loading real components during initialization."""
        # Mock the classes to return our mock instances
        mock_prompt_refiner_instance = Mock()
        mock_prompt_refiner_class.return_value = mock_prompt_refiner_instance
        
        mock_generation_loop_instance = Mock()
        mock_generation_loop_instance.uncertainty_head = Mock()
        mock_generation_loop_instance.tokenizer = Mock()
        mock_generation_loop_instance.model = Mock()
        mock_generation_loop_instance.model.name_or_path = "test-model"
        mock_generation_loop_instance.model.num_parameters.return_value = 1000000
        mock_generation_loop_instance.model.device = "cpu"
        mock_generation_loop_class.return_value = mock_generation_loop_instance
        
        config = AugmentedLLMConfig(
            prompt_refiner_model="models/test_refiner",
            generation_model="gpt2"
        )
        
        augmented_llm = AugmentedLLM(config=config)
        
        # Verify components were loaded
        mock_prompt_refiner_class.assert_called_once_with(
            model_path="models/test_refiner",
            device=None
        )
        mock_generation_loop_class.assert_called_once_with(
            model="gpt2",
            tokenizer=None,
            uncertainty_head=None,
            device=None,
            torch_dtype=None
        )
        
        self.assertIsNotNone(augmented_llm.prompt_refiner)
        self.assertIsNotNone(augmented_llm.generation_loop)


if __name__ == '__main__':
    unittest.main()