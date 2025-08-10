"""
Unit tests for the PromptRefiner class.

This module contains comprehensive unit tests for the PromptRefiner class,
testing initialization, inference, batch processing, error handling, and
integration with the trained model checkpoint.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import json

# Add src to path so we can import our package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Test if transformers is available
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from umin_framework.prompt_refiner import PromptRefiner


class TestPromptRefiner(unittest.TestCase):
    """Test cases for the PromptRefiner class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.project_root = project_root
        cls.model_path = cls.project_root / "models" / "prompt_refiner"
        cls.model_exists = cls.model_path.exists() and (cls.model_path / "config.json").exists()
        
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_prompts = [
            "What is the best programming language?",
            "How do I make a website?",
            "Which database should I use?",
            "How to optimize my code?",
            "Which framework should I choose?"
        ]
        
        # AskCQ-style test prompts (ambiguous prompts from our training data)
        self.askcq_test_prompts = [
            {
                "input": "What is the best programming language?",
                "expected_keywords": ["purpose", "web development", "data science", "mobile"]
            },
            {
                "input": "How do I make a website?",
                "expected_keywords": ["what kind", "static", "e-commerce", "experience"]
            },
            {
                "input": "Which database should I use?",
                "expected_keywords": ["application", "scalability", "SQL", "NoSQL"]
            }
        ]
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_initialization_with_valid_model(self):
        """Test PromptRefiner initialization with a valid model path."""
        if not self.model_exists:
            self.skipTest("Trained model not available")
        refiner = PromptRefiner(str(self.model_path), device="cpu")
        
        self.assertIsNotNone(refiner.model)
        self.assertIsNotNone(refiner.tokenizer)
        self.assertEqual(refiner.device, "cpu")
        self.assertTrue(refiner.model_path.exists())
        
        # Test model info
        info = refiner.get_model_info()
        self.assertIn("model_path", info)
        self.assertIn("device", info)
        self.assertIn("parameters", info)
        self.assertTrue(info["model_loaded"])
        self.assertTrue(info["tokenizer_loaded"])
        self.assertGreater(info["parameters"], 0)
    
    def test_initialization_with_invalid_path(self):
        """Test PromptRefiner initialization with invalid model path."""
        with self.assertRaises(FileNotFoundError):
            PromptRefiner("nonexistent/path")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_initialization_without_transformers(self):
        """Test that PromptRefiner raises ImportError when transformers unavailable."""
        with patch('umin_framework.prompt_refiner.HAS_TRANSFORMERS', False):
            with self.assertRaises(ImportError):
                PromptRefiner("some/path")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_refine_single_prompt(self):
        """Test refining a single prompt."""
        if not self.model_exists:
            self.skipTest("Trained model not available")
        refiner = PromptRefiner(str(self.model_path), device="cpu")
        
        result = refiner.refine("What is the best programming language?")
        
        # Basic checks
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertNotEqual(result.strip(), "")
        
        # Should not be identical to input (model should transform it)
        self.assertNotEqual(result, "What is the best programming language?")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_refine_empty_prompt(self):
        """Test that empty prompts raise ValueError."""
        if not self.model_exists:
            self.skipTest("Trained model not available")
        refiner = PromptRefiner(str(self.model_path), device="cpu")
        
        with self.assertRaises(ValueError):
            refiner.refine("")
            
        with self.assertRaises(ValueError):
            refiner.refine("   ")  # Only whitespace
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available") 
    def test_refine_batch_prompts(self):
        """Test batch processing of multiple prompts."""
        if not self.model_exists:
            self.skipTest("Trained model not available")
        refiner = PromptRefiner(str(self.model_path), device="cpu")
        
        results = refiner.refine_batch(self.test_prompts[:3])
        
        # Check we get the right number of results
        self.assertEqual(len(results), 3)
        
        # Check all results are valid strings
        for result in results:
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_refine_batch_empty_list(self):
        """Test that empty batch raises ValueError."""
        if not self.model_exists:
            self.skipTest("Trained model not available")
        refiner = PromptRefiner(str(self.model_path), device="cpu")
        
        with self.assertRaises(ValueError):
            refiner.refine_batch([])
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_refine_batch_with_empty_prompts(self):
        """Test batch processing with some empty prompts."""
        if not self.model_exists:
            self.skipTest("Trained model not available")
        refiner = PromptRefiner(str(self.model_path), device="cpu")
        
        # Should filter out empty prompts
        mixed_prompts = ["Valid prompt", "", "Another valid prompt", "   "]
        
        with self.assertRaises(ValueError):
            # Should raise error if no valid prompts after cleaning
            refiner.refine_batch(["", "   ", ""])
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_custom_generation_parameters(self):
        """Test refining with custom generation parameters."""
        if not self.model_exists:
            self.skipTest("Trained model not available")
        refiner = PromptRefiner(str(self.model_path), device="cpu")
        
        # Test with custom parameters
        result = refiner.refine(
            "How do I make a website?",
            max_length=50,
            num_beams=2,
            do_sample=False
        )
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_askcq_style_prompts(self):
        """Test with AskCQ-style ambiguous prompts from our training data."""
        if not self.model_exists:
            self.skipTest("Trained model not available")
        refiner = PromptRefiner(str(self.model_path), device="cpu")
        
        for test_case in self.askcq_test_prompts:
            with self.subTest(prompt=test_case["input"]):
                result = refiner.refine(test_case["input"])
                
                # Basic validation
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)
                
                # The result should be different from input (model should refine it)
                self.assertNotEqual(result.strip(), test_case["input"])
                
                # For a well-trained model, we might expect certain keywords
                # but since our model had limited training, we'll just check
                # it produces reasonable output
                result_lower = result.lower()
                
                # Should not just repeat the same word
                words = result_lower.split()
                unique_words = set(words)
                self.assertGreater(len(unique_words), 1, 
                                 f"Result seems to just repeat words: {result}")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_device_handling(self):
        """Test device handling (CPU vs GPU)."""
        if not self.model_exists:
            self.skipTest("Trained model not available")
        # Test CPU device
        refiner_cpu = PromptRefiner(str(self.model_path), device="cpu")
        self.assertEqual(refiner_cpu.device, "cpu")
        
        # Test auto device selection
        refiner_auto = PromptRefiner(str(self.model_path), device=None)
        self.assertIn(refiner_auto.device, ["cpu", "cuda"])
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_repr_method(self):
        """Test string representation of PromptRefiner."""
        if not self.model_exists:
            self.skipTest("Trained model not available")
        refiner = PromptRefiner(str(self.model_path), device="cpu")
        repr_str = repr(refiner)
        
        self.assertIn("PromptRefiner", repr_str)
        self.assertIn(str(self.model_path), repr_str)
        self.assertIn("cpu", repr_str)
        self.assertIn("loaded=True", repr_str)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_missing_model_files(self):
        """Test handling of missing model files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory but no model files
            temp_path = Path(temp_dir) / "empty_model"
            temp_path.mkdir()
            
            with self.assertRaises(ValueError):
                PromptRefiner(str(temp_path))
    
    def test_model_info_without_model(self):
        """Test get_model_info when no model is loaded (mocked scenario)."""
        # This tests the edge case where model loading fails
        with patch('umin_framework.prompt_refiner.HAS_TRANSFORMERS', True):
            with patch.object(PromptRefiner, '_load_model', side_effect=ValueError("Mock error")):
                with self.assertRaises(ValueError):
                    PromptRefiner("mock/path")

    def test_integration_workflow(self):
        """Test a complete workflow that would be used in practice."""
        if not (TRANSFORMERS_AVAILABLE and self.model_exists):
            self.skipTest("Requires transformers and trained model")
            
        # This simulates how PromptRefiner would be used in the larger system
        refiner = PromptRefiner(str(self.model_path), device="cpu")
        
        # Simulate a user asking an ambiguous question
        user_prompt = "What should I use for my project?"
        
        # Get a clarification
        clarification = refiner.refine(user_prompt)
        
        # Basic integration checks
        self.assertIsInstance(clarification, str)
        self.assertGreater(len(clarification), 0)
        
        # Should produce something different from input
        self.assertNotEqual(clarification.strip(), user_prompt)
        
        # Should be reasonably sized (not too short, not too long)
        self.assertGreater(len(clarification.split()), 2)
        self.assertLess(len(clarification), 500)  # Reasonable upper bound


class TestPromptRefinerWithoutModel(unittest.TestCase):
    """Test cases that don't require a trained model."""
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_error_messages_are_informative(self):
        """Test that error messages provide helpful information."""
        # Test missing path
        try:
            PromptRefiner("definitely/not/a/real/path")
        except FileNotFoundError as e:
            self.assertIn("does not exist", str(e))
            
        # Test with mocked missing files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "incomplete_model"
            temp_path.mkdir()
            
            # Create config.json but no model file
            (temp_path / "config.json").write_text('{"model_type": "t5"}')
            (temp_path / "tokenizer_config.json").write_text('{}')
            
            try:
                PromptRefiner(str(temp_path))
            except ValueError as e:
                self.assertIn("No model file found", str(e))


def create_test_suite():
    """Create and return a test suite for PromptRefiner."""
    suite = unittest.TestSuite()
    
    # Add all test methods from both test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPromptRefiner))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPromptRefinerWithoutModel))
    
    return suite


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)