"""
Unit tests for the UncertaintyHead class.

This module contains comprehensive unit tests for the UncertaintyHead class,
testing initialization, uncertainty scoring methods, batch processing, 
error handling, and integration with transformers models.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

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

from umin_framework.uncertainty_head import UncertaintyHead


class TestUncertaintyHead(unittest.TestCase):
    """Test cases for the UncertaintyHead class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.project_root = project_root
        # Use a very small model for testing
        cls.test_model_name = "distilgpt2"
        
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_sequences = [
            "The capital of France is",
            "I think that artificial intelligence",
            "The weather today is sunny",
            "Machine learning models can"
        ]
        
        self.short_sequence = "Hello world"
        self.empty_sequences = ["", "   ", "\t\n"]
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_initialization_with_model_name(self):
        """Test UncertaintyHead initialization with model name."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        self.assertIsNotNone(ue_head.model)
        self.assertIsNotNone(ue_head.tokenizer)
        self.assertEqual(ue_head.device, "cpu")
        self.assertEqual(ue_head.model_name, self.test_model_name)
        
        # Test model info
        info = ue_head.get_model_info()
        self.assertIn("model_name", info)
        self.assertIn("device", info)
        self.assertIn("parameters", info)
        self.assertTrue(info["model_loaded"])
        self.assertTrue(info["tokenizer_loaded"])
        self.assertGreater(info["parameters"], 0)
        self.assertGreater(info["vocab_size"], 0)
    
    def test_initialization_without_transformers(self):
        """Test that UncertaintyHead raises ImportError when transformers unavailable."""
        with patch('umin_framework.uncertainty_head.HAS_TRANSFORMERS', False):
            with self.assertRaises(ImportError):
                UncertaintyHead("some/model")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_device_auto_selection(self):
        """Test automatic device selection."""
        ue_head = UncertaintyHead(self.test_model_name, device=None)
        self.assertIn(ue_head.device, ["cpu", "cuda"])
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_score_single_sequence_entropy(self):
        """Test scoring a single sequence with entropy method."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        scores = ue_head.score(self.short_sequence, method="entropy")
        
        # Basic validation
        self.assertIsInstance(scores, list)
        self.assertGreater(len(scores), 0)
        
        # All scores should be non-negative (entropy is always >= 0)
        for score in scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            
        # Entropy scores should be reasonable (typically 0-15 for language models)
        for score in scores:
            self.assertLess(score, 20.0, "Entropy score seems too high")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_score_different_methods(self):
        """Test all uncertainty scoring methods."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        methods = ["entropy", "max_prob", "margin", "variance"]
        
        for method in methods:
            with self.subTest(method=method):
                scores = ue_head.score(self.short_sequence, method=method)
                
                self.assertIsInstance(scores, list)
                self.assertGreater(len(scores), 0)
                
                # All scores should be valid numbers
                for score in scores:
                    self.assertIsInstance(score, float)
                    self.assertFalse(torch.isnan(torch.tensor(score)), f"NaN score in {method}")
                    self.assertFalse(torch.isinf(torch.tensor(score)), f"Inf score in {method}")
                
                # Method-specific validations
                if method == "max_prob" or method == "margin":
                    # These should be between 0 and 1 (converted to uncertainty)
                    for score in scores:
                        self.assertGreaterEqual(score, 0.0)
                        self.assertLessEqual(score, 1.0)
                elif method == "entropy":
                    # Entropy should be non-negative
                    for score in scores:
                        self.assertGreaterEqual(score, 0.0)
                elif method == "variance":
                    # Variance should be non-negative
                    for score in scores:
                        self.assertGreaterEqual(score, 0.0)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_score_with_tokens(self):
        """Test scoring with token return enabled."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        scores, tokens = ue_head.score(self.short_sequence, return_tokens=True)
        
        # Check that we get both scores and tokens
        self.assertIsInstance(scores, list)
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(scores), len(tokens))
        
        # Check that all tokens are strings
        for token in tokens:
            self.assertIsInstance(token, str)
        
        # Check that tokens can be reconstructed
        reconstructed = "".join(tokens)
        # Note: tokenizer may add special tokens and spacing, so we check words are present
        original_words = self.short_sequence.strip().split()
        for word in original_words:
            self.assertIn(word, reconstructed, f"Word '{word}' not found in reconstructed text")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_score_batch_sequences(self):
        """Test batch processing of multiple sequences."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        batch_scores = ue_head.score(self.test_sequences[:3], method="entropy")
        
        # Should return a list of lists
        self.assertIsInstance(batch_scores, list)
        self.assertEqual(len(batch_scores), 3)
        
        # Each element should be a list of scores
        for scores in batch_scores:
            self.assertIsInstance(scores, list)
            self.assertGreater(len(scores), 0)
            for score in scores:
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_score_empty_sequence_error(self):
        """Test that empty sequences raise ValueError."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        for empty_seq in self.empty_sequences:
            with self.subTest(sequence=repr(empty_seq)):
                with self.assertRaises(ValueError):
                    ue_head.score(empty_seq)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_score_invalid_method_error(self):
        """Test that invalid methods raise ValueError."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        invalid_methods = ["invalid", "random", "", "ENTROPY"]
        
        for method in invalid_methods:
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    ue_head.score(self.short_sequence, method=method)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_score_generation(self):
        """Test scoring generated sequences separately from prompts."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        prompt = "The capital of France is"
        generation = " Paris, which is located"
        
        gen_scores, gen_tokens = ue_head.score_generation(prompt, generation)
        
        # Should get scores only for generated tokens
        self.assertIsInstance(gen_scores, list)
        self.assertIsInstance(gen_tokens, list)
        self.assertEqual(len(gen_scores), len(gen_tokens))
        self.assertGreater(len(gen_scores), 0)
        
        # Generated tokens should roughly match the generation
        gen_text = "".join(gen_tokens)
        self.assertTrue(any(word in gen_text for word in ["Paris", "which", "located"]))
        
        # All scores should be valid
        for score in gen_scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_uncertainty_threshold_functionality(self):
        """Test uncertainty threshold setting and detection."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        # Set threshold
        threshold = 5.0
        ue_head.set_uncertainty_threshold(threshold)
        self.assertEqual(ue_head.uncertainty_threshold, threshold)
        
        # Test high uncertainty detection
        scores = ue_head.score(self.short_sequence, method="entropy")
        high_uncertainty = ue_head.is_high_uncertainty(scores)
        
        self.assertIsInstance(high_uncertainty, list)
        self.assertEqual(len(high_uncertainty), len(scores))
        
        # Check that the flags are boolean and consistent with threshold
        for i, (score, is_high) in enumerate(zip(scores, high_uncertainty)):
            self.assertIsInstance(is_high, bool)
            if score > threshold:
                self.assertTrue(is_high, f"Score {score} > {threshold} should be high uncertainty")
            else:
                self.assertFalse(is_high, f"Score {score} <= {threshold} should not be high uncertainty")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_uncertainty_threshold_override(self):
        """Test overriding uncertainty threshold in detection."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        scores = [1.0, 3.0, 6.0, 8.0]
        
        # Test with different thresholds
        low_threshold = 2.0
        high_threshold = 7.0
        
        low_high = ue_head.is_high_uncertainty(scores, threshold=low_threshold)
        high_high = ue_head.is_high_uncertainty(scores, threshold=high_threshold)
        
        # More scores should be high with lower threshold
        self.assertGreaterEqual(sum(low_high), sum(high_high))
        
        # Check specific expectations
        self.assertEqual(low_high, [False, True, True, True])  # Scores > 2.0
        self.assertEqual(high_high, [False, False, False, True])  # Scores > 7.0
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_max_length_truncation(self):
        """Test that max_length parameter works correctly."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        long_sequence = " ".join(["word"] * 100)  # Very long sequence
        max_len = 10
        
        scores = ue_head.score(long_sequence, max_length=max_len)
        
        # Should be truncated to roughly max_length tokens
        self.assertLessEqual(len(scores), max_len + 2)  # Allow some buffer for special tokens
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_repr_method(self):
        """Test string representation of UncertaintyHead."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        repr_str = repr(ue_head)
        
        self.assertIn("UncertaintyHead", repr_str)
        self.assertIn(self.test_model_name, repr_str)
        self.assertIn("cpu", repr_str)
        self.assertIn("loaded=True", repr_str)
    
    def test_invalid_model_path(self):
        """Test handling of invalid model paths."""
        if not TRANSFORMERS_AVAILABLE:
            self.skipTest("Transformers not available")
            
        with self.assertRaises(ValueError):
            UncertaintyHead("definitely/not/a/real/model/path")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_consistency_across_runs(self):
        """Test that the same input gives consistent results."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        # Run scoring multiple times
        sequence = "The quick brown fox jumps"
        
        scores1 = ue_head.score(sequence, method="entropy")
        scores2 = ue_head.score(sequence, method="entropy")
        
        # Should be identical (models are deterministic in eval mode)
        self.assertEqual(len(scores1), len(scores2))
        for s1, s2 in zip(scores1, scores2):
            self.assertAlmostEqual(s1, s2, places=5)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available") 
    def test_different_uncertainty_methods_give_different_results(self):
        """Test that different uncertainty methods produce different results."""
        ue_head = UncertaintyHead(self.test_model_name, device="cpu")
        
        sequence = "This is a test sequence for uncertainty"
        
        entropy_scores = ue_head.score(sequence, method="entropy")
        max_prob_scores = ue_head.score(sequence, method="max_prob")
        margin_scores = ue_head.score(sequence, method="margin")
        
        # Methods should give different results
        self.assertNotEqual(entropy_scores, max_prob_scores)
        self.assertNotEqual(entropy_scores, margin_scores)
        self.assertNotEqual(max_prob_scores, margin_scores)
        
        # But all should have same length (same tokens)
        self.assertEqual(len(entropy_scores), len(max_prob_scores))
        self.assertEqual(len(entropy_scores), len(margin_scores))


class TestUncertaintyHeadEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_very_short_input(self):
        """Test with very short inputs."""
        ue_head = UncertaintyHead("distilgpt2", device="cpu")
        
        short_inputs = ["a", ".", "1"]
        
        for inp in short_inputs:
            with self.subTest(input=inp):
                scores = ue_head.score(inp)
                self.assertGreater(len(scores), 0)
                for score in scores:
                    self.assertIsInstance(score, float)
                    self.assertGreaterEqual(score, 0.0)
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "Transformers library not available")
    def test_special_characters_input(self):
        """Test with special characters and non-ASCII input."""
        ue_head = UncertaintyHead("distilgpt2", device="cpu")
        
        special_inputs = [
            "Hello! @#$%^&*()",
            "Café naïve résumé",
            "测试中文文本",
            "🤖 AI is cool! 🚀"
        ]
        
        for inp in special_inputs:
            with self.subTest(input=inp):
                try:
                    scores = ue_head.score(inp)
                    self.assertIsInstance(scores, list)
                    self.assertGreater(len(scores), 0)
                except Exception as e:
                    # Some tokenizers may not handle all character sets
                    self.assertIn("tokenizer", str(e).lower())


def create_test_suite():
    """Create and return a test suite for UncertaintyHead."""
    suite = unittest.TestSuite()
    
    # Add all test methods
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUncertaintyHead))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUncertaintyHeadEdgeCases))
    
    return suite


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run tests with verbose output
    unittest.main(verbosity=2)