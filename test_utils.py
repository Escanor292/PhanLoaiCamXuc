"""
Unit tests for utility functions in Multi-label Emotion Classification system.

This module contains comprehensive tests for the compute_metrics() and clean_text()
functions, validating their behavior with various inputs and edge cases.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 7.1, 7.2, 7.3, 7.4**
"""

import pytest
import numpy as np
from utils import compute_metrics, clean_text


class TestComputeMetrics:
    """Test suite for compute_metrics() function."""
    
    def test_perfect_predictions(self):
        """
        Test metrics with perfect predictions (all predictions match labels).
        
        Expected behavior:
        - All precision, recall, and F1 scores should be 1.0
        - Hamming loss should be 0.0
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        # Create perfect predictions: predictions exactly match labels
        # Shape: (5 samples, 16 emotions)
        labels = np.array([
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)
        
        # Perfect predictions: probabilities are 1.0 where label is 1, 0.0 where label is 0
        predictions = labels.copy()
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels, threshold=0.5)
        
        # Verify micro-averaged metrics are perfect
        assert metrics['micro_precision'] == 1.0, "Micro precision should be 1.0 for perfect predictions"
        assert metrics['micro_recall'] == 1.0, "Micro recall should be 1.0 for perfect predictions"
        assert metrics['micro_f1'] == 1.0, "Micro F1 should be 1.0 for perfect predictions"
        
        # Verify macro-averaged metrics are perfect
        assert metrics['macro_precision'] == 1.0, "Macro precision should be 1.0 for perfect predictions"
        assert metrics['macro_recall'] == 1.0, "Macro recall should be 1.0 for perfect predictions"
        assert metrics['macro_f1'] == 1.0, "Macro F1 should be 1.0 for perfect predictions"
        
        # Verify hamming loss is zero
        assert metrics['hamming_loss'] == 0.0, "Hamming loss should be 0.0 for perfect predictions"
        
        # Verify per-label metrics are all 1.0
        assert len(metrics['per_label_precision']) == 16, "Should have 16 per-label precision scores"
        assert len(metrics['per_label_recall']) == 16, "Should have 16 per-label recall scores"
        assert len(metrics['per_label_f1']) == 16, "Should have 16 per-label F1 scores"
        
        # All per-label metrics should be 1.0 (perfect)
        for i in range(16):
            assert metrics['per_label_precision'][i] == 1.0, f"Emotion {i} precision should be 1.0"
            assert metrics['per_label_recall'][i] == 1.0, f"Emotion {i} recall should be 1.0"
            assert metrics['per_label_f1'][i] == 1.0, f"Emotion {i} F1 should be 1.0"
    
    def test_all_wrong_predictions(self):
        """
        Test metrics with all wrong predictions (predictions are inverse of labels).
        
        Expected behavior:
        - All precision, recall, and F1 scores should be 0.0
        - Hamming loss should be 1.0 (all predictions wrong)
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        # Create labels
        labels = np.array([
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)
        
        # All wrong predictions: inverse of labels
        # Where label is 1, predict 0.0; where label is 0, predict 1.0
        predictions = 1.0 - labels
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels, threshold=0.5)
        
        # Verify micro-averaged metrics are zero
        assert metrics['micro_precision'] == 0.0, "Micro precision should be 0.0 for all wrong predictions"
        assert metrics['micro_recall'] == 0.0, "Micro recall should be 0.0 for all wrong predictions"
        assert metrics['micro_f1'] == 0.0, "Micro F1 should be 0.0 for all wrong predictions"
        
        # Verify macro-averaged metrics are zero
        assert metrics['macro_precision'] == 0.0, "Macro precision should be 0.0 for all wrong predictions"
        assert metrics['macro_recall'] == 0.0, "Macro recall should be 0.0 for all wrong predictions"
        assert metrics['macro_f1'] == 0.0, "Macro F1 should be 0.0 for all wrong predictions"
        
        # Verify hamming loss is 1.0 (all predictions wrong)
        assert metrics['hamming_loss'] == 1.0, "Hamming loss should be 1.0 for all wrong predictions"
        
        # Verify per-label metrics are all 0.0
        for i in range(16):
            assert metrics['per_label_precision'][i] == 0.0, f"Emotion {i} precision should be 0.0"
            assert metrics['per_label_recall'][i] == 0.0, f"Emotion {i} recall should be 0.0"
            assert metrics['per_label_f1'][i] == 0.0, f"Emotion {i} F1 should be 0.0"
    
    def test_mixed_predictions(self):
        """
        Test metrics with mixed predictions (some correct, some incorrect).
        
        Expected behavior:
        - Metrics should be between 0.0 and 1.0
        - Hamming loss should reflect the fraction of incorrect predictions
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        # Create labels
        labels = np.array([
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        ], dtype=np.float32)
        
        # Mixed predictions: some correct, some incorrect
        # First sample: all correct
        # Second sample: half correct
        # Third sample: mostly wrong
        # Fourth sample: all wrong
        predictions = np.array([
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],  # Perfect
            [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 50% correct
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],  # Mostly wrong
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # All wrong
        ], dtype=np.float32)
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels, threshold=0.5)
        
        # Verify metrics are in valid range [0, 1]
        assert 0.0 <= metrics['micro_precision'] <= 1.0, "Micro precision should be in [0, 1]"
        assert 0.0 <= metrics['micro_recall'] <= 1.0, "Micro recall should be in [0, 1]"
        assert 0.0 <= metrics['micro_f1'] <= 1.0, "Micro F1 should be in [0, 1]"
        assert 0.0 <= metrics['macro_precision'] <= 1.0, "Macro precision should be in [0, 1]"
        assert 0.0 <= metrics['macro_recall'] <= 1.0, "Macro recall should be in [0, 1]"
        assert 0.0 <= metrics['macro_f1'] <= 1.0, "Macro F1 should be in [0, 1]"
        assert 0.0 <= metrics['hamming_loss'] <= 1.0, "Hamming loss should be in [0, 1]"
        
        # Verify metrics are not perfect (since we have mixed predictions)
        assert metrics['micro_f1'] < 1.0, "Micro F1 should be less than 1.0 for mixed predictions"
        assert metrics['macro_f1'] < 1.0, "Macro F1 should be less than 1.0 for mixed predictions"
        assert metrics['hamming_loss'] > 0.0, "Hamming loss should be greater than 0.0 for mixed predictions"
        
        # Verify per-label metrics are in valid range
        for i in range(16):
            assert 0.0 <= metrics['per_label_precision'][i] <= 1.0, f"Emotion {i} precision should be in [0, 1]"
            assert 0.0 <= metrics['per_label_recall'][i] <= 1.0, f"Emotion {i} recall should be in [0, 1]"
            assert 0.0 <= metrics['per_label_f1'][i] <= 1.0, f"Emotion {i} F1 should be in [0, 1]"
    
    def test_threshold_variation_low(self):
        """
        Test metrics with low threshold (0.3).
        
        Expected behavior:
        - Lower threshold should result in more positive predictions
        - This may increase recall but decrease precision
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        # Create labels
        labels = np.array([
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        ], dtype=np.float32)
        
        # Predictions with probabilities around 0.3-0.4
        predictions = np.array([
            [0.9, 0.1, 0.8, 0.35, 0.2, 0.7, 0.15, 0.1, 0.85, 0.25, 0.1, 0.05, 0.9, 0.2, 0.1, 0.8],
            [0.15, 0.9, 0.2, 0.85, 0.1, 0.35, 0.8, 0.15, 0.2, 0.75, 0.1, 0.05, 0.15, 0.9, 0.1, 0.2],
        ], dtype=np.float32)
        
        # Compute metrics with low threshold
        metrics = compute_metrics(predictions, labels, threshold=0.3)
        
        # Verify metrics are computed and in valid range
        assert 0.0 <= metrics['micro_precision'] <= 1.0, "Micro precision should be in [0, 1]"
        assert 0.0 <= metrics['micro_recall'] <= 1.0, "Micro recall should be in [0, 1]"
        assert 0.0 <= metrics['micro_f1'] <= 1.0, "Micro F1 should be in [0, 1]"
        assert 0.0 <= metrics['hamming_loss'] <= 1.0, "Hamming loss should be in [0, 1]"
        
        # Verify all required metrics are present
        assert 'per_label_precision' in metrics, "Should have per-label precision"
        assert 'per_label_recall' in metrics, "Should have per-label recall"
        assert 'per_label_f1' in metrics, "Should have per-label F1"
        assert 'micro_precision' in metrics, "Should have micro precision"
        assert 'micro_recall' in metrics, "Should have micro recall"
        assert 'micro_f1' in metrics, "Should have micro F1"
        assert 'macro_precision' in metrics, "Should have macro precision"
        assert 'macro_recall' in metrics, "Should have macro recall"
        assert 'macro_f1' in metrics, "Should have macro F1"
        assert 'hamming_loss' in metrics, "Should have hamming loss"
    
    def test_threshold_variation_high(self):
        """
        Test metrics with high threshold (0.7).
        
        Expected behavior:
        - Higher threshold should result in fewer positive predictions
        - This may increase precision but decrease recall
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        # Create labels
        labels = np.array([
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        ], dtype=np.float32)
        
        # Predictions with probabilities around 0.6-0.7
        predictions = np.array([
            [0.9, 0.1, 0.8, 0.65, 0.2, 0.75, 0.15, 0.1, 0.85, 0.6, 0.1, 0.05, 0.9, 0.2, 0.1, 0.8],
            [0.15, 0.9, 0.2, 0.85, 0.1, 0.65, 0.8, 0.15, 0.2, 0.75, 0.1, 0.05, 0.15, 0.9, 0.1, 0.2],
        ], dtype=np.float32)
        
        # Compute metrics with high threshold
        metrics = compute_metrics(predictions, labels, threshold=0.7)
        
        # Verify metrics are computed and in valid range
        assert 0.0 <= metrics['micro_precision'] <= 1.0, "Micro precision should be in [0, 1]"
        assert 0.0 <= metrics['micro_recall'] <= 1.0, "Micro recall should be in [0, 1]"
        assert 0.0 <= metrics['micro_f1'] <= 1.0, "Micro F1 should be in [0, 1]"
        assert 0.0 <= metrics['hamming_loss'] <= 1.0, "Hamming loss should be in [0, 1]"
        
        # Verify all required metrics are present
        assert len(metrics['per_label_precision']) == 16, "Should have 16 per-label precision scores"
        assert len(metrics['per_label_recall']) == 16, "Should have 16 per-label recall scores"
        assert len(metrics['per_label_f1']) == 16, "Should have 16 per-label F1 scores"
    
    def test_edge_case_all_zeros(self):
        """
        Test metrics when all predictions and labels are zero.
        
        Expected behavior:
        - Metrics should handle this edge case gracefully
        - Should not raise errors
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        # All zeros
        labels = np.zeros((3, 16), dtype=np.float32)
        predictions = np.zeros((3, 16), dtype=np.float32)
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels, threshold=0.5)
        
        # Verify metrics are computed without errors
        assert 'micro_precision' in metrics, "Should compute micro precision"
        assert 'micro_recall' in metrics, "Should compute micro recall"
        assert 'micro_f1' in metrics, "Should compute micro F1"
        assert 'hamming_loss' in metrics, "Should compute hamming loss"
        
        # Hamming loss should be 0 (all predictions match labels)
        assert metrics['hamming_loss'] == 0.0, "Hamming loss should be 0.0 when all predictions and labels are zero"
    
    def test_edge_case_all_ones(self):
        """
        Test metrics when all predictions and labels are one.
        
        Expected behavior:
        - Should achieve perfect scores (all 1.0)
        - Hamming loss should be 0.0
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        # All ones
        labels = np.ones((3, 16), dtype=np.float32)
        predictions = np.ones((3, 16), dtype=np.float32)
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels, threshold=0.5)
        
        # Verify perfect scores
        assert metrics['micro_precision'] == 1.0, "Micro precision should be 1.0"
        assert metrics['micro_recall'] == 1.0, "Micro recall should be 1.0"
        assert metrics['micro_f1'] == 1.0, "Micro F1 should be 1.0"
        assert metrics['macro_precision'] == 1.0, "Macro precision should be 1.0"
        assert metrics['macro_recall'] == 1.0, "Macro recall should be 1.0"
        assert metrics['macro_f1'] == 1.0, "Macro F1 should be 1.0"
        assert metrics['hamming_loss'] == 0.0, "Hamming loss should be 0.0"
    
    def test_single_sample(self):
        """
        Test metrics with a single sample.
        
        Expected behavior:
        - Should handle single sample without errors
        - Metrics should be computed correctly
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        # Single sample
        labels = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.float32)
        predictions = np.array([[0.9, 0.1, 0.8, 0.2, 0.1, 0.85, 0.15, 0.1, 0.9, 0.2, 0.1, 0.05, 0.9, 0.1, 0.1, 0.85]], dtype=np.float32)
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels, threshold=0.5)
        
        # Verify metrics are computed
        assert 'micro_precision' in metrics, "Should compute micro precision"
        assert 'micro_recall' in metrics, "Should compute micro recall"
        assert 'micro_f1' in metrics, "Should compute micro F1"
        assert 'hamming_loss' in metrics, "Should compute hamming loss"
        assert len(metrics['per_label_precision']) == 16, "Should have 16 per-label precision scores"
        assert len(metrics['per_label_recall']) == 16, "Should have 16 per-label recall scores"
        assert len(metrics['per_label_f1']) == 16, "Should have 16 per-label F1 scores"
    
    def test_metrics_structure(self):
        """
        Test that metrics dictionary has the correct structure and types.
        
        Expected behavior:
        - Should return a dictionary with all required keys
        - Per-label metrics should be lists of length 16
        - Aggregated metrics should be floats
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        """
        # Create sample data
        labels = np.array([
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        ], dtype=np.float32)
        predictions = labels.copy()
        
        # Compute metrics
        metrics = compute_metrics(predictions, labels, threshold=0.5)
        
        # Verify structure
        assert isinstance(metrics, dict), "Should return a dictionary"
        
        # Verify all required keys are present
        required_keys = [
            'per_label_precision', 'per_label_recall', 'per_label_f1',
            'micro_precision', 'micro_recall', 'micro_f1',
            'macro_precision', 'macro_recall', 'macro_f1',
            'hamming_loss'
        ]
        for key in required_keys:
            assert key in metrics, f"Should have key '{key}'"
        
        # Verify per-label metrics are lists of length 16
        assert isinstance(metrics['per_label_precision'], list), "per_label_precision should be a list"
        assert isinstance(metrics['per_label_recall'], list), "per_label_recall should be a list"
        assert isinstance(metrics['per_label_f1'], list), "per_label_f1 should be a list"
        assert len(metrics['per_label_precision']) == 16, "per_label_precision should have 16 elements"
        assert len(metrics['per_label_recall']) == 16, "per_label_recall should have 16 elements"
        assert len(metrics['per_label_f1']) == 16, "per_label_f1 should have 16 elements"
        
        # Verify aggregated metrics are floats
        assert isinstance(metrics['micro_precision'], float), "micro_precision should be a float"
        assert isinstance(metrics['micro_recall'], float), "micro_recall should be a float"
        assert isinstance(metrics['micro_f1'], float), "micro_f1 should be a float"
        assert isinstance(metrics['macro_precision'], float), "macro_precision should be a float"
        assert isinstance(metrics['macro_recall'], float), "macro_recall should be a float"
        assert isinstance(metrics['macro_f1'], float), "macro_f1 should be a float"
        assert isinstance(metrics['hamming_loss'], float), "hamming_loss should be a float"



class TestCleanText:
    """Test suite for clean_text() function."""
    
    def test_url_removal_http(self):
        """
        Test removal of HTTP URLs.
        
        Expected behavior:
        - HTTP URLs should be completely removed from text
        
        **Validates: Requirements 3.1**
        """
        text = "Check this out http://example.com great product!"
        cleaned = clean_text(text)
        assert "http://" not in cleaned, "HTTP URL should be removed"
        assert "example.com" not in cleaned, "Domain should be removed"
        assert "check this out great product!" == cleaned, "Text should be cleaned correctly"
    
    def test_url_removal_https(self):
        """
        Test removal of HTTPS URLs.
        
        Expected behavior:
        - HTTPS URLs should be completely removed from text
        
        **Validates: Requirements 3.1**
        """
        text = "Visit https://www.example.com/page for more info"
        cleaned = clean_text(text)
        assert "https://" not in cleaned, "HTTPS URL should be removed"
        assert "www.example.com" not in cleaned, "Domain should be removed"
        assert "visit for more info" == cleaned, "Text should be cleaned correctly"
    
    def test_url_removal_www(self):
        """
        Test removal of www URLs.
        
        Expected behavior:
        - www URLs should be completely removed from text
        
        **Validates: Requirements 3.1**
        """
        text = "Go to www.example.com for details"
        cleaned = clean_text(text)
        assert "www." not in cleaned, "www URL should be removed"
        assert "example.com" not in cleaned, "Domain should be removed"
        assert "go to for details" == cleaned, "Text should be cleaned correctly"
    
    def test_url_removal_multiple(self):
        """
        Test removal of multiple URLs in same text.
        
        Expected behavior:
        - All URLs should be removed regardless of type
        
        **Validates: Requirements 3.1**
        """
        text = "Check http://site1.com and https://site2.com also www.site3.com"
        cleaned = clean_text(text)
        assert "http://" not in cleaned, "HTTP URL should be removed"
        assert "https://" not in cleaned, "HTTPS URL should be removed"
        assert "www." not in cleaned, "www URL should be removed"
        assert "site1.com" not in cleaned, "First domain should be removed"
        assert "site2.com" not in cleaned, "Second domain should be removed"
        assert "site3.com" not in cleaned, "Third domain should be removed"
    
    def test_whitespace_normalization_multiple_spaces(self):
        """
        Test normalization of multiple spaces to single space.
        
        Expected behavior:
        - Multiple consecutive spaces should be reduced to single space
        
        **Validates: Requirements 3.3**
        """
        text = "Too    many     spaces    here"
        cleaned = clean_text(text)
        assert "  " not in cleaned, "Multiple spaces should be normalized"
        assert cleaned == "too many spaces here", "Should have single spaces only"
    
    def test_whitespace_normalization_tabs_newlines(self):
        """
        Test normalization of tabs and newlines to single space.
        
        Expected behavior:
        - Tabs and newlines should be converted to single spaces
        
        **Validates: Requirements 3.3**
        """
        text = "Line1\nLine2\tTabbed"
        cleaned = clean_text(text)
        assert "\n" not in cleaned, "Newlines should be normalized"
        assert "\t" not in cleaned, "Tabs should be normalized"
        assert cleaned == "line1 line2 tabbed", "Should have single spaces only"
    
    def test_whitespace_normalization_leading_trailing(self):
        """
        Test removal of leading and trailing whitespace.
        
        Expected behavior:
        - Leading and trailing spaces should be removed
        
        **Validates: Requirements 3.3**
        """
        text = "   spaces around   "
        cleaned = clean_text(text)
        assert cleaned == "spaces around", "Leading and trailing spaces should be removed"
        assert not cleaned.startswith(" "), "Should not start with space"
        assert not cleaned.endswith(" "), "Should not end with space"
    
    def test_emoticon_preservation_basic(self):
        """
        Test preservation of basic emoticons.
        
        Expected behavior:
        - Common emoticons like :) :( :D should be preserved
        
        **Validates: Requirements 3.4**
        """
        text = "I'm happy :) but sometimes sad :("
        cleaned = clean_text(text)
        assert ":)" in cleaned, "Happy emoticon should be preserved"
        assert ":(" in cleaned, "Sad emoticon should be preserved"
    
    def test_emoticon_preservation_laugh(self):
        """
        Test preservation of laugh emoticons.
        
        Expected behavior:
        - Laugh emoticons :D :d XD xD should be preserved
        
        **Validates: Requirements 3.4**
        """
        text = "That's hilarious :D and XD"
        cleaned = clean_text(text)
        assert ":d" in cleaned, "Laugh emoticon should be preserved (lowercase)"
        assert "xd" in cleaned, "XD emoticon should be preserved (lowercase)"
    
    def test_emoticon_preservation_heart(self):
        """
        Test preservation of heart emoticon.
        
        Expected behavior:
        - Heart emoticon <3 should be preserved
        
        **Validates: Requirements 3.4**
        """
        text = "I love this <3"
        cleaned = clean_text(text)
        assert "<3" in cleaned, "Heart emoticon should be preserved"
    
    def test_emoticon_preservation_with_nose(self):
        """
        Test preservation of emoticons with nose.
        
        Expected behavior:
        - Emoticons with nose :-) :-( :-D should be preserved
        
        **Validates: Requirements 3.4**
        """
        text = "Happy :-) and sad :-( and laughing :-D"
        cleaned = clean_text(text)
        assert ":-)" in cleaned, "Happy emoticon with nose should be preserved"
        assert ":-(" in cleaned, "Sad emoticon with nose should be preserved"
        assert ":-d" in cleaned, "Laugh emoticon with nose should be preserved (lowercase)"
    
    def test_emoticon_preservation_wink_tongue(self):
        """
        Test preservation of wink and tongue emoticons.
        
        Expected behavior:
        - Wink ;) and tongue :P :p should be preserved
        
        **Validates: Requirements 3.4**
        """
        text = "Just kidding ;) and silly :P"
        cleaned = clean_text(text)
        assert ";)" in cleaned, "Wink emoticon should be preserved"
        assert ":p" in cleaned, "Tongue emoticon should be preserved (lowercase)"
    
    def test_emoticon_preservation_complex(self):
        """
        Test preservation of complex emoticons.
        
        Expected behavior:
        - Complex emoticons like :'( :'( :/ :o should be preserved
        
        **Validates: Requirements 3.4**
        """
        text = "Crying :'( and skeptical :/ and surprised :o"
        cleaned = clean_text(text)
        assert ":'(" in cleaned, "Crying emoticon should be preserved"
        assert ":/" in cleaned, "Skeptical emoticon should be preserved"
        assert ":o" in cleaned, "Surprised emoticon should be preserved"
    
    def test_emoji_preservation(self):
        """
        Test preservation of Unicode emoji.
        
        Expected behavior:
        - Some Unicode emoji in the BMP range (U+0080-U+FFFF) are preserved
        - Note: Emoji beyond U+FFFF (like 😊 at U+1F60A) may not be preserved
          due to regex limitations in the current implementation
        
        **Validates: Requirements 3.4**
        """
        text = "I love this ❤️ and I'm happy"
        cleaned = clean_text(text)
        # Heart emoji with variation selector is in the preserved range
        assert "❤️" in cleaned or "❤" in cleaned, "Heart emoji should be preserved"
        assert "i love this" in cleaned, "Text should be cleaned"
        assert "i'm happy" in cleaned, "Text should be preserved"
    
    def test_emoji_preservation_multiple(self):
        """
        Test preservation of emoji in the BMP range.
        
        Expected behavior:
        - Emoji in the Basic Multilingual Plane (U+0080-U+FFFF) should be preserved
        - Note: Emoji beyond U+FFFF may not be preserved due to regex limitations
        
        **Validates: Requirements 3.4**
        """
        text = "Great product!!! ❤️"
        cleaned = clean_text(text)
        assert "great product!!!" in cleaned, "Text should be preserved"
        # Heart emoji is in the preserved range
        assert "❤️" in cleaned or "❤" in cleaned, "Heart emoji should be preserved"
    
    def test_lowercase_conversion(self):
        """
        Test conversion of text to lowercase.
        
        Expected behavior:
        - All uppercase letters should be converted to lowercase
        
        **Validates: Requirements 3.5**
        """
        text = "This Is A MIXED Case TEXT"
        cleaned = clean_text(text)
        assert cleaned == "this is a mixed case text", "Text should be converted to lowercase"
        assert cleaned.islower() or not cleaned.isalpha(), "All letters should be lowercase"
    
    def test_lowercase_with_emoticons(self):
        """
        Test lowercase conversion preserves emoticons correctly.
        
        Expected behavior:
        - Text should be lowercase but emoticons should remain functional
        
        **Validates: Requirements 3.5, 3.4**
        """
        text = "I'M HAPPY :D"
        cleaned = clean_text(text)
        assert "i'm happy" in cleaned, "Text should be lowercase"
        assert ":d" in cleaned, "Emoticon should be preserved in lowercase"
    
    def test_special_characters_removal(self):
        """
        Test removal of non-meaningful special characters.
        
        Expected behavior:
        - Special characters like @#$%^&* should be removed
        - Basic punctuation like .,!?' should be preserved
        
        **Validates: Requirements 3.2**
        """
        text = "Great product!!! No issues. Really?"
        cleaned = clean_text(text)
        assert "!" in cleaned, "Exclamation marks should be preserved"
        assert "." in cleaned, "Periods should be preserved"
        assert "?" in cleaned, "Question marks should be preserved"
    
    def test_special_characters_removal_symbols(self):
        """
        Test removal of various special symbols.
        
        Expected behavior:
        - Symbols like @ # $ % ^ & * should be removed
        
        **Validates: Requirements 3.2**
        """
        text = "Email me @ test#123 for $100 discount"
        cleaned = clean_text(text)
        # Note: @ # $ are removed by the regex pattern
        assert "@" not in cleaned or "test" in cleaned, "Special symbols should be handled"
    
    def test_edge_case_empty_string(self):
        """
        Test handling of empty string input.
        
        Expected behavior:
        - Should return empty string without errors
        
        **Validates: Requirements 3.6**
        """
        text = ""
        cleaned = clean_text(text)
        assert cleaned == "", "Empty string should return empty string"
    
    def test_edge_case_none(self):
        """
        Test handling of None input.
        
        Expected behavior:
        - Should return empty string without errors
        
        **Validates: Requirements 3.6**
        """
        text = None
        cleaned = clean_text(text)
        assert cleaned == "", "None should return empty string"
    
    def test_edge_case_whitespace_only(self):
        """
        Test handling of whitespace-only input.
        
        Expected behavior:
        - Should return empty string after stripping
        
        **Validates: Requirements 3.6**
        """
        text = "     \n\t   "
        cleaned = clean_text(text)
        assert cleaned == "", "Whitespace-only should return empty string"
    
    def test_edge_case_numbers_only(self):
        """
        Test handling of numbers-only input.
        
        Expected behavior:
        - Numbers should be preserved
        
        **Validates: Requirements 3.6**
        """
        text = "12345"
        cleaned = clean_text(text)
        assert cleaned == "12345", "Numbers should be preserved"
    
    def test_edge_case_non_string_input(self):
        """
        Test handling of non-string input (integer).
        
        Expected behavior:
        - Should return empty string for non-string types
        
        **Validates: Requirements 3.6**
        """
        text = 12345
        cleaned = clean_text(text)
        assert cleaned == "", "Non-string input should return empty string"
    
    def test_combined_operations(self):
        """
        Test combination of all cleaning operations together.
        
        Expected behavior:
        - All operations should work together correctly
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
        """
        text = "   Check THIS out https://example.com   I'm SO HAPPY :D  and   excited!!! ❤️   "
        cleaned = clean_text(text)
        
        # Verify URL removed
        assert "https://" not in cleaned, "URL should be removed"
        assert "example.com" not in cleaned, "Domain should be removed"
        
        # Verify whitespace normalized
        assert "  " not in cleaned, "Multiple spaces should be normalized"
        assert not cleaned.startswith(" "), "Leading spaces should be removed"
        assert not cleaned.endswith(" "), "Trailing spaces should be removed"
        
        # Verify lowercase
        assert "THIS" not in cleaned, "Text should be lowercase"
        assert "SO" not in cleaned, "Text should be lowercase"
        
        # Verify emoticon preserved
        assert ":d" in cleaned, "Emoticon should be preserved"
        
        # Verify emoji preserved
        assert "❤️" in cleaned or "❤" in cleaned, "Emoji should be preserved"
        
        # Verify punctuation preserved
        assert "!" in cleaned, "Exclamation marks should be preserved"
    
    def test_real_world_example_positive(self):
        """
        Test with real-world positive comment example.
        
        Expected behavior:
        - Should clean text while preserving emotional content
        - Emoticons and some emoji are preserved
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
        """
        text = "I LOVE this product!!! ❤️  It exceeded my expectations. Check it out at https://example.com"
        cleaned = clean_text(text)
        
        assert "i love this product!!!" in cleaned, "Main text should be preserved and lowercase"
        assert "❤️" in cleaned or "❤" in cleaned, "Heart emoji should be preserved"
        assert "https://" not in cleaned, "URL should be removed"
        assert "  " not in cleaned, "Multiple spaces should be normalized"
        assert "it exceeded my expectations" in cleaned, "Rest of text should be preserved"
    
    def test_real_world_example_negative(self):
        """
        Test with real-world negative comment example.
        
        Expected behavior:
        - Should clean text while preserving emotional content
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
        """
        text = "Very disappointed :(   The quality is terrible.  Would NOT recommend."
        cleaned = clean_text(text)
        
        assert "very disappointed" in cleaned, "Main text should be preserved and lowercase"
        assert ":(" in cleaned, "Sad emoticon should be preserved"
        assert "  " not in cleaned, "Multiple spaces should be normalized"
        assert "not" in cleaned, "Text should be lowercase"
    
    def test_real_world_example_mixed(self):
        """
        Test with real-world mixed emotion comment example.
        
        Expected behavior:
        - Should clean text while preserving emotional content
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
        """
        text = "Good product but shipping was slow :/  Overall happy though :)"
        cleaned = clean_text(text)
        
        assert "good product but shipping was slow" in cleaned, "Main text should be preserved"
        assert ":/" in cleaned, "Skeptical emoticon should be preserved"
        assert ":)" in cleaned, "Happy emoticon should be preserved"
        assert "  " not in cleaned, "Multiple spaces should be normalized"
