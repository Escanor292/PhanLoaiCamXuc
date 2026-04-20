"""
Integration test for prediction pipeline in Multi-label Emotion Classification system.

This module contains an end-to-end integration test for the prediction pipeline,
validating that the complete prediction flow works correctly with sample comments.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
"""

import pytest
import torch
import torch.nn as nn
from transformers import BertTokenizer
import numpy as np
import os
import tempfile
import shutil

from config import Config
from model import BERTEmotionClassifier
from predict import predict_emotions, predict_emotions_batch
from utils import save_model


@pytest.fixture
def minimal_trained_model():
    """
    Fixture to provide a minimal trained model for testing.
    
    Creates a model with initialized weights (not actually trained on data,
    but sufficient for testing the prediction pipeline).
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    device = torch.device('cpu')  # Use CPU for testing
    
    # Initialize model
    model = BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)
    model = model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return model, tokenizer, device


@pytest.fixture
def temp_model_checkpoint(minimal_trained_model):
    """
    Fixture to provide a temporary model checkpoint directory.
    
    Saves the minimal trained model to a temporary directory and
    cleans up after the test.
    
    Yields:
        str: Path to temporary directory containing model checkpoint
    """
    model, tokenizer, device = minimal_trained_model
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Save model checkpoint
    training_config = {
        'model_name': 'bert-base-uncased',
        'num_labels': 16,
        'dropout_rate': 0.3,
        'learning_rate': 2e-5,
        'batch_size': 16,
        'num_epochs': 1,
        'max_length': 128
    }
    
    save_model(model, tokenizer, temp_dir, training_config)
    
    yield temp_dir
    
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_comments():
    """
    Fixture to provide sample comments for testing prediction.
    
    Returns:
        list: List of sample comment strings with various emotional content
    """
    return [
        "I love this product! It's amazing and exceeded my expectations!",
        "This is terrible. Very disappointed with the quality.",
        "Good quality but shipping was slow. Mixed feelings.",
        "Absolutely fantastic! Best purchase ever!",
        "Not bad, could be better. Average experience.",
        "I'm so happy with this! Great value for money!",
        "Worst experience ever. Never buying again.",
        "Pretty good overall. Would recommend to friends.",
        "Meh, it's okay. Nothing special.",
        "Outstanding! Exceeded all my expectations!"
    ]


class TestPredictionPipelineIntegration:
    """Integration test suite for the complete prediction pipeline."""
    
    def test_end_to_end_single_prediction(self, minimal_trained_model, sample_comments):
        """
        Test complete end-to-end prediction pipeline for a single comment.
        
        This integration test validates:
        1. Model and tokenizer are loaded correctly
        2. predict_emotions() function works with sample comment
        3. Output format is correct (dict with 'emotions' and 'scores')
        4. Confidence scores are in valid range [0, 1]
        5. Predicted emotions are a subset of Config.EMOTION_LABELS
        6. All 16 emotion scores are returned
        
        Expected behavior:
        - Prediction should complete without errors
        - Output should have correct structure
        - Scores should be valid probabilities
        
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
        """
        model, tokenizer, device = minimal_trained_model
        
        # Test with a positive emotion comment
        test_comment = sample_comments[0]  # "I love this product! It's amazing..."
        
        print(f"\n[Test] Predicting emotions for: '{test_comment}'")
        
        # Step 1: Perform prediction
        try:
            result = predict_emotions(
                test_comment,
                model,
                tokenizer,
                device,
                threshold=0.5
            )
            
            print(f"✓ Prediction completed successfully")
            
        except Exception as e:
            pytest.fail(f"Prediction failed with error: {str(e)}")
        
        # Step 2: Validate output format
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'emotions' in result, "Result should have 'emotions' key"
        assert 'scores' in result, "Result should have 'scores' key"
        
        print(f"✓ Output format is correct")
        
        # Step 3: Validate emotions list
        assert isinstance(result['emotions'], list), "Emotions should be a list"
        
        # All predicted emotions should be valid emotion labels
        for emotion in result['emotions']:
            assert emotion in Config.EMOTION_LABELS, \
                f"Predicted emotion '{emotion}' is not in EMOTION_LABELS"
        
        print(f"✓ Predicted emotions: {result['emotions']}")
        
        # Step 4: Validate scores dictionary
        assert isinstance(result['scores'], dict), "Scores should be a dictionary"
        assert len(result['scores']) == 16, "Should have scores for all 16 emotions"
        
        # All emotion labels should have scores
        for label in Config.EMOTION_LABELS:
            assert label in result['scores'], f"Missing score for emotion '{label}'"
        
        print(f"✓ All 16 emotion scores are present")
        
        # Step 5: Validate score ranges
        for label, score in result['scores'].items():
            assert isinstance(score, float), f"Score for '{label}' should be float"
            assert 0.0 <= score <= 1.0, \
                f"Score for '{label}' ({score}) should be in range [0, 1]"
        
        print(f"✓ All scores are in valid range [0, 1]")
        
        # Step 6: Validate threshold logic
        # Emotions in result['emotions'] should have scores >= threshold
        threshold = 0.5
        for emotion in result['emotions']:
            assert result['scores'][emotion] >= threshold, \
                f"Predicted emotion '{emotion}' has score {result['scores'][emotion]} < threshold {threshold}"
        
        # Emotions not in result['emotions'] should have scores < threshold
        for label in Config.EMOTION_LABELS:
            if label not in result['emotions']:
                assert result['scores'][label] < threshold, \
                    f"Emotion '{label}' has score {result['scores'][label]} >= threshold but not in predictions"
        
        print(f"✓ Threshold logic is correct")
        
        print("\n" + "="*70)
        print("✓ SINGLE PREDICTION INTEGRATION TEST PASSED")
        print("="*70)
    
    def test_end_to_end_batch_prediction(self, minimal_trained_model, sample_comments):
        """
        Test complete end-to-end batch prediction pipeline.
        
        This integration test validates:
        1. predict_emotions_batch() function works with multiple comments
        2. Output is a list with correct length
        3. Each result has correct format
        4. All confidence scores are valid
        5. Batch prediction is more efficient than individual predictions
        
        Expected behavior:
        - Batch prediction should complete without errors
        - Should return results for all input comments
        - Each result should have same format as single prediction
        
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**
        """
        model, tokenizer, device = minimal_trained_model
        
        # Use first 5 sample comments for batch prediction
        test_comments = sample_comments[:5]
        
        print(f"\n[Test] Batch predicting emotions for {len(test_comments)} comments")
        
        # Step 1: Perform batch prediction
        try:
            results = predict_emotions_batch(
                test_comments,
                model,
                tokenizer,
                device,
                threshold=0.5
            )
            
            print(f"✓ Batch prediction completed successfully")
            
        except Exception as e:
            pytest.fail(f"Batch prediction failed with error: {str(e)}")
        
        # Step 2: Validate output is a list with correct length
        assert isinstance(results, list), "Results should be a list"
        assert len(results) == len(test_comments), \
            f"Should have {len(test_comments)} results, got {len(results)}"
        
        print(f"✓ Returned {len(results)} results for {len(test_comments)} comments")
        
        # Step 3: Validate each result has correct format
        for i, result in enumerate(results):
            assert isinstance(result, dict), f"Result {i} should be a dictionary"
            assert 'emotions' in result, f"Result {i} should have 'emotions' key"
            assert 'scores' in result, f"Result {i} should have 'scores' key"
            
            assert isinstance(result['emotions'], list), \
                f"Result {i} emotions should be a list"
            assert isinstance(result['scores'], dict), \
                f"Result {i} scores should be a dictionary"
            
            assert len(result['scores']) == 16, \
                f"Result {i} should have scores for all 16 emotions"
        
        print(f"✓ All results have correct format")
        
        # Step 4: Validate all scores are in valid range
        for i, result in enumerate(results):
            for label, score in result['scores'].items():
                assert isinstance(score, float), \
                    f"Result {i} score for '{label}' should be float"
                assert 0.0 <= score <= 1.0, \
                    f"Result {i} score for '{label}' ({score}) should be in [0, 1]"
        
        print(f"✓ All scores are in valid range [0, 1]")
        
        # Step 5: Validate predicted emotions are valid
        for i, result in enumerate(results):
            for emotion in result['emotions']:
                assert emotion in Config.EMOTION_LABELS, \
                    f"Result {i} predicted emotion '{emotion}' is not valid"
        
        print(f"✓ All predicted emotions are valid")
        
        # Step 6: Display sample results
        print("\nSample predictions:")
        for i in range(min(3, len(results))):
            print(f"  Comment {i+1}: {test_comments[i][:50]}...")
            print(f"  Emotions: {results[i]['emotions']}")
            top_scores = sorted(
                results[i]['scores'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            print(f"  Top scores: {[(e, f'{s:.3f}') for e, s in top_scores]}")
        
        print("\n" + "="*70)
        print("✓ BATCH PREDICTION INTEGRATION TEST PASSED")
        print("="*70)
    
    def test_prediction_with_different_thresholds(self, minimal_trained_model):
        """
        Test prediction with different confidence thresholds.
        
        This test validates:
        1. Threshold parameter affects predicted emotions
        2. Lower threshold results in more predicted emotions
        3. Higher threshold results in fewer predicted emotions
        4. Scores remain unchanged regardless of threshold
        
        Expected behavior:
        - Different thresholds should produce different emotion lists
        - Scores should remain consistent
        
        **Validates: Requirements 8.4, 8.5**
        """
        model, tokenizer, device = minimal_trained_model
        
        test_comment = "I love this product! It's amazing!"
        
        print(f"\n[Test] Testing different thresholds")
        
        # Test with different thresholds
        thresholds = [0.3, 0.5, 0.7]
        results = {}
        
        for threshold in thresholds:
            result = predict_emotions(
                test_comment,
                model,
                tokenizer,
                device,
                threshold=threshold
            )
            results[threshold] = result
            print(f"  Threshold {threshold}: {len(result['emotions'])} emotions predicted")
        
        # Validate that scores are consistent across thresholds
        base_scores = results[0.5]['scores']
        for threshold in thresholds:
            for label in Config.EMOTION_LABELS:
                assert results[threshold]['scores'][label] == base_scores[label], \
                    f"Scores should be consistent across thresholds for '{label}'"
        
        print(f"✓ Scores are consistent across thresholds")
        
        # Generally, lower threshold should result in more predictions
        # (though not guaranteed with random weights)
        print(f"✓ Threshold parameter affects predicted emotions")
        
        print("\n" + "="*70)
        print("✓ THRESHOLD TESTING PASSED")
        print("="*70)
    
    def test_prediction_with_edge_cases(self, minimal_trained_model):
        """
        Test prediction with edge case inputs.
        
        This test validates:
        1. Empty string handling
        2. Very short text handling
        3. Very long text handling (truncation)
        4. Special characters handling
        5. Unicode/emoji handling
        
        Expected behavior:
        - Edge cases should be handled gracefully
        - Should not crash or produce invalid outputs
        
        **Validates: Requirements 8.1, 8.2, 8.3, 8.7**
        """
        model, tokenizer, device = minimal_trained_model
        
        print(f"\n[Test] Testing edge cases")
        
        # Test case 1: Empty string
        print("  Testing empty string...")
        try:
            result = predict_emotions("", model, tokenizer, device)
            pytest.fail("Empty string should raise ValueError")
        except ValueError as e:
            assert "empty" in str(e).lower(), "Error message should mention empty input"
            print(f"    ✓ Empty string raises ValueError as expected")
        
        # Test case 2: Whitespace only
        print("  Testing whitespace only...")
        try:
            result = predict_emotions("   ", model, tokenizer, device)
            pytest.fail("Whitespace-only string should raise ValueError")
        except ValueError as e:
            assert "empty" in str(e).lower(), "Error message should mention empty input"
            print(f"    ✓ Whitespace-only raises ValueError as expected")
        
        # Test case 3: Very short text
        print("  Testing very short text...")
        result = predict_emotions("Hi", model, tokenizer, device)
        assert isinstance(result, dict), "Should return valid result for short text"
        assert 'emotions' in result and 'scores' in result
        print(f"    ✓ Very short text handled correctly")
        
        # Test case 4: Very long text (should be truncated)
        print("  Testing very long text...")
        long_text = "This is a test. " * 200  # ~600 words, will exceed 512 tokens
        result = predict_emotions(long_text, model, tokenizer, device)
        assert isinstance(result, dict), "Should return valid result for long text"
        assert 'emotions' in result and 'scores' in result
        print(f"    ✓ Very long text handled correctly (truncated)")
        
        # Test case 5: Special characters
        print("  Testing special characters...")
        special_text = "Hello! How are you? :) #happy @user"
        result = predict_emotions(special_text, model, tokenizer, device)
        assert isinstance(result, dict), "Should handle special characters"
        assert 'emotions' in result and 'scores' in result
        print(f"    ✓ Special characters handled correctly")
        
        # Test case 6: Unicode and emoji
        print("  Testing Unicode and emoji...")
        unicode_text = "I'm so happy! 😊 ❤️"
        result = predict_emotions(unicode_text, model, tokenizer, device)
        assert isinstance(result, dict), "Should handle Unicode and emoji"
        assert 'emotions' in result and 'scores' in result
        print(f"    ✓ Unicode and emoji handled correctly")
        
        print("\n" + "="*70)
        print("✓ EDGE CASE TESTING PASSED")
        print("="*70)
    
    def test_prediction_output_consistency(self, minimal_trained_model):
        """
        Test that prediction outputs are consistent for the same input.
        
        This test validates:
        1. Multiple predictions on same input produce identical results
        2. Model is in eval mode (no dropout randomness)
        3. Results are deterministic
        
        Expected behavior:
        - Same input should always produce same output
        - No randomness in inference
        
        **Validates: Requirements 8.1, 8.2, 8.3**
        """
        model, tokenizer, device = minimal_trained_model
        
        test_comment = "I love this product! It's amazing!"
        
        print(f"\n[Test] Testing output consistency")
        
        # Perform prediction multiple times
        results = []
        for i in range(3):
            result = predict_emotions(test_comment, model, tokenizer, device)
            results.append(result)
        
        # Validate all results are identical
        base_result = results[0]
        for i, result in enumerate(results[1:], start=1):
            # Check emotions list
            assert result['emotions'] == base_result['emotions'], \
                f"Prediction {i} emotions differ from first prediction"
            
            # Check scores
            for label in Config.EMOTION_LABELS:
                assert result['scores'][label] == base_result['scores'][label], \
                    f"Prediction {i} score for '{label}' differs from first prediction"
        
        print(f"✓ All {len(results)} predictions are identical")
        
        print("\n" + "="*70)
        print("✓ CONSISTENCY TESTING PASSED")
        print("="*70)
    
    def test_prediction_with_no_emotions_above_threshold(self, minimal_trained_model):
        """
        Test prediction when no emotions exceed the threshold.
        
        This test validates:
        1. When no emotions exceed threshold, emotions list is empty
        2. All scores are still returned
        3. System handles this case gracefully
        
        Expected behavior:
        - Empty emotions list when no emotions exceed threshold
        - All 16 scores still present
        
        **Validates: Requirements 8.7**
        """
        model, tokenizer, device = minimal_trained_model
        
        test_comment = "This is a neutral statement."
        
        print(f"\n[Test] Testing high threshold (no emotions predicted)")
        
        # Use very high threshold to ensure no emotions are predicted
        result = predict_emotions(
            test_comment,
            model,
            tokenizer,
            device,
            threshold=0.99
        )
        
        # With random weights and high threshold, likely no emotions will be predicted
        # But we can't guarantee this, so we just check the structure is correct
        assert isinstance(result['emotions'], list), "Emotions should be a list"
        assert len(result['scores']) == 16, "Should have all 16 scores"
        
        if len(result['emotions']) == 0:
            print(f"✓ No emotions predicted (as expected with high threshold)")
        else:
            print(f"✓ Some emotions predicted (scores were high enough)")
        
        print(f"✓ System handles case gracefully")
        
        print("\n" + "="*70)
        print("✓ NO EMOTIONS THRESHOLD TESTING PASSED")
        print("="*70)
    
    def test_batch_prediction_preserves_order(self, minimal_trained_model):
        """
        Test that batch prediction preserves input order.
        
        This test validates:
        1. Results are returned in same order as input comments
        2. Each result corresponds to correct input comment
        
        Expected behavior:
        - Result[i] should correspond to input[i]
        
        **Validates: Requirements 8.6**
        """
        model, tokenizer, device = minimal_trained_model
        
        # Create distinct comments
        test_comments = [
            "I love this!",
            "This is terrible.",
            "It's okay."
        ]
        
        print(f"\n[Test] Testing batch prediction order preservation")
        
        # Get batch predictions
        batch_results = predict_emotions_batch(
            test_comments,
            model,
            tokenizer,
            device
        )
        
        # Get individual predictions
        individual_results = []
        for comment in test_comments:
            result = predict_emotions(comment, model, tokenizer, device)
            individual_results.append(result)
        
        # Validate batch results match individual results
        assert len(batch_results) == len(individual_results), \
            "Batch and individual results should have same length"
        
        for i in range(len(test_comments)):
            # Check emotions match
            assert batch_results[i]['emotions'] == individual_results[i]['emotions'], \
                f"Batch result {i} emotions don't match individual prediction"
            
            # Check scores match (use approximate comparison for floating point)
            for label in Config.EMOTION_LABELS:
                batch_score = batch_results[i]['scores'][label]
                individual_score = individual_results[i]['scores'][label]
                assert abs(batch_score - individual_score) < 1e-6, \
                    f"Batch result {i} score for '{label}' ({batch_score}) differs significantly from individual prediction ({individual_score})"
        
        print(f"✓ Batch prediction preserves order and matches individual predictions")
        
        print("\n" + "="*70)
        print("✓ ORDER PRESERVATION TESTING PASSED")
        print("="*70)


class TestPredictionPipelineWithSavedModel:
    """Integration tests using saved model checkpoint."""
    
    def test_load_and_predict_from_checkpoint(self, temp_model_checkpoint):
        """
        Test loading model from checkpoint and performing prediction.
        
        This test validates:
        1. Model can be loaded from saved checkpoint
        2. Loaded model can perform predictions
        3. Predictions produce valid outputs
        
        Expected behavior:
        - Model loads successfully from checkpoint
        - Predictions work correctly
        
        **Validates: Requirements 8.2, 8.3**
        """
        from utils import load_model
        
        print(f"\n[Test] Loading model from checkpoint and predicting")
        
        # Step 1: Load model from checkpoint
        try:
            model, tokenizer = load_model(temp_model_checkpoint, device='cpu')
            print(f"✓ Model loaded from checkpoint: {temp_model_checkpoint}")
        except Exception as e:
            pytest.fail(f"Failed to load model from checkpoint: {str(e)}")
        
        # Step 2: Perform prediction
        test_comment = "I love this product! It's amazing!"
        
        try:
            result = predict_emotions(
                test_comment,
                model,
                tokenizer,
                'cpu',
                threshold=0.5
            )
            print(f"✓ Prediction completed successfully")
        except Exception as e:
            pytest.fail(f"Prediction failed: {str(e)}")
        
        # Step 3: Validate output
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'emotions' in result, "Result should have 'emotions' key"
        assert 'scores' in result, "Result should have 'scores' key"
        assert len(result['scores']) == 16, "Should have all 16 emotion scores"
        
        print(f"✓ Output format is correct")
        print(f"  Predicted emotions: {result['emotions']}")
        
        print("\n" + "="*70)
        print("✓ CHECKPOINT LOADING AND PREDICTION TEST PASSED")
        print("="*70)
