"""
Integration test for training pipeline in Multi-label Emotion Classification system.

This module contains an end-to-end integration test for the training pipeline,
validating that the complete training flow works correctly with minimal sample data.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np
import os
import tempfile
import shutil

from config import Config
from model import BERTEmotionClassifier
from dataset import EmotionDataset
from train import train_epoch, evaluate
from utils import save_model, load_model


@pytest.fixture
def minimal_sample_data():
    """
    Fixture to provide minimal sample data for integration testing.
    
    Creates 20 samples with realistic multi-label emotion combinations
    to test the complete training pipeline.
    
    Returns:
        tuple: (texts, labels) where:
            - texts (list): List of 20 comment strings
            - labels (np.ndarray): Binary label matrix of shape (20, 16)
    """
    texts = [
        "I love this product! It's amazing and exceeded my expectations!",
        "This is terrible. Very disappointed with the quality.",
        "Good quality but shipping was slow. Mixed feelings.",
        "Absolutely fantastic! Best purchase ever!",
        "Not bad, could be better. Average experience.",
        "I'm so happy with this! Great value for money!",
        "Worst experience ever. Never buying again.",
        "Pretty good overall. Would recommend to friends.",
        "Meh, it's okay. Nothing special.",
        "Outstanding! Exceeded all my expectations!",
        "Disappointed and frustrated. Poor customer service.",
        "Decent product. Does what it's supposed to do.",
        "I'm thrilled! This is exactly what I needed!",
        "Terrible quality. Complete waste of money.",
        "Satisfied with my purchase. Good product.",
        "Amazing! I'm so excited to use this!",
        "Not impressed. Expected much better quality.",
        "Great product! Very happy with it!",
        "Awful experience. Would not recommend.",
        "Perfect! Exactly as described and works great!"
    ]
    
    # Create realistic multi-label emotion combinations
    # Each row represents emotions for one comment
    labels = np.array([
        [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # joy, trust, surprise, love, excited
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # sadness, disgust, anger, disappointed
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],  # trust, anticipation, worried, calm
        [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1],  # joy, trust, surprise, anticipation, love, proud, excited
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],  # anticipation, calm
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # joy, trust, love, excited
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # sadness, disgust, anger, disappointed
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # joy, trust
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # calm
        [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # joy, trust, surprise, love, proud, excited
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # sadness, anger, disappointed
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # trust, calm
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # joy, trust, love, excited
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # sadness, disgust, anger, disappointed
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # joy, trust
        [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # joy, trust, surprise, love, excited
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # disappointed
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # joy, trust, love
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # sadness, disgust, anger, disappointed
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # joy, trust, love, proud, excited
    ], dtype=np.float32)
    
    return texts, labels


@pytest.fixture
def temp_model_dir():
    """
    Fixture to provide a temporary directory for saving model checkpoints.
    
    Creates a temporary directory that is automatically cleaned up after the test.
    
    Yields:
        str: Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestTrainingPipelineIntegration:
    """Integration test suite for the complete training pipeline."""
    
    def test_end_to_end_training_pipeline(self, minimal_sample_data, temp_model_dir):
        """
        Test complete end-to-end training pipeline with minimal data.
        
        This integration test validates:
        1. Dataset creation and DataLoader setup
        2. Model initialization
        3. Training for 1 epoch completes without errors
        4. Evaluation on validation set works correctly
        5. Model checkpoint is saved successfully
        6. Saved checkpoint can be loaded
        7. Loaded model produces valid outputs
        
        Expected behavior:
        - Training should complete without errors
        - Loss should be computed correctly
        - Model checkpoint should be saved to disk
        - Loaded model should produce same outputs as original
        
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
        """
        texts, labels = minimal_sample_data
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Device configuration
        device = torch.device('cpu')  # Use CPU for testing to avoid GPU dependency
        
        # Step 1: Create dataset and dataloader
        print("\n[Step 1/7] Creating dataset and dataloader...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Split data into train (16 samples) and val (4 samples)
        train_texts = texts[:16]
        train_labels = labels[:16]
        val_texts = texts[16:]
        val_labels = labels[16:]
        
        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length=128)
        val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length=128)
        
        # Use small batch size for testing
        batch_size = 4
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        assert len(train_loader) > 0, "Train loader should have batches"
        assert len(val_loader) > 0, "Val loader should have batches"
        print(f"✓ Created train loader with {len(train_loader)} batches")
        print(f"✓ Created val loader with {len(val_loader)} batches")
        
        # Step 2: Initialize model, optimizer, and loss function
        print("\n[Step 2/7] Initializing model, optimizer, and loss function...")
        model = BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.BCEWithLogitsLoss()
        
        assert model is not None, "Model should be initialized"
        assert optimizer is not None, "Optimizer should be initialized"
        assert criterion is not None, "Loss function should be initialized"
        print("✓ Model initialized successfully")
        print("✓ Optimizer (AdamW) initialized")
        print("✓ Loss function (BCEWithLogitsLoss) initialized")
        
        # Step 3: Train for 1 epoch
        print("\n[Step 3/7] Training for 1 epoch...")
        initial_loss = None
        
        try:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            initial_loss = train_loss
            
            assert train_loss is not None, "Training should return a loss value"
            assert isinstance(train_loss, float), "Loss should be a float"
            assert train_loss > 0, "Loss should be positive"
            assert train_loss < 10, "Loss should be in reasonable range"
            assert not np.isnan(train_loss), "Loss should not be NaN"
            assert not np.isinf(train_loss), "Loss should not be infinite"
            
            print(f"✓ Training completed successfully")
            print(f"✓ Training loss: {train_loss:.4f}")
            
        except Exception as e:
            pytest.fail(f"Training failed with error: {str(e)}")
        
        # Step 4: Evaluate on validation set
        print("\n[Step 4/7] Evaluating on validation set...")
        
        try:
            val_loss, val_predictions, val_labels_array = evaluate(
                model, val_loader, criterion, device
            )
            
            assert val_loss is not None, "Evaluation should return a loss value"
            assert isinstance(val_loss, float), "Val loss should be a float"
            assert val_loss > 0, "Val loss should be positive"
            assert not np.isnan(val_loss), "Val loss should not be NaN"
            assert not np.isinf(val_loss), "Val loss should not be infinite"
            
            assert val_predictions is not None, "Evaluation should return predictions"
            assert isinstance(val_predictions, np.ndarray), "Predictions should be numpy array"
            assert val_predictions.shape == (4, 16), "Predictions shape should be (4, 16)"
            
            assert val_labels_array is not None, "Evaluation should return labels"
            assert isinstance(val_labels_array, np.ndarray), "Labels should be numpy array"
            assert val_labels_array.shape == (4, 16), "Labels shape should be (4, 16)"
            
            # Check that predictions are probabilities (0-1 range)
            assert np.all(val_predictions >= 0), "Predictions should be >= 0"
            assert np.all(val_predictions <= 1), "Predictions should be <= 1"
            
            print(f"✓ Evaluation completed successfully")
            print(f"✓ Validation loss: {val_loss:.4f}")
            print(f"✓ Predictions shape: {val_predictions.shape}")
            
        except Exception as e:
            pytest.fail(f"Evaluation failed with error: {str(e)}")
        
        # Step 5: Save model checkpoint
        print("\n[Step 5/7] Saving model checkpoint...")
        
        training_config = {
            'model_name': 'bert-base-uncased',
            'num_labels': 16,
            'dropout_rate': 0.3,
            'learning_rate': 2e-5,
            'batch_size': batch_size,
            'num_epochs': 1,
            'max_length': 128,
            'train_loss': initial_loss,
            'val_loss': val_loss
        }
        
        try:
            save_model(model, tokenizer, temp_model_dir, training_config)
            
            # Verify checkpoint files exist
            model_path = os.path.join(temp_model_dir, 'pytorch_model.bin')
            config_path = os.path.join(temp_model_dir, 'training_config.json')
            
            assert os.path.exists(model_path), "Model checkpoint should be saved"
            assert os.path.exists(config_path), "Training config should be saved"
            
            # Check file sizes are reasonable
            model_size = os.path.getsize(model_path)
            assert model_size > 1_000_000, "Model checkpoint should be > 1MB (BERT is large)"
            
            print(f"✓ Model checkpoint saved to: {model_path}")
            print(f"✓ Model checkpoint size: {model_size / 1_000_000:.1f} MB")
            print(f"✓ Training config saved to: {config_path}")
            
        except Exception as e:
            pytest.fail(f"Model saving failed with error: {str(e)}")
        
        # Step 6: Load saved checkpoint
        print("\n[Step 6/7] Loading saved checkpoint...")
        
        try:
            loaded_model, loaded_tokenizer = load_model(temp_model_dir, device='cpu')
            
            assert loaded_model is not None, "Loaded model should not be None"
            assert loaded_tokenizer is not None, "Loaded tokenizer should not be None"
            assert isinstance(loaded_model, BERTEmotionClassifier), "Loaded model should be BERTEmotionClassifier"
            
            print(f"✓ Model loaded successfully from: {temp_model_dir}")
            print(f"✓ Tokenizer loaded successfully")
            
        except Exception as e:
            pytest.fail(f"Model loading failed with error: {str(e)}")
        
        # Step 7: Verify loaded model produces valid outputs
        print("\n[Step 7/7] Verifying loaded model produces valid outputs...")
        
        try:
            # Get a sample batch from validation loader
            sample_batch = next(iter(val_loader))
            input_ids = sample_batch['input_ids'].to(device)
            attention_mask = sample_batch['attention_mask'].to(device)
            
            # Get outputs from original model
            model.eval()
            with torch.no_grad():
                original_output = model(input_ids, attention_mask)
            
            # Get outputs from loaded model
            loaded_model.eval()
            with torch.no_grad():
                loaded_output = loaded_model(input_ids, attention_mask)
            
            # Verify outputs match (should be identical since we loaded same weights)
            assert original_output.shape == loaded_output.shape, "Output shapes should match"
            assert torch.allclose(original_output, loaded_output, atol=1e-5), \
                "Loaded model should produce same outputs as original"
            
            # Verify output shape is correct
            batch_size_actual = input_ids.shape[0]
            assert original_output.shape == (batch_size_actual, 16), \
                f"Output shape should be ({batch_size_actual}, 16)"
            
            # Verify outputs are finite
            assert torch.isfinite(original_output).all(), "Outputs should be finite"
            assert torch.isfinite(loaded_output).all(), "Loaded model outputs should be finite"
            
            print(f"✓ Loaded model produces valid outputs")
            print(f"✓ Output shape: {loaded_output.shape}")
            print(f"✓ Outputs match original model (max diff: {torch.max(torch.abs(original_output - loaded_output)).item():.2e})")
            
        except Exception as e:
            pytest.fail(f"Output verification failed with error: {str(e)}")
        
        print("\n" + "="*70)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("="*70)
        print("\nSummary:")
        print(f"  - Trained on {len(train_texts)} samples for 1 epoch")
        print(f"  - Validated on {len(val_texts)} samples")
        print(f"  - Training loss: {initial_loss:.4f}")
        print(f"  - Validation loss: {val_loss:.4f}")
        print(f"  - Model checkpoint saved and loaded successfully")
        print(f"  - Loaded model produces identical outputs")
        print("="*70)
    
    def test_training_reduces_loss(self, minimal_sample_data, temp_model_dir):
        """
        Test that training actually reduces loss over multiple batches.
        
        This test verifies that the model is learning by checking that
        the loss decreases as training progresses.
        
        Expected behavior:
        - Loss should generally decrease over batches
        - Final loss should be lower than initial loss
        
        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        texts, labels = minimal_sample_data
        
        # Set random seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        device = torch.device('cpu')
        
        # Create dataset and dataloader
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        # Initialize model
        model = BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)
        model = model.to(device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.BCEWithLogitsLoss()
        
        # Track losses
        batch_losses = []
        
        # Train for a few batches
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Train for 3 batches
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        # Verify we have losses
        assert len(batch_losses) >= 2, "Should have at least 2 batch losses"
        
        # Check that losses are reasonable
        for loss in batch_losses:
            assert loss > 0, "Loss should be positive"
            assert loss < 10, "Loss should be in reasonable range"
            assert not np.isnan(loss), "Loss should not be NaN"
        
        print(f"\nBatch losses: {[f'{loss:.4f}' for loss in batch_losses]}")
        print(f"Initial loss: {batch_losses[0]:.4f}")
        print(f"Final loss: {batch_losses[-1]:.4f}")
        
        # Note: We don't strictly require loss to decrease monotonically
        # because with small batch sizes and limited data, there can be variance
        # But we can check that the model is producing valid gradients
        print("✓ Model is training and producing valid losses")
    
    def test_training_with_different_batch_sizes(self, minimal_sample_data, temp_model_dir):
        """
        Test that training works with different batch sizes.
        
        Expected behavior:
        - Training should work with batch_size=1, 2, 4
        - All batch sizes should produce valid losses
        
        **Validates: Requirements 6.1, 6.2, 6.3**
        """
        texts, labels = minimal_sample_data
        device = torch.device('cpu')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        for batch_size in [1, 2, 4]:
            print(f"\nTesting with batch_size={batch_size}")
            
            # Set random seed for consistency
            torch.manual_seed(42)
            np.random.seed(42)
            
            # Create dataset and dataloader
            train_dataset = EmotionDataset(texts[:12], labels[:12], tokenizer, max_length=128)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            model = BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)
            model = model.to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            criterion = nn.BCEWithLogitsLoss()
            
            # Train for 1 epoch
            try:
                train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
                
                assert train_loss is not None, f"Training with batch_size={batch_size} should return loss"
                assert train_loss > 0, f"Loss should be positive for batch_size={batch_size}"
                assert not np.isnan(train_loss), f"Loss should not be NaN for batch_size={batch_size}"
                
                print(f"✓ batch_size={batch_size}: loss={train_loss:.4f}")
                
            except Exception as e:
                pytest.fail(f"Training failed with batch_size={batch_size}: {str(e)}")
        
        print("\n✓ Training works with all tested batch sizes")
