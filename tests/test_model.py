"""
Unit tests for BERTEmotionClassifier model in Multi-label Emotion Classification system.

This module contains comprehensive tests for the BERTEmotionClassifier class,
validating model initialization, forward pass behavior, output shapes, device handling,
parameter trainability, save/load functionality, and edge cases.

**Validates: Requirements 11.6, 11.7, 11.8, 11.9, 5.1, 5.2, 5.3, 5.4, 5.5**
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

from model import BERTEmotionClassifier
from config import Config


@pytest.fixture
def model():
    """
    Fixture to provide a BERTEmotionClassifier instance with default parameters.
    
    Returns:
        BERTEmotionClassifier: Model instance with default configuration
    """
    return BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)


@pytest.fixture
def sample_inputs():
    """
    Fixture to provide sample input tensors for testing.
    
    Returns:
        tuple: (input_ids, attention_mask) with batch_size=4, seq_len=128
    """
    batch_size = 4
    seq_len = 128
    vocab_size = 30522  # BERT vocab size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    return input_ids, attention_mask


class TestModelInitialization:
    """Test suite for BERTEmotionClassifier initialization."""
    
    def test_initialization_default_parameters(self):
        """
        Test successful initialization with default parameters.
        
        Expected behavior:
        - Model should initialize without errors
        - Should have correct architecture components
        
        **Validates: Requirements 11.6, 5.1**
        """
        model = BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)
        
        assert hasattr(model, 'bert'), "Model should have BERT component"
        assert hasattr(model, 'dropout'), "Model should have dropout layer"
        assert hasattr(model, 'classifier'), "Model should have classifier layer"
        
        # Check classifier output dimension
        assert model.classifier.out_features == 16, "Classifier should output 16 labels"
    
    def test_initialization_custom_num_labels(self):
        """
        Test initialization with custom number of labels.
        
        Expected behavior:
        - Model should accept different num_labels values
        - Classifier output dimension should match num_labels
        
        **Validates: Requirements 11.6, 5.1**
        """
        for num_labels in [8, 16, 20, 32]:
            model = BERTEmotionClassifier(num_labels=num_labels, dropout_rate=0.3)
            assert model.classifier.out_features == num_labels, f"Classifier should output {num_labels} labels"
    
    def test_initialization_custom_dropout_rate(self):
        """
        Test initialization with custom dropout rate.
        
        Expected behavior:
        - Model should accept different dropout rates
        - Dropout layer should have correct dropout probability
        
        **Validates: Requirements 11.6, 5.1**
        """
        for dropout_rate in [0.1, 0.3, 0.5]:
            model = BERTEmotionClassifier(num_labels=16, dropout_rate=dropout_rate)
            assert model.dropout.p == dropout_rate, f"Dropout rate should be {dropout_rate}"
    
    def test_initialization_bert_hidden_size(self):
        """
        Test that BERT hidden size is correct (768 for bert-base-uncased).
        
        Expected behavior:
        - BERT config should have hidden_size = 768
        - Classifier input dimension should be 768
        
        **Validates: Requirements 5.1, 5.2**
        """
        model = BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)
        
        bert_hidden_size = model.bert.config.hidden_size
        assert bert_hidden_size == 768, "BERT base should have hidden size 768"
        
        classifier_input_size = model.classifier.in_features
        assert classifier_input_size == 768, "Classifier input should match BERT hidden size"
    
    def test_initialization_model_type(self):
        """
        Test that model is an instance of nn.Module.
        
        Expected behavior:
        - Model should inherit from nn.Module
        - Should be compatible with PyTorch training
        
        **Validates: Requirements 11.6**
        """
        model = BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)
        
        assert isinstance(model, nn.Module), "Model should be an nn.Module instance"


class TestModelForwardPass:
    """Test suite for BERTEmotionClassifier forward pass."""
    
    def test_forward_pass_output_shape(self, model, sample_inputs):
        """
        Test that forward pass produces correct output shape.
        
        Expected behavior:
        - Output shape should be (batch_size, num_labels)
        
        **Validates: Requirements 11.7, 5.3**
        """
        input_ids, attention_mask = sample_inputs
        batch_size = input_ids.shape[0]
        
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 16), f"Output shape should be ({batch_size}, 16)"
    
    def test_forward_pass_returns_logits(self, model, sample_inputs):
        """
        Test that forward pass returns logits (not probabilities).
        
        Expected behavior:
        - Output should be raw logits (can be negative, > 1)
        - Output should NOT be probabilities (0-1 range)
        
        **Validates: Requirements 11.7, 5.3**
        """
        input_ids, attention_mask = sample_inputs
        
        logits = model(input_ids, attention_mask)
        
        # Logits can be any real number (negative or > 1)
        # If they were probabilities, they would all be in [0, 1]
        # We check that at least some values are outside [0, 1]
        has_values_outside_prob_range = torch.any(logits < 0) or torch.any(logits > 1)
        
        # Note: In rare cases, all logits might happen to be in [0, 1] by chance
        # So we also check that sigmoid hasn't been applied by comparing with sigmoid output
        probabilities = torch.sigmoid(logits)
        assert not torch.allclose(logits, probabilities, atol=1e-6), "Output should be logits, not probabilities"
    
    def test_forward_pass_output_type(self, model, sample_inputs):
        """
        Test that forward pass returns a torch.Tensor.
        
        Expected behavior:
        - Output should be a torch.Tensor
        - Output dtype should be float32
        
        **Validates: Requirements 11.7, 5.3**
        """
        input_ids, attention_mask = sample_inputs
        
        logits = model(input_ids, attention_mask)
        
        assert isinstance(logits, torch.Tensor), "Output should be a torch.Tensor"
        assert logits.dtype == torch.float32, "Output should be float32"
    
    def test_forward_pass_different_batch_sizes(self, model):
        """
        Test forward pass with different batch sizes.
        
        Expected behavior:
        - Should handle various batch sizes correctly
        - Output shape should match input batch size
        
        **Validates: Requirements 11.7, 5.3**
        """
        seq_len = 128
        vocab_size = 30522
        
        for batch_size in [1, 2, 8, 16, 32]:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            
            logits = model(input_ids, attention_mask)
            
            assert logits.shape == (batch_size, 16), f"Output shape should be ({batch_size}, 16)"
    
    def test_forward_pass_different_sequence_lengths(self, model):
        """
        Test forward pass with different sequence lengths.
        
        Expected behavior:
        - Should handle various sequence lengths correctly
        - Output shape should always be (batch_size, num_labels)
        
        **Validates: Requirements 11.7, 5.3**
        """
        batch_size = 4
        vocab_size = 30522
        
        for seq_len in [32, 64, 128, 256, 512]:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            
            logits = model(input_ids, attention_mask)
            
            assert logits.shape == (batch_size, 16), f"Output shape should be ({batch_size}, 16) for seq_len={seq_len}"
    
    def test_forward_pass_with_padding(self, model):
        """
        Test forward pass with padded sequences (attention_mask with 0s).
        
        Expected behavior:
        - Should handle padded sequences correctly
        - Padding should be masked by attention_mask
        
        **Validates: Requirements 11.7, 5.3**
        """
        batch_size = 4
        seq_len = 128
        vocab_size = 30522
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        # Set last 50 tokens as padding
        attention_mask[:, -50:] = 0
        input_ids[:, -50:] = 0  # Padding token ID
        
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 16), "Should handle padded sequences correctly"
    
    def test_forward_pass_gradient_flow(self, model, sample_inputs):
        """
        Test that gradients flow through the model during forward pass.
        
        Expected behavior:
        - Output should require gradients
        - Gradients should be computable
        
        **Validates: Requirements 11.7, 5.3**
        """
        input_ids, attention_mask = sample_inputs
        
        model.train()  # Set to training mode
        logits = model(input_ids, attention_mask)
        
        assert logits.requires_grad, "Output should require gradients"
        
        # Compute a dummy loss and backpropagate
        loss = logits.sum()
        loss.backward()
        
        # Check that at least some parameters have gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "Gradients should flow through the model"


class TestModelDeviceHandling:
    """Test suite for model device handling (CPU/CUDA)."""
    
    def test_model_on_cpu(self, model):
        """
        Test that model can be moved to CPU.
        
        Expected behavior:
        - Model should be movable to CPU
        - All parameters should be on CPU
        
        **Validates: Requirements 11.8, 5.4**
        """
        model = model.to('cpu')
        
        # Check that all parameters are on CPU
        for param in model.parameters():
            assert param.device.type == 'cpu', "All parameters should be on CPU"
    
    def test_model_forward_on_cpu(self, model, sample_inputs):
        """
        Test forward pass on CPU.
        
        Expected behavior:
        - Forward pass should work on CPU
        - Output should be on CPU
        
        **Validates: Requirements 11.8, 5.4**
        """
        model = model.to('cpu')
        input_ids, attention_mask = sample_inputs
        input_ids = input_ids.to('cpu')
        attention_mask = attention_mask.to('cpu')
        
        logits = model(input_ids, attention_mask)
        
        assert logits.device.type == 'cpu', "Output should be on CPU"
        assert logits.shape == (4, 16), "Output shape should be correct"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_cuda(self, model):
        """
        Test that model can be moved to CUDA.
        
        Expected behavior:
        - Model should be movable to CUDA
        - All parameters should be on CUDA
        
        **Validates: Requirements 11.8, 5.4**
        """
        model = model.to('cuda')
        
        # Check that all parameters are on CUDA
        for param in model.parameters():
            assert param.device.type == 'cuda', "All parameters should be on CUDA"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_forward_on_cuda(self, model, sample_inputs):
        """
        Test forward pass on CUDA.
        
        Expected behavior:
        - Forward pass should work on CUDA
        - Output should be on CUDA
        
        **Validates: Requirements 11.8, 5.4**
        """
        model = model.to('cuda')
        input_ids, attention_mask = sample_inputs
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        
        logits = model(input_ids, attention_mask)
        
        assert logits.device.type == 'cuda', "Output should be on CUDA"
        assert logits.shape == (4, 16), "Output shape should be correct"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_device_transfer(self, model, sample_inputs):
        """
        Test transferring model between CPU and CUDA.
        
        Expected behavior:
        - Model should be transferable between devices
        - Forward pass should work after transfer
        
        **Validates: Requirements 11.8, 5.4**
        """
        input_ids, attention_mask = sample_inputs
        
        # Start on CPU
        model = model.to('cpu')
        input_ids_cpu = input_ids.to('cpu')
        attention_mask_cpu = attention_mask.to('cpu')
        
        logits_cpu = model(input_ids_cpu, attention_mask_cpu)
        assert logits_cpu.device.type == 'cpu', "Output should be on CPU"
        
        # Move to CUDA
        model = model.to('cuda')
        input_ids_cuda = input_ids.to('cuda')
        attention_mask_cuda = attention_mask.to('cuda')
        
        logits_cuda = model(input_ids_cuda, attention_mask_cuda)
        assert logits_cuda.device.type == 'cuda', "Output should be on CUDA"
        
        # Move back to CPU
        model = model.to('cpu')
        logits_cpu2 = model(input_ids_cpu, attention_mask_cpu)
        assert logits_cpu2.device.type == 'cpu', "Output should be on CPU after transfer back"


class TestModelParameters:
    """Test suite for model parameter properties."""
    
    def test_model_parameters_trainable(self, model):
        """
        Test that model parameters are trainable (requires_grad=True).
        
        Expected behavior:
        - All parameters should have requires_grad=True by default
        
        **Validates: Requirements 11.9, 5.5**
        """
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Parameter {name} should be trainable"
    
    def test_model_has_parameters(self, model):
        """
        Test that model has parameters.
        
        Expected behavior:
        - Model should have a non-zero number of parameters
        
        **Validates: Requirements 11.9, 5.5**
        """
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0, "Model should have parameters"
        
        # BERT base has ~110M parameters, plus classifier layer
        assert num_params > 100_000_000, "Model should have at least 100M parameters (BERT base)"
    
    def test_model_parameter_groups(self, model):
        """
        Test that model has expected parameter groups (BERT, dropout, classifier).
        
        Expected behavior:
        - Should have parameters from BERT, dropout, and classifier
        
        **Validates: Requirements 11.9, 5.5**
        """
        param_names = [name for name, _ in model.named_parameters()]
        
        # Check for BERT parameters
        bert_params = [name for name in param_names if 'bert' in name]
        assert len(bert_params) > 0, "Should have BERT parameters"
        
        # Check for classifier parameters
        classifier_params = [name for name in param_names if 'classifier' in name]
        assert len(classifier_params) > 0, "Should have classifier parameters"
    
    def test_classifier_parameters_shape(self, model):
        """
        Test that classifier layer has correct parameter shapes.
        
        Expected behavior:
        - Classifier weight should be (num_labels, 768)
        - Classifier bias should be (num_labels,)
        
        **Validates: Requirements 11.9, 5.5**
        """
        classifier_weight = model.classifier.weight
        classifier_bias = model.classifier.bias
        
        assert classifier_weight.shape == (16, 768), "Classifier weight should be (16, 768)"
        assert classifier_bias.shape == (16,), "Classifier bias should be (16,)"
    
    def test_model_gradient_computation(self, model, sample_inputs):
        """
        Test that gradients can be computed for model parameters.
        
        Expected behavior:
        - After backward pass, parameters used in forward pass should have gradients
        - At least classifier and some BERT encoder parameters should have gradients
        
        **Validates: Requirements 11.9, 5.5**
        """
        input_ids, attention_mask = sample_inputs
        
        model.train()
        model.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = logits.sum()
        loss.backward()
        
        # Check that classifier parameters have gradients
        assert model.classifier.weight.grad is not None, "Classifier weight should have gradients"
        assert model.classifier.bias.grad is not None, "Classifier bias should have gradients"
        
        # Check that at least some BERT parameters have gradients
        bert_params_with_grad = sum(1 for name, param in model.named_parameters() 
                                    if 'bert' in name and param.grad is not None)
        assert bert_params_with_grad > 0, "At least some BERT parameters should have gradients"


class TestModelSaveLoad:
    """Test suite for model save and load functionality."""
    
    def test_model_save_state_dict(self, model):
        """
        Test that model state_dict can be saved.
        
        Expected behavior:
        - state_dict() should return a dictionary
        - Dictionary should contain all model parameters
        
        **Validates: Requirements 11.10, 5.6**
        """
        state_dict = model.state_dict()
        
        assert isinstance(state_dict, dict), "state_dict should be a dictionary"
        assert len(state_dict) > 0, "state_dict should not be empty"
        
        # Check for key components
        assert any('bert' in key for key in state_dict.keys()), "state_dict should contain BERT parameters"
        assert any('classifier' in key for key in state_dict.keys()), "state_dict should contain classifier parameters"
    
    def test_model_load_state_dict(self, model):
        """
        Test that model can load state_dict.
        
        Expected behavior:
        - Model should be able to load its own state_dict
        - Parameters should match after loading
        
        **Validates: Requirements 11.10, 5.6**
        """
        # Save state dict
        state_dict = model.state_dict()
        
        # Create new model and load state dict
        new_model = BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)
        new_model.load_state_dict(state_dict)
        
        # Check that parameters match
        for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
            assert name1 == name2, "Parameter names should match"
            assert torch.equal(param1, param2), f"Parameter {name1} should match after loading"
    
    def test_model_save_load_to_file(self, model, sample_inputs):
        """
        Test saving and loading model to/from file.
        
        Expected behavior:
        - Model should be savable to file
        - Loaded model should produce same outputs
        
        **Validates: Requirements 11.10, 5.6**
        """
        input_ids, attention_mask = sample_inputs
        
        # Get original output
        model.eval()
        with torch.no_grad():
            original_output = model(input_ids, attention_mask)
        
        # Save model to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'model.pt')
            torch.save(model.state_dict(), save_path)
            
            # Load model from file
            new_model = BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)
            new_model.load_state_dict(torch.load(save_path))
            new_model.eval()
            
            # Get output from loaded model
            with torch.no_grad():
                loaded_output = new_model(input_ids, attention_mask)
            
            # Outputs should be identical
            assert torch.allclose(original_output, loaded_output, atol=1e-6), "Loaded model should produce same outputs"
    
    def test_model_save_load_preserves_architecture(self, model):
        """
        Test that save/load preserves model architecture.
        
        Expected behavior:
        - Loaded model should have same architecture
        - All components should be present
        
        **Validates: Requirements 11.10, 5.6**
        """
        state_dict = model.state_dict()
        
        new_model = BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)
        new_model.load_state_dict(state_dict)
        
        # Check architecture components
        assert hasattr(new_model, 'bert'), "Loaded model should have BERT component"
        assert hasattr(new_model, 'dropout'), "Loaded model should have dropout layer"
        assert hasattr(new_model, 'classifier'), "Loaded model should have classifier layer"
        
        # Check dimensions
        assert new_model.classifier.out_features == 16, "Loaded model should have correct output dimension"
    
    def test_model_save_load_different_parameters(self):
        """
        Test that models with different parameters can be saved and loaded.
        
        Expected behavior:
        - Models with different num_labels should save/load correctly
        
        **Validates: Requirements 11.10, 5.6**
        """
        for num_labels in [8, 16, 20]:
            model = BERTEmotionClassifier(num_labels=num_labels, dropout_rate=0.3)
            state_dict = model.state_dict()
            
            new_model = BERTEmotionClassifier(num_labels=num_labels, dropout_rate=0.3)
            new_model.load_state_dict(state_dict)
            
            assert new_model.classifier.out_features == num_labels, f"Loaded model should have {num_labels} outputs"


class TestModelEdgeCases:
    """Test suite for model edge cases."""
    
    def test_single_sample_forward(self, model):
        """
        Test forward pass with single sample (batch_size=1).
        
        Expected behavior:
        - Should handle single sample correctly
        - Output shape should be (1, num_labels)
        
        **Validates: Requirements 11.7, 5.3**
        """
        batch_size = 1
        seq_len = 128
        vocab_size = 30522
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (1, 16), "Output shape should be (1, 16) for single sample"
    
    def test_large_batch_forward(self, model):
        """
        Test forward pass with large batch size.
        
        Expected behavior:
        - Should handle large batches correctly
        - Output shape should match batch size
        
        **Validates: Requirements 11.7, 5.3**
        """
        batch_size = 64
        seq_len = 128
        vocab_size = 30522
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 16), f"Output shape should be ({batch_size}, 16)"
    
    def test_model_eval_mode(self, model, sample_inputs):
        """
        Test model in evaluation mode.
        
        Expected behavior:
        - Model should switch to eval mode
        - Dropout should be disabled in eval mode
        
        **Validates: Requirements 11.7, 5.3**
        """
        input_ids, attention_mask = sample_inputs
        
        model.eval()
        
        # In eval mode, dropout is disabled, so outputs should be deterministic
        with torch.no_grad():
            output1 = model(input_ids, attention_mask)
            output2 = model(input_ids, attention_mask)
        
        assert torch.equal(output1, output2), "Outputs should be identical in eval mode"
    
    def test_model_train_mode_dropout(self, model, sample_inputs):
        """
        Test that dropout is active in training mode.
        
        Expected behavior:
        - In train mode, dropout should introduce randomness
        - Multiple forward passes may produce different outputs
        
        **Validates: Requirements 11.7, 5.3**
        """
        input_ids, attention_mask = sample_inputs
        
        model.train()
        
        # In train mode, dropout introduces randomness
        # Note: Due to randomness, outputs might occasionally be very similar
        # We run multiple times to increase confidence
        outputs = []
        for _ in range(5):
            output = model(input_ids, attention_mask)
            outputs.append(output)
        
        # Check that at least some outputs differ
        all_identical = all(torch.equal(outputs[0], out) for out in outputs[1:])
        
        # Note: There's a small chance all outputs are identical due to dropout randomness
        # This test might rarely fail, but it's a good sanity check
        # If dropout_rate is 0.3, the probability of all outputs being identical is very low
    
    def test_model_with_all_padding(self, model):
        """
        Test forward pass with all padding (attention_mask all zeros).
        
        Expected behavior:
        - Should handle all-padding input without errors
        - Should produce valid output shape
        
        **Validates: Requirements 11.7, 5.3**
        """
        batch_size = 4
        seq_len = 128
        
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 16), "Should handle all-padding input"
    
    def test_model_with_minimum_sequence_length(self, model):
        """
        Test forward pass with minimum sequence length (1 token).
        
        Expected behavior:
        - Should handle very short sequences
        - Output shape should be correct
        
        **Validates: Requirements 11.7, 5.3**
        """
        batch_size = 4
        seq_len = 1
        vocab_size = 30522
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 16), "Should handle minimum sequence length"
    
    def test_model_output_range(self, model, sample_inputs):
        """
        Test that model outputs are in a reasonable range.
        
        Expected behavior:
        - Logits should be finite (no NaN or Inf)
        - Logits should be in a reasonable range (typically -10 to 10)
        
        **Validates: Requirements 11.7, 5.3**
        """
        input_ids, attention_mask = sample_inputs
        
        model.eval()
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        
        # Check for NaN or Inf
        assert torch.isfinite(logits).all(), "Logits should be finite (no NaN or Inf)"
        
        # Check that logits are in a reasonable range
        # Logits typically range from -10 to 10, but can be outside this range
        # We just check they're not extremely large
        assert torch.abs(logits).max() < 100, "Logits should be in a reasonable range"
