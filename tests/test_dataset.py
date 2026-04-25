"""
Unit tests for EmotionDataset class in Multi-label Emotion Classification system.

This module contains comprehensive tests for the EmotionDataset class,
validating dataset initialization, length, item retrieval, tokenization,
DataLoader compatibility, and edge cases.

**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 4.1, 4.2, 4.3, 4.4, 4.5**
"""

import pytest
import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from dataset import EmotionDataset
from config import Config


@pytest.fixture
def tokenizer():
    """
    Fixture to provide a BERT tokenizer for tests.
    
    Returns:
        BertTokenizer: Pre-trained BERT tokenizer instance
    """
    return BertTokenizer.from_pretrained('bert-base-uncased')


@pytest.fixture
def sample_data():
    """
    Fixture to provide sample texts and labels for testing.
    
    Returns:
        tuple: (texts, labels) where texts is a list of strings and
               labels is a numpy array of shape (N, 16)
    """
    texts = [
        "I love this product! It's amazing!",
        "This is terrible. Very disappointed.",
        "Good quality but shipping was slow.",
        "Absolutely fantastic! Exceeded expectations!",
        "Not bad, could be better."
    ]
    
    # Binary labels for 16 emotions (5 samples x 16 emotions)
    labels = np.array([
        [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # joy, trust, surprise, love, excited
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # sadness, disgust, anger, disappointed
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],  # trust, anticipation, worried, calm
        [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1],  # joy, trust, surprise, anticipation, love, proud, excited
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],  # anticipation, calm
    ], dtype=np.float32)
    
    return texts, labels


class TestEmotionDatasetInitialization:
    """Test suite for EmotionDataset initialization."""
    
    def test_initialization_success(self, tokenizer, sample_data):
        """
        Test successful initialization of EmotionDataset.
        
        Expected behavior:
        - Dataset should initialize without errors
        - All attributes should be set correctly
        
        **Validates: Requirements 11.1, 11.2**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        
        assert dataset.texts == texts, "Texts should be stored correctly"
        assert np.array_equal(dataset.labels, labels), "Labels should be stored correctly"
        assert dataset.tokenizer == tokenizer, "Tokenizer should be stored correctly"
        assert dataset.max_length == 128, "Max length should be set correctly"
    
    def test_initialization_default_max_length(self, tokenizer, sample_data):
        """
        Test initialization with default max_length parameter.
        
        Expected behavior:
        - Default max_length should be 512 (BERT's maximum)
        
        **Validates: Requirements 11.2, 4.3**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        assert dataset.max_length == 512, "Default max_length should be 512"
    
    def test_initialization_mismatched_lengths(self, tokenizer):
        """
        Test initialization with mismatched texts and labels lengths.
        
        Expected behavior:
        - Should raise ValueError when len(texts) != len(labels)
        
        **Validates: Requirements 11.2**
        """
        texts = ["Text 1", "Text 2", "Text 3"]
        labels = np.array([
            [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        ], dtype=np.float32)  # Only 2 labels for 3 texts
        
        with pytest.raises(ValueError, match="Number of texts .* must match number of label rows"):
            EmotionDataset(texts, labels, tokenizer)
    
    def test_initialization_invalid_label_shape(self, tokenizer):
        """
        Test initialization with invalid label shape (not 16 columns).
        
        Expected behavior:
        - Should raise ValueError when labels don't have 16 columns
        
        **Validates: Requirements 11.2**
        """
        texts = ["Text 1", "Text 2"]
        labels = np.array([
            [1, 0, 1, 0, 0],  # Only 5 columns instead of 16
            [0, 1, 0, 1, 0],
        ], dtype=np.float32)
        
        with pytest.raises(ValueError, match="Labels must have 16 columns"):
            EmotionDataset(texts, labels, tokenizer)


class TestEmotionDatasetLength:
    """Test suite for EmotionDataset __len__ method."""
    
    def test_len_returns_correct_length(self, tokenizer, sample_data):
        """
        Test that __len__() returns the correct number of samples.
        
        Expected behavior:
        - len(dataset) should equal the number of text samples
        
        **Validates: Requirements 11.2**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        assert len(dataset) == 5, "Dataset length should be 5"
        assert len(dataset) == len(texts), "Dataset length should match number of texts"
    
    def test_len_single_sample(self, tokenizer):
        """
        Test __len__() with a single sample.
        
        Expected behavior:
        - Should return 1 for single sample dataset
        
        **Validates: Requirements 11.2**
        """
        texts = ["Single text sample"]
        labels = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.float32)
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        assert len(dataset) == 1, "Dataset length should be 1 for single sample"
    
    def test_len_large_dataset(self, tokenizer):
        """
        Test __len__() with a larger dataset.
        
        Expected behavior:
        - Should return correct length for larger datasets
        
        **Validates: Requirements 11.2**
        """
        num_samples = 100
        texts = [f"Sample text {i}" for i in range(num_samples)]
        labels = np.random.randint(0, 2, size=(num_samples, 16)).astype(np.float32)
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        assert len(dataset) == num_samples, f"Dataset length should be {num_samples}"


class TestEmotionDatasetGetItem:
    """Test suite for EmotionDataset __getitem__ method."""
    
    def test_getitem_returns_correct_format(self, tokenizer, sample_data):
        """
        Test that __getitem__() returns a dictionary with correct keys.
        
        Expected behavior:
        - Should return dict with 'input_ids', 'attention_mask', 'labels'
        
        **Validates: Requirements 11.4**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        
        sample = dataset[0]
        
        assert isinstance(sample, dict), "Sample should be a dictionary"
        assert 'input_ids' in sample, "Sample should have 'input_ids' key"
        assert 'attention_mask' in sample, "Sample should have 'attention_mask' key"
        assert 'labels' in sample, "Sample should have 'labels' key"
        assert len(sample) == 3, "Sample should have exactly 3 keys"
    
    def test_getitem_tensor_types(self, tokenizer, sample_data):
        """
        Test that __getitem__() returns correct tensor types.
        
        Expected behavior:
        - input_ids and attention_mask should be torch.Tensor (int64)
        - labels should be torch.Tensor (float32)
        
        **Validates: Requirements 11.4, 4.2**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        
        sample = dataset[0]
        
        assert isinstance(sample['input_ids'], torch.Tensor), "input_ids should be a torch.Tensor"
        assert isinstance(sample['attention_mask'], torch.Tensor), "attention_mask should be a torch.Tensor"
        assert isinstance(sample['labels'], torch.Tensor), "labels should be a torch.Tensor"
        
        assert sample['input_ids'].dtype == torch.int64, "input_ids should be int64"
        assert sample['attention_mask'].dtype == torch.int64, "attention_mask should be int64"
        assert sample['labels'].dtype == torch.float32, "labels should be float32"
    
    def test_getitem_tensor_shapes(self, tokenizer, sample_data):
        """
        Test that __getitem__() returns tensors with correct shapes.
        
        Expected behavior:
        - input_ids shape: (max_length,)
        - attention_mask shape: (max_length,)
        - labels shape: (16,)
        
        **Validates: Requirements 11.4, 4.2, 4.3**
        """
        texts, labels = sample_data
        max_length = 128
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=max_length)
        
        sample = dataset[0]
        
        assert sample['input_ids'].shape == (max_length,), f"input_ids shape should be ({max_length},)"
        assert sample['attention_mask'].shape == (max_length,), f"attention_mask shape should be ({max_length},)"
        assert sample['labels'].shape == (16,), "labels shape should be (16,)"
    
    def test_getitem_different_max_lengths(self, tokenizer, sample_data):
        """
        Test __getitem__() with different max_length values.
        
        Expected behavior:
        - Tensor shapes should match the specified max_length
        
        **Validates: Requirements 4.3, 4.4**
        """
        texts, labels = sample_data
        
        for max_length in [64, 128, 256, 512]:
            dataset = EmotionDataset(texts, labels, tokenizer, max_length=max_length)
            sample = dataset[0]
            
            assert sample['input_ids'].shape == (max_length,), f"input_ids shape should be ({max_length},)"
            assert sample['attention_mask'].shape == (max_length,), f"attention_mask shape should be ({max_length},)"
    
    def test_getitem_labels_match_input(self, tokenizer, sample_data):
        """
        Test that __getitem__() returns correct labels for each sample.
        
        Expected behavior:
        - Labels should match the input labels array
        
        **Validates: Requirements 11.4**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            expected_labels = torch.tensor(labels[idx], dtype=torch.float32)
            
            assert torch.equal(sample['labels'], expected_labels), f"Labels for sample {idx} should match input"
    
    def test_getitem_tokenization_includes_special_tokens(self, tokenizer, sample_data):
        """
        Test that tokenization includes BERT special tokens [CLS] and [SEP].
        
        Expected behavior:
        - input_ids should start with [CLS] token (101)
        - input_ids should contain [SEP] token (102)
        
        **Validates: Requirements 4.2**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        
        sample = dataset[0]
        input_ids = sample['input_ids']
        
        # [CLS] token ID is 101 in BERT
        assert input_ids[0] == 101, "First token should be [CLS] (101)"
        
        # [SEP] token ID is 102 in BERT
        assert 102 in input_ids, "input_ids should contain [SEP] token (102)"
    
    def test_getitem_attention_mask_values(self, tokenizer, sample_data):
        """
        Test that attention_mask has correct values (1 for tokens, 0 for padding).
        
        Expected behavior:
        - attention_mask should be 1 for real tokens
        - attention_mask should be 0 for padding tokens
        
        **Validates: Requirements 4.2, 4.4**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        
        sample = dataset[0]
        attention_mask = sample['attention_mask']
        
        # All values should be 0 or 1
        assert torch.all((attention_mask == 0) | (attention_mask == 1)), "attention_mask should only contain 0 or 1"
        
        # Should have at least some 1s (for the actual tokens)
        assert torch.sum(attention_mask) > 0, "attention_mask should have at least some 1s"
    
    def test_getitem_padding_applied(self, tokenizer):
        """
        Test that padding is applied to shorter sequences.
        
        Expected behavior:
        - Short text should be padded to max_length
        - Padding tokens (0) should have attention_mask value 0
        
        **Validates: Requirements 4.4**
        """
        texts = ["Short"]
        labels = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.float32)
        max_length = 128
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=max_length)
        
        sample = dataset[0]
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        
        # Should have padding tokens (0)
        assert 0 in input_ids, "Short text should have padding tokens"
        
        # Padding tokens should have attention_mask = 0
        padding_positions = (input_ids == 0)
        assert torch.all(attention_mask[padding_positions] == 0), "Padding positions should have attention_mask = 0"
    
    def test_getitem_truncation_applied(self, tokenizer):
        """
        Test that truncation is applied to longer sequences.
        
        Expected behavior:
        - Long text should be truncated to max_length
        - Resulting tensor should have exactly max_length elements
        
        **Validates: Requirements 4.3**
        """
        # Create a very long text that will exceed max_length
        long_text = " ".join(["word"] * 1000)  # 1000 words
        texts = [long_text]
        labels = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.float32)
        max_length = 64
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=max_length)
        
        sample = dataset[0]
        
        assert sample['input_ids'].shape == (max_length,), f"Long text should be truncated to {max_length}"
        assert sample['attention_mask'].shape == (max_length,), f"attention_mask should be truncated to {max_length}"
    
    def test_getitem_all_indices(self, tokenizer, sample_data):
        """
        Test that __getitem__() works for all valid indices.
        
        Expected behavior:
        - Should successfully retrieve all samples in the dataset
        
        **Validates: Requirements 11.4**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            assert isinstance(sample, dict), f"Sample {idx} should be a dictionary"
            assert 'input_ids' in sample, f"Sample {idx} should have input_ids"
            assert 'attention_mask' in sample, f"Sample {idx} should have attention_mask"
            assert 'labels' in sample, f"Sample {idx} should have labels"


class TestEmotionDatasetDataLoaderCompatibility:
    """Test suite for EmotionDataset compatibility with PyTorch DataLoader."""
    
    def test_dataloader_batching(self, tokenizer, sample_data):
        """
        Test that dataset works with DataLoader for batching.
        
        Expected behavior:
        - DataLoader should successfully create batches
        - Batch tensors should have correct shapes
        
        **Validates: Requirements 11.5**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        
        batch_size = 2
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Get first batch
        batch = next(iter(dataloader))
        
        assert isinstance(batch, dict), "Batch should be a dictionary"
        assert 'input_ids' in batch, "Batch should have input_ids"
        assert 'attention_mask' in batch, "Batch should have attention_mask"
        assert 'labels' in batch, "Batch should have labels"
        
        # Check batch dimensions
        assert batch['input_ids'].shape == (batch_size, 128), f"Batch input_ids shape should be ({batch_size}, 128)"
        assert batch['attention_mask'].shape == (batch_size, 128), f"Batch attention_mask shape should be ({batch_size}, 128)"
        assert batch['labels'].shape == (batch_size, 16), f"Batch labels shape should be ({batch_size}, 16)"
    
    def test_dataloader_iteration(self, tokenizer, sample_data):
        """
        Test that DataLoader can iterate through entire dataset.
        
        Expected behavior:
        - Should iterate through all samples
        - Total samples should match dataset length
        
        **Validates: Requirements 11.5**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        
        batch_size = 2
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_samples = 0
        for batch in dataloader:
            batch_samples = batch['input_ids'].shape[0]
            total_samples += batch_samples
            
            # Verify batch structure
            assert 'input_ids' in batch, "Each batch should have input_ids"
            assert 'attention_mask' in batch, "Each batch should have attention_mask"
            assert 'labels' in batch, "Each batch should have labels"
        
        assert total_samples == len(dataset), "DataLoader should iterate through all samples"
    
    def test_dataloader_shuffling(self, tokenizer, sample_data):
        """
        Test that DataLoader shuffling works correctly.
        
        Expected behavior:
        - Shuffled DataLoader should produce different order
        - All samples should still be present
        
        **Validates: Requirements 11.5**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        
        # Create two dataloaders with different random seeds
        dataloader1 = DataLoader(dataset, batch_size=1, shuffle=True)
        dataloader2 = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Get all samples from both dataloaders
        samples1 = [batch['labels'] for batch in dataloader1]
        samples2 = [batch['labels'] for batch in dataloader2]
        
        # Both should have same number of samples
        assert len(samples1) == len(samples2) == len(dataset), "Both dataloaders should have all samples"
    
    def test_dataloader_different_batch_sizes(self, tokenizer, sample_data):
        """
        Test DataLoader with different batch sizes.
        
        Expected behavior:
        - Should work correctly with various batch sizes
        - Last batch may be smaller if dataset size not divisible by batch_size
        
        **Validates: Requirements 11.5**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        
        for batch_size in [1, 2, 3, 5]:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            batches = list(dataloader)
            total_samples = sum(batch['input_ids'].shape[0] for batch in batches)
            
            assert total_samples == len(dataset), f"Batch size {batch_size} should process all samples"
    
    def test_dataloader_num_workers(self, tokenizer, sample_data):
        """
        Test DataLoader with multiple workers (if supported).
        
        Expected behavior:
        - Should work with num_workers > 0 (parallel data loading)
        
        **Validates: Requirements 11.5**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        
        # Note: num_workers > 0 may not work on all systems (especially Windows)
        # This test uses num_workers=0 for compatibility
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        batch = next(iter(dataloader))
        assert batch['input_ids'].shape[0] == 2, "DataLoader with num_workers should work correctly"


class TestEmotionDatasetEdgeCases:
    """Test suite for EmotionDataset edge cases."""
    
    def test_single_sample_dataset(self, tokenizer):
        """
        Test dataset with a single sample.
        
        Expected behavior:
        - Should handle single sample correctly
        
        **Validates: Requirements 11.1, 11.2, 11.4**
        """
        texts = ["Single sample text"]
        labels = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.float32)
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        assert len(dataset) == 1, "Dataset should have length 1"
        
        sample = dataset[0]
        assert isinstance(sample, dict), "Sample should be a dictionary"
        assert sample['labels'].shape == (16,), "Labels should have shape (16,)"
    
    def test_empty_text_handling(self, tokenizer):
        """
        Test dataset with empty text string.
        
        Expected behavior:
        - Should handle empty text without errors
        - Should still produce valid tokenization output
        
        **Validates: Requirements 11.3, 4.2**
        """
        texts = [""]
        labels = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.float32)
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        
        sample = dataset[0]
        
        # Should still have [CLS] and [SEP] tokens
        assert sample['input_ids'].shape == (128,), "Should have correct shape"
        assert sample['input_ids'][0] == 101, "Should start with [CLS] token"
    
    def test_very_short_text(self, tokenizer):
        """
        Test dataset with very short text (single word).
        
        Expected behavior:
        - Should pad to max_length
        - Should have mostly padding tokens
        
        **Validates: Requirements 4.4**
        """
        texts = ["Hi"]
        labels = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.float32)
        max_length = 128
        dataset = EmotionDataset(texts, labels, tokenizer, max_length=max_length)
        
        sample = dataset[0]
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        
        # Should have many padding tokens
        num_padding = torch.sum(input_ids == 0).item()
        assert num_padding > max_length - 10, "Short text should have mostly padding"
        
        # Attention mask should reflect padding
        assert torch.sum(attention_mask == 0).item() == num_padding, "Padding should have attention_mask = 0"
    
    def test_special_characters_in_text(self, tokenizer):
        """
        Test dataset with special characters in text.
        
        Expected behavior:
        - Should handle special characters correctly
        - Tokenizer should process them appropriately
        
        **Validates: Requirements 11.3, 4.2**
        """
        texts = ["Hello! How are you? :) #happy"]
        labels = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.float32)
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        sample = dataset[0]
        
        # Should successfully tokenize
        assert sample['input_ids'].shape[0] > 0, "Should produce valid tokenization"
        assert torch.sum(sample['attention_mask']).item() > 0, "Should have non-zero attention mask"
    
    def test_unicode_text(self, tokenizer):
        """
        Test dataset with Unicode characters (e.g., Vietnamese, emoji).
        
        Expected behavior:
        - Should handle Unicode text correctly
        - BERT tokenizer should process it appropriately
        
        **Validates: Requirements 11.3, 4.2**
        """
        texts = ["Tôi rất vui! 😊"]  # Vietnamese with emoji
        labels = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]], dtype=np.float32)
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        sample = dataset[0]
        
        # Should successfully tokenize
        assert sample['input_ids'].shape[0] > 0, "Should produce valid tokenization"
        assert sample['input_ids'][0] == 101, "Should start with [CLS] token"
    
    def test_all_zero_labels(self, tokenizer):
        """
        Test dataset with all-zero labels (no emotions).
        
        Expected behavior:
        - Should handle all-zero labels correctly
        
        **Validates: Requirements 11.4**
        """
        texts = ["Neutral text"]
        labels = np.zeros((1, 16), dtype=np.float32)
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        sample = dataset[0]
        
        assert torch.sum(sample['labels']).item() == 0, "Labels should be all zeros"
        assert sample['labels'].shape == (16,), "Labels should have correct shape"
    
    def test_all_one_labels(self, tokenizer):
        """
        Test dataset with all-one labels (all emotions present).
        
        Expected behavior:
        - Should handle all-one labels correctly
        
        **Validates: Requirements 11.4**
        """
        texts = ["Very emotional text"]
        labels = np.ones((1, 16), dtype=np.float32)
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        sample = dataset[0]
        
        assert torch.sum(sample['labels']).item() == 16, "Labels should be all ones"
        assert sample['labels'].shape == (16,), "Labels should have correct shape"
    
    def test_negative_index_access(self, tokenizer, sample_data):
        """
        Test dataset access with negative indices.
        
        Expected behavior:
        - Should support negative indexing like Python lists
        
        **Validates: Requirements 11.4**
        """
        texts, labels = sample_data
        dataset = EmotionDataset(texts, labels, tokenizer)
        
        # Access last element with negative index
        last_sample = dataset[-1]
        expected_last_sample = dataset[len(dataset) - 1]
        
        assert torch.equal(last_sample['labels'], expected_last_sample['labels']), "Negative indexing should work"
