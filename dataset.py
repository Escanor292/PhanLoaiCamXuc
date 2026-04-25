"""
Dataset module for Multi-label Emotion Classification system.

This module contains the PyTorch Dataset class for loading and preprocessing
emotion classification data. The EmotionDataset class handles tokenization
on-the-fly during data loading for memory efficiency.
"""

import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    """
    Custom PyTorch Dataset for multi-label emotion classification.
    
    This dataset class handles text tokenization on-the-fly during data loading,
    which is more memory-efficient than pre-tokenizing all data. It's designed
    to work with BERT tokenizers and returns dictionaries compatible with
    PyTorch DataLoader for batching.
    
    Attributes:
        texts (list of str): List of comment texts to classify
        labels (numpy.ndarray): Binary label matrix of shape (N, 16) where N is
                               the number of samples. Each row contains 16 binary
                               values (0 or 1) indicating presence of each emotion.
        tokenizer (BertTokenizer): BERT tokenizer instance for text tokenization
        max_length (int): Maximum sequence length for tokenization. Sequences
                         longer than this will be truncated, shorter ones will
                         be padded. Default is 512 (BERT's maximum).
    
    Examples:
        >>> from transformers import BertTokenizer
        >>> import numpy as np
        >>> 
        >>> texts = ["I love this!", "This is terrible"]
        >>> labels = np.array([[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
        ...                    [0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0]])
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> 
        >>> dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        >>> print(f"Dataset size: {len(dataset)}")
        Dataset size: 2
        >>> 
        >>> sample = dataset[0]
        >>> print(f"Keys: {sample.keys()}")
        Keys: dict_keys(['input_ids', 'attention_mask', 'labels'])
        >>> print(f"Input shape: {sample['input_ids'].shape}")
        Input shape: torch.Size([128])
        >>> print(f"Labels shape: {sample['labels'].shape}")
        Labels shape: torch.Size([16])
    
    Notes:
        - Tokenization is performed on-the-fly in __getitem__() to save memory
        - The tokenizer adds special tokens ([CLS], [SEP]) automatically
        - Padding is applied to max_length for consistent batch sizes
        - Tensors are squeezed to remove the batch dimension added by tokenizer
        - Compatible with torch.utils.data.DataLoader for batching and shuffling
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize the EmotionDataset.
        
        Args:
            texts (list of str): List of comment texts. Each text should be a
                                string containing the comment to classify.
            labels (numpy.ndarray): Binary label matrix of shape (N, 16) where
                                   N equals len(texts). Each row contains 16
                                   binary values (0 or 1) for the 16 emotions.
            tokenizer (BertTokenizer): BERT tokenizer instance from transformers
                                      library. Should be loaded with
                                      BertTokenizer.from_pretrained().
            max_length (int, optional): Maximum sequence length for tokenization.
                                       Sequences longer than this are truncated,
                                       shorter ones are padded. Default is 512
                                       (BERT's maximum input length).
        
        Raises:
            ValueError: If len(texts) != len(labels) or if labels shape is invalid
        
        Examples:
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> texts = ["Happy comment", "Sad comment"]
            >>> labels = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            ...                    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]])
            >>> dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
        """
        # Validate inputs
        if len(texts) != len(labels):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match number of label rows ({len(labels)})"
            )
        
        if labels.shape[1] != 16:
            raise ValueError(
                f"Labels must have 16 columns (one per emotion), got {labels.shape[1]}"
            )
        
        # Store instance variables
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        This method is required by PyTorch's Dataset interface and is used by
        DataLoader to determine the dataset size.
        
        Returns:
            int: Number of samples (comments) in the dataset
        
        Examples:
            >>> dataset = EmotionDataset(texts, labels, tokenizer)
            >>> print(len(dataset))
            100
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single tokenized sample with its labels.
        
        This method is called by DataLoader to retrieve individual samples.
        It performs tokenization on-the-fly, which is more memory-efficient
        than pre-tokenizing all data.
        
        The tokenization process:
        1. Retrieves the text at the given index
        2. Tokenizes using BERT tokenizer with:
           - max_length: Truncates or pads to this length
           - truncation=True: Truncates sequences longer than max_length
           - padding='max_length': Pads shorter sequences to max_length
           - return_tensors='pt': Returns PyTorch tensors
        3. Squeezes tensors to remove batch dimension (from [1, seq_len] to [seq_len])
        4. Retrieves labels and converts to float32 tensor
        
        Args:
            idx (int): Index of the sample to retrieve (0 <= idx < len(dataset))
        
        Returns:
            dict: Dictionary containing:
                - 'input_ids' (torch.Tensor): Token IDs of shape (max_length,)
                                             Values are integers representing tokens
                - 'attention_mask' (torch.Tensor): Attention mask of shape (max_length,)
                                                  Values are 1 for real tokens, 0 for padding
                - 'labels' (torch.Tensor): Emotion labels of shape (16,)
                                          Values are 0 or 1 for each emotion
        
        Examples:
            >>> dataset = EmotionDataset(texts, labels, tokenizer, max_length=128)
            >>> sample = dataset[0]
            >>> 
            >>> print(sample['input_ids'].shape)
            torch.Size([128])
            >>> print(sample['attention_mask'].shape)
            torch.Size([128])
            >>> print(sample['labels'].shape)
            torch.Size([16])
            >>> 
            >>> # Check data types
            >>> print(sample['input_ids'].dtype)
            torch.int64
            >>> print(sample['labels'].dtype)
            torch.float32
        
        Notes:
            - The tokenizer automatically adds [CLS] at the start and [SEP] at the end
            - Padding tokens have ID 0 and attention_mask value 0
            - The squeeze() operation removes the batch dimension added by return_tensors='pt'
        """
        # Get text at index
        text = self.texts[idx]
        
        # Tokenize text
        # The tokenizer returns a dictionary with 'input_ids', 'attention_mask', and 'token_type_ids'
        # return_tensors='pt' returns PyTorch tensors with shape [1, seq_length]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Get labels at index and convert to tensor
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Return dictionary with squeezed tensors
        # squeeze() removes the batch dimension: [1, seq_length] -> [seq_length]
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Shape: (max_length,)
            'attention_mask': encoding['attention_mask'].squeeze(0),  # Shape: (max_length,)
            'labels': label  # Shape: (16,)
        }
