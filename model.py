"""
Model module for Multi-label Emotion Classification system.

This module defines the BERTEmotionClassifier class, which implements a BERT-based
neural network for multi-label emotion classification. The model uses a pre-trained
bert-base-uncased model with a custom classification head.
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BERTEmotionClassifier(nn.Module):
    """
    BERT-based multi-label emotion classifier.
    
    This model uses a pre-trained BERT base model followed by a dropout layer
    and a linear classification head to predict 16 emotion labels simultaneously.
    
    Architecture:
        BERT Base (bert-base-uncased)
        → Dropout (configurable rate, default 0.3)
        → Linear (768 → num_labels)
        → Sigmoid (applied during inference, not in forward pass)
    
    The model outputs raw logits (unnormalized scores) during the forward pass.
    Sigmoid activation is applied during inference or implicitly in BCEWithLogitsLoss
    during training for numerical stability.
    
    Attributes:
        bert (BertModel): Pre-trained BERT model from Hugging Face
        dropout (nn.Dropout): Dropout layer for regularization
        classifier (nn.Linear): Linear classification head
    
    Example:
        >>> model = BERTEmotionClassifier(num_labels=16, dropout_rate=0.3)
        >>> input_ids = torch.randint(0, 1000, (8, 128))  # batch_size=8, seq_len=128
        >>> attention_mask = torch.ones(8, 128)
        >>> logits = model(input_ids, attention_mask)
        >>> print(logits.shape)  # torch.Size([8, 16])
    """
    
    def __init__(self, num_labels=16, dropout_rate=0.3):
        """
        Initialize the BERTEmotionClassifier.
        
        Args:
            num_labels (int, optional): Number of emotion labels to predict. 
                Defaults to 16.
            dropout_rate (float, optional): Dropout rate for regularization. 
                Defaults to 0.3.
        
        Raises:
            ConnectionError: If BERT model download fails due to network issues.
            RuntimeError: If BERT model initialization fails.
        """
        super(BERTEmotionClassifier, self).__init__()
        
        # Load pre-trained BERT model with error handling
        try:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        except Exception as e:
            # Check if it's a connection-related error
            if "connection" in str(e).lower() or "network" in str(e).lower():
                raise ConnectionError(
                    f"Failed to download BERT model due to network issues: {str(e)}. "
                    f"Please check your internet connection and try again. "
                    f"If the problem persists, you may need to download the model manually."
                )
            else:
                raise RuntimeError(
                    f"Failed to initialize BERT model: {str(e)}. "
                    f"This may be due to corrupted cache or incompatible transformers version."
                )
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head: 768 (BERT hidden size) → num_labels
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        This method processes input through BERT, extracts the [CLS] token
        representation, applies dropout, and passes through the classification
        head to produce logits.
        
        Args:
            input_ids (torch.Tensor): Token IDs from BERT tokenizer.
                Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Attention mask indicating which tokens
                are padding (0) and which are real tokens (1).
                Shape: (batch_size, sequence_length)
        
        Returns:
            torch.Tensor: Raw logits (unnormalized scores) for each emotion label.
                Shape: (batch_size, num_labels)
                Note: Sigmoid activation is NOT applied here. Apply torch.sigmoid()
                during inference or use BCEWithLogitsLoss during training.
        
        Example:
            >>> model = BERTEmotionClassifier(num_labels=16)
            >>> input_ids = torch.randint(0, 1000, (4, 64))
            >>> attention_mask = torch.ones(4, 64)
            >>> logits = model(input_ids, attention_mask)
            >>> probabilities = torch.sigmoid(logits)  # Convert to probabilities
        """
        # Pass inputs through BERT model
        # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size=768)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token representation (first token)
        # [CLS] token aggregates sequence-level information for classification
        # Shape: (batch_size, 768)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout for regularization
        # Shape: (batch_size, 768)
        cls_output = self.dropout(cls_output)
        
        # Pass through classification head to get logits
        # Shape: (batch_size, num_labels)
        logits = self.classifier(cls_output)
        
        return logits
