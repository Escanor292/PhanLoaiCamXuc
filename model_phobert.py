"""
Enhanced Model with PhoBERT + BiLSTM + Attention for Vietnamese Emotion Classification

Architecture:
    PhoBERT (vinai/phobert-base)
    → BiLSTM (bidirectional LSTM)
    → Self-Attention Layer
    → Dropout
    → Linear Classification Head
    → Sigmoid

This architecture combines:
- PhoBERT: Pre-trained on Vietnamese corpus (better for Vietnamese text)
- BiLSTM: Captures sequential dependencies in both directions
- Attention: Focuses on important parts of the text
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class AttentionLayer(nn.Module):
    """
    Self-attention mechanism to focus on important parts of the sequence.
    
    This layer computes attention weights for each position in the sequence
    and creates a weighted sum representation.
    """
    
    def __init__(self, hidden_size):
        """
        Initialize attention layer.
        
        Args:
            hidden_size (int): Size of hidden representations
        """
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        """
        Apply attention mechanism.
        
        Args:
            lstm_output (torch.Tensor): Output from BiLSTM
                Shape: (batch_size, sequence_length, hidden_size)
        
        Returns:
            tuple: (context_vector, attention_weights)
                - context_vector: Weighted sum of lstm_output
                  Shape: (batch_size, hidden_size)
                - attention_weights: Attention scores
                  Shape: (batch_size, sequence_length)
        """
        # Calculate attention scores
        # Shape: (batch_size, sequence_length, 1)
        attention_scores = self.attention(lstm_output)
        
        # Apply softmax to get attention weights
        # Shape: (batch_size, sequence_length, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention weights to lstm_output
        # Shape: (batch_size, hidden_size)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context_vector, attention_weights.squeeze(-1)


class PhoBERTEmotionClassifier(nn.Module):
    """
    Enhanced emotion classifier using PhoBERT + BiLSTM + Attention.
    
    This model is specifically designed for Vietnamese text and uses:
    1. PhoBERT: Vietnamese pre-trained BERT model
    2. BiLSTM: Captures sequential patterns in both directions
    3. Attention: Focuses on important words/phrases
    4. Classification head: Maps to emotion labels
    
    Architecture Flow:
        Input Text
        → PhoBERT Tokenizer
        → PhoBERT Encoder (768-dim)
        → BiLSTM (768 → 256*2=512)
        → Self-Attention (512 → 512)
        → Dropout (0.3)
        → Linear (512 → num_labels)
        → Sigmoid
    
    Attributes:
        phobert (AutoModel): PhoBERT base model
        lstm (nn.LSTM): Bidirectional LSTM layer
        attention (AttentionLayer): Self-attention mechanism
        dropout (nn.Dropout): Dropout for regularization
        classifier (nn.Linear): Final classification layer
    """
    
    def __init__(self, num_labels=16, dropout_rate=0.3, lstm_hidden_size=256):
        """
        Initialize PhoBERT-based emotion classifier.
        
        Args:
            num_labels (int): Number of emotion labels. Default: 16
            dropout_rate (float): Dropout rate for regularization. Default: 0.3
            lstm_hidden_size (int): Hidden size for LSTM. Default: 256
                Note: BiLSTM will output lstm_hidden_size * 2
        """
        super(PhoBERTEmotionClassifier, self).__init__()
        
        # Load PhoBERT model
        try:
            self.phobert = AutoModel.from_pretrained('vinai/phobert-base')
            print("✅ PhoBERT model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load PhoBERT: {e}")
            print("Falling back to multilingual BERT...")
            self.phobert = AutoModel.from_pretrained('bert-base-multilingual-cased')
        
        # PhoBERT output size
        self.hidden_size = 768
        
        # BiLSTM layer
        # Input: (batch, seq_len, 768)
        # Output: (batch, seq_len, lstm_hidden_size * 2)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # No dropout for single layer LSTM
        )
        
        # Attention layer
        self.attention = AttentionLayer(lstm_hidden_size * 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        # Input: lstm_hidden_size * 2 (BiLSTM output)
        # Output: num_labels
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        
        # Store config
        self.lstm_hidden_size = lstm_hidden_size
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the enhanced model.
        
        Args:
            input_ids (torch.Tensor): Token IDs from PhoBERT tokenizer
                Shape: (batch_size, sequence_length)
            attention_mask (torch.Tensor): Attention mask
                Shape: (batch_size, sequence_length)
        
        Returns:
            torch.Tensor: Raw logits for each emotion label
                Shape: (batch_size, num_labels)
        """
        # 1. PhoBERT encoding
        # Output shape: (batch_size, sequence_length, 768)
        phobert_output = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get all hidden states (not just [CLS])
        # Shape: (batch_size, sequence_length, 768)
        sequence_output = phobert_output.last_hidden_state
        
        # 2. BiLSTM layer
        # Output shape: (batch_size, sequence_length, lstm_hidden_size * 2)
        lstm_output, (hidden, cell) = self.lstm(sequence_output)
        
        # 3. Attention layer
        # Output shape: (batch_size, lstm_hidden_size * 2)
        context_vector, attention_weights = self.attention(lstm_output)
        
        # 4. Dropout
        context_vector = self.dropout(context_vector)
        
        # 5. Classification
        # Output shape: (batch_size, num_labels)
        logits = self.classifier(context_vector)
        
        return logits
    
    def get_attention_weights(self, input_ids, attention_mask):
        """
        Get attention weights for visualization.
        
        This method is useful for understanding which parts of the text
        the model focuses on when making predictions.
        
        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor): Attention mask
        
        Returns:
            torch.Tensor: Attention weights
                Shape: (batch_size, sequence_length)
        """
        with torch.no_grad():
            # Forward pass through PhoBERT and LSTM
            phobert_output = self.phobert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            sequence_output = phobert_output.last_hidden_state
            lstm_output, _ = self.lstm(sequence_output)
            
            # Get attention weights
            _, attention_weights = self.attention(lstm_output)
            
        return attention_weights


class HybridEmotionClassifier(nn.Module):
    """
    Hybrid model combining PhoBERT [CLS] token and BiLSTM+Attention.
    
    This architecture uses both:
    1. [CLS] token from PhoBERT (global sentence representation)
    2. BiLSTM + Attention (sequential + focused representation)
    
    The two representations are concatenated before classification.
    """
    
    def __init__(self, num_labels=16, dropout_rate=0.3, lstm_hidden_size=256):
        """
        Initialize hybrid classifier.
        
        Args:
            num_labels (int): Number of emotion labels
            dropout_rate (float): Dropout rate
            lstm_hidden_size (int): LSTM hidden size
        """
        super(HybridEmotionClassifier, self).__init__()
        
        # Load PhoBERT
        try:
            self.phobert = AutoModel.from_pretrained('vinai/phobert-base')
            print("✅ PhoBERT model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load PhoBERT: {e}")
            print("Falling back to multilingual BERT...")
            self.phobert = AutoModel.from_pretrained('bert-base-multilingual-cased')
        
        self.hidden_size = 768
        
        # BiLSTM + Attention branch
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = AttentionLayer(lstm_hidden_size * 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        # Input: 768 ([CLS]) + lstm_hidden_size*2 (BiLSTM+Attention)
        combined_size = self.hidden_size + lstm_hidden_size * 2
        self.classifier = nn.Linear(combined_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass combining [CLS] and BiLSTM+Attention.
        
        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor): Attention mask
        
        Returns:
            torch.Tensor: Logits for emotion labels
        """
        # PhoBERT encoding
        phobert_output = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Branch 1: [CLS] token representation
        cls_output = phobert_output.last_hidden_state[:, 0, :]
        
        # Branch 2: BiLSTM + Attention
        sequence_output = phobert_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        context_vector, _ = self.attention(lstm_output)
        
        # Combine both representations
        combined = torch.cat([cls_output, context_vector], dim=1)
        
        # Dropout and classification
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        
        return logits
