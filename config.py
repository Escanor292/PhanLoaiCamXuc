"""
Configuration module for Multi-label Emotion Classification system.

This module contains all hyperparameters, paths, and constants used throughout
the system. Modify values here to experiment with different configurations.
"""

import torch


class Config:
    """
    Centralized configuration class for the emotion classification system.
    
    Attributes:
        MODEL_NAME (str): Pre-trained BERT model identifier
        HIDDEN_SIZE (int): BERT hidden dimension size
        NUM_LABELS (int): Number of emotion labels to predict
        DROPOUT_RATE (float): Dropout rate for regularization
        LEARNING_RATE (float): Learning rate for optimizer
        BATCH_SIZE (int): Batch size for training and evaluation
        NUM_EPOCHS (int): Number of training epochs
        MAX_LENGTH (int): Maximum token sequence length
        PREDICTION_THRESHOLD (float): Confidence threshold for predictions
        RANDOM_SEED (int): Random seed for reproducibility
        DATA_DIR (str): Directory for data files
        MODEL_SAVE_DIR (str): Directory for saved model checkpoints
        TRAIN_SPLIT (float): Proportion of data for training
        VAL_SPLIT (float): Proportion of data for validation
        TEST_SPLIT (float): Proportion of data for testing
        EMOTION_LABELS (list): List of 16 emotion label names
        DEVICE (str): Device for computation (cuda or cpu)
    """
    
    # Model Configuration
    MODEL_NAME = "bert-base-uncased"
    HIDDEN_SIZE = 768  # BERT base hidden dimension
    NUM_LABELS = 16
    DROPOUT_RATE = 0.3
    
    # Training Configuration
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    MAX_LENGTH = 512  # Maximum token sequence length
    
    # Data Configuration
    DATA_DIR = "data/"
    MODEL_SAVE_DIR = "saved_model/"
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Transfer Learning Configuration
    USE_TRANSFER_LEARNING = True  # Load best model from registry
    BASE_MODEL_ID = None  # None = auto-select best model, or specify model_id
    
    # Prediction Configuration
    PREDICTION_THRESHOLD = 0.5
    
    # Emotion Labels (16 emotions)
    EMOTION_LABELS = [
        "joy",
        "trust",
        "fear",
        "surprise",
        "sadness",
        "disgust",
        "anger",
        "anticipation",
        "love",
        "worried",
        "disappointed",
        "proud",
        "embarrassed",
        "jealous",
        "calm",
        "excited"
    ]
    
    # Vietnamese translations for emotion labels
    EMOTION_LABELS_VI = {
        "joy": "vui vẻ",
        "trust": "tin tưởng",
        "fear": "sợ hãi",
        "surprise": "ngạc nhiên",
        "sadness": "buồn bã",
        "disgust": "ghê tởm",
        "anger": "tức giận",
        "anticipation": "mong đợi",
        "love": "yêu thương",
        "worried": "lo lắng",
        "disappointed": "thất vọng",
        "proud": "tự hào",
        "embarrassed": "xấu hổ",
        "jealous": "ghen tị",
        "calm": "bình tĩnh",
        "excited": "phấn khích"
    }
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
