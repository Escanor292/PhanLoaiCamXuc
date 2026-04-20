"""
Training module for Multi-label Emotion Classification system.

This module orchestrates the model training process, including:
- Training loop for one epoch
- Evaluation on validation/test sets
- Main training pipeline with data loading, model initialization, and checkpointing
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import warnings

from config import Config
from model import BERTEmotionClassifier
from dataset import EmotionDataset
from utils import load_data, compute_metrics, plot_training_curves, save_model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    
    This function performs one complete pass through the training data,
    computing loss, performing backpropagation, and updating model weights.
    It includes gradient clipping to prevent exploding gradients and displays
    progress using tqdm.
    
    Args:
        model (BERTEmotionClassifier): Model instance to train
        dataloader (DataLoader): Training data loader providing batches
        optimizer (torch.optim.Optimizer): AdamW optimizer for weight updates
        criterion (nn.Module): BCEWithLogitsLoss for computing loss
        device (torch.device): Device to run training on (cuda or cpu)
    
    Returns:
        float: Average training loss across all batches in the epoch
    
    Example:
        >>> model = BERTEmotionClassifier(num_labels=16)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        >>> criterion = nn.BCEWithLogitsLoss()
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> avg_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        >>> print(f"Training loss: {avg_loss:.4f}")
        Training loss: 0.3245
    
    Notes:
        - Model is set to training mode (dropout enabled)
        - Gradients are clipped to max norm of 1.0 to prevent exploding gradients
        - Progress bar shows current batch, total batches, and running loss
        - Loss is accumulated and averaged across all batches
    """
    # Set model to training mode (enables dropout)
    model.train()
    
    # Initialize loss accumulator
    total_loss = 0.0
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    # Iterate through batches
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute logits
        logits = model(input_ids, attention_mask)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        # Max norm of 1.0 is standard for BERT fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Update progress bar with current loss
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate average loss across all batches
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on validation or test data.
    
    This function performs inference on the provided dataset without computing
    gradients (no backpropagation). It computes the average loss and collects
    all predictions and true labels for metric calculation.
    
    Args:
        model (BERTEmotionClassifier): Model instance to evaluate
        dataloader (DataLoader): Validation or test data loader
        criterion (nn.Module): BCEWithLogitsLoss for computing loss
        device (torch.device): Device to run evaluation on (cuda or cpu)
    
    Returns:
        tuple: (average_loss, predictions, true_labels) where:
            - average_loss (float): Mean loss across all batches
            - predictions (np.ndarray): Predicted probabilities of shape (N, 16)
                                       where N is the number of samples
            - true_labels (np.ndarray): True binary labels of shape (N, 16)
    
    Example:
        >>> model = BERTEmotionClassifier(num_labels=16)
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> criterion = nn.BCEWithLogitsLoss()
        >>> avg_loss, preds, labels = evaluate(model, val_loader, criterion, device)
        >>> print(f"Validation loss: {avg_loss:.4f}")
        Validation loss: 0.2845
        >>> metrics = compute_metrics(preds, labels)
        >>> print(f"Micro F1: {metrics['micro_f1']:.4f}")
        Micro F1: 0.7234
    
    Notes:
        - Model is set to evaluation mode (dropout disabled)
        - No gradients are computed (torch.no_grad() context)
        - Sigmoid activation is applied to logits to get probabilities
        - All predictions and labels are collected in memory
    """
    # Set model to evaluation mode (disables dropout)
    model.eval()
    
    # Initialize loss accumulator and prediction/label lists
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Create progress bar
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        
        # Iterate through batches
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass: compute logits
            logits = model(input_ids, attention_mask)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(logits)
            
            # Move predictions and labels to CPU and convert to numpy
            all_predictions.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate all predictions and labels
    predictions = np.vstack(all_predictions)
    true_labels = np.vstack(all_labels)
    
    return avg_loss, predictions, true_labels


def main():
    """
    Main training pipeline for emotion classification model.
    
    This function orchestrates the complete training process:
    1. Sets random seeds for reproducibility
    2. Detects and configures device (GPU or CPU)
    3. Loads data from CSV and splits into train/val/test
    4. Creates datasets and dataloaders
    5. Initializes model, optimizer, and loss function
    6. Trains for specified number of epochs
    7. Evaluates on validation set after each epoch
    8. Saves best model checkpoint based on validation loss
    9. Generates and saves training curves plot
    10. Logs all hyperparameters and final metrics
    
    The function uses configuration from Config class and saves outputs
    to the directory specified in Config.MODEL_SAVE_DIR.
    
    Returns:
        None: Outputs are saved to disk (model checkpoint, training curves, logs)
    
    Raises:
        FileNotFoundError: If data file is not found
        ValueError: If data is invalid or empty
        RuntimeError: If training fails or diverges
    
    Example:
        >>> # Configure hyperparameters in config.py, then run:
        >>> main()
        Setting random seeds for reproducibility...
        Device: cuda
        Loading data from data/sample_comments.csv...
        Loaded 1000 samples
        Splitting data: 700 train, 150 val, 150 test
        Initializing model...
        Starting training for 5 epochs...
        Epoch 1/5: train_loss=0.4523, val_loss=0.3845
        Epoch 2/5: train_loss=0.3234, val_loss=0.2956
        ...
        Best model saved with validation loss: 0.2456
        Training complete!
    
    Notes:
        - Random seeds are set for Python, NumPy, and PyTorch
        - GPU is used if available, otherwise falls back to CPU with warning
        - Model checkpoint is saved only when validation loss improves
        - Training curves plot is saved to MODEL_SAVE_DIR/training_curves.png
        - All hyperparameters are logged and saved with the model
    """
    print("=" * 70)
    print("Multi-label Emotion Classification - Training Pipeline")
    print("=" * 70)
    
    # 1. Set random seeds for reproducibility
    print("\n[1/10] Setting random seeds for reproducibility...")
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)
    print(f"Random seed: {Config.RANDOM_SEED}")
    
    # 2. Detect and configure device
    print("\n[2/10] Detecting device...")
    device = torch.device(Config.DEVICE)
    if Config.DEVICE == "cuda" and torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    elif Config.DEVICE == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "CUDA requested but not available. Falling back to CPU. "
            "Training will be significantly slower."
        )
        device = torch.device("cpu")
        print(f"Device: {device} (GPU not available)")
    else:
        print(f"Device: {device}")
    
    # 3. Load data from CSV
    print("\n[3/10] Loading data...")
    data_path = os.path.join(Config.DATA_DIR, "sample_comments.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset file not found at '{data_path}'. "
            f"Please ensure the file exists or run generate_sample_data.py first."
        )
    
    texts, labels = load_data(data_path)
    print(f"Loaded {len(texts)} samples")
    print(f"Label shape: {labels.shape}")
    
    # 4. Split data into train/val/test
    print("\n[4/10] Splitting data...")
    n_samples = len(texts)
    n_train = int(n_samples * Config.TRAIN_SPLIT)
    n_val = int(n_samples * Config.VAL_SPLIT)
    n_test = n_samples - n_train - n_val
    
    # Create indices and shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Split texts and labels
    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    test_texts = [texts[i] for i in test_indices]
    
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]
    
    print(f"Train: {len(train_texts)} samples")
    print(f"Validation: {len(val_texts)} samples")
    print(f"Test: {len(test_texts)} samples")
    
    # 5. Create datasets and dataloaders
    print("\n[5/10] Creating datasets and dataloaders...")
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, Config.MAX_LENGTH)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, Config.MAX_LENGTH)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, Config.MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 6. Initialize model, optimizer, and loss function
    print("\n[6/10] Initializing model...")
    model = BERTEmotionClassifier(
        num_labels=Config.NUM_LABELS,
        dropout_rate=Config.DROPOUT_RATE
    )
    model = model.to(device)
    print(f"Model: BERTEmotionClassifier")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE
    )
    print(f"Optimizer: AdamW (lr={Config.LEARNING_RATE})")
    
    # Initialize loss function
    criterion = nn.BCEWithLogitsLoss()
    print(f"Loss function: BCEWithLogitsLoss")
    
    # 7. Training loop
    print("\n[7/10] Starting training...")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print("=" * 70)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 70)
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss, val_predictions, val_labels_array = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Compute validation metrics
        val_metrics = compute_metrics(val_predictions, val_labels_array, threshold=Config.PREDICTION_THRESHOLD)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Micro-F1: {val_metrics['micro_f1']:.4f}")
        print(f"  Val Macro-F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Val Hamming Loss: {val_metrics['hamming_loss']:.4f}")
        
        # 8. Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"\n  ✓ New best validation loss! Saving model...")
            
            # Create save directory if it doesn't exist
            os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
            
            # Prepare training configuration
            training_config = {
                'model_name': Config.MODEL_NAME,
                'num_labels': Config.NUM_LABELS,
                'dropout_rate': Config.DROPOUT_RATE,
                'learning_rate': Config.LEARNING_RATE,
                'batch_size': Config.BATCH_SIZE,
                'num_epochs': Config.NUM_EPOCHS,
                'max_length': Config.MAX_LENGTH,
                'prediction_threshold': Config.PREDICTION_THRESHOLD,
                'random_seed': Config.RANDOM_SEED,
                'best_epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'train_samples': len(train_texts),
                'val_samples': len(val_texts),
                'test_samples': len(test_texts)
            }
            
            # Save model
            save_model(model, tokenizer, Config.MODEL_SAVE_DIR, training_config)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    # 9. Generate and save training curves
    print("\n[8/10] Generating training curves...")
    plot_path = os.path.join(Config.MODEL_SAVE_DIR, "training_curves.png")
    plot_training_curves(train_losses, val_losses, plot_path)
    
    # 10. Final evaluation on test set
    print("\n[9/10] Evaluating on test set...")
    test_loss, test_predictions, test_labels_array = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_predictions, test_labels_array, threshold=Config.PREDICTION_THRESHOLD)
    
    print(f"\nTest Set Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Micro-F1: {test_metrics['micro_f1']:.4f}")
    print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Test Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    
    # Print per-label F1 scores
    print(f"\nPer-Label F1 Scores:")
    for i, emotion in enumerate(Config.EMOTION_LABELS):
        f1 = test_metrics['per_label_f1'][i]
        print(f"  {emotion:15s}: {f1:.4f}")
    
    # 11. Log final summary
    print("\n[10/10] Training summary:")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final test loss: {test_loss:.4f}")
    print(f"  Model saved to: {Config.MODEL_SAVE_DIR}")
    print(f"  Training curves saved to: {plot_path}")
    
    print("\n" + "=" * 70)
    print("All done! You can now use predict.py for inference.")
    print("=" * 70)


if __name__ == "__main__":
    main()
