"""
Training script for PhoBERT-based emotion classifier

Usage:
    python train_phobert.py --model_type phobert
    python train_phobert.py --model_type hybrid
    python train_phobert.py --model_type phobert --epochs 10 --batch_size 16
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from config import Config
from dataset import EmotionDataset
from model_phobert import PhoBERTEmotionClassifier, HybridEmotionClassifier
from model_registry import ModelRegistry


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PhoBERT emotion classifier')
    
    parser.add_argument('--model_type', type=str, default='phobert',
                       choices=['phobert', 'hybrid'],
                       help='Model architecture: phobert or hybrid')
    parser.add_argument('--data_file', type=str, default='data/member_an.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--lstm_hidden_size', type=int, default=256,
                       help='Hidden size for LSTM layer')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--experiment_name', type=str, default='PhoBERT Training',
                       help='Name of the experiment')
    parser.add_argument('--person', type=str, default='team',
                       help='Person running the experiment')
    
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.sigmoid(logits).cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    
    # Hamming loss
    pred_binary = (all_predictions > 0.5).astype(int)
    hamming_loss = np.mean(pred_binary != all_labels)
    
    # F1 scores
    from sklearn.metrics import f1_score
    micro_f1 = f1_score(all_labels, pred_binary, average='micro', zero_division=0)
    macro_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)
    
    return {
        'loss': avg_loss,
        'hamming_loss': hamming_loss,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Training curves saved to {save_path}")


def main():
    """Main training function"""
    args = parse_args()
    
    print("=" * 60)
    print("🚀 PhoBERT Emotion Classifier Training")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Data File: {args.data_file}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"LSTM Hidden Size: {args.lstm_hidden_size}")
    print(f"Max Length: {args.max_length}")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("\n📥 Loading PhoBERT tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        print("✅ PhoBERT tokenizer loaded")
    except Exception as e:
        print(f"⚠️ Failed to load PhoBERT tokenizer: {e}")
        print("Falling back to multilingual BERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Load datasets
    print(f"\n📊 Loading data from {args.data_file}...")
    train_dataset = EmotionDataset(
        args.data_file,
        tokenizer,
        max_length=args.max_length,
        split='train'
    )
    val_dataset = EmotionDataset(
        args.data_file,
        tokenizer,
        max_length=args.max_length,
        split='val'
    )
    test_dataset = EmotionDataset(
        args.data_file,
        tokenizer,
        max_length=args.max_length,
        split='test'
    )
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    print(f"✅ Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    print(f"\n🏗️ Initializing {args.model_type} model...")
    if args.model_type == 'phobert':
        model = PhoBERTEmotionClassifier(
            num_labels=len(Config.EMOTION_LABELS),
            dropout_rate=args.dropout_rate,
            lstm_hidden_size=args.lstm_hidden_size
        )
    else:  # hybrid
        model = HybridEmotionClassifier(
            num_labels=len(Config.EMOTION_LABELS),
            dropout_rate=args.dropout_rate,
            lstm_hidden_size=args.lstm_hidden_size
        )
    
    model.to(device)
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("\n🎯 Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_metrics['loss'])
        
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Hamming Loss: {val_metrics['hamming_loss']:.4f}")
        print(f"  Micro F1: {val_metrics['micro_f1']:.4f}")
        print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch + 1
            print(f"  ⭐ New best model!")
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("📊 Final Evaluation on Test Set")
    print("="*60)
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    print(f"Micro F1: {test_metrics['micro_f1']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"phobert_{args.model_type}_{timestamp}"
    save_dir = f"experiments/{model_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), f"{save_dir}/pytorch_model.bin")
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    # Save training curves
    plot_training_curves(train_losses, val_losses, f"{save_dir}/training_curves.png")
    
    # Save config
    import json
    config = {
        'model_type': args.model_type,
        'experiment_name': args.experiment_name,
        'model_name': 'vinai/phobert-base',
        'num_labels': len(Config.EMOTION_LABELS),
        'dropout_rate': args.dropout_rate,
        'lstm_hidden_size': args.lstm_hidden_size,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'max_length': args.max_length,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_loss': test_metrics['loss'],
        'hamming_loss': test_metrics['hamming_loss'],
        'micro_f1': test_metrics['micro_f1'],
        'macro_f1': test_metrics['macro_f1']
    }
    
    with open(f"{save_dir}/training_config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save results
    with open(f"{save_dir}/results.txt", 'w', encoding='utf-8') as f:
        f.write(f"PhoBERT Emotion Classification Results\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Person: {args.person}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Learning Rate: {args.learning_rate}\n")
        f.write(f"  LSTM Hidden Size: {args.lstm_hidden_size}\n")
        f.write(f"  Dropout Rate: {args.dropout_rate}\n")
        f.write(f"  Max Length: {args.max_length}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Best Epoch: {best_epoch}\n")
        f.write(f"  Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"  Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}\n")
        f.write(f"  Micro F1: {test_metrics['micro_f1']:.4f}\n")
        f.write(f"  Macro F1: {test_metrics['macro_f1']:.4f}\n")
    
    print(f"\n✅ Model saved to {save_dir}")
    
    # Register model
    print("\n📝 Registering model...")
    registry = ModelRegistry()
    registry.register_model(
        model_path=save_dir,
        metrics={
            'test_loss': test_metrics['loss'],
            'hamming_loss': test_metrics['hamming_loss'],
            'micro_f1': test_metrics['micro_f1'],
            'macro_f1': test_metrics['macro_f1']
        },
        metadata={
            'person': args.person,
            'experiment_name': args.experiment_name,
            'model_type': args.model_type,
            'model_name': 'vinai/phobert-base',
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'lstm_hidden_size': args.lstm_hidden_size,
            'dropout_rate': args.dropout_rate,
            'max_length': args.max_length,
            'data_file': args.data_file
        }
    )
    
    print("\n" + "="*60)
    print("✅ Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
