"""
Training Script with Command-Line Arguments
Allows multiple people to train with different configurations without modifying config.py

Usage:
    python train_with_args.py --data data/my_data.csv --output saved_model/my_model/ --epochs 10
"""

import fix_encoding  # Fix Windows emoji encoding
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm
from dotenv import load_dotenv

from config import Config
from model import BERTEmotionClassifier
from model_phobert import PhoBERTEmotionClassifier, HybridEmotionClassifier
from dataset import EmotionDataset
from utils import load_data, compute_metrics, plot_training_curves, save_model
from model_registry import ModelRegistry

# Load environment variables from .env file
load_dotenv()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train Multi-label Emotion Classification Model')
    
    # Data arguments
    parser.add_argument('--data', type=str, default='data/sample_comments.csv',
                        help='Path to training data CSV file')
    parser.add_argument('--output', type=str, default=Config.MODEL_SAVE_DIR,
                        help='Directory to save model checkpoints')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='phobert',
                        choices=['bert', 'phobert', 'hybrid'],
                        help='Model architecture: bert (old), phobert (new - default), or hybrid')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Pre-trained model name (auto-selected based on model-type if not specified)')
    parser.add_argument('--dropout', type=float, default=Config.DROPOUT_RATE,
                        help='Dropout rate')
    parser.add_argument('--lstm-hidden-size', type=int, default=256,
                        help='LSTM hidden size (only for phobert/hybrid models)')
    parser.add_argument('--max-length', type=int, default=Config.MAX_LENGTH,
                        help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=Config.RANDOM_SEED,
                        help='Random seed for reproducibility')
    
    # Data split arguments
    parser.add_argument('--train-split', type=float, default=0.7,
                        help='Proportion of data for training')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Proportion of data for validation')
    
    # Other arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--experiment-name', type=str, default='default',
                        help='Name for this experiment (for logging)')
    parser.add_argument('--register-model', action='store_true',
                        help='Register model to central registry after training')
    parser.add_argument('--transfer-from', type=str, default=None,
                        help='Model ID to transfer learn from (enables transfer learning)')
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            
            all_predictions.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    predictions = np.vstack(all_predictions)
    labels = np.vstack(all_labels)
    
    return avg_loss, predictions, labels


def main():
    """Main training function."""
    args = parse_args()
    
    # Auto-select model name based on model type
    if args.model_name is None:
        if args.model_type == 'bert':
            args.model_name = Config.MODEL_NAME  # bert-base-uncased
        elif args.model_type in ['phobert', 'hybrid']:
            args.model_name = 'vinai/phobert-base'
    
    print("="*70)
    print("MULTI-LABEL EMOTION CLASSIFICATION - TRAINING")
    print("="*70)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"\nConfiguration:")
    print(f"  Model Type: {args.model_type.upper()} {'🆕' if args.model_type != 'bert' else ''}")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Base Model: {args.model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Dropout: {args.dropout}")
    if args.model_type in ['phobert', 'hybrid']:
        print(f"  LSTM Hidden Size: {args.lstm_hidden_size}")
    print(f"  Max Length: {args.max_length}")
    print(f"  Random Seed: {args.seed}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"  Device: {device}")
    
    if device.type == 'cpu':
        print("\n⚠ Warning: Training on CPU. This will be slow!")
        print("  Consider using GPU for faster training.")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("="*70)
    
    texts, labels = load_data(args.data)
    print(f"✓ Loaded {len(texts)} samples")
    print(f"✓ Number of emotion labels: {labels.shape[1]}")
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels,
        test_size=(1 - args.train_split),
        random_state=args.seed
    )
    
    val_size = args.val_split / (1 - args.train_split)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=(1 - val_size),
        random_state=args.seed
    )
    
    print(f"\nData Split:")
    print(f"  Training: {len(train_texts)} samples ({len(train_texts)/len(texts)*100:.1f}%)")
    print(f"  Validation: {len(val_texts)} samples ({len(val_texts)/len(texts)*100:.1f}%)")
    print(f"  Test: {len(test_texts)} samples ({len(test_texts)/len(texts)*100:.1f}%)")
    
    # Create datasets and dataloaders
    print(f"\n{'='*70}")
    print("CREATING DATASETS")
    print("="*70)
    
    # Load appropriate tokenizer
    if args.model_type == 'bert':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    else:
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as e:
            print(f"⚠️ Failed to load {args.model_name} tokenizer: {e}")
            print("Falling back to multilingual BERT...")
            tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"✓ Created datasets and dataloaders")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize model
    print(f"\n{'='*70}")
    print("INITIALIZING MODEL")
    print("="*70)
    
    # Check if transfer learning is requested
    transfer_learning = args.transfer_from is not None
    base_model_loaded = False
    
    if transfer_learning:
        print(f"\n🔄 TRANSFER LEARNING MODE")
        print(f"   Loading base model: {args.transfer_from}")
        
        try:
            from transfer_learning import load_base_model_for_transfer
            
            # Try to load base model
            base_model, base_tokenizer, base_info = load_base_model_for_transfer(
                args.model_type, 
                device
            )
            
            if base_model is not None:
                model = base_model
                base_model_loaded = True
                print(f"\n✅ Transfer Learning ENABLED")
                print(f"   Base model: {base_info['model_id']}")
                print(f"   Will fine-tune with new data")
            else:
                print(f"\n⚠️  Could not load base model")
                print(f"   Will train from scratch instead")
                transfer_learning = False
        except Exception as e:
            print(f"\n⚠️  Transfer learning failed: {e}")
            print(f"   Will train from scratch instead")
            transfer_learning = False
    
    # If not transfer learning or failed, initialize new model
    if not base_model_loaded:
        if args.model_type == 'bert':
            print("📦 Loading BERT base model (English)...")
            model = BERTEmotionClassifier(
                num_labels=len(Config.EMOTION_LABELS),
                dropout_rate=args.dropout
            )
        elif args.model_type == 'phobert':
            print("📦 Loading PhoBERT + BiLSTM + Attention model (Vietnamese)...")
            model = PhoBERTEmotionClassifier(
                num_labels=len(Config.EMOTION_LABELS),
                dropout_rate=args.dropout,
                lstm_hidden_size=args.lstm_hidden_size
            )
        else:  # hybrid
            print("📦 Loading Hybrid PhoBERT model (Vietnamese)...")
            model = HybridEmotionClassifier(
                num_labels=len(Config.EMOTION_LABELS),
                dropout_rate=args.dropout,
                lstm_hidden_size=args.lstm_hidden_size
            )
        
        model = model.to(device)
    
    print(f"\n✓ Model initialized")
    print(f"  Architecture: {args.model_type.upper()}")
    if transfer_learning and base_model_loaded:
        print(f"  Mode: Transfer Learning (Fine-tuning) ⭐")
    else:
        print(f"  Mode: Training from scratch")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer and loss
    # Use lower learning rate for transfer learning
    if transfer_learning and base_model_loaded:
        effective_lr = args.lr * 0.5  # Half the learning rate for fine-tuning
        print(f"\n🔧 Transfer Learning: Using lower learning rate ({effective_lr:.2e})")
        optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    if transfer_learning and base_model_loaded:
        print("MODE: TRANSFER LEARNING (Fine-tuning existing model)")
    else:
        print("MODE: FROM SCRATCH (Using PhoBERT pre-trained weights)")
    print("="*70)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_predictions, val_labels_array = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Compute metrics (predictions first, then labels - matching utils.py signature)
        metrics = compute_metrics(val_predictions, val_labels_array, Config.PREDICTION_THRESHOLD)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Micro F1: {metrics['micro_f1']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"\n✓ New best model! Saving checkpoint...")
            
            training_config = {
                'experiment_name': args.experiment_name,
                'model_type': args.model_type,
                'model_name': args.model_name,
                'num_labels': len(Config.EMOTION_LABELS),
                'dropout_rate': args.dropout,
                'lstm_hidden_size': args.lstm_hidden_size if args.model_type != 'bert' else None,
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'num_epochs': args.epochs,
                'max_length': args.max_length,
                'random_seed': args.seed,
                'best_epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'micro_f1': metrics['micro_f1'],
                'macro_f1': metrics['macro_f1']
            }
            
            save_model(model, tokenizer, args.output, training_config)
    
    # Plot training curves
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print("="*70)
    
    plot_path = os.path.join(args.output, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, plot_path)
    print(f"✓ Training curves saved to: {plot_path}")
    
    # Final evaluation on test set
    print(f"\n{'='*70}")
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    test_loss, test_predictions, test_labels_array = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_predictions, test_labels_array, Config.PREDICTION_THRESHOLD)
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Micro F1: {test_metrics['micro_f1']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")
    
    print(f"\nPer-Label F1 Scores:")
    for label, f1 in zip(Config.EMOTION_LABELS, test_metrics['per_label_f1']):
        print(f"  {label:15s}: {f1:.4f}")
    
    # Save final results
    results_file = os.path.join(args.output, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Data: {args.data}\n")
        f.write(f"  Model: {args.model_name}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Dropout: {args.dropout}\n\n")
        f.write(f"Test Results:\n")
        f.write(f"  Test Loss: {test_loss:.4f}\n")
        f.write(f"  Micro F1: {test_metrics['micro_f1']:.4f}\n")
        f.write(f"  Macro F1: {test_metrics['macro_f1']:.4f}\n")
        f.write(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}\n\n")
        f.write(f"Per-Label F1 Scores:\n")
        for label, f1 in zip(Config.EMOTION_LABELS, test_metrics['per_label_f1']):
            f.write(f"  {label:15s}: {f1:.4f}\n")
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Register model to registry if requested
    if args.register_model:
        print(f"\n{'='*70}")
        print("REGISTERING MODEL")
        print("="*70)
        
        registry = ModelRegistry()
        
        metrics = {
            'macro_f1': float(test_metrics['macro_f1']),
            'micro_f1': float(test_metrics['micro_f1']),
            'test_loss': float(test_loss),
            'hamming_loss': float(test_metrics['hamming_loss'])
        }
        
        metadata = {
            'person': os.getenv('USER', os.getenv('USERNAME', 'unknown')),
            'experiment_name': args.experiment_name,
            'model_type': args.model_type,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'dropout_rate': args.dropout,
            'lstm_hidden_size': args.lstm_hidden_size if args.model_type != 'bert' else None,
            'max_length': args.max_length,
            'data_file': args.data,
            'model_name': args.model_name
        }
        
        model_id = registry.register_model(args.output, metrics, metadata)
        
        print(f"\n✓ Model registered with ID: {model_id}")
        print(f"✓ Check registry: python model_registry.py list")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {args.output}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
