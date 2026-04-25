"""
Minimal Training Script - No matplotlib, no data tracker
Direct training for environments with Application Control Policy
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss
from tqdm import tqdm
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model_phobert import HybridEmotionClassifier
from dataset import EmotionDataset
from transformers import AutoTokenizer

def load_data(csv_file):
    """Load data from CSV file."""
    df = pd.read_csv(csv_file)
    texts = df['text'].tolist()
    labels = df[Config.EMOTION_LABELS].values.astype(np.float32)
    return texts, labels

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
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    hamming = hamming_loss(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, macro_f1, micro_f1, hamming

def save_model(model, tokenizer, save_dir, config_dict):
    """Save model and tokenizer."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    # Save config
    import json
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"✅ Model saved to: {save_dir}")

def main():
    print("=" * 80)
    print("🚀 MINIMAL TRAINING - HYBRID PHOBERT")
    print("=" * 80)
    print()
    
    # Settings
    data_file = 'data/member_khanh.csv'
    epochs = 5
    batch_size = 16
    learning_rate = 2e-5
    lstm_hidden_size = 256
    max_length = 256
    
    print(f"✅ Data: {data_file}")
    print(f"✅ Model: Hybrid PhoBERT + BiLSTM + Attention")
    print(f"✅ Epochs: {epochs}")
    print(f"✅ Batch Size: {batch_size}")
    print(f"✅ Learning Rate: {learning_rate}")
    print(f"✅ LSTM Hidden: {lstm_hidden_size}")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    print()
    
    # Load data
    print("📊 Loading data...")
    texts, labels = load_data(data_file)
    print(f"   Total samples: {len(texts)}")
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    print(f"   Train: {len(train_texts)}")
    print(f"   Val: {len(val_texts)}")
    print(f"   Test: {len(test_texts)}")
    print()
    
    # Load tokenizer
    print("🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    print("   ✅ Tokenizer loaded")
    print()
    
    # Create datasets
    print("📦 Creating datasets...")
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("   ✅ Datasets created")
    print()
    
    # Create model
    print("🤖 Creating model...")
    model = HybridEmotionClassifier(
        num_labels=len(Config.EMOTION_LABELS),
        dropout_rate=Config.DROPOUT_RATE,
        lstm_hidden_size=lstm_hidden_size
    )
    model.to(device)
    print("   ✅ Model created")
    print()
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    print("🚀 Starting training...")
    print("=" * 80)
    
    best_val_f1 = 0
    
    for epoch in range(epochs):
        print(f"\n📍 Epoch {epoch + 1}/{epochs}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"   Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_macro_f1, val_micro_f1, val_hamming = evaluate(
            model, val_loader, criterion, device
        )
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val Macro F1: {val_macro_f1:.4f}")
        print(f"   Val Micro F1: {val_micro_f1:.4f}")
        print(f"   Val Hamming Loss: {val_hamming:.4f}")
        
        # Save best model
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            print(f"   ⭐ New best model! Saving...")
            save_model(model, tokenizer, Config.MODEL_SAVE_DIR, {
                'model_type': 'hybrid',
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'lstm_hidden_size': lstm_hidden_size,
                'max_length': max_length
            })
    
    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    
    test_loss, test_macro_f1, test_micro_f1, test_hamming = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n✅ Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Macro F1: {test_macro_f1:.4f}")
    print(f"   Micro F1: {test_micro_f1:.4f}")
    print(f"   Hamming Loss: {test_hamming:.4f}")
    
    # Register model
    print("\n" + "=" * 80)
    print("📝 REGISTERING MODEL")
    print("=" * 80)
    
    try:
        from model_registry import ModelRegistry
        
        registry = ModelRegistry()
        person = os.getenv('USER', os.getenv('USERNAME', 'khanh'))
        
        model_id = registry.register_model(
            model_path=Config.MODEL_SAVE_DIR,
            metrics={
                'test_loss': test_loss,
                'macro_f1': test_macro_f1,
                'micro_f1': test_micro_f1,
                'hamming_loss': test_hamming
            },
            metadata={
                'person': person,
                'experiment_name': f'Khanh Training - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                'model_type': 'hybrid',
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'lstm_hidden_size': lstm_hidden_size
            }
        )
        
        if model_id:
            print(f"✅ Model registered: {model_id}")
        else:
            print("⚠️  Model not registered (not better than current best)")
    
    except Exception as e:
        print(f"⚠️  Failed to register model: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Check results: python model_registry.py list")
    print("2. Push to GitHub: git add model_registry/ && git commit -m 'Training results' && git push")
    print()

if __name__ == "__main__":
    main()
