"""
Transfer Learning Module

This module handles loading pre-trained models from the registry
and fine-tuning them with new data.

Benefits:
- Faster training (model already knows basics)
- Better performance (keeps old knowledge)
- Needs less data (builds on existing knowledge)
"""

import fix_encoding  # Fix Windows emoji encoding
import os
import torch
from transformers import AutoTokenizer

from config import Config
from model_phobert import PhoBERTEmotionClassifier, HybridEmotionClassifier
from model import BERTEmotionClassifier
from model_registry import ModelRegistry


def load_base_model_for_transfer(model_type='hybrid', device='cpu'):
    """
    Load the best model from registry for transfer learning.
    
    Args:
        model_type (str): Type of model ('bert', 'phobert', 'hybrid')
        device (str): Device to load model on
    
    Returns:
        tuple: (model, tokenizer, base_model_info) or (None, None, None) if no base model
    """
    print("\n" + "="*80)
    print("🔍 TRANSFER LEARNING: Looking for base model...")
    print("="*80)
    
    # Get model registry
    registry = ModelRegistry()
    
    # Try to find best model of same type
    best_model = None
    all_models = registry.registry.get('models', [])
    
    # Filter by model type
    if all_models:
        for model_info in all_models:
            metadata = model_info.get('metadata', {})
            if metadata.get('model_type') == model_type:
                if best_model is None or model_info['metrics']['test_loss'] < best_model['metrics']['test_loss']:
                    best_model = model_info
    
    # If no model of same type, get overall best
    if best_model is None:
        best_model = registry.get_best_model()
    
    if best_model is None:
        print("📝 No existing models found.")
        print("✅ Will train from scratch (using PhoBERT pre-trained weights)")
        print("="*80)
        return None, None, None
    
    # Show base model info
    print(f"\n🏆 Found Base Model:")
    print(f"   • Model ID: {best_model['model_id']}")
    print(f"   • Model Type: {best_model['metadata'].get('model_type', 'bert')}")
    print(f"   • Test Loss: {best_model['metrics']['test_loss']:.4f}")
    print(f"   • Macro F1: {best_model['metrics']['macro_f1']:.4f}")
    print(f"   • Person: {best_model['metadata'].get('person', 'Unknown')}")
    print(f"   • Date: {best_model.get('registered_at', 'Unknown')}")
    
    model_path = best_model['path']
    
    # Check if model files exist
    model_file = os.path.join(model_path, 'pytorch_model.bin')
    if not os.path.exists(model_file):
        print(f"\n⚠️  Model files not found at: {model_path}")
        print("✅ Will train from scratch instead")
        print("="*80)
        return None, None, None
    
    try:
        print(f"\n📥 Loading model from: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("   ✅ Tokenizer loaded")
        
        # Determine model architecture from metadata
        base_model_type = best_model['metadata'].get('model_type', 'bert')
        
        # Initialize model with same architecture
        if base_model_type == 'bert':
            model = BERTEmotionClassifier(
                num_labels=len(Config.EMOTION_LABELS),
                dropout_rate=Config.DROPOUT_RATE
            )
        elif base_model_type == 'phobert':
            lstm_hidden = best_model['metadata'].get('lstm_hidden_size', 256)
            model = PhoBERTEmotionClassifier(
                num_labels=len(Config.EMOTION_LABELS),
                dropout_rate=Config.DROPOUT_RATE,
                lstm_hidden_size=lstm_hidden
            )
        else:  # hybrid
            lstm_hidden = best_model['metadata'].get('lstm_hidden_size', 256)
            model = HybridEmotionClassifier(
                num_labels=len(Config.EMOTION_LABELS),
                dropout_rate=Config.DROPOUT_RATE,
                lstm_hidden_size=lstm_hidden
            )
        
        # Load weights
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print("   ✅ Model weights loaded")
        
        print("\n🎯 Transfer Learning ENABLED!")
        print("   • Model will start with existing knowledge")
        print("   • Training will be faster")
        print("   • Performance will be better")
        print("="*80)
        
        return model, tokenizer, best_model
        
    except Exception as e:
        print(f"\n⚠️  Failed to load base model: {e}")
        print("✅ Will train from scratch instead")
        print("="*80)
        return None, None, None


def should_use_transfer_learning():
    """
    Check if transfer learning should be used.
    
    Returns:
        bool: True if transfer learning should be used
    """
    # Check config
    if hasattr(Config, 'USE_TRANSFER_LEARNING'):
        return Config.USE_TRANSFER_LEARNING
    
    # Default: use transfer learning
    return True


def get_transfer_learning_settings(base_model_info):
    """
    Get optimal training settings for transfer learning.
    
    Args:
        base_model_info (dict): Information about base model
    
    Returns:
        dict: Training settings
    """
    if base_model_info is None:
        # Training from scratch - use default settings
        return {
            'epochs': 10,
            'learning_rate': 2e-5,
            'description': 'Training from scratch'
        }
    
    # Transfer learning - use lower learning rate and fewer epochs
    return {
        'epochs': 5,  # Fewer epochs needed
        'learning_rate': 1e-5,  # Lower learning rate to preserve knowledge
        'description': 'Fine-tuning with transfer learning'
    }


def print_transfer_learning_info():
    """Print information about transfer learning."""
    print("\n" + "="*80)
    print("📚 ABOUT TRANSFER LEARNING")
    print("="*80)
    print()
    print("Transfer Learning = Model học từ model cũ")
    print()
    print("Ví dụ:")
    print("  Model cũ: Đã học 1000 câu")
    print("  Model mới: Học thêm 100 câu MỚI")
    print("  → Model mới biết: 1000 + 100 = 1100 câu! ✅")
    print()
    print("Lợi ích:")
    print("  ✅ Training nhanh hơn (3-5 epochs thay vì 10-15)")
    print("  ✅ Performance tốt hơn (giữ kiến thức cũ)")
    print("  ✅ Cần ít data hơn (chỉ cần data mới)")
    print()
    print("So sánh:")
    print("  Training từ đầu:  PhoBERT → Train 10 epochs → Model mới")
    print("  Transfer Learning: Model cũ → Fine-tune 5 epochs → Model mới ⭐")
    print("="*80)


if __name__ == '__main__':
    # Test transfer learning
    print_transfer_learning_info()
    
    # Try to load base model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, base_info = load_base_model_for_transfer('hybrid', device)
    
    if model is not None:
        print("\n✅ Transfer learning ready!")
        print(f"   Base model: {base_info['model_id']}")
        print(f"   Model type: {base_info['metadata'].get('model_type', 'unknown')}")
    else:
        print("\n📝 No base model found - will train from scratch")
