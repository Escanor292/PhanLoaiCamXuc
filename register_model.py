"""
Register the trained model to registry
"""

import fix_encoding  # Fix Windows emoji encoding
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_registry import ModelRegistry

def main():
    print("=" * 80)
    print("📝 REGISTERING MODEL")
    print("=" * 80)
    print()
    
    model_path = 'saved_model/'
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
        print("❌ Model not found at:", model_path)
        return
    
    # Load training config
    config_file = os.path.join(model_path, 'training_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print("✅ Training config loaded")
    else:
        print("⚠️  Training config not found, using defaults")
        config = {}
    
    # Get person name
    person = os.getenv('USER', os.getenv('USERNAME', 'khanh'))
    
    # Prompt for metrics (since we don't have them from the interrupted training)
    print("\n📊 Please provide the final test metrics:")
    print("(If you don't know, use approximate values from validation)")
    print()
    
    try:
        test_loss = float(input("Test Loss (e.g., 0.4353): ") or "0.4353")
        macro_f1 = float(input("Macro F1 (e.g., 0.0475): ") or "0.0475")
        micro_f1 = float(input("Micro F1 (e.g., 0.3333): ") or "0.3333")
        hamming_loss = float(input("Hamming Loss (e.g., 0.1533): ") or "0.1533")
    except ValueError:
        print("❌ Invalid input, using default values")
        test_loss = 0.4353
        macro_f1 = 0.0475
        micro_f1 = 0.3333
        hamming_loss = 0.1533
    
    print()
    print("📝 Registering model with:")
    print(f"   • Test Loss: {test_loss:.4f}")
    print(f"   • Macro F1: {macro_f1:.4f}")
    print(f"   • Micro F1: {micro_f1:.4f}")
    print(f"   • Hamming Loss: {hamming_loss:.4f}")
    print()
    
    # Register model
    try:
        registry = ModelRegistry()
        
        model_id = registry.register_model(
            model_path=model_path,
            metrics={
                'test_loss': test_loss,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1,
                'hamming_loss': hamming_loss
            },
            metadata={
                'person': person,
                'experiment_name': f'Khanh Training - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                'model_type': config.get('model_type', 'hybrid'),
                'learning_rate': config.get('learning_rate', 2e-5),
                'batch_size': config.get('batch_size', 16),
                'epochs': config.get('epochs', 5),
                'lstm_hidden_size': config.get('lstm_hidden_size', 256)
            }
        )
        
        if model_id:
            print(f"✅ Model registered successfully: {model_id}")
            print()
            print("Next steps:")
            print("1. Check results: python model_registry.py list")
            print("2. Push to GitHub:")
            print("   git add data/member_khanh.csv model_registry/")
            print("   git commit -m 'Training results from Khanh: 500 samples'")
            print("   git push")
        else:
            print("⚠️  Model not registered (not better than current best)")
            print("   Current best has lower test loss")
    
    except Exception as e:
        print(f"❌ Failed to register model: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
