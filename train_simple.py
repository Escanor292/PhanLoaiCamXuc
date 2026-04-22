"""
Simple Training Script - Auto Transfer Learning for Team Members

This is the SIMPLEST way for team members to train a model.
Just run: python train_simple.py

What it does automatically:
1. Merges ALL CSV files in data/ directory
2. Loads the BEST existing model (Transfer Learning)
3. Trains for 3 epochs (optimal for Transfer Learning)
4. Registers the new model
5. Auto-deploys if it's the best model

No parameters needed - everything is optimized!
"""

import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from train import main
from model_registry import ModelRegistry


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("🚀 SIMPLE TRAINING - AUTO TRANSFER LEARNING")
    print("=" * 80)
    print()
    print("✅ Transfer Learning: ENABLED (learns from best model)")
    print("✅ Auto Data Merge: ENABLED (uses all CSV files)")
    print("✅ Auto Deploy: ENABLED (if model is best)")
    print("✅ Optimized Settings: 3 epochs, LR=2e-5")
    print()
    print("Just sit back and relax! 🎯")
    print("=" * 80)


def check_data_files():
    """Check if there are CSV files to train on."""
    import glob
    
    csv_files = glob.glob(os.path.join(Config.DATA_DIR, "*.csv"))
    csv_files = [f for f in csv_files if "TEMPLATE" not in f.upper()]
    
    if not csv_files:
        print("❌ ERROR: No CSV files found in data/ directory!")
        print()
        print("Please add your data first:")
        print("1. Create data/member_YourName.csv")
        print("2. Follow the format in HUONG_DAN_DONG_GOP_DATA.md")
        print("3. Then run this script again")
        return False
    
    print(f"📊 Found {len(csv_files)} data file(s):")
    for csv_file in sorted(csv_files):
        print(f"   • {os.path.basename(csv_file)}")
    print()
    
    return True


def show_current_best():
    """Show current best model info."""
    try:
        registry = ModelRegistry()
        best_model = registry.get_best_model()
        
        if best_model:
            print("🏆 Current Best Model:")
            print(f"   • Model ID: {best_model['model_id']}")
            print(f"   • Test Loss: {best_model['metrics']['test_loss']:.4f}")
            print(f"   • Macro F1: {best_model['metrics']['macro_f1']:.4f}")
            print(f"   • Person: {best_model['metadata'].get('person', 'Unknown')}")
            print()
            print("🎯 Your new model will learn from this best model!")
        else:
            print("📝 No existing models found. Training from scratch...")
        
        print()
        
    except Exception as e:
        print(f"⚠️  Could not load model registry: {e}")
        print("Training will start from scratch...")
        print()


def main_simple():
    """Main function for simple training."""
    print_banner()
    
    # Check if data files exist
    if not check_data_files():
        return
    
    # Show current best model
    show_current_best()
    
    # Confirm settings
    print("🔧 Training Settings:")
    print(f"   • Transfer Learning: {'✅ ENABLED' if Config.USE_TRANSFER_LEARNING else '❌ DISABLED'}")
    print(f"   • Auto Data Merge: {'✅ ENABLED' if Config.AUTO_MERGE_DATA else '❌ DISABLED'}")
    print(f"   • Epochs: {Config.NUM_EPOCHS}")
    print(f"   • Learning Rate: {Config.LEARNING_RATE}")
    print(f"   • Batch Size: {Config.BATCH_SIZE}")
    print()
    
    # Start training
    print("🚀 Starting training...")
    print("=" * 80)
    
    try:
        # Run main training function
        main()
        
        print("=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("✅ Your model has been trained and registered!")
        print("✅ If it's the best model, it's automatically deployed!")
        print()
        print("Next steps:")
        print("1. Check results: python model_registry.py list")
        print("2. Test model: python my_test.py")
        print("3. Commit results: git add model_registry/ && git commit -m 'Training results' && git push")
        print()
        
    except Exception as e:
        print("=" * 80)
        print("❌ TRAINING FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check if you have enough disk space")
        print("2. Check if your CSV files are valid")
        print("3. Try: python train_incremental.py --no-transfer")
        print()


if __name__ == "__main__":
    main_simple()