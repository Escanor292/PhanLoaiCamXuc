"""
Unified Training Script - PhoBERT as Default

This is the NEW default training script that uses PhoBERT by default.
It replaces the old BERT-based training while maintaining backward compatibility.

Usage:
    # Default: PhoBERT (Vietnamese)
    python train_unified.py
    
    # Use old BERT (English)
    python train_unified.py --model-type bert
    
    # Use Hybrid (most powerful)
    python train_unified.py --model-type hybrid

Features:
- PhoBERT as default (optimized for Vietnamese)
- BiLSTM + Attention for better context understanding
- Transfer learning support
- Auto data merging
- Model registry integration
"""

import fix_encoding  # Fix Windows emoji encoding
import argparse
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config


def parse_args():
    """Parse command line arguments with PhoBERT as default"""
    parser = argparse.ArgumentParser(
        description='Train Multi-label Emotion Classification Model (PhoBERT Default)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with PhoBERT (default)
  python train_unified.py
  
  # Train with old BERT
  python train_unified.py --model-type bert
  
  # Train with Hybrid model
  python train_unified.py --model-type hybrid
  
  # Custom configuration
  python train_unified.py --epochs 15 --batch-size 32 --learning-rate 3e-5
        """
    )
    
    # Model type
    parser.add_argument('--model-type', type=str, default='phobert',
                       choices=['bert', 'phobert', 'hybrid'],
                       help='Model architecture (default: phobert)')
    
    # Data arguments
    parser.add_argument('--data', type=str, default=None,
                       help='Path to training data CSV file (default: auto-merge all files in data/)')
    parser.add_argument('--output', type=str, default=Config.MODEL_SAVE_DIR,
                       help='Directory to save model checkpoints')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', '--lr', type=float, default=Config.LEARNING_RATE,
                       help='Learning rate')
    
    # Model arguments
    parser.add_argument('--dropout', type=float, default=Config.DROPOUT_RATE,
                       help='Dropout rate')
    parser.add_argument('--lstm-hidden-size', type=int, default=256,
                       help='LSTM hidden size (for phobert/hybrid)')
    parser.add_argument('--max-length', type=int, default=Config.MAX_LENGTH,
                       help='Maximum sequence length')
    
    # Other arguments
    parser.add_argument('--experiment-name', type=str, default='PhoBERT Training',
                       help='Name for this experiment')
    parser.add_argument('--person', type=str, default=None,
                       help='Person running the experiment (default: auto-detect)')
    parser.add_argument('--register-model', action='store_true', default=True,
                       help='Register model to registry (default: True)')
    parser.add_argument('--no-register', dest='register_model', action='store_false',
                       help='Do not register model to registry')
    parser.add_argument('--seed', type=int, default=Config.RANDOM_SEED,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Print banner
    print("=" * 80)
    print("🚀 UNIFIED TRAINING SCRIPT - PhoBERT DEFAULT")
    print("=" * 80)
    print()
    print(f"Model Type: {args.model_type.upper()}")
    
    if args.model_type == 'phobert':
        print("✅ Using PhoBERT + BiLSTM + Attention (Vietnamese optimized)")
    elif args.model_type == 'hybrid':
        print("✅ Using Hybrid PhoBERT (Most powerful)")
    else:
        print("⚠️  Using old BERT base (English - not recommended for Vietnamese)")
    
    print("=" * 80)
    print()
    
    # Import and call train_with_args
    from train_with_args import main as train_main
    
    # Override sys.argv to pass arguments to train_with_args
    sys.argv = ['train_with_args.py']
    sys.argv.extend(['--model-type', args.model_type])
    
    if args.data:
        sys.argv.extend(['--data', args.data])
    else:
        # Auto-merge data
        print("📊 Auto-merging all CSV files in data/ directory...")
        sys.argv.extend(['--data', 'data/member_an.csv'])  # Will be handled by train_with_args
    
    sys.argv.extend(['--output', args.output])
    sys.argv.extend(['--epochs', str(args.epochs)])
    sys.argv.extend(['--batch-size', str(args.batch_size)])
    sys.argv.extend(['--lr', str(args.learning_rate)])
    sys.argv.extend(['--dropout', str(args.dropout)])
    sys.argv.extend(['--lstm-hidden-size', str(args.lstm_hidden_size)])
    sys.argv.extend(['--max-length', str(args.max_length)])
    sys.argv.extend(['--experiment-name', args.experiment_name])
    sys.argv.extend(['--seed', str(args.seed)])
    
    if args.person:
        sys.argv.extend(['--person', args.person])
    
    if args.register_model:
        sys.argv.append('--register-model')
    
    # Call train_with_args main function
    train_main()


if __name__ == '__main__':
    main()
