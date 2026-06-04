"""
Quick Training Script - Bypass DataTracker
Train directly without incremental tracking
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

def main():
    # Determine available data file
    data_file = 'data/member_khanh.csv'
    if not os.path.exists(data_file):
        # Fallback to member_khanh1.csv
        fallback_file = 'data/member_khanh1.csv'
        if os.path.exists(fallback_file):
            data_file = fallback_file
        else:
            import glob
            member_files = glob.glob('data/member_*.csv')
            if member_files:
                data_file = sorted(member_files)[0]
            else:
                data_file = 'data/sample_comments.csv'
                
    print("=" * 80)
    print("🚀 QUICK TRAINING - HYBRID PHOBERT")
    print("=" * 80)
    print()
    print(f"✅ Training on: {data_file}")
    print("✅ Model: Hybrid PhoBERT + BiLSTM + Attention")
    print("✅ Epochs: 5")
    print("✅ Learning Rate: 2e-5")
    print()
    
    # Get person name
    person = os.getenv('USER', os.getenv('USERNAME', 'khanh'))
    
    # Set up arguments
    sys.argv = [
        'train_with_args.py',
        '--model-type', 'hybrid',
        '--data', data_file,
        '--epochs', '5',
        '--batch-size', '16',
        '--lr', '2e-5',
        '--lstm-hidden-size', '256',
        '--dropout', '0.3',
        '--max-length', '256',
        '--experiment-name', f'Khanh Training - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '--register-model'
    ]
    
    print("🚀 Starting training...")
    print("=" * 80)
    print()
    
    # Import and run training
    from train_with_args import main as train_main
    train_main()
    
    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETED!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Check results: python model_registry.py list")
    print("2. Test model: python demo_phobert.py --mode interactive")
    print()

if __name__ == "__main__":
    main()
