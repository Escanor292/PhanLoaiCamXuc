"""
Incremental Training Script - Train with Transfer Learning

This script makes it easy to train a new model using transfer learning
from the best existing model. It automatically:
1. Loads the best model from registry
2. Merges all CSV files in data/ directory
3. Trains the model with new data
4. Registers the new model

Usage:
    python train_incremental.py
    python train_incremental.py --epochs 10
    python train_incremental.py --no-transfer  # Train from scratch
    python train_incremental.py --base-model model_20260422_124631
"""

import argparse
from config import Config
from train import main


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train emotion classification model with transfer learning"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {Config.NUM_EPOCHS})"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size (default: {Config.BATCH_SIZE})"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"Learning rate (default: {Config.LEARNING_RATE})"
    )
    
    parser.add_argument(
        "--no-transfer",
        action="store_true",
        help="Disable transfer learning and train from scratch"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Specific model ID to use as base (default: best model)"
    )
    
    return parser.parse_args()


def main_with_args():
    """Main function with command line argument support."""
    args = parse_args()
    
    # Update config based on arguments
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
        print(f"Setting epochs to: {args.epochs}")
    
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
        print(f"Setting batch size to: {args.batch_size}")
    
    if args.lr is not None:
        Config.LEARNING_RATE = args.lr
        print(f"Setting learning rate to: {args.lr}")
    
    if args.no_transfer:
        Config.USE_TRANSFER_LEARNING = False
        print("Transfer learning disabled")
    
    if args.base_model is not None:
        Config.BASE_MODEL_ID = args.base_model
        print(f"Using base model: {args.base_model}")
    
    # Run training
    main()


if __name__ == "__main__":
    main_with_args()
