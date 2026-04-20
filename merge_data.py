"""
Merge Data Script - Merge multiple CSV files into master dataset

Usage:
    python merge_data.py
    
    Or specify custom files:
    python merge_data.py --files data/file1.csv data/file2.csv --output data/master.csv
"""

import pandas as pd
import os
import argparse
from config import Config


def validate_dataframe(df, filename):
    """
    Validate dataframe has correct format.
    
    Args:
        df: DataFrame to validate
        filename: Name of file (for error messages)
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if 'text' column exists
    if 'text' not in df.columns:
        print(f"✗ Error in {filename}: Missing 'text' column")
        return False
    
    # Check if all emotion columns exist
    missing_emotions = []
    for emotion in Config.EMOTION_LABELS:
        if emotion not in df.columns:
            missing_emotions.append(emotion)
    
    if missing_emotions:
        print(f"✗ Error in {filename}: Missing emotion columns: {missing_emotions}")
        return False
    
    # Check if emotion columns contain only 0 or 1
    for emotion in Config.EMOTION_LABELS:
        unique_values = df[emotion].unique()
        if not all(val in [0, 1] for val in unique_values):
            print(f"✗ Error in {filename}: Column '{emotion}' contains values other than 0 or 1")
            return False
    
    # Check if each row has at least one emotion
    emotion_cols = [col for col in df.columns if col in Config.EMOTION_LABELS]
    row_sums = df[emotion_cols].sum(axis=1)
    
    if (row_sums == 0).any():
        zero_emotion_count = (row_sums == 0).sum()
        print(f"⚠ Warning in {filename}: {zero_emotion_count} rows have no emotions")
    
    return True


def merge_datasets(data_files, output_file, remove_duplicates=True, validate=True):
    """
    Merge multiple CSV files into one master dataset.
    
    Args:
        data_files: List of CSV file paths
        output_file: Output master dataset path
        remove_duplicates: Whether to remove duplicate texts
        validate: Whether to validate data format
        
    Returns:
        DataFrame: Merged dataset
    """
    print("="*70)
    print("MERGING DATASETS")
    print("="*70)
    
    all_data = []
    total_samples = 0
    
    for file in data_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                
                # Validate if requested
                if validate:
                    if not validate_dataframe(df, file):
                        print(f"⚠ Skipping {file} due to validation errors")
                        continue
                
                print(f"✓ Loaded {file}: {len(df)} samples")
                all_data.append(df)
                total_samples += len(df)
                
            except Exception as e:
                print(f"✗ Error loading {file}: {e}")
        else:
            print(f"⚠ Warning: {file} not found, skipping")
    
    if not all_data:
        print("\n✗ No valid data files found!")
        return None
    
    # Merge all dataframes
    print(f"\nMerging {len(all_data)} files...")
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates if requested
    duplicates_removed = 0
    if remove_duplicates:
        original_len = len(master_df)
        master_df = master_df.drop_duplicates(subset=['text'], keep='first')
        duplicates_removed = original_len - len(master_df)
        
        if duplicates_removed > 0:
            print(f"✓ Removed {duplicates_removed} duplicate texts")
    
    # Save master dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    master_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("MERGE COMPLETE")
    print("="*70)
    print(f"Total samples loaded: {total_samples}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Final dataset size: {len(master_df)}")
    print(f"Output: {output_file}")
    print("="*70)
    
    # Show statistics
    print(f"\nDataset Statistics:")
    print(f"  Total comments: {len(master_df)}")
    
    # Count samples per emotion
    emotion_cols = [col for col in master_df.columns if col in Config.EMOTION_LABELS]
    print(f"\nSamples per emotion:")
    for emotion in Config.EMOTION_LABELS:
        if emotion in master_df.columns:
            count = master_df[emotion].sum()
            percentage = count / len(master_df) * 100
            print(f"  {emotion:15s}: {int(count):4d} ({percentage:5.1f}%)")
    
    # Show emotion distribution
    print(f"\nEmotion distribution:")
    emotions_per_sample = master_df[emotion_cols].sum(axis=1)
    print(f"  Min emotions per sample: {emotions_per_sample.min()}")
    print(f"  Max emotions per sample: {emotions_per_sample.max()}")
    print(f"  Avg emotions per sample: {emotions_per_sample.mean():.2f}")
    
    return master_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Merge multiple CSV files into master dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge default files
  python merge_data.py
  
  # Merge specific files
  python merge_data.py --files data/file1.csv data/file2.csv
  
  # Specify output file
  python merge_data.py --output data/custom_master.csv
  
  # Keep duplicates
  python merge_data.py --keep-duplicates
  
  # Skip validation
  python merge_data.py --no-validate
        """
    )
    
    parser.add_argument(
        '--files',
        nargs='+',
        help='List of CSV files to merge (default: auto-detect in data/ folder)'
    )
    parser.add_argument(
        '--output',
        default='data/master_dataset.csv',
        help='Output master dataset path (default: data/master_dataset.csv)'
    )
    parser.add_argument(
        '--keep-duplicates',
        action='store_true',
        help='Keep duplicate texts (default: remove duplicates)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip data validation (default: validate)'
    )
    
    args = parser.parse_args()
    
    # Determine which files to merge
    if args.files:
        data_files = args.files
    else:
        # Auto-detect CSV files in data/ folder
        data_dir = 'data'
        if os.path.exists(data_dir):
            data_files = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith('.csv') and f != 'master_dataset.csv'
            ]
            
            if not data_files:
                print("No CSV files found in data/ folder")
                print("Please specify files with --files argument")
                return
            
            print(f"Auto-detected {len(data_files)} CSV files in data/ folder:")
            for f in data_files:
                print(f"  - {f}")
            print()
        else:
            print("data/ folder not found")
            print("Please specify files with --files argument")
            return
    
    # Merge datasets
    master_df = merge_datasets(
        data_files=data_files,
        output_file=args.output,
        remove_duplicates=not args.keep_duplicates,
        validate=not args.no_validate
    )
    
    if master_df is not None:
        print(f"\n✓ Master dataset created successfully!")
        print(f"✓ You can now train with: python train_with_args.py --data {args.output}")


if __name__ == "__main__":
    main()
