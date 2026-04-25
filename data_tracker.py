"""
Data Tracker - Track which data has been trained

This module tracks:
- Which CSV files have been trained
- How many samples from each file
- Which specific samples (by text hash)

This enables incremental training - only train on NEW data!
"""

import fix_encoding  # Fix Windows emoji encoding
import json
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime


class DataTracker:
    """
    Track training data to enable incremental training.
    
    Features:
    - Track which files have been trained
    - Track which samples have been trained (by hash)
    - Detect new data automatically
    - Extract only new samples for training
    """
    
    def __init__(self, tracker_file='model_registry/data_tracker.json'):
        """
        Initialize data tracker.
        
        Args:
            tracker_file: Path to tracker JSON file
        """
        self.tracker_file = Path(tracker_file)
        self.tracker_file.parent.mkdir(exist_ok=True)
        
        self.tracker = self._load_tracker()
    
    def _load_tracker(self):
        """Load tracker from file."""
        if self.tracker_file.exists():
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        return {
            'files': {},  # {filename: {samples: count, hashes: [...]}}
            'total_trained_samples': 0,
            'last_training': None,
            'created_at': datetime.now().isoformat()
        }
    
    def _save_tracker(self):
        """Save tracker to file."""
        with open(self.tracker_file, 'w') as f:
            json.dump(self.tracker, f, indent=2)
    
    def _hash_text(self, text):
        """Create hash of text for tracking."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_new_data(self, csv_files):
        """
        Get only NEW data from CSV files that hasn't been trained yet.
        
        Args:
            csv_files: List of CSV file paths
        
        Returns:
            tuple: (new_data_df, stats_dict)
                - new_data_df: DataFrame with only new samples
                - stats_dict: Statistics about new vs old data
        """
        print("\n" + "="*80)
        print("[CHECK] CHECKING FOR NEW DATA")
        print("="*80)
        
        all_new_data = []
        stats = {
            'total_files': len(csv_files),
            'files_with_new_data': 0,
            'total_samples': 0,
            'new_samples': 0,
            'already_trained': 0,
            'by_file': {}
        }
        
        for csv_file in csv_files:
            filename = Path(csv_file).name
            
            # Load CSV
            try:
                df = pd.read_csv(csv_file)
                total_in_file = len(df)
                stats['total_samples'] += total_in_file
            except Exception as e:
                print(f"⚠️  Error loading {filename}: {e}")
                continue
            
            # Get trained hashes for this file
            trained_hashes = set()
            if filename in self.tracker['files']:
                trained_hashes = set(self.tracker['files'][filename].get('hashes', []))
            
            # Find new samples
            new_samples = []
            for idx, row in df.iterrows():
                text = str(row.get('text', ''))
                text_hash = self._hash_text(text)
                
                if text_hash not in trained_hashes:
                    new_samples.append(row)
            
            new_count = len(new_samples)
            old_count = total_in_file - new_count
            
            stats['by_file'][filename] = {
                'total': total_in_file,
                'new': new_count,
                'already_trained': old_count
            }
            
            stats['new_samples'] += new_count
            stats['already_trained'] += old_count
            
            if new_count > 0:
                stats['files_with_new_data'] += 1
                all_new_data.extend(new_samples)
                print(f"[NEW] {filename}:")
                print(f"   * Total: {total_in_file} samples")
                print(f"   * New: {new_count} samples")
                print(f"   * Already trained: {old_count} samples")
            else:
                print(f"[SKIP] {filename}:")
                print(f"   * Total: {total_in_file} samples")
                print(f"   * All samples already trained - SKIP")
        
        # Create DataFrame from new data
        if all_new_data:
            new_data_df = pd.DataFrame(all_new_data)
        else:
            new_data_df = pd.DataFrame()
        
        # Print summary
        print(f"\n[SUMMARY]")
        print(f"   * Total files: {stats['total_files']}")
        print(f"   * Files with new data: {stats['files_with_new_data']}")
        print(f"   * Total samples: {stats['total_samples']}")
        print(f"   * New samples: {stats['new_samples']} [NEW]")
        print(f"   * Already trained: {stats['already_trained']}")
        
        if stats['new_samples'] == 0:
            print(f"\n[WARN] NO NEW DATA FOUND!")
            print(f"   All data has been trained already.")
            print(f"   Please add new data before training.")
        else:
            print(f"\n[OK] Found {stats['new_samples']} new samples to train!")
        
        print("="*80)
        
        return new_data_df, stats
    
    def mark_as_trained(self, csv_files, trained_df):
        """
        Mark data as trained after successful training.
        
        Args:
            csv_files: List of CSV files that were used
            trained_df: DataFrame that was actually trained
        """
        print("\n" + "="*80)
        print("[SAVE] UPDATING DATA TRACKER")
        print("="*80)
        
        # Get hashes of trained samples
        trained_hashes = set()
        for idx, row in trained_df.iterrows():
            text = str(row.get('text', ''))
            text_hash = self._hash_text(text)
            trained_hashes.add(text_hash)
        
        # Update tracker for each file
        for csv_file in csv_files:
            filename = Path(csv_file).name
            
            try:
                df = pd.read_csv(csv_file)
            except:
                continue
            
            # Get all hashes from this file
            file_hashes = set()
            for idx, row in df.iterrows():
                text = str(row.get('text', ''))
                text_hash = self._hash_text(text)
                
                # Only add if it was in trained data
                if text_hash in trained_hashes:
                    file_hashes.add(text_hash)
            
            # Update or create file entry
            if filename not in self.tracker['files']:
                self.tracker['files'][filename] = {
                    'samples': 0,
                    'hashes': [],
                    'first_trained': datetime.now().isoformat()
                }
            
            # Add new hashes
            existing_hashes = set(self.tracker['files'][filename]['hashes'])
            new_hashes = file_hashes - existing_hashes
            
            if new_hashes:
                self.tracker['files'][filename]['hashes'].extend(list(new_hashes))
                self.tracker['files'][filename]['samples'] = len(self.tracker['files'][filename]['hashes'])
                self.tracker['files'][filename]['last_updated'] = datetime.now().isoformat()
                
                print(f"[OK] {filename}:")
                print(f"   * Added {len(new_hashes)} new samples to tracker")
                print(f"   * Total tracked: {self.tracker['files'][filename]['samples']} samples")
        
        # Update global stats
        self.tracker['total_trained_samples'] = sum(
            f['samples'] for f in self.tracker['files'].values()
        )
        self.tracker['last_training'] = datetime.now().isoformat()
        
        self._save_tracker()
        
        print(f"\n[STATS] Total tracked samples: {self.tracker['total_trained_samples']}")
        print("="*80)
    
    def get_stats(self):
        """Get training statistics."""
        print("\n" + "="*80)
        print("DATA TRACKER STATISTICS")
        print("="*80)
        
        print(f"\nTotal trained samples: {self.tracker['total_trained_samples']}")
        print(f"Last training: {self.tracker.get('last_training', 'Never')}")
        print(f"\nFiles tracked: {len(self.tracker['files'])}")
        
        if self.tracker['files']:
            print("\nPer-file statistics:")
            for filename, info in sorted(self.tracker['files'].items()):
                print(f"\n  {filename}:")
                print(f"    • Samples trained: {info['samples']}")
                print(f"    • First trained: {info.get('first_trained', 'Unknown')}")
                print(f"    • Last updated: {info.get('last_updated', 'Unknown')}")
        
        print("\n" + "="*80)
    
    def reset(self):
        """Reset tracker (clear all history)."""
        print("\n[WARN] Resetting data tracker...")
        self.tracker = {
            'files': {},
            'total_trained_samples': 0,
            'last_training': None,
            'created_at': datetime.now().isoformat()
        }
        self._save_tracker()
        print("[OK] Data tracker reset complete")


# CLI Interface
if __name__ == '__main__':
    import sys
    
    tracker = DataTracker()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'stats':
            tracker.get_stats()
        
        elif command == 'reset':
            confirm = input("Are you sure you want to reset tracker? (yes/no): ")
            if confirm.lower() == 'yes':
                tracker.reset()
            else:
                print("Reset cancelled")
        
        elif command == 'check':
            # Check for new data
            import glob
            csv_files = glob.glob('data/*.csv')
            csv_files = [f for f in csv_files if 'TEMPLATE' not in f.upper()]
            
            new_data, stats = tracker.get_new_data(csv_files)
            print(f"\nNew samples ready for training: {len(new_data)}")
        
        else:
            print(f"Unknown command: {command}")
            print("Usage:")
            print("  python data_tracker.py stats   - Show statistics")
            print("  python data_tracker.py check   - Check for new data")
            print("  python data_tracker.py reset   - Reset tracker")
    else:
        tracker.get_stats()
