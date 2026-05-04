
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from config import Config

def check_data_stats(file_path):
    df = pd.read_csv(file_path)
    labels = df[Config.EMOTION_LABELS].values
    
    total_elements = labels.size
    positive_elements = np.sum(labels)
    zero_elements = total_elements - positive_elements
    
    pos_ratio = positive_elements / total_elements
    neg_ratio = zero_elements / total_elements
    
    avg_pos_per_sample = positive_elements / len(df)
    
    print(f"File: {file_path}")
    print(f"Total samples: {len(df)}")
    print(f"Positive ratio: {pos_ratio:.4f}")
    print(f"Negative ratio: {neg_ratio:.4f}")
    print(f"Avg positive labels per sample: {avg_pos_per_sample:.2f}")
    
    # Per label stats
    print("\nPer Label Positive Count:")
    for i, label in enumerate(Config.EMOTION_LABELS):
        count = np.sum(labels[:, i])
        print(f"  {label:15s}: {count}")

if __name__ == "__main__":
    check_data_stats("data/sample_comments copy.csv")
