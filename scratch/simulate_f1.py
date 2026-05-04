
import numpy as np
from sklearn.metrics import f1_score

def simulate_zero_f1():
    # 100 samples, 16 labels
    # Positive ratio ~0.15
    n_samples = 15 # test set size
    n_labels = 16
    
    # Random true labels with 0.15 prob
    np.random.seed(42)
    y_true = (np.random.rand(n_samples, n_labels) < 0.15).astype(int)
    
    # Weak model: predicts positive with 0.10 prob (bias towards 0)
    # But since it's "weak", the TPs will be low
    y_prob = np.random.rand(n_samples, n_labels) * 0.45 # All < 0.5
    
    y_pred = (y_prob > 0.5).astype(int)
    
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"Test samples: {n_samples}")
    print(f"Total positive labels: {np.sum(y_true)}")
    print(f"Total predicted positive: {np.sum(y_pred)}")
    print(f"Micro F1: {micro:.4f}")
    print(f"Macro F1: {macro:.4f}")
    
    # Now try threshold 0.3
    y_pred_03 = (y_prob > 0.3).astype(int)
    micro_03 = f1_score(y_true, y_pred_03, average='micro', zero_division=0)
    print(f"Micro F1 (threshold 0.3): {micro_03:.4f}")

if __name__ == "__main__":
    simulate_zero_f1()
