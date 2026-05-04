
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_registry import ModelRegistry
from dataset import EmotionDataset
from config import Config
from transformers import AutoTokenizer
from model_phobert import HybridEmotionClassifier

def test_model_predictions():
    registry = ModelRegistry()
    best_model = registry.get_best_model()
    
    if not best_model:
        print("No model found in registry.")
        return

    print(f"Testing model: {best_model['model_id']}")
    model_path = best_model['path']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridEmotionClassifier(num_labels=16)
    model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
    model.to(device)
    model.eval()
    
    # Test texts
    test_texts = [
        "Tôi rất vui và hạnh phúc!",
        "Thật là thất vọng quá đi.",
        "Tôi lo lắng về kết quả này.",
        "Tuyệt vời quá!"
    ]
    
    print("\nPredictions:")
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        print(f"\nText: {text}")
        found = False
        for i, prob in enumerate(probs):
            if prob > 0.3: # Using lower threshold to see if there's ANY signal
                print(f"  - {Config.EMOTION_LABELS[i]}: {prob:.4f}")
                found = True
        if not found:
            max_idx = np.argmax(probs)
            print(f"  (No predictions > 0.3. Max is {Config.EMOTION_LABELS[max_idx]}: {probs[max_idx]:.4f})")

if __name__ == "__main__":
    test_model_predictions()
