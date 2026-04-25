"""
Demo script for PhoBERT emotion classifier with attention visualization

Usage:
    python demo_phobert.py
    python demo_phobert.py --model_path experiments/phobert_phobert_20260422_150000
"""

import argparse
import torch
from transformers import AutoTokenizer
import numpy as np

from config import Config
from model_phobert import PhoBERTEmotionClassifier, HybridEmotionClassifier


def load_model(model_path, model_type='phobert', device='cpu'):
    """Load trained PhoBERT model"""
    print(f"📥 Loading {model_type} model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Initialize model
    if model_type == 'phobert':
        model = PhoBERTEmotionClassifier(num_labels=len(Config.EMOTION_LABELS))
    else:
        model = HybridEmotionClassifier(num_labels=len(Config.EMOTION_LABELS))
    
    # Load weights
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=device))
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, tokenizer


def predict_with_attention(model, tokenizer, text, threshold=0.4, device='cpu'):
    """
    Predict emotions and get attention weights
    
    Args:
        model: Trained model
        tokenizer: PhoBERT tokenizer
        text (str): Input text
        threshold (float): Confidence threshold
        device: torch device
    
    Returns:
        dict: Predictions with attention weights
    """
    # Tokenize
    inputs = tokenizer(
        text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Get attention weights (if model supports it)
    attention_weights = None
    if hasattr(model, 'get_attention_weights'):
        attention_weights = model.get_attention_weights(input_ids, attention_mask)
        attention_weights = attention_weights.squeeze().cpu().numpy()
    
    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())
    
    # Process results
    emotions = {}
    detected_emotions = []
    
    for i, emotion in enumerate(Config.EMOTION_LABELS):
        score = float(probabilities[i])
        emotions[emotion] = {
            'score': score,
            'vietnamese': Config.EMOTION_LABELS_VI.get(emotion, emotion),
            'detected': score >= threshold
        }
        
        if score >= threshold:
            detected_emotions.append({
                'emotion': emotion,
                'vietnamese': Config.EMOTION_LABELS_VI.get(emotion, emotion),
                'score': score
            })
    
    # Sort by score
    detected_emotions.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'text': text,
        'emotions': emotions,
        'detected_emotions': detected_emotions,
        'tokens': tokens,
        'attention_weights': attention_weights,
        'threshold': threshold
    }


def visualize_attention(tokens, attention_weights, top_k=10):
    """
    Visualize attention weights
    
    Args:
        tokens (list): List of tokens
        attention_weights (np.array): Attention weights
        top_k (int): Number of top tokens to show
    """
    if attention_weights is None:
        print("⚠️ Attention weights not available for this model")
        return
    
    # Filter out padding tokens
    valid_indices = [i for i, token in enumerate(tokens) if token not in ['<pad>', '<s>', '</s>']]
    valid_tokens = [tokens[i] for i in valid_indices]
    valid_weights = attention_weights[valid_indices]
    
    # Get top k tokens by attention weight
    top_indices = np.argsort(valid_weights)[-top_k:][::-1]
    
    print("\n" + "="*60)
    print("🔍 Attention Visualization (Top Important Words)")
    print("="*60)
    
    max_weight = valid_weights[top_indices[0]]
    for idx in top_indices:
        token = valid_tokens[idx]
        weight = valid_weights[idx]
        bar_length = int((weight / max_weight) * 40)
        bar = "█" * bar_length
        print(f"{token:20s} {bar} {weight:.4f}")


def interactive_demo(model, tokenizer, device):
    """Interactive demo mode"""
    print("\n" + "="*60)
    print("🎭 PhoBERT Emotion Classifier - Interactive Demo")
    print("="*60)
    print("Enter Vietnamese text to analyze emotions")
    print("Type 'quit' or 'exit' to stop")
    print("="*60)
    
    while True:
        print("\n📝 Enter text (Vietnamese):")
        text = input("> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not text:
            print("⚠️ Please enter some text")
            continue
        
        # Predict
        result = predict_with_attention(model, tokenizer, text, threshold=0.4, device=device)
        
        # Display results
        print("\n" + "="*60)
        print("📊 Emotion Analysis Results")
        print("="*60)
        print(f"Text: {text}")
        print("\n🎭 Detected Emotions:")
        
        if result['detected_emotions']:
            for emotion in result['detected_emotions']:
                print(f"  • {emotion['vietnamese']:15s} ({emotion['emotion']:12s}) - {emotion['score']:.2%}")
        else:
            print("  No emotions detected above threshold")
        
        # Show all emotions
        print("\n📈 All Emotion Scores:")
        sorted_emotions = sorted(
            result['emotions'].items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        for emotion, data in sorted_emotions[:8]:  # Top 8
            bar_length = int(data['score'] * 30)
            bar = "█" * bar_length
            print(f"  {data['vietnamese']:15s} {bar} {data['score']:.2%}")
        
        # Visualize attention
        if result['attention_weights'] is not None:
            visualize_attention(result['tokens'], result['attention_weights'], top_k=10)


def batch_demo(model, tokenizer, device):
    """Demo with predefined examples"""
    examples = [
        "Tôi rất vui vì được gặp bạn!",
        "Sản phẩm này thật tệ, tôi rất thất vọng",
        "Tôi lo lắng về kỳ thi sắp tới",
        "Wow, điều này thật tuyệt vời và bất ngờ!",
        "Tôi ghét những người không giữ lời hứa",
        "Cảm ơn bạn rất nhiều, tôi rất biết ơn",
        "Tôi buồn vì phải chia tay người yêu",
        "Tôi tự hào về thành tích của mình"
    ]
    
    print("\n" + "="*60)
    print("🎭 PhoBERT Emotion Classifier - Batch Demo")
    print("="*60)
    
    for i, text in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}/{len(examples)}")
        print(f"{'='*60}")
        
        result = predict_with_attention(model, tokenizer, text, threshold=0.4, device=device)
        
        print(f"📝 Text: {text}")
        print("\n🎭 Detected Emotions:")
        
        if result['detected_emotions']:
            for emotion in result['detected_emotions']:
                print(f"  • {emotion['vietnamese']:15s} - {emotion['score']:.2%}")
        else:
            print("  No emotions detected")
        
        # Show attention for first few examples
        if i <= 3 and result['attention_weights'] is not None:
            visualize_attention(result['tokens'], result['attention_weights'], top_k=5)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PhoBERT Emotion Classifier Demo')
    parser.add_argument('--model_path', type=str, default='saved_model',
                       help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='phobert',
                       choices=['phobert', 'hybrid'],
                       help='Model architecture type')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'batch'],
                       help='Demo mode: interactive or batch')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.model_type, device)
    
    # Run demo
    if args.mode == 'interactive':
        interactive_demo(model, tokenizer, device)
    else:
        batch_demo(model, tokenizer, device)


if __name__ == '__main__':
    main()
