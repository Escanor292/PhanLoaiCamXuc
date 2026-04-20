"""
Demo Prediction - Test prediction without training

This script demonstrates prediction using a pre-trained model.
If no model exists, it will show you how to get one.
"""

import os
import sys

def check_model_exists():
    """Check if model exists."""
    model_dir = 'saved_model'
    required_files = ['pytorch_model.bin', 'tokenizer.json', 'training_config.json']
    
    if not os.path.exists(model_dir):
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            return False
    
    return True

def show_no_model_message():
    """Show message when no model exists."""
    print("="*70)
    print("NO MODEL FOUND")
    print("="*70)
    print()
    print("You need a trained model to run predictions.")
    print()
    print("Options:")
    print()
    print("1. Train a model (SLOW - 30-60 minutes on CPU):")
    print("   python train_with_args.py --epochs 2 --register-model")
    print()
    print("2. Get model from team member:")
    print("   - Ask a team member who has trained a model")
    print("   - Copy their 'saved_model/' folder to your project")
    print()
    print("3. Download from GitHub (if available):")
    print("   - Check if team has uploaded model to GitHub")
    print("   - git pull to get the latest model")
    print()
    print("="*70)

def run_demo():
    """Run demo prediction."""
    print("="*70)
    print("DEMO DỰ ĐOÁN - PHÂN LOẠI CẢM XÚC")
    print("="*70)
    print()
    
    # Check if model exists
    if not check_model_exists():
        show_no_model_message()
        return
    
    # Import prediction module
    try:
        from predict import predict_emotions
        from utils import load_model
        from config import Config
    except ImportError as e:
        print(f"Lỗi import module: {e}")
        print("Đảm bảo bạn đã cài đặt tất cả dependencies:")
        print("  pip install -r requirements.txt")
        return
    
    # Load model
    print("Đang tải model...")
    try:
        model, tokenizer = load_model('saved_model')
        print("✓ Model đã tải thành công!")
        print()
    except Exception as e:
        print(f"✗ Lỗi khi tải model: {e}")
        return
    
    # Demo predictions
    test_cases = [
        "I love this product! It's amazing!",
        "This is terrible. I'm very disappointed.",
        "I'm so worried about the exam tomorrow.",
        "Congratulations! I'm so proud of you!",
        "This is okay, nothing special.",
    ]
    
    print("="*70)
    print("CÁC DỰ ĐOÁN DEMO")
    print("="*70)
    print()
    
    for i, text in enumerate(test_cases, 1):
        print(f"{i}. Văn bản: \"{text}\"")
        print("-" * 70)
        
        try:
            result = predict_emotions(text, model, tokenizer, 'cpu')
            
            # Convert emotions to Vietnamese
            emotions_vi = [Config.EMOTION_LABELS_VI.get(e, e) for e in result['emotions']]
            print(f"   Cảm xúc dự đoán: {', '.join(emotions_vi) if emotions_vi else 'Không có'}")
            print(f"   Top 3 điểm tin cậy:")
            
            # Sort by confidence
            sorted_scores = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
            for emotion, score in sorted_scores[:3]:
                emotion_vi = Config.EMOTION_LABELS_VI.get(emotion, emotion)
                print(f"     • {emotion_vi:15s}: {score:.4f}")
            
            print()
        except Exception as e:
            print(f"   ✗ Lỗi: {e}")
            print()
    
    print("="*70)
    print("DEMO HOÀN TẤT")
    print("="*70)
    print()
    print("Để thử văn bản của bạn, chạy:")
    print("  python predict.py")
    print()

if __name__ == "__main__":
    run_demo()
