"""
Simplified Manual Testing Script

This script runs a subset of manual tests to verify the prediction system works correctly.
It's designed to be lightweight and work even with limited system resources.
"""

import sys
from config import Config

# Test cases - simplified version
QUICK_TESTS = [
    # Positive emotions
    ("I love this product! It's amazing!", ["joy", "love", "excited"]),
    ("I'm so proud of this achievement!", ["proud", "joy"]),
    
    # Negative emotions
    ("This is terrible. I'm very disappointed.", ["disappointed", "anger", "sadness"]),
    ("I'm worried about the outcome.", ["worried", "fear"]),
    
    # Mixed emotions
    ("I'm excited but also worried.", ["excited", "worried"]),
    
    # Neutral/calm
    ("The product works as described.", ["calm", "trust"]),
    
    # Vietnamese
    ("Tôi rất vui với sản phẩm này!", ["joy", "love"]),
    
    # Edge cases
    ("ok", ["calm"]),
    ("!!!", []),
]


def run_quick_tests():
    """Run quick manual tests."""
    print("\n" + "="*70)
    print("QUICK MANUAL TESTING")
    print("="*70)
    
    try:
        from utils import load_model
        from predict import predict_emotions
        
        print(f"\nLoading model from: {Config.MODEL_SAVE_DIR}")
        model, tokenizer = load_model(Config.MODEL_SAVE_DIR, Config.DEVICE)
        print(f"✓ Model loaded successfully on {Config.DEVICE}\n")
        
    except FileNotFoundError:
        print("\n✗ Model not found. Please train the model first:")
        print("  python train.py")
        return False
    except Exception as e:
        print(f"\n✗ Error loading model: {str(e)}")
        print("\nThis may be due to:")
        print("  - Insufficient memory")
        print("  - Corrupted model checkpoint")
        print("  - Incompatible transformers version")
        print("\nPlease refer to MANUAL_TESTING_REPORT.md for complete test documentation.")
        return False
    
    print("Running tests...\n")
    passed = 0
    total = len(QUICK_TESTS)
    
    for i, (text, expected) in enumerate(QUICK_TESTS, 1):
        try:
            result = predict_emotions(text, model, tokenizer, Config.DEVICE)
            predicted = result['emotions']
            
            # Check if any expected emotion was found
            found = set(predicted) & set(expected) if expected else not predicted
            status = "✓" if found else "⚠"
            
            if found:
                passed += 1
            
            print(f"{status} Test {i}/{total}: \"{text[:40]}{'...' if len(text) > 40 else ''}\"")
            print(f"  Expected: {', '.join(expected) if expected else 'None'}")
            print(f"  Predicted: {', '.join(predicted) if predicted else 'None'}")
            
            if predicted:
                top_emotion = max(predicted, key=lambda e: result['scores'][e])
                print(f"  Top: {top_emotion} ({result['scores'][top_emotion]:.3f})")
            print()
            
        except Exception as e:
            print(f"✗ Test {i}/{total} failed: {str(e)}\n")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*70)
    
    print("\n✓ Quick testing complete!")
    print("✓ See MANUAL_TESTING_REPORT.md for comprehensive test documentation")
    
    return True


if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)
