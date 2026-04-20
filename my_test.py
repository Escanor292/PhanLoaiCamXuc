"""
Test dự đoán cảm xúc với văn bản của bạn
"""

from predict import predict_emotions
from utils import load_model
from config import Config

# Load model
print("Đang tải model...")
model, tokenizer = load_model('saved_model')
print("✓ Model đã tải!\n")

# ========================================
# THAY ĐỔI VĂN BẢN Ở ĐÂY
# ========================================
test_texts = [
    "đụ má vãi lồn",
]

print("="*70)
print("KẾT QUẢ DỰ ĐOÁN CẢM XÚC")
print("="*70)
print()

# Test từng văn bản
for i, text in enumerate(test_texts, 1):
    print(f"{i}. Văn bản: \"{text}\"")
    print("-" * 70)
    
    result = predict_emotions(text, model, tokenizer, 'cpu')
    
    # Hiển thị cảm xúc
    emotions_vi = [Config.EMOTION_LABELS_VI.get(e, e) for e in result['emotions']]
    print(f"   Cảm xúc dự đoán: {', '.join(emotions_vi) if emotions_vi else 'Không có'}")
    
    # Hiển thị top 5
    sorted_scores = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
    print(f"   Top 5 điểm tin cậy:")
    for emotion, score in sorted_scores[:5]:
        emotion_vi = Config.EMOTION_LABELS_VI.get(emotion, emotion)
        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"     {emotion_vi:15s} [{bar}] {score:.3f}")
    
    print()

print("="*70)
print("HOÀN TẤT!")
print("="*70)
print()
print("Để test văn bản khác:")
print("1. Mở file my_test.py")
print("2. Sửa danh sách test_texts")
print("3. Chạy lại: python my_test.py")
