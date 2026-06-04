"""
Script tạo biểu đồ từ SỐ LIỆU THỰC TẾ trong dự án

QUAN TRỌNG:
- Hiện tại CHỈ CÓ 1 MODEL (đã dùng Transfer Learning)
- CHƯA CÓ dữ liệu training từ đầu để so sánh
- Script này sẽ vẽ biểu đồ từ dữ liệu THỰC TẾ có sẵn

ĐỂ CÓ BIỂU ĐỒ SO SÁNH THỰC TẾ, CẦN:
1. Training 1 model TỪ ĐẦU (không dùng Transfer Learning)
2. Lưu kết quả vào file
3. So sánh với model hiện tại
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Thiết lập font hỗ trợ tiếng Việt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_model_results():
    """Đọc kết quả thực tế từ model registry"""
    
    registry_path = 'model_registry/registry.json'
    
    if not os.path.exists(registry_path):
        print("❌ Không tìm thấy model_registry/registry.json")
        return None
    
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    if not registry['models']:
        print("❌ Chưa có model nào trong registry")
        return None
    
    # Lấy model tốt nhất
    best_model_id = registry['best_model']
    best_model = None
    
    for model in registry['models']:
        if model['model_id'] == best_model_id:
            best_model = model
            break
    
    return best_model


def create_per_emotion_f1_chart():
    """
    Biểu đồ F1 Score cho từng cảm xúc (DỮ LIỆU THỰC TẾ)
    
    Nguồn: model_registry/models/model_20260523_121847/results.txt
    """
    
    # Dữ liệu THỰC TẾ từ results.txt
    emotions = [
        'joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust',
        'anger', 'anticipation', 'love', 'worried', 'disappointed',
        'proud', 'embarrassed', 'jealous', 'calm', 'excited'
    ]
    
    f1_scores = [
        0.8941, 0.4600, 0.5542, 0.6241, 0.6732, 0.6875,
        0.7654, 0.6207, 0.5497, 0.8163, 0.7579,
        0.4390, 0.8966, 0.6977, 0.6126, 0.7172
    ]
    
    # Sắp xếp theo F1 score (cao → thấp)
    sorted_data = sorted(zip(emotions, f1_scores), key=lambda x: x[1], reverse=True)
    emotions_sorted, f1_sorted = zip(*sorted_data)
    
    # Tạo màu: xanh (>0.7), vàng (0.5-0.7), đỏ (<0.5)
    colors = []
    for score in f1_sorted:
        if score >= 0.7:
            colors.append('#4ECDC4')  # Xanh - Tốt
        elif score >= 0.5:
            colors.append('#FFD93D')  # Vàng - Trung bình
        else:
            colors.append('#FF6B6B')  # Đỏ - Cần cải thiện
    
    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.barh(emotions_sorted, f1_sorted, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Thêm giá trị F1
    for i, (bar, score) in enumerate(zip(bars, f1_sorted)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f} ({score*100:.2f}%)',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Thêm đường ngưỡng
    ax.axvline(x=0.7, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Tot (>0.7)')
    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Trung binh (>0.5)')
    
    ax.set_xlabel('F1 Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cam xuc', fontsize=14, fontweight='bold')
    ax.set_title('F1 SCORE CHO 16 CAM XUC (DU LIEU THUC TE)\nModel: model_20260523_121847', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([0, 1.0])
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Thêm thống kê
    avg_f1 = np.mean(f1_sorted)
    stats_text = f"""
    THONG KE:
    • Macro F1: {avg_f1:.4f}
    • Tot nhat: {emotions_sorted[0]} ({f1_sorted[0]:.4f})
    • Yeu nhat: {emotions_sorted[-1]} ({f1_sorted[-1]:.4f})
    • >0.7: {sum(1 for s in f1_sorted if s >= 0.7)}/16
    • <0.5: {sum(1 for s in f1_sorted if s < 0.5)}/16
    """
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('f1_per_emotion_REAL.png', dpi=300, bbox_inches='tight')
    print("✅ Đã tạo: f1_per_emotion_REAL.png")
    plt.close()


def create_metrics_summary():
    """
    Biểu đồ tổng hợp các metrics (DỮ LIỆU THỰC TẾ)
    
    Công thức:
    - Macro F1 = Average của F1 từng label
    - Micro F1 = F1 tính trên tất cả predictions
    - Hamming Loss = Tỷ lệ labels dự đoán sai
    """
    
    model = load_model_results()
    
    if not model:
        print("❌ Không thể load model results")
        return
    
    metrics = model['metrics']
    
    # Dữ liệu
    metric_names = ['Macro F1', 'Micro F1', 'Test Loss', 'Hamming Loss']
    values = [
        metrics['macro_f1'],
        metrics['micro_f1'],
        metrics['test_loss'],
        metrics['hamming_loss']
    ]
    
    # Tạo 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: F1 Scores
    f1_names = ['Macro F1', 'Micro F1']
    f1_values = [metrics['macro_f1'], metrics['micro_f1']]
    
    bars1 = ax1.bar(f1_names, f1_values, color=['#4ECDC4', '#95E1D3'], 
                    alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars1, f1_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{val:.4f}\n({val*100:.2f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1 SCORES', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Muc tieu (0.7)')
    ax1.legend()
    
    # Subplot 2: Loss Metrics
    loss_names = ['Test Loss', 'Hamming Loss']
    loss_values = [metrics['test_loss'], metrics['hamming_loss']]
    
    bars2 = ax2.bar(loss_names, loss_values, color=['#FF6B6B', '#FFA07A'], 
                    alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars2, loss_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('LOSS METRICS (Thap hon = Tot hon)', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, max(loss_values) * 1.3])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'METRICS TONG HOP - Model: {model["model_id"]}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('metrics_summary_REAL.png', dpi=300, bbox_inches='tight')
    print("✅ Đã tạo: metrics_summary_REAL.png")
    plt.close()


def create_training_info():
    """
    Biểu đồ thông tin training (DỮ LIỆU THỰC TẾ)
    """
    
    model = load_model_results()
    
    if not model:
        return
    
    metadata = model['metadata']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Tạo bảng thông tin
    info_text = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║          THONG TIN TRAINING THUC TE                      ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  MODEL ID: {model['model_id']}              ║
    ║                                                          ║
    ║  NGUOI TRAINING: {metadata['person']}                              ║
    ║  EXPERIMENT: {metadata['experiment_name'][:30]}...       ║
    ║                                                          ║
    ║  ─────────────── THAM SO ───────────────                ║
    ║                                                          ║
    ║  • Model Type:        {metadata['model_type']}                      ║
    ║  • Base Model:        {metadata['model_name'][:25]}...  ║
    ║  • Epochs:            {metadata['num_epochs']}                             ║
    ║  • Batch Size:        {metadata['batch_size']}                            ║
    ║  • Learning Rate:     {metadata['learning_rate']}                      ║
    ║  • Dropout:           {metadata['dropout_rate']}                           ║
    ║  • LSTM Hidden:       {metadata['lstm_hidden_size']}                          ║
    ║  • Max Length:        {metadata['max_length']} tokens                      ║
    ║                                                          ║
    ║  ─────────────── KET QUA ───────────────                ║
    ║                                                          ║
    ║  • Macro F1:          {model['metrics']['macro_f1']:.4f} (67.29%)              ║
    ║  • Micro F1:          {model['metrics']['micro_f1']:.4f} (68.57%)              ║
    ║  • Test Loss:         {model['metrics']['test_loss']:.4f}                        ║
    ║  • Hamming Loss:      {model['metrics']['hamming_loss']:.4f}                        ║
    ║                                                          ║
    ║  ─────────────── THOI GIAN ──────────────               ║
    ║                                                          ║
    ║  • Registered:        {model['registered_at'][:19]}         ║
    ║  • Status:            {model['status'].upper()}                      ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    
    CONG THUC METRICS:
    
    1. MACRO F1 = Average(F1_label1, F1_label2, ..., F1_label16)
       → Trung bình F1 của 16 cảm xúc
       → Mỗi label có trọng số bằng nhau
    
    2. MICRO F1 = F1(tất cả predictions)
       → Tính F1 trên toàn bộ predictions
       → Labels nhiều samples có trọng số cao hơn
    
    3. HAMMING LOSS = (Số labels dự đoán sai) / (Tổng số labels)
       → Tỷ lệ labels bị dự đoán sai
       → Càng thấp càng tốt
    
    4. TEST LOSS = Binary Cross Entropy Loss
       → Loss function cho multi-label classification
       → Càng thấp càng tốt
    """
    
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('training_info_REAL.png', dpi=300, bbox_inches='tight')
    print("✅ Đã tạo: training_info_REAL.png")
    plt.close()


def create_note_for_comparison():
    """
    Tạo note giải thích tại sao chưa có biểu đồ so sánh
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    note_text = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║              LUU Y VE BIEU DO SO SANH                             ║
    ║                                                                   ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  HIEN TAI: CHI CO 1 MODEL (da dung Transfer Learning)            ║
    ║                                                                   ║
    ║  Model ID: model_20260523_121847                                  ║
    ║  • Epochs: 5                                                      ║
    ║  • Macro F1: 0.6729 (67.29%)                                      ║
    ║  • Thoi gian: ~50-60 phut                                         ║
    ║                                                                   ║
    ║  ─────────────────────────────────────────────────────────────   ║
    ║                                                                   ║
    ║  DE CO BIEU DO SO SANH THUC TE, CAN:                              ║
    ║                                                                   ║
    ║  1. Training 1 model TU DAU (khong dung Transfer Learning)        ║
    ║     • Sua config.py: USE_TRANSFER_LEARNING = False                ║
    ║     • Chay: python train_simple.py                                ║
    ║     • Luu ket qua vao file rieng                                  ║
    ║                                                                   ║
    ║  2. So sanh 2 models:                                             ║
    ║     • Model A: Training tu dau                                    ║
    ║     • Model B: Transfer Learning (hien tai)                       ║
    ║                                                                   ║
    ║  3. Ve bieu do so sanh:                                           ║
    ║     • Epochs: A vs B                                              ║
    ║     • Thoi gian: A vs B                                           ║
    ║     • F1 Score: A vs B                                            ║
    ║                                                                   ║
    ║  ─────────────────────────────────────────────────────────────   ║
    ║                                                                   ║
    ║  DU KIEN (dua tren kinh nghiem):                                  ║
    ║                                                                   ║
    ║  Training tu dau:                                                 ║
    ║  • Epochs: 15-20                                                  ║
    ║  • Thoi gian: 3-4 gio                                             ║
    ║  • F1 Score: 0.60-0.65                                            ║
    ║                                                                   ║
    ║  Transfer Learning (hien tai):                                    ║
    ║  • Epochs: 5                                                      ║
    ║  • Thoi gian: 50-60 phut                                          ║
    ║  • F1 Score: 0.6729                                               ║
    ║                                                                   ║
    ║  → Cai thien: -67% thoi gian, +10% F1 Score                      ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    
    HUONG DAN TRAINING TU DAU:
    
    1. Backup model hien tai:
       cp -r saved_model saved_model_backup
    
    2. Sua config.py:
       USE_TRANSFER_LEARNING = False
    
    3. Training:
       python train_simple.py
    
    4. Luu ket qua:
       cp model_registry/models/model_*/results.txt results_from_scratch.txt
    
    5. So sanh:
       python compare_models.py
    """
    
    ax.text(0.5, 0.5, note_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='center', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('NOTE_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Đã tạo: NOTE_comparison.png")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("TAO BIEU DO TU SO LIEU THUC TE")
    print("="*70)
    
    print("\n1. Biểu đồ F1 Score cho 16 cảm xúc...")
    create_per_emotion_f1_chart()
    
    print("\n2. Biểu đồ tổng hợp metrics...")
    create_metrics_summary()
    
    print("\n3. Thông tin training...")
    create_training_info()
    
    print("\n4. Note về biểu đồ so sánh...")
    create_note_for_comparison()
    
    print("\n" + "="*70)
    print("✅ HOAN THANH! Da tao 4 file PNG:")
    print("   1. f1_per_emotion_REAL.png - F1 cho 16 cảm xúc")
    print("   2. metrics_summary_REAL.png - Tổng hợp metrics")
    print("   3. training_info_REAL.png - Thông tin training")
    print("   4. NOTE_comparison.png - Lưu ý về so sánh")
    print("="*70)
    print("\n💡 Để có biểu đồ so sánh thực tế:")
    print("   1. Training 1 model từ đầu (USE_TRANSFER_LEARNING = False)")
    print("   2. Lưu kết quả")
    print("   3. Chạy lại script này để so sánh")
