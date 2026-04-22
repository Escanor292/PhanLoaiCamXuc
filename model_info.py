"""
Model Info - Xem thông tin model nhanh chóng
"""

import json
from datetime import datetime
from model_registry import ModelRegistry


def show_latest_model():
    """Hiển thị model mới nhất (theo thời gian)."""
    
    registry = ModelRegistry()
    
    with open('model_registry/registry.json', 'r') as f:
        data = json.load(f)
    
    if not data['models']:
        print("❌ Chưa có model nào!")
        return
    
    # Sort theo registered_at (thời gian đăng ký)
    models = sorted(data['models'], key=lambda x: x['registered_at'], reverse=True)
    latest = models[0]
    
    print("=" * 70)
    print("🆕 MODEL MỚI NHẤT (Latest)")
    print("=" * 70)
    print(f"Model ID:     {latest['model_id']}")
    print(f"Người tạo:    {latest['metadata'].get('person', 'Unknown')}")
    print(f"Experiment:   {latest['metadata'].get('experiment_name', 'N/A')}")
    print(f"Thời gian:    {latest['registered_at']}")
    print()
    print("Metrics:")
    print(f"  • Test Loss:     {latest['metrics']['test_loss']:.4f}")
    print(f"  • Macro F1:      {latest['metrics']['macro_f1']:.4f}")
    print(f"  • Hamming Loss:  {latest['metrics']['hamming_loss']:.4f}")
    print()
    print("Training Config:")
    print(f"  • Learning Rate: {latest['metadata'].get('learning_rate', 'N/A')}")
    print(f"  • Batch Size:    {latest['metadata'].get('batch_size', 'N/A')}")
    print(f"  • Epochs:        {latest['metadata'].get('num_epochs', 'N/A')}")
    print(f"  • Data:          {latest['metadata'].get('data_file', 'N/A')}")
    print("=" * 70)


def show_best_model():
    """Hiển thị model tốt nhất (theo test loss)."""
    
    registry = ModelRegistry()
    
    with open('model_registry/registry.json', 'r') as f:
        data = json.load(f)
    
    if not data['models']:
        print("❌ Chưa có model nào!")
        return
    
    # Sort theo test_loss (thấp nhất = tốt nhất)
    models = sorted(data['models'], key=lambda x: x['metrics']['test_loss'])
    best = models[0]
    
    print("=" * 70)
    print("🏆 MODEL TỐT NHẤT (Best)")
    print("=" * 70)
    print(f"Model ID:     {best['model_id']}")
    print(f"Người tạo:    {best['metadata'].get('person', 'Unknown')}")
    print(f"Experiment:   {best['metadata'].get('experiment_name', 'N/A')}")
    print(f"Thời gian:    {best['registered_at']}")
    print()
    print("Metrics:")
    print(f"  • Test Loss:     {best['metrics']['test_loss']:.4f} ⭐ LOWEST")
    print(f"  • Macro F1:      {best['metrics']['macro_f1']:.4f}")
    print(f"  • Hamming Loss:  {best['metrics']['hamming_loss']:.4f}")
    print()
    print("Training Config:")
    print(f"  • Learning Rate: {best['metadata'].get('learning_rate', 'N/A')}")
    print(f"  • Batch Size:    {best['metadata'].get('batch_size', 'N/A')}")
    print(f"  • Epochs:        {best['metadata'].get('num_epochs', 'N/A')}")
    print(f"  • Data:          {best['metadata'].get('data_file', 'N/A')}")
    print("=" * 70)


def show_production_model():
    """Hiển thị model đang dùng trong production."""
    
    import os
    
    if not os.path.exists('saved_model/training_config.json'):
        print("❌ Không tìm thấy production model!")
        return
    
    with open('saved_model/training_config.json', 'r') as f:
        config = json.load(f)
    
    print("=" * 70)
    print("🚀 MODEL PRODUCTION (Đang Dùng)")
    print("=" * 70)
    print(f"Experiment:   {config.get('experiment_name', 'N/A')}")
    print(f"Base Model:   {config.get('model_name', 'N/A')}")
    print()
    print("Metrics:")
    print(f"  • Val Loss:      {config.get('best_val_loss', 'N/A'):.4f}")
    print(f"  • Macro F1:      {config.get('macro_f1', 0):.4f}")
    print(f"  • Micro F1:      {config.get('micro_f1', 0):.4f}")
    print()
    print("Training Config:")
    print(f"  • Learning Rate: {config.get('learning_rate', 'N/A')}")
    print(f"  • Batch Size:    {config.get('batch_size', 'N/A')}")
    print(f"  • Epochs:        {config.get('num_epochs', 'N/A')}")
    print(f"  • Best Epoch:    {config.get('best_epoch', 'N/A')}")
    print()
    print("📍 Location: saved_model/")
    print("=" * 70)


def show_summary():
    """Hiển thị tổng quan tất cả models."""
    
    with open('model_registry/registry.json', 'r') as f:
        data = json.load(f)
    
    if not data['models']:
        print("❌ Chưa có model nào!")
        return
    
    # Sort theo test_loss
    models = sorted(data['models'], key=lambda x: x['metrics']['test_loss'])
    
    print("=" * 90)
    print("📊 TỔNG QUAN TẤT CẢ MODELS")
    print("=" * 90)
    print(f"{'#':<3} {'Model ID':<25} {'Person':<12} {'Test Loss':<12} {'Time':<20}")
    print("-" * 90)
    
    for i, model in enumerate(models, 1):
        model_id = model['model_id']
        person = model['metadata'].get('person', 'Unknown')[:11]
        test_loss = model['metrics']['test_loss']
        time = model['registered_at'][:19]
        
        # Đánh dấu best
        marker = "⭐" if i == 1 else "  "
        
        print(f"{marker} {i:<2} {model_id:<25} {person:<12} {test_loss:<12.4f} {time:<20}")
    
    print("=" * 90)
    print(f"Total: {len(models)} models")
    print(f"Best:  {models[0]['model_id']} (Test Loss: {models[0]['metrics']['test_loss']:.4f})")
    print(f"Latest: {sorted(data['models'], key=lambda x: x['registered_at'], reverse=True)[0]['model_id']}")
    print("=" * 90)


def main():
    """Main function."""
    
    import sys
    
    if len(sys.argv) < 2:
        print("🔍 MODEL INFO - Xem Thông Tin Model")
        print("=" * 50)
        print("Usage:")
        print("  python model_info.py latest      # Model mới nhất")
        print("  python model_info.py best        # Model tốt nhất")
        print("  python model_info.py production  # Model đang dùng")
        print("  python model_info.py summary     # Tổng quan tất cả")
        print("  python model_info.py all         # Hiển thị tất cả")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'latest':
        show_latest_model()
    elif command == 'best':
        show_best_model()
    elif command == 'production':
        show_production_model()
    elif command == 'summary':
        show_summary()
    elif command == 'all':
        show_latest_model()
        print()
        show_best_model()
        print()
        show_production_model()
        print()
        show_summary()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use: latest, best, production, summary, or all")


if __name__ == "__main__":
    main()