"""
Simple Training Script - Hybrid PhoBERT for Team Members

This is the SIMPLEST way for team members to train a model.
Just run: python train_simple.py

What it does automatically:
1. Uses HYBRID PhoBERT model (most powerful for Vietnamese)
2. Merges ALL CSV files in data/ directory
3. Trains for 5 epochs (optimal for Hybrid model)
4. Registers the new model
5. Auto-deploys if it's the best model

No parameters needed - everything is optimized!
"""

import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model_registry import ModelRegistry
from model_sharing import ModelSharing


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("🚀 HUẤN LUYỆN ĐƠN GIẢN - HYBRID PHOBERT (MẠNH NHẤT)")
    print("=" * 80)
    print()
    print("✅ Mô hình: Hybrid PhoBERT + BiLSTM + Attention")
    print("✅ Tối ưu cho: Tiếng Việt")
    print("✅ Huấn luyện tăng cường: Chỉ học dữ liệu MỚI ⭐")
    print("✅ Tự động gộp dữ liệu: ĐÃ BẬT (có thể tùy chọn) ⭐")
    print("✅ Transfer Learning: ĐÃ BẬT (học từ mô hình tốt nhất) ⭐")
    print("✅ Lưu mô hình tốt nhất: ĐÃ BẬT (tiết kiệm bộ nhớ) ⭐")
    print("✅ Tự động triển khai: ĐÃ BẬT (nếu đạt kết quả tốt nhất)")
    print("✅ Cài đặt: 5 epochs, LR=2e-5, LSTM=256")
    print()
    print("Bạn chỉ cần ngồi đợi kết quả! 🎯")
    print("Quá trình này có thể tốn thời gian nhưng sẽ cho kết quả TỐT NHẤT!")
    print("=" * 80)


def check_data_files():
    """Check if there are CSV files to train on."""
    import glob
    
    csv_files = glob.glob(os.path.join(Config.DATA_DIR, "*.csv"))
    csv_files = [f for f in csv_files if "TEMPLATE" not in f.upper() and "merged_temp" not in f.lower()]
    csv_files = sorted(csv_files)
    
    if not csv_files:
        print("❌ LỖI: Không tìm thấy file CSV nào trong thư mục data/!")
        print()
        print("Vui lòng thêm dữ liệu của bạn trước:")
        print("1. Tạo file data/member_TenCuaBan.csv")
        print("2. Làm theo định dạng trong HUONG_DAN_DONG_GOP_DATA.md")
        print("3. Sau đó chạy lại script này")
        return False, []
    
    print(f"📊 Tìm thấy {len(csv_files)} file dữ liệu:")
    for i, csv_file in enumerate(csv_files):
        print(f"   {i+1}. {os.path.basename(csv_file)}")
    print()
    
    return True, csv_files


def merge_all_data(csv_files):
    """
    Merge all CSV files and extract only NEW data that hasn't been trained.
    
    Args:
        csv_files (list): List of CSV file paths
    
    Returns:
        tuple: (merged_file_path, new_samples_count, stats)
    """
    print("\n" + "="*80)
    print("📦 AUTO-MERGING DATA + INCREMENTAL TRAINING")
    print("="*80)
    
    # Use data tracker to get only new data
    from data_tracker import DataTracker
    tracker = DataTracker()
    
    new_data_df, stats = tracker.get_new_data(csv_files)
    
    if len(new_data_df) == 0:
        print("\n❌ NO NEW DATA TO TRAIN!")
        print("   All data has been trained already.")
        print("\n💡 To train anyway:")
        print("   1. Add new data to CSV files")
        print("   2. Or reset tracker: python data_tracker.py reset")
        return None, 0, stats
    
    print(f"\n✅ Proceeding with {len(new_data_df)} NEW samples")
    
    # Save to temporary file
    merged_file = os.path.join(Config.DATA_DIR, "merged_temp.csv")
    new_data_df.to_csv(merged_file, index=False)
    print(f"   • Saved to: {merged_file}")
    print("="*80)
    
    return merged_file, len(new_data_df), stats


def show_current_best():
    """Show current best model info."""
    try:
        registry = ModelRegistry()
        best_model = registry.get_best_model()
        
        if best_model:
            print("🏆 Current Best Model:")
            print(f"   • Model ID: {best_model['model_id']}")
            print(f"   • Test Loss: {best_model['metrics']['test_loss']:.4f}")
            print(f"   • Macro F1: {best_model['metrics']['macro_f1']:.4f}")
            print(f"   • Person: {best_model['metadata'].get('person', 'Unknown')}")
            print()
            print("🎯 Your new model will learn from this best model!")
        else:
            print("📝 No existing models found. Training from scratch...")
        
        print()
        
    except Exception as e:
        print(f"⚠️  Could not load model registry: {e}")
        print("Training will start from scratch...")
        print()


def main_simple():
    """Main function for simple training."""
    print_banner()
    
    # Check if data files exist
    has_data, all_csv_files = check_data_files()
    if not has_data:
        return
    
    # Bước lựa chọn file CSV tương tác
    csv_files = all_csv_files
    if len(all_csv_files) > 1:
        print("❓ Bạn muốn dùng những file nào để huấn luyện?")
        print("   A. Dùng TẤT CẢ các file (Mặc định - nhấn Enter)")
        print("   B. Chọn các file cụ thể (nhập số thứ tự, ví dụ: 1,3)")
        
        try:
            choice = input("\n👉 Lựa chọn của bạn: ").strip().upper()
            
            if choice and choice != 'A':
                # Xử lý nhập số (ví dụ: "1,2") hoặc nhập tên file
                indices = []
                for part in choice.replace(',', ' ').split():
                    if part.isdigit():
                        indices.append(int(part) - 1)
                
                selected_files = [all_csv_files[i] for i in indices if 0 <= i < len(all_csv_files)]
                
                if selected_files:
                    csv_files = selected_files
                    print(f"\n✅ Đã chọn {len(csv_files)} file để huấn luyện.")
                else:
                    print("\n⚠️  Không có file hợp lệ được chọn. Sẽ dùng TẤT CẢ các file.")
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️  Đã hủy lựa chọn. Sẽ dùng TẤT CẢ các file.")
        except Exception as e:
            print(f"\n⚠️  Lỗi khi chọn file ({e}). Sẽ dùng TẤT CẢ các file.")
    
    # Merge all data files and get only NEW data
    try:
        merged_file, new_samples_count, merge_stats = merge_all_data(csv_files)
        
        if merged_file is None or new_samples_count == 0:
            print("\n" + "="*80)
            print("⏭️  BỎ QUA HUẤN LUYỆN - Không có dữ liệu mới")
            print("="*80)
            return
    except Exception as e:
        print(f"❌ LỖI: Thất bại khi xử lý dữ liệu: {e}")
        return
    
    # Show current best model
    show_current_best()
    
    # Confirm settings
    print("🔧 Cài đặt huấn luyện:")
    print(f"   • Loại mô hình: Hybrid PhoBERT (Mạnh nhất)")
    print(f"   • Kiến trúc: PhoBERT + BiLSTM + Attention")
    print(f"   • Số lượng file: {len(csv_files)} file đã chọn")
    print(f"   • Số mẫu mới: {new_samples_count} mẫu ⭐")
    print(f"   • Epochs: 5 (tối ưu cho Hybrid)")
    print(f"   • Learning Rate: {Config.LEARNING_RATE}")
    print(f"   • Batch Size: {Config.BATCH_SIZE}")
    print(f"   • LSTM Hidden Size: 256")
    print()
    
    # Start training
    print("🚀 Bắt đầu huấn luyện với Hybrid PhoBERT...")
    print("⏱️  Quá trình này có thể tốn thời gian nhưng sẽ cho kết quả TỐT NHẤT!")
    print("=" * 80)
    
    try:
        # Import transfer learning module
        from transfer_learning import (
            load_base_model_for_transfer,
            should_use_transfer_learning,
            get_transfer_learning_settings,
            print_transfer_learning_info
        )
        
        # Show transfer learning info
        print_transfer_learning_info()
        
        # Check if transfer learning is enabled
        use_transfer = should_use_transfer_learning()
        
        if use_transfer:
            # Try to load base model
            device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
            base_model, base_tokenizer, base_info = load_base_model_for_transfer('hybrid', device)
            
            # Get optimal settings
            settings = get_transfer_learning_settings(base_info)
            epochs = settings['epochs']
            learning_rate = settings['learning_rate']
            
            print(f"\n🔧 Training Settings ({settings['description']}):")
            print(f"   • Epochs: {epochs}")
            print(f"   • Learning Rate: {learning_rate}")
            print()
        else:
            epochs = 10
            learning_rate = Config.LEARNING_RATE
            base_model = None
            print("\n⚠️  Transfer Learning DISABLED")
            print("   Training from scratch...")
        
        # Import and run train_with_args
        from train_with_args import main as train_main
        
        # Get person name from environment or prompt
        person = os.getenv('USER', os.getenv('USERNAME', 'team_member'))
        
        # Set up arguments for hybrid training
        sys.argv = [
            'train_with_args.py',
            '--model-type', 'hybrid',
            '--data', merged_file,  # Use merged file!
            '--epochs', str(epochs),
            '--batch-size', str(Config.BATCH_SIZE),
            '--lr', str(learning_rate),
            '--lstm-hidden-size', '256',
            '--dropout', str(Config.DROPOUT_RATE),
            '--max-length', '256',
            '--experiment-name', f'Hybrid Training by {person} - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            '--register-model'
        ]
        
        # Add transfer learning flag if base model exists
        if base_model is not None:
            sys.argv.extend(['--transfer-from', base_info['model_id']])
        
        # Run training
        train_main()
        
        # Mark data as trained
        print("\n" + "="*80)
        print("📝 Marking data as trained...")
        print("="*80)
        
        from data_tracker import DataTracker
        import pandas as pd
        
        tracker = DataTracker()
        trained_df = pd.read_csv(merged_file)
        tracker.mark_as_trained(csv_files, trained_df)
        
        print("=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        
        if base_model is not None:
            print("✅ Your Hybrid PhoBERT model has been fine-tuned with Transfer Learning!")
            print("✅ Model learned from previous knowledge + new data!")
        else:
            print("✅ Your Hybrid PhoBERT model has been trained from scratch!")
        
        print("✅ If it's the best model, it's automatically deployed!")
        
        # Auto-sync to Hugging Face
        if getattr(Config, 'AUTO_SYNC_CLOUD', False):
            print("\n" + "="*80)
            print("🔄 AUTO-SYNCING TO HUGGING FACE")
            print("="*80)
            try:
                sharing = ModelSharing()
                if sharing.sync_best_model():
                    print("✅ Model synced to Hugging Face successfully!")
                else:
                    print("⚠️  Model sync to Hugging Face failed.")
            except Exception as e:
                print(f"⚠️  Error during auto-sync: {e}")
            print("="*80)
        
        print()
        print("📊 Model Features:")
        print("   • PhoBERT: Optimized for Vietnamese")
        print("   • BiLSTM: Understands context in both directions")
        print("   • Attention: Focuses on important words")
        if base_model is not None:
            print("   • Transfer Learning: Keeps old knowledge + learns new ⭐")
        print(f"   • Auto-Merge: Trained on {len(csv_files)} files ⭐")
        print()
        print("Next steps:")
        print("1. Check results: python model_registry.py list")
        print("2. Compare models: python compare_experiments.py")
        print("3. Test model: python demo_phobert.py --mode interactive")
        print("4. Commit results: git add model_registry/ && git commit -m 'Hybrid PhoBERT training results' && git push")
        print()
        
    except Exception as e:
        print("=" * 80)
        print("❌ TRAINING FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check if you have enough disk space")
        print("2. Check if your CSV files are valid")
        print("3. Check internet connection (PhoBERT needs to download)")
        print("4. Try with PhoBERT instead: python train_unified.py --model-type phobert")
        print("5. Or use old BERT: python train_unified.py --model-type bert")
        print()
    
    finally:
        # Clean up temporary merged file
        if len(csv_files) > 1:
            try:
                if os.path.exists(merged_file):
                    os.remove(merged_file)
                    print(f"\n🧹 Cleaned up temporary file: {merged_file}")
            except Exception as e:
                print(f"\n⚠️  Could not remove temporary file: {e}")


if __name__ == "__main__":
    main_simple()