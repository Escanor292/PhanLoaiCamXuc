# 🚀 Cheat Sheet - Tất Cả Lệnh Hữu Ích

## ⚡ Lệnh Thường Dùng Nhất

```bash
# Training (CHỈ CẦN LỆNH NÀY!)
python train_simple.py

# Xem model mới nhất
python model_info.py latest

# Xem model tốt nhất
python model_info.py best

# Xem tổng quan
python model_info.py summary
```

---

## 📊 Xem Thông Tin Model

### Model Info (Nhanh)
```bash
python model_info.py latest      # Model mới nhất (theo thời gian)
python model_info.py best        # Model tốt nhất (test loss thấp nhất)
python model_info.py production  # Model đang dùng trong production
python model_info.py summary     # Tổng quan tất cả models
python model_info.py all         # Hiển thị tất cả thông tin
```

### Model Registry (Chi tiết)
```bash
python model_registry.py list                    # Xem tất cả models
python model_registry.py info --model-id <id>    # Xem chi tiết 1 model
python model_registry.py deploy --model-id <id>  # Deploy model cụ thể
```

---

## 🎓 Training

### Training Đơn Giản (Khuyến nghị)
```bash
python train_simple.py
# Tự động: merge data + transfer learning + deploy + upload
```

### Training Với Tùy Chọn
```bash
python train_incremental.py --epochs 3
python train_incremental.py --epochs 5 --lr 1e-5
python train_incremental.py --no-transfer --epochs 5  # Train từ đầu
python train_incremental.py --base-model model_20260422_124631
```

### Training Cũ (Không khuyến nghị)
```bash
python train.py  # Train cơ bản
python train_with_args.py --data data/file.csv --register-model
```

---

## ☁️ Model Sharing

### Upload/Download
```bash
python model_sharing.py sync                      # Upload model tốt nhất
python model_sharing.py upload --model-id <id>    # Upload model cụ thể
python model_sharing.py download --model-id <id>  # Download model cụ thể
```

### Cấu Hình
```bash
python model_sharing.py config --method huggingface --repo "org/repo"
python switch_repo.py  # Chuyển đổi giữa các repo
```

### Setup Hugging Face
```bash
python setup_hf_login.py  # Setup login HF
python test_org_access.py # Test quyền organization
```

---

## 📁 Data Management

### Merge Data
```bash
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv
```

### Generate Sample Data
```bash
python generate_sample_data.py
```

---

## 🧪 Testing

### Test Model
```bash
python my_test.py           # Test đơn giản
python demo_prediction.py   # Demo prediction
python manual_testing.py    # Manual testing
```

### Run Tests
```bash
python -m pytest tests/                    # Chạy tất cả tests
python -m pytest tests/test_model.py       # Test model
python -m pytest tests/test_dataset.py     # Test dataset
```

---

## 🔧 Git Workflow

### Pull & Push
```bash
git pull                                          # Pull code mới
git add .                                         # Add tất cả files
git commit -m "Message"                           # Commit
git push                                          # Push lên GitHub
```

### Kiểm Tra Trước Khi Push
```bash
python check_before_push.py  # Kiểm tra không xóa file
git status                   # Xem files thay đổi
git diff                     # Xem nội dung thay đổi
```

---

## 🔍 Utilities

### Check Model Size
```bash
# PowerShell
Get-ChildItem saved_model/ | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}
```

### Check Python Packages
```bash
pip list                    # Xem tất cả packages
pip show transformers       # Xem chi tiết package
pip install -r requirements.txt  # Cài đặt dependencies
```

---

## 🎯 Workflow Hoàn Chỉnh

### Setup Lần Đầu
```bash
git clone https://github.com/Escanor292/PhanLoaiCamXuc.git
cd PhanLoaiCamXuc
pip install -r requirements.txt
python setup_hf_login.py
```

### Hàng Ngày
```bash
# 1. Pull code mới
git pull

# 2. Training
python train_simple.py

# 3. Xem kết quả
python model_info.py latest

# 4. Push kết quả
git add model_registry/
git commit -m "Training results"
git push
```

---

## 📊 So Sánh Models

### Compare Experiments
```bash
python compare_experiments.py experiments/exp1/ experiments/exp2/
```

---

## 🆘 Troubleshooting

### Lỗi Thường Gặp
```bash
# CUDA out of memory
python train_incremental.py --batch-size 8

# Authentication failed
python setup_hf_login.py

# Model not found
python model_sharing.py sync

# Git conflict
git status
git diff
# Sửa conflict thủ công, sau đó:
git add .
git commit -m "Resolve conflict"
git push
```

---

## 📚 Documentation

### Xem Hướng Dẫn
```bash
# Quick start
cat TRAINING_QUICK_START.md

# Full guide
cat HUONG_DAN_CHO_THANH_VIEN.md

# Setup complete
cat SETUP_COMPLETE.md

# Model sharing
cat TEAM_MODEL_SHARING.md
```

---

## 🔗 Links Quan Trọng

- **GitHub:** https://github.com/Escanor292/PhanLoaiCamXuc
- **Hugging Face:** https://huggingface.co/emotion-classification-vn/emotion-classification
- **Model Registry:** `model_registry/registry.json`

---

## 💡 Tips

### Xem Model Nhanh
```bash
# Thay vì chạy python model_registry.py list (dài)
python model_info.py summary  # Ngắn gọn hơn
```

### Training Nhanh
```bash
# Thay vì nhiều lệnh
python train_simple.py  # Chỉ 1 lệnh, tự động tất cả
```

### Check Before Push
```bash
# Luôn check trước khi push
python check_before_push.py
git status
```

---

**Lưu file này để tham khảo nhanh!** 📌