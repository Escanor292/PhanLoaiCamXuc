# Quick Start - Chạy Dự Án Trong 5 Phút

## 🎯 Mục Tiêu

Chạy dự án nhanh nhất có thể để xem kết quả.

---

## ⚡ Option 1: Test Prediction (30 giây)

**Dùng model đã train sẵn để predict.**

### Bước 1: Check Model

```bash
# Check xem có model chưa
ls saved_model/
```

**Nếu chưa có model:** Download từ team hoặc train (xem Option 2)

### Bước 2: Run Prediction

```bash
python predict.py
```

**Kết quả:**
```
======================================================================
EMOTION PREDICTION - INTERACTIVE MODE
======================================================================

Model loaded from: saved_model/
Ready for prediction!

Enter text (or 'quit' to exit): I love this product!

Predicted Emotions: joy, love, excited
Confidence Scores:
  joy         : 0.9234
  love        : 0.8567
  excited     : 0.7823
  ...

Enter text (or 'quit' to exit):
```

**Test với các câu:**
- "I love this product!" → joy, love, excited
- "This is terrible" → sadness, disgust, anger
- "I'm so worried about this" → worried, fear, sadness

---

## 🎓 Option 2: Training (Chậm - 30-60 phút)

**Train model từ đầu.**

### ⚠️ Lưu Ý Quan Trọng

**Training rất chậm trên CPU:**
- 100 samples, 2 epochs: ~30 phút
- 100 samples, 5 epochs: ~60 phút
- 1000 samples, 5 epochs: ~10 giờ

**Khuyến nghị:**
- ✅ Có GPU: Training nhanh (5-10 phút)
- ❌ Không có GPU: Dùng model của team hoặc train với epochs ít

### Bước 1: Training Nhanh (2 epochs)

```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output saved_model/ \
    --experiment-name "Quick Training" \
    --epochs 2 \
    --register-model
```

**Thời gian:** ~30 phút trên CPU

**Kết quả:**
```
======================================================================
TRAINING
======================================================================

Epoch 1/2
Training: 100%|████████████| 5/5 [15:00<00:00]
  Train Loss: 0.3456
  Val Loss: 0.2987
  Micro F1: 0.7823
  Macro F1: 0.7456

Epoch 2/2
Training: 100%|████████████| 5/5 [15:00<00:00]
  Train Loss: 0.1234
  Val Loss: 0.1567
  Micro F1: 0.8567
  Macro F1: 0.8234

======================================================================
TRAINING COMPLETE!
======================================================================
Model saved to: saved_model/
Test Macro F1: 0.8123
```

### Bước 2: Test Model

```bash
python predict.py
```

---

## 📊 Option 3: Collaborative Training (Team)

**Nhiều người cùng góp data và training.**

### Workflow

```
1. Mỗi người tạo data
   ↓
2. Merge data
   ↓
3. Training với config khác nhau
   ↓
4. So sánh kết quả
   ↓
5. Deploy model tốt nhất
```

### Bước 1: Tạo Data Của Bạn

```bash
# Copy template
copy data\sample_comments.csv data\member_TenBan_data.csv

# Edit file (thêm comments của bạn)
notepad data\member_TenBan_data.csv
```

### Bước 2: Merge Data

```bash
python merge_data.py
```

**Kết quả:**
```
======================================================================
MERGING DATASETS
======================================================================
✓ Loaded data/sample_comments.csv: 100 samples
✓ Loaded data/member_john_data.csv: 200 samples
✓ Loaded data/member_alice_data.csv: 150 samples

Merge complete!
Final dataset: 450 samples
Output: data/master_dataset.csv
======================================================================
```

### Bước 3: Training

```bash
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/my_exp/ \
    --experiment-name "My Experiment" \
    --lr 5e-5 \
    --register-model
```

### Bước 4: Xem Kết Quả

```bash
python model_registry.py list
```

**Kết quả:**
```
================================================================================
MODEL REGISTRY - Top 10 Models
================================================================================

⭐ 1. model_20260420_150000 [BEST]
   Metrics:
     • Macro F1:      0.8234
     • Micro F1:      0.8567
   Metadata:
     • Person:        alice
     • Experiment:    Member Alice - LR 1e-5

📦 2. model_20260420_143022 [REGISTERED]
   Metrics:
     • Macro F1:      0.8123
     • Micro F1:      0.8445
   Metadata:
     • Person:        john
     • Experiment:    Member John - LR 5e-5
```

### Bước 5: Deploy Best Model

```bash
python model_registry.py deploy --model-id model_20260420_150000
```

---

## 🔧 Troubleshooting

### Problem 1: Training quá chậm

**Solution:**
```bash
# Giảm epochs
python train_with_args.py --epochs 1 --register-model

# Hoặc giảm batch size
python train_with_args.py --batch-size 8 --register-model

# Hoặc dùng model của team
# (Xin model từ người đã train)
```

### Problem 2: Không có model để predict

**Solution:**
```bash
# Option 1: Train model (chậm)
python train_with_args.py --epochs 2 --register-model

# Option 2: Xin model từ team
# Copy folder saved_model/ từ người khác
```

### Problem 3: Out of memory

**Solution:**
```bash
# Giảm batch size
python train_with_args.py --batch-size 4 --register-model

# Hoặc giảm max length
python train_with_args.py --max-length 128 --register-model
```

---

## 📋 Commands Tóm Tắt

### Prediction (Nhanh)
```bash
python predict.py
```

### Training (Chậm)
```bash
# Quick training (2 epochs)
python train_with_args.py --epochs 2 --register-model

# Full training (5 epochs)
python train_with_args.py --register-model
```

### Collaborative
```bash
# Merge data
python merge_data.py

# Training
python train_with_args.py --data data/master_dataset.csv --register-model

# View models
python model_registry.py list

# Deploy
python model_registry.py deploy --model-id <model_id>
```

---

## 🎯 Khuyến Nghị

### Nếu bạn muốn:

**1. Test nhanh (30 giây)**
→ Dùng model của team + `python predict.py`

**2. Train model (30-60 phút)**
→ `python train_with_args.py --epochs 2 --register-model`

**3. Collaborative training**
→ Đọc `HUONG_DAN_CHO_THANH_VIEN.md`

---

## 📚 Đọc Thêm

- **README.md** - Project overview
- **HUONG_DAN_CHO_THANH_VIEN.md** - Hướng dẫn chi tiết cho team

---

**Created:** 2026-04-20  
**Version:** 1.0
