# Quick Start: Collaborative Training

## 🚀 5 Bước Đơn Giản

### Bước 1: Mỗi Người Tạo Data Của Mình

```bash
# Member 1
# Tạo file: data/member1_data.csv

# Member 2
# Tạo file: data/member2_data.csv

# Member 3
# Tạo file: data/member3_data.csv
```

**Format:**
```csv
text,joy,trust,fear,surprise,sadness,disgust,anger,anticipation,love,worried,disappointed,proud,embarrassed,jealous,calm,excited
"Comment 1",1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1
"Comment 2",0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0
```

---

### Bước 2: Merge Tất Cả Data

```bash
python merge_data.py
```

**Output:**
```
======================================================================
MERGING DATASETS
======================================================================
✓ Loaded data/sample_comments.csv: 100 samples
✓ Loaded data/member1_data.csv: 200 samples
✓ Loaded data/member2_data.csv: 150 samples
✓ Loaded data/member3_data.csv: 300 samples

Merging 4 files...
✓ Removed 5 duplicate texts

======================================================================
MERGE COMPLETE
======================================================================
Total samples loaded: 750
Duplicates removed: 5
Final dataset size: 745
Output: data/master_dataset.csv
======================================================================
```

---

### Bước 3: Mỗi Người Train Với Config Khác Nhau

**Member 1:**
```bash
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/member1_lr5e5/ \
    --experiment-name "Member 1 - LR 5e-5" \
    --lr 5e-5 \
    --register-model
```

**Member 2:**
```bash
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/member2_lr1e5/ \
    --experiment-name "Member 2 - LR 1e-5" \
    --lr 1e-5 \
    --register-model
```

**Member 3:**
```bash
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/member3_bs32/ \
    --experiment-name "Member 3 - Batch Size 32" \
    --batch-size 32 \
    --register-model
```

---

### Bước 4: Xem Kết Quả

```bash
python model_registry.py list
```

**Output:**
```
================================================================================
MODEL REGISTRY - Top 10 Models (sorted by macro_f1)
================================================================================

⭐ 1. model_20260420_150000 [BEST]
   ────────────────────────────────────────────────────────────────────────────
   Metrics:
     • Macro F1:      0.8234
     • Micro F1:      0.8567
   Metadata:
     • Person:        member2
     • Experiment:    Member 2 - LR 1e-5

📦 2. model_20260420_143000 [REGISTERED]
   ────────────────────────────────────────────────────────────────────────────
   Metrics:
     • Macro F1:      0.8123
     • Micro F1:      0.8445
   Metadata:
     • Person:        member1
     • Experiment:    Member 1 - LR 5e-5
```

---

### Bước 5: Deploy Model Tốt Nhất

```bash
python model_registry.py deploy --model-id model_20260420_150000
```

**Hoặc enable auto-deploy:**
```bash
export AUTO_DEPLOY=true
# Model tốt nhất sẽ tự động deploy
```

---

## ✅ Checklist

- [ ] Mỗi người đã tạo data của mình
- [ ] Data đã được merge vào master dataset
- [ ] Mỗi người train với config khác nhau
- [ ] Tất cả đều train trên cùng master dataset
- [ ] Đã so sánh kết quả
- [ ] Đã deploy model tốt nhất

---

## 💡 Tips

### Tip 1: Validate Data Trước Khi Merge

```bash
# Check format
python -c "import pandas as pd; df = pd.read_csv('data/member1_data.csv'); print(df.head())"
```

### Tip 2: Backup Master Dataset

```bash
# Backup trước khi merge mới
cp data/master_dataset.csv data/master_dataset_backup.csv
```

### Tip 3: Parallel Training

```bash
# Chạy nhiều training cùng lúc
python train_with_args.py --lr 1e-5 --register-model &
python train_with_args.py --lr 2e-5 --register-model &
python train_with_args.py --lr 5e-5 --register-model &
wait
```

---

## 🆘 Troubleshooting

### Problem: Merge data bị lỗi

**Solution:**
```bash
# Check format của từng file
python merge_data.py --no-validate --files data/member1_data.csv
```

### Problem: Training bị lỗi

**Solution:**
```bash
# Check master dataset
python -c "import pandas as pd; df = pd.read_csv('data/master_dataset.csv'); print(df.info())"
```

---

## 📚 Đọc Thêm

- **COLLABORATIVE_TRAINING_WORKFLOW.md** - Hướng dẫn chi tiết
- **DEPLOYMENT_GUIDE.md** - Multi-person training setup
- **MODEL_REGISTRY_INTEGRATION.md** - Model registry details

---

**Created:** 2026-04-20  
**Version:** 1.0
