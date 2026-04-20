# Collaborative Training Workflow - Góp Data + Train Cùng Nhau

## 🎯 Mục Tiêu

Mỗi người trong team:
1. **Góp data mới** của mình vào dataset chung
2. **Train model** với hyperparameter khác nhau trên dataset chung
3. **So sánh kết quả** và chọn model tốt nhất

## ⚠️ Nguyên Tắc Quan Trọng

### ❌ KHÔNG NÊN:
- ❌ Train chỉ trên data riêng của mình
- ❌ Train tiếp model với data cá nhân
- ❌ Bỏ qua data của người khác

### ✅ NÊN:
- ✅ Merge tất cả data vào dataset chung
- ✅ Train trên dataset chung (có data của tất cả mọi người)
- ✅ Mỗi người thử hyperparameter khác nhau
- ✅ So sánh kết quả trên cùng validation/test set

### 🚨 Tại Sao?

Nếu train riêng trên data cá nhân:
- **Lệch nhãn**: Mỗi người label khác nhau
- **Overfitting**: Model chỉ tốt trên data của 1 người
- **Quên data cũ**: Model mất khả năng predict data trước đó
- **Bias**: Model bị thiên lệch theo style của 1 người

---

## 📋 Workflow Chi Tiết

### Bước 1: Chốt Base Model Chung

**Lần đầu tiên:**
```bash
# Train base model với sample data
python train_with_args.py \
    --data data/sample_comments.csv \
    --output models/base_model_v1/ \
    --experiment-name "Base Model v1" \
    --epochs 5 \
    --register-model

# Đây sẽ là base model cho tất cả mọi người
```

**Kết quả:**
```
models/base_model_v1/
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── training_config.json
```

**Share với team:**
```bash
# Option 1: Git LFS (recommended)
git lfs track "models/base_model_v1/*"
git add models/base_model_v1/
git commit -m "Add base model v1"
git push

# Option 2: Google Drive / Dropbox
# Upload folder models/base_model_v1/ lên cloud

# Option 3: Shared network drive
# Copy vào shared folder
```

---

### Bước 2: Mỗi Người Góp Data Mới

#### 2.1. Chuẩn Bị Data Cá Nhân

**Member 1:**
```bash
# Tạo file data của mình
# data/member1_data.csv (200 comments)
```

**Member 2:**
```bash
# data/member2_data.csv (150 comments)
```

**Member 3:**
```bash
# data/member3_data.csv (300 comments)
```

**Format data:**
```csv
text,joy,trust,fear,surprise,sadness,disgust,anger,anticipation,love,worried,disappointed,proud,embarrassed,jealous,calm,excited
"Comment của member 1",1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1
"Comment của member 2",0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0
```

#### 2.2. Merge Data Vào Dataset Chung

**Tạo script merge data:**

```python
# merge_data.py
import pandas as pd
import os

def merge_datasets(data_files, output_file):
    """
    Merge multiple CSV files into one master dataset.
    
    Args:
        data_files: List of CSV file paths
        output_file: Output master dataset path
    """
    print("="*70)
    print("MERGING DATASETS")
    print("="*70)
    
    all_data = []
    
    for file in data_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            print(f"✓ Loaded {file}: {len(df)} samples")
            all_data.append(df)
        else:
            print(f"⚠ Warning: {file} not found, skipping")
    
    # Merge all dataframes
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates (based on text)
    original_len = len(master_df)
    master_df = master_df.drop_duplicates(subset=['text'], keep='first')
    duplicates_removed = original_len - len(master_df)
    
    # Save master dataset
    master_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print("MERGE COMPLETE")
    print("="*70)
    print(f"Total samples: {len(master_df)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Output: {output_file}")
    print("="*70)
    
    return master_df

if __name__ == "__main__":
    # List all data files
    data_files = [
        'data/sample_comments.csv',      # Original sample data
        'data/member1_data.csv',         # Member 1's data
        'data/member2_data.csv',         # Member 2's data
        'data/member3_data.csv',         # Member 3's data
    ]
    
    # Merge into master dataset
    master_df = merge_datasets(
        data_files=data_files,
        output_file='data/master_dataset.csv'
    )
    
    # Show statistics
    print(f"\nDataset Statistics:")
    print(f"  Total comments: {len(master_df)}")
    
    # Count samples per emotion
    emotion_cols = [col for col in master_df.columns if col != 'text']
    print(f"\nSamples per emotion:")
    for emotion in emotion_cols:
        count = master_df[emotion].sum()
        print(f"  {emotion:15s}: {count:4d} ({count/len(master_df)*100:.1f}%)")
```

**Chạy merge:**
```bash
python merge_data.py
```

**Kết quả:**
```
data/master_dataset.csv  # 750 comments (100 + 200 + 150 + 300)
```

#### 2.3. Share Master Dataset

```bash
# Commit master dataset
git add data/master_dataset.csv
git commit -m "Update master dataset with new contributions"
git push

# Hoặc upload lên shared drive
```

---

### Bước 3: Mỗi Người Train Với Config Khác Nhau

**Quan trọng:** Tất cả đều train trên **cùng master dataset**, chỉ khác **hyperparameters**.

#### Member 1: Thử Learning Rate Cao

```bash
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/member1_lr5e5/ \
    --experiment-name "Member 1 - LR 5e-5" \
    --lr 5e-5 \
    --epochs 5 \
    --batch-size 16 \
    --register-model
```

#### Member 2: Thử Learning Rate Thấp

```bash
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/member2_lr1e5/ \
    --experiment-name "Member 2 - LR 1e-5" \
    --lr 1e-5 \
    --epochs 5 \
    --batch-size 16 \
    --register-model
```

#### Member 3: Thử Batch Size Lớn

```bash
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/member3_bs32/ \
    --experiment-name "Member 3 - Batch Size 32" \
    --lr 2e-5 \
    --epochs 5 \
    --batch-size 32 \
    --register-model
```

#### Member 4: Thử Nhiều Epochs

```bash
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/member4_ep10/ \
    --experiment-name "Member 4 - 10 Epochs" \
    --lr 2e-5 \
    --epochs 10 \
    --batch-size 16 \
    --register-model
```

#### Member 5: Thử Dropout Cao

```bash
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/member5_drop05/ \
    --experiment-name "Member 5 - Dropout 0.5" \
    --lr 2e-5 \
    --epochs 5 \
    --batch-size 16 \
    --dropout 0.5 \
    --register-model
```

---

### Bước 4: So Sánh Kết Quả

#### 4.1. Xem Tất Cả Models

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
     • Test Loss:     0.2145
     • Hamming Loss:  0.0723
   Metadata:
     • Person:        member2
     • Experiment:    Member 2 - LR 1e-5
     • Learning Rate: 1e-05
     • Batch Size:    16
     • Epochs:        5

📦 2. model_20260420_143000 [REGISTERED]
   ────────────────────────────────────────────────────────────────────────────
   Metrics:
     • Macro F1:      0.8123
     • Micro F1:      0.8445
     • Test Loss:     0.2234
     • Hamming Loss:  0.0756
   Metadata:
     • Person:        member4
     • Experiment:    Member 4 - 10 Epochs
     • Learning Rate: 2e-05
     • Batch Size:    16
     • Epochs:        10

📦 3. model_20260420_140000 [REGISTERED]
   ────────────────────────────────────────────────────────────────────────────
   Metrics:
     • Macro F1:      0.8012
     • Micro F1:      0.8334
     • Test Loss:     0.2345
     • Hamming Loss:  0.0789
   Metadata:
     • Person:        member1
     • Experiment:    Member 1 - LR 5e-5
     • Learning Rate: 5e-05
     • Batch Size:    16
     • Epochs:        5
```

#### 4.2. So Sánh Chi Tiết

```bash
# Tạo comparison report
python compare_experiments.py \
    experiments/member1_lr5e5/ \
    experiments/member2_lr1e5/ \
    experiments/member3_bs32/ \
    experiments/member4_ep10/ \
    experiments/member5_drop05/
```

**Output:**
```
================================================================================
EXPERIMENT COMPARISON
================================================================================

Experiment                    | Macro F1 | Micro F1 | Test Loss | Config
------------------------------|----------|----------|-----------|------------------
Member 2 - LR 1e-5           | 0.8234   | 0.8567   | 0.2145    | lr=1e-5, bs=16
Member 4 - 10 Epochs         | 0.8123   | 0.8445   | 0.2234    | lr=2e-5, ep=10
Member 1 - LR 5e-5           | 0.8012   | 0.8334   | 0.2345    | lr=5e-5, bs=16
Member 3 - Batch Size 32     | 0.7956   | 0.8289   | 0.2456    | lr=2e-5, bs=32
Member 5 - Dropout 0.5       | 0.7845   | 0.8123   | 0.2567    | lr=2e-5, drop=0.5

================================================================================
BEST MODEL: Member 2 - LR 1e-5
Macro F1: 0.8234
Improvement over baseline: +8.2%
================================================================================
```

---

### Bước 5: Deploy Model Tốt Nhất

#### 5.1. Manual Deploy

```bash
# Deploy model của Member 2 (best)
python model_registry.py deploy --model-id model_20260420_150000
```

#### 5.2. Auto Deploy

```bash
# Enable auto-deploy
export AUTO_DEPLOY=true

# Model tốt nhất sẽ tự động deploy khi training
```

---

## 📊 Ví Dụ Thực Tế

### Scenario: Team 5 Người

**Tuần 1:**

**Monday:**
- Member 1: Label 200 comments → `data/member1_week1.csv`
- Member 2: Label 150 comments → `data/member2_week1.csv`
- Member 3: Label 300 comments → `data/member3_week1.csv`
- Member 4: Label 100 comments → `data/member4_week1.csv`
- Member 5: Label 250 comments → `data/member5_week1.csv`

**Tuesday:**
```bash
# Merge data
python merge_data.py

# Result: data/master_dataset.csv (1000 comments)
```

**Wednesday - Friday:**
```bash
# Member 1: Train với lr=5e-5
python train_with_args.py --data data/master_dataset.csv --lr 5e-5 --register-model

# Member 2: Train với lr=1e-5
python train_with_args.py --data data/master_dataset.csv --lr 1e-5 --register-model

# Member 3: Train với batch-size=32
python train_with_args.py --data data/master_dataset.csv --batch-size 32 --register-model

# Member 4: Train với epochs=10
python train_with_args.py --data data/master_dataset.csv --epochs 10 --register-model

# Member 5: Train với dropout=0.5
python train_with_args.py --data data/master_dataset.csv --dropout 0.5 --register-model
```

**Friday EOD:**
```bash
# So sánh kết quả
python model_registry.py list

# Deploy best model
python model_registry.py deploy --model-id <best_model_id>
```

**Tuần 2:**

Lặp lại với data mới:
- Merge data tuần 2 vào master dataset
- Train lại với configs khác
- So sánh và deploy

---

## 🔧 Tools Hỗ Trợ

### 1. Script Merge Data

Đã tạo sẵn: `merge_data.py`

```bash
# Sử dụng
python merge_data.py
```

### 2. Script So Sánh Experiments

Đã tạo sẵn: `compare_experiments.py`

```bash
# Sử dụng
python compare_experiments.py experiments/*/
```

### 3. Model Registry

Đã tích hợp sẵn:

```bash
# List models
python model_registry.py list

# Deploy model
python model_registry.py deploy --model-id <model_id>

# Check best model
python model_registry.py best
```

---

## 📋 Checklist Cho Mỗi Vòng Training

### Trước Khi Training

- [ ] Tất cả members đã góp data mới
- [ ] Data đã được merge vào master dataset
- [ ] Master dataset đã được validate (format, duplicates)
- [ ] Master dataset đã được share với team
- [ ] Mỗi người đã chọn hyperparameter khác nhau

### Trong Quá Trình Training

- [ ] Tất cả đều train trên cùng master dataset
- [ ] Sử dụng `--register-model` flag
- [ ] Đặt tên experiment rõ ràng
- [ ] Monitor training progress

### Sau Khi Training

- [ ] So sánh kết quả trên registry
- [ ] Phân tích model nào tốt nhất và tại sao
- [ ] Deploy model tốt nhất
- [ ] Document kết quả và insights
- [ ] Share findings với team

---

## 🎯 Best Practices

### 1. Data Management

✅ **DO:**
- Merge tất cả data vào master dataset
- Remove duplicates
- Validate data format
- Version control master dataset
- Backup data thường xuyên

❌ **DON'T:**
- Train trên data riêng
- Bỏ qua data của người khác
- Không check duplicates
- Không validate format

### 2. Hyperparameter Search

✅ **DO:**
- Mỗi người thử config khác nhau
- Document config đã thử
- Systematic search (grid search hoặc random search)
- Share insights về config tốt

❌ **DON'T:**
- Tất cả dùng cùng config
- Random thử không có kế hoạch
- Không document kết quả
- Không share findings

### 3. Model Selection

✅ **DO:**
- So sánh trên cùng validation/test set
- Xem nhiều metrics (macro F1, micro F1, per-label F1)
- Consider tradeoffs (accuracy vs speed)
- Test model trên real data

❌ **DON'T:**
- Chỉ xem 1 metric
- Không test trên real data
- Deploy mà không validate
- Không backup model cũ

---

## 🚀 Quick Start

### Lần Đầu Tiên

```bash
# 1. Train base model
python train_with_args.py \
    --data data/sample_comments.csv \
    --output models/base_model_v1/ \
    --experiment-name "Base Model v1" \
    --register-model

# 2. Share base model với team
git add models/base_model_v1/
git commit -m "Add base model v1"
git push
```

### Mỗi Vòng Training

```bash
# 1. Góp data
# Tạo data/memberX_data.csv

# 2. Merge data
python merge_data.py

# 3. Train với config khác nhau
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/my_exp/ \
    --experiment-name "My Experiment" \
    --lr 2e-5 \
    --register-model

# 4. So sánh kết quả
python model_registry.py list

# 5. Deploy best model
python model_registry.py deploy --model-id <best_model_id>
```

---

## 📚 Tài Liệu Liên Quan

- **DEPLOYMENT_GUIDE.md** - Multi-person training setup
- **CONTINUOUS_LEARNING_GUIDE.md** - Advanced training methods
- **MODEL_REGISTRY_INTEGRATION.md** - Model registry details
- **AUTO_UPGRADE_SUMMARY.md** - Auto-upgrade guide

---

## 💡 Tips & Tricks

### Tip 1: Systematic Hyperparameter Search

```bash
# Grid search example
for lr in 1e-5 2e-5 5e-5; do
    for bs in 16 32; do
        python train_with_args.py \
            --data data/master_dataset.csv \
            --output experiments/lr${lr}_bs${bs}/ \
            --experiment-name "LR ${lr} BS ${bs}" \
            --lr $lr \
            --batch-size $bs \
            --register-model
    done
done
```

### Tip 2: Parallel Training

```bash
# Member 1
python train_with_args.py --lr 1e-5 --register-model &

# Member 2
python train_with_args.py --lr 2e-5 --register-model &

# Member 3
python train_with_args.py --lr 5e-5 --register-model &

# Wait for all to complete
wait
```

### Tip 3: Auto-Deploy Best Model

```bash
# Enable auto-deploy
export AUTO_DEPLOY=true

# Bây giờ model tốt nhất sẽ tự động deploy
python train_with_args.py --register-model ...
```

---

**Created:** 2026-04-20  
**Version:** 1.0  
**Status:** Complete ✅
