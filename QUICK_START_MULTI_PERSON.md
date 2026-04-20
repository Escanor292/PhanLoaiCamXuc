# Quick Start Guide - Multi-Person Training

Hướng dẫn nhanh để nhiều người có thể training cùng lúc.

## 🚀 Setup Nhanh (5 phút)

### Bước 1: Clone Project

```bash
# Clone repository
git clone https://github.com/your-username/emotion-classification.git
cd emotion-classification

# Tạo virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Bước 2: Chuẩn Bị Dữ Liệu

**Option A: Sử dụng sample data có sẵn**
```bash
# Data đã có sẵn trong data/sample_comments.csv
# Có thể dùng luôn để test
```

**Option B: Sử dụng dữ liệu riêng**
```bash
# Copy dữ liệu của bạn vào thư mục data/
cp /path/to/your/data.csv data/my_data.csv
```

### Bước 3: Chạy Training

**Cách 1: Training đơn giản (dùng config mặc định)**
```bash
python train.py
```

**Cách 2: Training với custom config**
```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output saved_model/my_experiment/ \
    --experiment-name "My First Experiment" \
    --epochs 5 \
    --batch-size 16 \
    --lr 2e-5
```

## 👥 Scenarios Cho Nhiều Người

### Scenario 1: Mỗi Người Training Trên Máy Riêng

**Người A:**
```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_a/exp001_baseline/ \
    --experiment-name "Person A - Baseline" \
    --epochs 5
```

**Người B:**
```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_b/exp001_higher_lr/ \
    --experiment-name "Person B - Higher LR" \
    --epochs 5 \
    --lr 5e-5
```

**Người C:**
```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_c/exp001_larger_batch/ \
    --experiment-name "Person C - Larger Batch" \
    --epochs 5 \
    --batch-size 32
```

### Scenario 2: Chia Dữ Liệu

**Bước 1: Chia dữ liệu**
```python
# split_data.py
import pandas as pd

df = pd.read_csv('data/all_comments.csv')
n_parts = 3
part_size = len(df) // n_parts

for i in range(n_parts):
    start = i * part_size
    end = (i + 1) * part_size if i < n_parts - 1 else len(df)
    df_part = df.iloc[start:end]
    df_part.to_csv(f'data/part_{i+1}.csv', index=False)
```

**Bước 2: Mỗi người train trên 1 phần**
```bash
# Người 1
python train_with_args.py --data data/part_1.csv --output saved_model/part1/

# Người 2
python train_with_args.py --data data/part_2.csv --output saved_model/part2/

# Người 3
python train_with_args.py --data data/part_3.csv --output saved_model/part3/
```

### Scenario 3: Thử Nghiệm Hyperparameters

**Grid Search Manual:**

```bash
# Experiment 1: LR = 1e-5
python train_with_args.py --lr 1e-5 --output experiments/lr_1e5/

# Experiment 2: LR = 2e-5
python train_with_args.py --lr 2e-5 --output experiments/lr_2e5/

# Experiment 3: LR = 5e-5
python train_with_args.py --lr 5e-5 --output experiments/lr_5e5/

# Experiment 4: Batch Size = 8
python train_with_args.py --batch-size 8 --output experiments/bs_8/

# Experiment 5: Batch Size = 32
python train_with_args.py --batch-size 32 --output experiments/bs_32/
```

## 📊 So Sánh Kết Quả

### Cách 1: Xem Results File

```bash
# Xem kết quả của experiment
cat experiments/person_a/exp001_baseline/results.txt
```

### Cách 2: So Sánh Nhiều Experiments

```bash
# Tạo script compare.py
python compare_experiments.py \
    experiments/person_a/exp001_baseline/ \
    experiments/person_b/exp001_higher_lr/ \
    experiments/person_c/exp001_larger_batch/
```

### Cách 3: Update Experiment Log

```bash
# Thêm kết quả vào experiment_log.md
nano experiments/experiment_log.md
```

## 🔄 Workflow Đề Xuất

### Daily Workflow

```
1. Morning (9:00 AM)
   ├── Pull latest code: git pull
   ├── Check experiment_log.md
   └── Plan experiments for the day

2. Training (9:30 AM - 5:00 PM)
   ├── Run experiments
   ├── Monitor training
   └── Document results

3. End of Day (5:00 PM)
   ├── Update experiment_log.md
   ├── Commit results (not models)
   ├── Share insights with team
   └── Plan next day experiments
```

### Weekly Workflow

```
Monday:
  - Team meeting: Review last week results
  - Plan experiments for the week
  - Assign tasks

Tuesday-Thursday:
  - Run experiments
  - Daily standup (15 min)
  - Share progress

Friday:
  - Compare all experiments
  - Select best model
  - Document findings
  - Plan next week
```

## 💡 Tips & Tricks

### Tip 1: Đặt Tên Experiment Rõ Ràng

❌ Bad:
```bash
--experiment-name "test1"
--output saved_model/model1/
```

✅ Good:
```bash
--experiment-name "PersonA_Baseline_LR2e5_BS16"
--output experiments/person_a/exp001_baseline_lr2e5_bs16/
```

### Tip 2: Luôn Set Random Seed

```bash
python train_with_args.py --seed 42
```

### Tip 3: Monitor Training

```bash
# Sử dụng tensorboard (nếu có)
tensorboard --logdir experiments/

# Hoặc watch training log
tail -f experiments/person_a/exp001/training.log
```

### Tip 4: Save Disk Space

```bash
# Chỉ giữ best model, xóa intermediate checkpoints
# Thêm vào .gitignore
saved_model/*/checkpoint_epoch_*.pt
```

### Tip 5: Backup Models

```bash
# Backup lên Google Drive hoặc cloud storage
rclone copy experiments/ gdrive:emotion-models/
```

## 🐛 Troubleshooting

### Problem: Out of Memory

**Solution:**
```bash
# Giảm batch size
python train_with_args.py --batch-size 8

# Hoặc giảm max length
python train_with_args.py --max-length 256
```

### Problem: Training Quá Chậm

**Solution:**
```bash
# Check device
python -c "import torch; print(torch.cuda.is_available())"

# Nếu có GPU nhưng không dùng
python train_with_args.py --device cuda

# Nếu không có GPU, giảm data
python train_with_args.py --data data/small_sample.csv
```

### Problem: Model Không Học

**Solution:**
```bash
# Thử learning rate cao hơn
python train_with_args.py --lr 5e-5

# Hoặc train lâu hơn
python train_with_args.py --epochs 10
```

### Problem: Overfitting

**Solution:**
```bash
# Tăng dropout
python train_with_args.py --dropout 0.5

# Hoặc thêm data
# Hoặc early stopping (đã có sẵn trong code)
```

## 📝 Checklist Trước Khi Training

- [ ] Đã pull latest code từ Git
- [ ] Đã activate virtual environment
- [ ] Đã check data có đúng format không
- [ ] Đã tạo thư mục output
- [ ] Đã set experiment name rõ ràng
- [ ] Đã set random seed
- [ ] Đã check GPU available (nếu có)
- [ ] Đã document objective trong experiment log

## 📝 Checklist Sau Khi Training

- [ ] Đã check results.txt
- [ ] Đã xem training curves
- [ ] Đã update experiment_log.md
- [ ] Đã commit code changes (không commit models)
- [ ] Đã share insights với team
- [ ] Đã backup model (nếu cần)
- [ ] Đã plan next experiment

## 🎯 Example Commands

### Training Cơ Bản
```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output saved_model/baseline/ \
    --epochs 5
```

### Training Với Full Options
```bash
python train_with_args.py \
    --data data/my_data.csv \
    --output experiments/person_a/exp001/ \
    --experiment-name "PersonA_Exp001_Baseline" \
    --model-name bert-base-uncased \
    --epochs 10 \
    --batch-size 32 \
    --lr 2e-5 \
    --dropout 0.3 \
    --max-length 512 \
    --seed 42 \
    --device cuda
```

### Training Nhanh (Test)
```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output test_output/ \
    --epochs 1 \
    --batch-size 8
```

## 📚 Tài Liệu Tham Khảo

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Hướng dẫn chi tiết
- [experiments/README.md](experiments/README.md) - Experiment tracking
- [README.md](README.md) - Project documentation

## 🆘 Cần Giúp Đỡ?

1. Check [README.md](README.md) troubleshooting section
2. Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
3. Hỏi trong team chat
4. Tạo issue trên GitHub

---

**Happy Training! 🚀**

Nếu có câu hỏi, liên hệ team lead hoặc tạo issue trên GitHub.
