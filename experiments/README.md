# 🔬 Experiment Tracking

Hướng dẫn theo dõi và quản lý các experiments trong dự án.

---

## 📁 Cấu Trúc Thư Mục

```
experiments/
├── README.md                    # File này
├── experiment_log.md            # Log tất cả experiments
├── member_an/                   # Experiments của An
│   ├── exp001_baseline/
│   └── exp002_tuning/
├── member_dat/                  # Experiments của Dat
└── shared/                      # Experiments chung
    └── best_models/
```

---

## 🎯 Quick Start

### 1. Tạo Experiment Mới

```bash
# Tạo thư mục
mkdir -p experiments/member_an/exp001_baseline

# Training với experiment tracking
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/member_an/exp001_baseline/ \
    --experiment-name "An - Baseline Model" \
    --epochs 5 \
    --batch-size 16
```

### 2. Document Experiment

Tạo `experiments/member_an/exp001_baseline/README.md`:

```markdown
# Experiment 001 - Baseline

**Người thực hiện:** An  
**Ngày:** 2026-04-25  
**Mục đích:** Tạo baseline model

## Configuration
- Data: 100 samples
- Model: PhoBERT Hybrid
- Epochs: 5
- Batch Size: 16
- Learning Rate: 2e-5

## Results
- Macro F1: 0.756
- Micro F1: 0.812
- Test Loss: 0.289

## Insights
- Model học tốt, không overfitting
- Có thể cải thiện bằng data augmentation
```

### 3. Update Experiment Log

Thêm vào `experiments/experiment_log.md`:

```markdown
## Exp001 - Baseline (An)
- **Date:** 2026-04-25
- **Status:** ✅ Completed
- **Results:** Macro F1 = 0.756
- **Path:** experiments/member_an/exp001_baseline/
```

---

## 📝 Quy Tắc Đặt Tên

### Experiment Folders
```
expXXX_description/
```

**Ví dụ:**
- `exp001_baseline` - Baseline model
- `exp002_higher_lr` - Thử learning rate cao
- `exp003_data_aug` - Data augmentation

### Model Checkpoints
```
model_v{version}_{description}.pt
```

**Ví dụ:**
- `model_v1.0_baseline.pt`
- `model_v1.1_improved.pt`

---

## 📋 Template Experiment README

```markdown
# Experiment XXX - [Description]

**Experimenter:** [Tên]  
**Date:** [YYYY-MM-DD]  
**Status:** [Planning/Running/Completed/Failed]

## Objective
[Mục đích của experiment]

## Hypothesis
[Giả thuyết: Nếu làm X thì Y sẽ cải thiện]

## Configuration
- Data: [path và số lượng samples]
- Model: PhoBERT Hybrid
- Epochs: [số]
- Batch Size: [số]
- Learning Rate: [số]
- Other params: [...]

## Results
| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Loss | X.XXX | X.XXX | X.XXX |
| Macro F1 | X.XXX | X.XXX | X.XXX |
| Micro F1 | X.XXX | X.XXX | X.XXX |

## Analysis
### What Worked
- [Điều gì hoạt động tốt]

### What Didn't Work
- [Điều gì không hoạt động]

### Insights
- [Những insight thu được]

## Next Steps
- [ ] [Action item 1]
- [ ] [Action item 2]
```

---

## ✅ Best Practices

### 1. Luôn Document
- Document mọi experiment, kể cả failed
- Failed experiments cũng có giá trị

### 2. Reproducibility
- Lưu random seed
- Lưu exact configuration
- Lưu version dependencies

### 3. Compare Experiments
```bash
python compare_experiments.py \
    experiments/member_an/exp001_baseline/ \
    experiments/member_an/exp002_tuning/
```

### 4. Share Results
- Update `experiment_log.md` sau mỗi experiment
- Share insights với team

## 📋 Experiment Checklist

**Trước khi chạy:**
- [ ] Tạo thư mục experiment
- [ ] Document objective
- [ ] Set random seed
- [ ] Check data quality

**Sau khi chạy:**
- [ ] Lưu model checkpoint
- [ ] Document results
- [ ] Update experiment_log.md
- [ ] Share insights với team

---

## ❓ FAQ

**Q: Tôi nên chạy bao nhiêu experiments?**  
A: 3-5 experiments cho mỗi idea để test hypothesis.

**Q: Có nên lưu tất cả checkpoints?**  
A: Chỉ lưu best model. Xóa checkpoints trung gian.

**Q: Experiment failed, có nên document?**  
A: Có! Document lý do fail và lessons learned.

---

**Happy Experimenting! 🚀**
