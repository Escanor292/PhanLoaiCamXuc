# Experiment Log

Log tất cả experiments của team. Mỗi experiment mới thêm vào đầu file.

---

## Template

```markdown
## ExpXXX - [Title] ([Person])
- **Date:** YYYY-MM-DD
- **Status:** ✅ Completed / 🔄 Running / ❌ Failed / 📋 Planned
- **Objective:** [Mục đích]
- **Key Changes:** [Thay đổi chính so với baseline]
- **Results:**
  - Macro F1: X.XXX
  - Micro F1: X.XXX
  - Val Loss: X.XXX
- **Insights:** [Insights quan trọng]
- **Path:** experiments/person_x/expXXX_name/
- **Next:** [Experiment tiếp theo]
```

---

## Experiments

### Baseline Experiments

## Exp001 - Baseline Model (Person A)
- **Date:** 2026-04-20
- **Status:** 📋 Planned
- **Objective:** Tạo baseline model với default hyperparameters
- **Configuration:**
  - Model: bert-base-uncased
  - Data: 100 samples
  - Epochs: 5
  - Batch Size: 16
  - Learning Rate: 2e-5
- **Results:** TBD
- **Path:** experiments/person_a/exp001_baseline/

---

### Hyperparameter Tuning

_Experiments về tuning hyperparameters sẽ được thêm vào đây_

---

### Data Experiments

_Experiments về data augmentation, cleaning, etc. sẽ được thêm vào đây_

---

### Model Architecture

_Experiments về thay đổi model architecture sẽ được thêm vào đây_

---

### Failed Experiments

_Lưu lại các experiments failed để học từ mistakes_

---

## Summary Statistics

| Metric | Best Value | Experiment | Person |
|--------|------------|------------|--------|
| Macro F1 | TBD | - | - |
| Micro F1 | TBD | - | - |
| Val Loss | TBD | - | - |

## Lessons Learned

### What Works
- TBD

### What Doesn't Work
- TBD

### Best Practices Discovered
- TBD

---

**Last Updated:** 2026-04-20
