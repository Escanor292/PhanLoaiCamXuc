# Experiment Tracking

Thư mục này dùng để track các experiments của từng người trong team.

## Cấu Trúc

```
experiments/
├── README.md                    # File này
├── experiment_log.md            # Log tất cả experiments
├── person_a/                    # Experiments của người A
│   ├── exp001_baseline/
│   ├── exp002_tuning/
│   └── ...
├── person_b/                    # Experiments của người B
│   ├── exp001_baseline/
│   └── ...
└── shared/                      # Experiments chung
    └── best_models/
```

## Quy Tắc Đặt Tên

### Experiment Folders
```
expXXX_description/
```

Ví dụ:
- `exp001_baseline` - Baseline model
- `exp002_higher_lr` - Thử learning rate cao hơn
- `exp003_multilingual` - Model đa ngôn ngữ

### Model Checkpoints
```
model_v{version}_{description}.pt
```

Ví dụ:
- `model_v1.0_baseline.pt`
- `model_v1.1_improved.pt`
- `model_v2.0_final.pt`

## Cách Sử Dụng

### 1. Tạo Experiment Mới

```bash
# Tạo thư mục cho experiment
mkdir -p experiments/person_a/exp001_baseline

# Chạy training với output vào thư mục đó
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_a/exp001_baseline/ \
    --experiment-name "Person A - Baseline" \
    --epochs 5 \
    --batch-size 16 \
    --lr 2e-5
```

### 2. Document Experiment

Tạo file `experiments/person_a/exp001_baseline/README.md`:

```markdown
# Experiment 001 - Baseline

## Thông Tin
- **Người thực hiện:** Người A
- **Ngày:** 2026-04-20
- **Mục đích:** Tạo baseline model để so sánh

## Configuration
- Data: data/sample_comments.csv (100 samples)
- Model: bert-base-uncased
- Epochs: 5
- Batch Size: 16
- Learning Rate: 2e-5
- Dropout: 0.3

## Kết Quả
- Train Loss: 0.234
- Val Loss: 0.289
- Test Macro F1: 0.756
- Test Micro F1: 0.812

## Nhận Xét
- Model học tốt, không có overfitting
- F1 score khá cao cho baseline
- Có thể cải thiện bằng cách tăng data

## Next Steps
- Thử tăng learning rate
- Thử data augmentation
```

### 3. Update Experiment Log

Thêm vào `experiments/experiment_log.md`:

```markdown
## Experiment 001 - Baseline (Person A)
- **Date:** 2026-04-20
- **Status:** ✅ Completed
- **Results:** Macro F1 = 0.756
- **Notes:** Good baseline, ready for improvements
- **Path:** experiments/person_a/exp001_baseline/
```

## Template Cho Experiment

```markdown
# Experiment XXX - [Description]

## Metadata
- **Experimenter:** [Tên]
- **Date:** [YYYY-MM-DD]
- **Status:** [Planning/Running/Completed/Failed]
- **Parent Experiment:** [exp_id nếu có]

## Objective
[Mục đích của experiment này]

## Hypothesis
[Giả thuyết: Nếu làm X thì Y sẽ cải thiện]

## Configuration
```yaml
data:
  path: data/sample_comments.csv
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

model:
  name: bert-base-uncased
  dropout: 0.3
  max_length: 512

training:
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  optimizer: AdamW
  seed: 42
```

## Results
| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Loss | 0.234 | 0.289 | 0.301 |
| Micro F1 | 0.856 | 0.812 | 0.798 |
| Macro F1 | 0.789 | 0.756 | 0.742 |
| Hamming Loss | 0.045 | 0.052 | 0.058 |

### Per-Label F1 Scores
| Emotion | F1 Score |
|---------|----------|
| joy | 0.823 |
| trust | 0.756 |
| fear | 0.689 |
| ... | ... |

## Analysis
### What Worked
- [Điều gì hoạt động tốt]

### What Didn't Work
- [Điều gì không hoạt động]

### Insights
- [Những insight thu được]

## Artifacts
- Model: `model_v1.0_baseline.pt`
- Training curves: `training_curves.png`
- Results: `results.txt`
- Logs: `training.log`

## Next Steps
- [ ] [Action item 1]
- [ ] [Action item 2]

## References
- [Link đến papers/resources liên quan]
```

## Best Practices

### 1. Luôn Document
- Document mọi experiment, kể cả failed experiments
- Failed experiments cũng có giá trị (biết được cái gì không work)

### 2. Reproducibility
- Lưu random seed
- Lưu exact configuration
- Lưu version của dependencies

### 3. Version Control
- Commit code trước khi chạy experiment
- Tag commit với experiment ID
```bash
git tag exp001_baseline
git push origin exp001_baseline
```

### 4. Compare Experiments
```bash
# So sánh 2 experiments
python compare_experiments.py \
    experiments/person_a/exp001_baseline/ \
    experiments/person_a/exp002_tuning/
```

### 5. Share Results
- Update experiment_log.md sau mỗi experiment
- Share insights trong team meetings
- Document lessons learned

## Experiment Checklist

Trước khi chạy experiment:
- [ ] Đã tạo thư mục experiment
- [ ] Đã document objective và hypothesis
- [ ] Đã set random seed
- [ ] Đã commit code changes
- [ ] Đã check data quality

Sau khi chạy experiment:
- [ ] Đã lưu model checkpoint
- [ ] Đã lưu training curves
- [ ] Đã document results
- [ ] Đã update experiment_log.md
- [ ] Đã share insights với team

## Tools

### MLflow (Optional)
```python
# Thêm vào train_with_args.py
import mlflow

mlflow.set_experiment(args.experiment_name)

with mlflow.start_run():
    mlflow.log_params({
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs
    })
    
    # ... training code ...
    
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_f1": test_f1
    })
    
    mlflow.pytorch.log_model(model, "model")
```

### Weights & Biases (Optional)
```python
import wandb

wandb.init(
    project="emotion-classification",
    name=args.experiment_name,
    config=vars(args)
)

# ... training code ...

wandb.log({
    "train_loss": train_loss,
    "val_loss": val_loss
})
```

## FAQ

**Q: Tôi nên chạy bao nhiêu experiments?**
A: Chạy đủ để test hypothesis của bạn. Thường 3-5 experiments cho mỗi idea.

**Q: Tôi có nên lưu tất cả model checkpoints?**
A: Chỉ lưu best model và một vài checkpoints quan trọng. Xóa các checkpoints trung gian.

**Q: Làm sao để compare experiments?**
A: Sử dụng experiment_log.md hoặc tools như MLflow/W&B.

**Q: Experiment failed, tôi có nên document không?**
A: Có! Failed experiments cũng quan trọng. Document lý do fail và lessons learned.

---

**Happy Experimenting! 🚀**
