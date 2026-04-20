# Model Registry Integration - Complete ✅

## Tóm Tắt

Hệ thống **Model Registry** đã được tích hợp hoàn chỉnh vào `train_with_args.py`. Bây giờ nhiều người có thể training và model chính sẽ **tự động nâng cấp** khi có model tốt hơn.

## ✅ Đã Hoàn Thành

### 1. Model Registry System (`model_registry.py`)
- ✅ Track tất cả trained models với metrics
- ✅ Tự động identify best model (by macro F1)
- ✅ Auto-deployment khi `AUTO_DEPLOY=true`
- ✅ Backup previous production models
- ✅ CLI interface để manage models

### 2. Training Script Integration (`train_with_args.py`)
- ✅ Added `--register-model` argument
- ✅ Import `ModelRegistry` class
- ✅ Register model sau khi training completes
- ✅ Capture metrics: macro_f1, micro_f1, test_loss, hamming_loss
- ✅ Capture metadata: person, experiment_name, hyperparameters
- ✅ Trigger auto-evaluation và deployment

### 3. Documentation
- ✅ Updated `CONTINUOUS_LEARNING_GUIDE.md`
- ✅ Updated `AUTO_UPGRADE_SUMMARY.md`
- ✅ Created `MODEL_REGISTRY_INTEGRATION.md` (this file)

## 🚀 Cách Sử Dụng

### Quick Start

```bash
# Person A training
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_a/exp001/ \
    --experiment-name "Person A - Baseline" \
    --register-model

# Person B training
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_b/exp001/ \
    --experiment-name "Person B - Higher LR" \
    --lr 5e-5 \
    --register-model

# Xem tất cả models
python model_registry.py list

# Deploy best model manually
python model_registry.py deploy --model-id <model_id>
```

### Enable Auto-Deployment

```bash
# Set environment variable
export AUTO_DEPLOY=true

# Bây giờ training sẽ tự động deploy nếu model tốt hơn
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/my_exp/ \
    --register-model
```

## 📊 Workflow

### Manual Selection (Mặc định)

```
Person A training → Model A (F1: 0.75) → Registry
Person B training → Model B (F1: 0.78) → Registry → 🎉 New Best!
Person C training → Model C (F1: 0.72) → Registry

↓ Manual action required
python model_registry.py deploy --model-id model_B
```

### Auto-Deployment (Với AUTO_DEPLOY=true)

```
Person A training → Model A (F1: 0.75) → Registry → Auto-deploy (first model)
Person B training → Model B (F1: 0.78) → Registry → 🎉 New Best! → Auto-deploy
Person C training → Model C (F1: 0.72) → Registry → (Not better, skip)

Production model: Model B (F1: 0.78) ✅
```

## 🔧 Technical Details

### Integration Points

**1. Import Statement (line ~13):**
```python
from model_registry import ModelRegistry
```

**2. Argument Parser (line ~60):**
```python
parser.add_argument('--register-model', action='store_true',
                    help='Register model to central registry after training')
```

**3. Registration Logic (end of main()):**
```python
if args.register_model:
    print(f"\n{'='*70}")
    print("REGISTERING MODEL")
    print("="*70)
    
    registry = ModelRegistry()
    
    metrics = {
        'macro_f1': float(test_metrics['macro_f1']),
        'micro_f1': float(test_metrics['micro_f1']),
        'test_loss': float(test_loss),
        'hamming_loss': float(test_metrics['hamming_loss'])
    }
    
    metadata = {
        'person': os.getenv('USER', os.getenv('USERNAME', 'unknown')),
        'experiment_name': args.experiment_name,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'dropout_rate': args.dropout,
        'max_length': args.max_length,
        'data_file': args.data,
        'model_name': args.model_name
    }
    
    model_id = registry.register_model(args.output, metrics, metadata)
    
    print(f"\n✓ Model registered with ID: {model_id}")
    print(f"✓ Check registry: python model_registry.py list")
```

### Registry Structure

```
model_registry/
├── registry.json              # Registry database
├── models/                    # All registered models
│   ├── model_20260420_120000/
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── training_config.json
│   ├── model_20260420_143022/
│   └── model_20260420_150000/
└── backups/                   # Backup of previous production models
    ├── backup_20260420_143022/
    └── backup_20260420_150000/
```

### Registry JSON Format

```json
{
  "models": [
    {
      "model_id": "model_20260420_143022",
      "path": "model_registry/models/model_20260420_143022",
      "metrics": {
        "macro_f1": 0.7823,
        "micro_f1": 0.8145,
        "test_loss": 0.2341,
        "hamming_loss": 0.0876
      },
      "metadata": {
        "person": "john",
        "experiment_name": "Higher LR",
        "learning_rate": 5e-05,
        "batch_size": 16,
        "num_epochs": 5,
        "dropout_rate": 0.3,
        "max_length": 128,
        "data_file": "data/sample_comments.csv",
        "model_name": "bert-base-uncased"
      },
      "registered_at": "2026-04-20T14:30:22.123456",
      "status": "production",
      "deployed_at": "2026-04-20T14:30:25.789012"
    }
  ],
  "production_model": "model_20260420_143022",
  "best_model": "model_20260420_143022",
  "created_at": "2026-04-20T12:00:00.000000"
}
```

## 📋 CLI Commands

### List Models

```bash
# List top 10 models (default)
python model_registry.py list

# List top 5 models
python model_registry.py list --top 5

# Sort by different metric
python model_registry.py list --sort-by micro_f1
```

### Deploy Model

```bash
# Deploy specific model
python model_registry.py deploy --model-id model_20260420_143022
```

### Check Models

```bash
# Get production model path
python model_registry.py production

# Get best model path
python model_registry.py best

# Get model info
python model_registry.py info --model-id model_20260420_143022
```

## 🔒 Safety Features

### 1. Automatic Backup
Mỗi khi deploy model mới, model cũ được backup:
```
model_registry/backups/backup_20260420_143022/
```

### 2. Rollback Support
```bash
# Nếu model mới có vấn đề, rollback về backup
cp -r model_registry/backups/backup_20260420_143022/* saved_model/
```

### 3. Manual Override
```bash
# Deploy một model cụ thể (override auto-selection)
python model_registry.py deploy --model-id model_20260420_120000
```

## 💡 Example Scenario

### 3 Người Training Cùng Lúc

**9:00 AM - Person A:**
```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_a/exp001/ \
    --experiment-name "Person A - Baseline" \
    --register-model

# Output:
# Model registered: model_20260420_090000
# Macro F1: 0.7500
# 🎉 New best model found! (first model)
# AUTO_DEPLOY enabled. Deploying...
# ✓ Model deployed to production
```

**10:00 AM - Person B:**
```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_b/exp001/ \
    --experiment-name "Person B - Higher LR" \
    --lr 5e-5 \
    --register-model

# Output:
# Model registered: model_20260420_100000
# Macro F1: 0.7800
# 🎉 New best model found!
# Previous Best: model_20260420_090000
# New Best: model_20260420_100000
# Improvement: +0.0300
# AUTO_DEPLOY enabled. Deploying...
# ✓ Backed up current model
# ✓ Model deployed to production
```

**11:00 AM - Person C:**
```bash
python train_with_args.py \
    --data data/sample_comments.csv \
    --output experiments/person_c/exp001/ \
    --experiment-name "Person C - Larger Batch" \
    --batch-size 32 \
    --register-model

# Output:
# Model registered: model_20260420_110000
# Macro F1: 0.7200
# (Not better than current best, no deployment)
```

**Result:**
- Production model: `model_20260420_100000` (Person B, F1: 0.78)
- Best model: `model_20260420_100000`
- All 3 models tracked in registry

## ⚙️ Configuration

### Environment Variables

```bash
# Enable auto-deployment
export AUTO_DEPLOY=true

# Disable auto-deployment (default)
export AUTO_DEPLOY=false
unset AUTO_DEPLOY
```

### Registry Location

Default: `model_registry/`

To change, edit `model_registry.py`:
```python
registry = ModelRegistry(registry_dir='custom_path')
```

## 🆘 Troubleshooting

### Problem: Model không tự động deploy

**Check:**
```bash
echo $AUTO_DEPLOY
# Should output: true
```

**Solution:**
```bash
export AUTO_DEPLOY=true
```

### Problem: Registry không tìm thấy model

**Check:**
```bash
python model_registry.py list
```

**Solution:**
```bash
# Re-train với --register-model flag
python train_with_args.py --register-model ...
```

### Problem: Import error "No module named 'model_registry'"

**Check:**
```bash
# Verify file exists
ls -la model_registry.py
```

**Solution:**
```bash
# Make sure you're in the project root directory
cd /path/to/project
python train_with_args.py --register-model ...
```

## 📚 Related Documentation

- **CONTINUOUS_LEARNING_GUIDE.md** - Comprehensive guide với 5 methods
- **AUTO_UPGRADE_SUMMARY.md** - Quick reference guide
- **DEPLOYMENT_GUIDE.md** - Multi-person training setup
- **QUICK_START_MULTI_PERSON.md** - 5-minute setup guide

## ✅ Testing Checklist

- [x] `model_registry.py` created and tested
- [x] `train_with_args.py` integration complete
- [x] `--register-model` flag works
- [x] Metrics captured correctly
- [x] Metadata captured correctly
- [x] Auto-selection works (best model by macro F1)
- [x] Auto-deployment works (when AUTO_DEPLOY=true)
- [x] Backup system works
- [x] CLI commands work (list, deploy, best, production, info)
- [x] Documentation updated

## 🎓 Next Steps

### For Users

1. **Try it out:**
   ```bash
   python train_with_args.py --register-model ...
   ```

2. **Enable auto-deploy:**
   ```bash
   export AUTO_DEPLOY=true
   ```

3. **Monitor registry:**
   ```bash
   python model_registry.py list
   ```

### For Advanced Users

1. **Setup Federated Learning** - See `CONTINUOUS_LEARNING_GUIDE.md` section 3
2. **Setup Online Learning** - See `CONTINUOUS_LEARNING_GUIDE.md` section 4
3. **Setup Model Ensemble** - See `CONTINUOUS_LEARNING_GUIDE.md` section 5

## 🎉 Summary

**Integration Complete!** ✅

Hệ thống model registry đã được tích hợp hoàn chỉnh. Bây giờ:

✅ Nhiều người có thể training cùng lúc
✅ Model tốt nhất tự động được chọn
✅ Có thể enable auto-deployment
✅ Tất cả models được track
✅ Backup tự động
✅ CLI interface để manage

**Bắt đầu ngay:**
```bash
python train_with_args.py --register-model ...
```

---

**Created:** 2026-04-20  
**Version:** 1.0  
**Status:** Complete ✅
