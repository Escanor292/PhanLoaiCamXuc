# GitHub Quick Start - 3 Bước Đơn Giản

## 🚀 Setup Lần Đầu (Owner)

### Bước 1: Chạy Setup Script

```powershell
# Chạy script tự động
.\setup_git.ps1
```

**Script sẽ:**
- ✅ Check Git installation
- ✅ Init Git repo
- ✅ Add remote (GitHub)
- ✅ Setup Git LFS (optional)
- ✅ Create .gitignore
- ✅ Commit và push

**Hoặc làm manual:**
```bash
# Init git
git init

# Add remote
git remote add origin https://github.com/Escanor292/PhanLoaiCamXuc.git

# Add files
git add .

# Commit
git commit -m "Initial commit"

# Push
git branch -M main
git push -u origin main
```

---

## 👥 Team Members Clone Repo

### Bước 2: Clone Repo

```bash
# Clone về máy
git clone https://github.com/Escanor292/PhanLoaiCamXuc.git

# Vào folder
cd PhanLoaiCamXuc

# Install dependencies
pip install -r requirements.txt
```

---

## 🔄 Daily Workflow

### Bước 3: Pull → Work → Push

```bash
# Morning: Pull latest
git pull

# Work: Add data, train, etc.
python merge_data.py
python train_with_args.py --data data/master_dataset.csv --register-model

# Evening: Commit và push
git add .
git commit -m "Add data and training results"
git push
```

---

## 📋 Collaborative Training Workflow

### Góp Data

```bash
# 1. Pull latest
git pull

# 2. Add your data
# Tạo file: data/memberX_data.csv

# 3. Merge data
python merge_data.py

# 4. Commit và push
git add data/
git commit -m "Add memberX data (200 comments)"
git push
```

### Training

```bash
# 1. Pull latest
git pull

# 2. Training
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/my_exp/ \
    --experiment-name "My Experiment" \
    --lr 5e-5 \
    --register-model

# 3. Commit results (chỉ commit registry, không commit model files)
git add model_registry/registry.json
git commit -m "Training: LR 5e-5 (F1: 0.80)"
git push
```

### Deploy Best Model

```bash
# 1. Pull latest
git pull

# 2. Xem models
python model_registry.py list

# 3. Deploy
python model_registry.py deploy --model-id <best_model_id>

# 4. Commit (nếu muốn share production model)
git add saved_model/
git add model_registry/registry.json
git commit -m "Deploy best model (F1: 0.82)"
git push
```

---

## 🆘 Common Issues

### Issue 1: Push bị reject

```bash
# Error: Updates were rejected

# Solution: Pull first
git pull
git push
```

### Issue 2: Merge conflict

```bash
# Error: CONFLICT in data/master_dataset.csv

# Solution:
# 1. Mở file và resolve conflict
# 2. Add và commit
git add data/master_dataset.csv
git commit -m "Resolve conflict"
git push
```

### Issue 3: Forgot to pull

```bash
# Already committed but can't push

# Solution:
git pull --rebase
git push
```

---

## ✅ Checklist

### Setup (One-time)
- [ ] Run `.\setup_git.ps1` hoặc setup manual
- [ ] Push to GitHub
- [ ] Share repo URL với team

### Daily Workflow
- [ ] Pull trước khi làm việc
- [ ] Commit thường xuyên
- [ ] Push sau khi xong việc

### Collaborative Training
- [ ] Pull latest
- [ ] Add data → Merge → Commit → Push
- [ ] Training → Commit results → Push
- [ ] Deploy best model

---

## 📚 Đọc Thêm

- **GITHUB_SETUP.md** - Hướng dẫn chi tiết
- **COLLABORATIVE_TRAINING_WORKFLOW.md** - Collaborative workflow
- **Git Documentation** - https://git-scm.com/doc

---

**Repo:** https://github.com/Escanor292/PhanLoaiCamXuc.git  
**Created:** 2026-04-20  
**Version:** 1.0
