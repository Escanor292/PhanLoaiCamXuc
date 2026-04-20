# GitHub Setup - Collaborative Training

## 🎯 Mục Tiêu

Setup GitHub repo để team có thể:
1. Share code và data
2. Collaborative training
3. Track models và experiments
4. Version control

## 📋 Prerequisites

- Git đã cài đặt
- GitHub account
- Repo: https://github.com/Escanor292/PhanLoaiCamXuc.git

---

## 🚀 Setup Lần Đầu (Owner)

### Bước 1: Init Git Repo

```bash
# Trong folder D:\PhanLoaiCamXuc
cd D:\PhanLoaiCamXuc

# Init git
git init

# Add remote
git remote add origin https://github.com/Escanor292/PhanLoaiCamXuc.git
```

### Bước 2: Update .gitignore

Đã có file `.gitignore` với nội dung:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.pytest_cache/

# Data files (optional - có thể commit sample data)
# data/*.csv

# Model files (large files - dùng Git LFS)
saved_model/*.bin
saved_model/*.pt
saved_model/*.pth

# Model registry (large files - dùng Git LFS)
model_registry/models/*/pytorch_model.bin
model_registry/backups/

# Experiments (optional - có thể commit hoặc ignore)
experiments/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
```

### Bước 3: Setup Git LFS (Cho Models Lớn)

**Git LFS giúp quản lý files lớn (models) trên GitHub.**

```bash
# Install Git LFS (nếu chưa có)
# Download từ: https://git-lfs.github.com/

# Init Git LFS
git lfs install

# Track model files
git lfs track "saved_model/*.bin"
git lfs track "saved_model/*.pt"
git lfs track "saved_model/*.pth"
git lfs track "model_registry/models/*/pytorch_model.bin"

# Add .gitattributes
git add .gitattributes
```

### Bước 4: Initial Commit

```bash
# Add all files
git add .

# Commit
git commit -m "Initial commit: Multi-label Emotion Classification System

- Complete training pipeline
- Model registry with auto-upgrade
- Collaborative training workflow
- Documentation and tools"

# Push to GitHub
git push -u origin main
```

**Nếu bị lỗi "main" không tồn tại:**
```bash
# Rename branch to main
git branch -M main

# Push
git push -u origin main
```

---

## 👥 Setup Cho Team Members

### Bước 1: Clone Repo

```bash
# Clone repo về máy
git clone https://github.com/Escanor292/PhanLoaiCamXuc.git

# Vào folder
cd PhanLoaiCamXuc
```

### Bước 2: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt
```

### Bước 3: Verify Setup

```bash
# Check git status
git status

# Check remote
git remote -v

# Pull latest
git pull
```

---

## 🔄 Collaborative Workflow

### Workflow 1: Góp Data

**Member 1:**
```bash
# 1. Pull latest
git pull

# 2. Tạo data của mình
# data/member1_data.csv

# 3. Add và commit
git add data/member1_data.csv
git commit -m "Add member1 data (200 comments)"
git push

# 4. Merge data
python merge_data.py

# 5. Commit master dataset
git add data/master_dataset.csv
git commit -m "Update master dataset with member1 data"
git push
```

**Member 2:**
```bash
# 1. Pull latest (có master dataset mới)
git pull

# 2. Tạo data của mình
# data/member2_data.csv

# 3. Add và commit
git add data/member2_data.csv
git commit -m "Add member2 data (150 comments)"
git push

# 4. Merge data
python merge_data.py

# 5. Commit master dataset
git add data/master_dataset.csv
git commit -m "Update master dataset with member2 data"
git push
```

### Workflow 2: Training

**Member 1:**
```bash
# 1. Pull latest
git pull

# 2. Training
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/member1_lr5e5/ \
    --experiment-name "Member 1 - LR 5e-5" \
    --lr 5e-5 \
    --register-model

# 3. Commit registry (không commit model files nếu quá lớn)
git add model_registry/registry.json
git commit -m "Add training result: Member 1 - LR 5e-5 (F1: 0.80)"
git push
```

**Member 2:**
```bash
# 1. Pull latest
git pull

# 2. Training
python train_with_args.py \
    --data data/master_dataset.csv \
    --output experiments/member2_lr1e5/ \
    --experiment-name "Member 2 - LR 1e-5" \
    --lr 1e-5 \
    --register-model

# 3. Commit registry
git add model_registry/registry.json
git commit -m "Add training result: Member 2 - LR 1e-5 (F1: 0.82)"
git push
```

### Workflow 3: Deploy Best Model

**Sau khi tất cả training xong:**

```bash
# 1. Pull latest
git pull

# 2. Xem models
python model_registry.py list

# 3. Deploy best model
python model_registry.py deploy --model-id model_20260420_150000

# 4. Commit production model (nếu dùng Git LFS)
git add saved_model/
git add model_registry/registry.json
git commit -m "Deploy best model: model_20260420_150000 (F1: 0.82)"
git push
```

---

## 📁 Git Structure

```
PhanLoaiCamXuc/
├── .git/                           # Git repo
├── .gitignore                      # Git ignore rules
├── .gitattributes                  # Git LFS tracking
├── data/
│   ├── sample_comments.csv         # ✅ Commit (sample data)
│   ├── member1_data.csv            # ✅ Commit (contributions)
│   ├── member2_data.csv            # ✅ Commit (contributions)
│   └── master_dataset.csv          # ✅ Commit (merged dataset)
├── model_registry/
│   ├── registry.json               # ✅ Commit (metadata only)
│   └── models/                     # ⚠️ Git LFS hoặc ignore
├── saved_model/                    # ⚠️ Git LFS hoặc ignore
├── experiments/                    # ❌ Ignore (quá lớn)
├── [code files]                    # ✅ Commit
└── [documentation]                 # ✅ Commit
```

**Legend:**
- ✅ Commit: Push lên GitHub
- ⚠️ Git LFS: Dùng Git LFS cho files lớn
- ❌ Ignore: Không commit (local only)

---

## 🔧 Git Commands Thường Dùng

### Pull Latest Changes

```bash
# Pull trước khi làm việc
git pull
```

### Add và Commit

```bash
# Add specific files
git add data/member1_data.csv
git add model_registry/registry.json

# Add all changes
git add .

# Commit với message
git commit -m "Add member1 data and training results"

# Push to GitHub
git push
```

### Check Status

```bash
# Xem files đã thay đổi
git status

# Xem diff
git diff

# Xem commit history
git log --oneline
```

### Resolve Conflicts

```bash
# Nếu có conflict khi pull
git pull
# CONFLICT in data/master_dataset.csv

# 1. Mở file và resolve conflict manually
# 2. Add resolved file
git add data/master_dataset.csv

# 3. Commit
git commit -m "Resolve merge conflict in master_dataset.csv"

# 4. Push
git push
```

---

## 🚨 Best Practices

### 1. Pull Trước Khi Làm Việc

```bash
# LUÔN pull trước khi bắt đầu
git pull
```

### 2. Commit Thường Xuyên

```bash
# Commit sau mỗi thay đổi quan trọng
git add .
git commit -m "Descriptive message"
git push
```

### 3. Write Good Commit Messages

**❌ Bad:**
```bash
git commit -m "update"
git commit -m "fix"
git commit -m "changes"
```

**✅ Good:**
```bash
git commit -m "Add member1 data (200 comments)"
git commit -m "Update master dataset with new contributions"
git commit -m "Train model with LR 5e-5 (F1: 0.80)"
git commit -m "Deploy best model (F1: 0.82)"
```

### 4. Không Commit Files Lớn

```bash
# Dùng Git LFS cho models
git lfs track "*.bin"

# Hoặc ignore trong .gitignore
echo "experiments/" >> .gitignore
```

### 5. Resolve Conflicts Cẩn Thận

```bash
# Nếu có conflict, đừng force push
# Resolve manually và commit
```

---

## 💡 Tips & Tricks

### Tip 1: Alias Cho Commands Thường Dùng

```bash
# Add aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'

# Sử dụng
git st      # thay vì git status
git ci -m "message"  # thay vì git commit -m "message"
```

### Tip 2: Ignore Local Experiments

```bash
# Thêm vào .gitignore
echo "experiments/" >> .gitignore
echo "my_local_tests/" >> .gitignore

# Commit .gitignore
git add .gitignore
git commit -m "Update .gitignore"
git push
```

### Tip 3: Stash Changes

```bash
# Nếu đang làm dở và cần pull
git stash

# Pull
git pull

# Apply stashed changes
git stash pop
```

### Tip 4: View File History

```bash
# Xem history của 1 file
git log --follow data/master_dataset.csv

# Xem changes trong 1 commit
git show <commit-hash>
```

---

## 🆘 Troubleshooting

### Problem: Push bị reject

```bash
# Error: Updates were rejected because the remote contains work

# Solution: Pull first
git pull
git push
```

### Problem: Merge conflict

```bash
# Error: CONFLICT in data/master_dataset.csv

# Solution:
# 1. Mở file và tìm conflict markers
# <<<<<<< HEAD
# Your changes
# =======
# Their changes
# >>>>>>> branch-name

# 2. Resolve manually
# 3. Add và commit
git add data/master_dataset.csv
git commit -m "Resolve merge conflict"
git push
```

### Problem: Forgot to pull before commit

```bash
# Already committed locally but can't push

# Solution:
git pull --rebase
git push
```

### Problem: Want to undo last commit

```bash
# Undo last commit but keep changes
git reset --soft HEAD~1

# Undo last commit and discard changes
git reset --hard HEAD~1
```

---

## 📚 Resources

- **Git Documentation**: https://git-scm.com/doc
- **GitHub Guides**: https://guides.github.com/
- **Git LFS**: https://git-lfs.github.com/
- **Pro Git Book**: https://git-scm.com/book/en/v2

---

## ✅ Quick Reference

### Daily Workflow

```bash
# Morning: Pull latest
git pull

# Work: Add data, train, etc.
# ...

# Evening: Commit và push
git add .
git commit -m "Today's work: added data and trained models"
git push
```

### Collaborative Training Workflow

```bash
# 1. Pull latest
git pull

# 2. Add your data
# data/memberX_data.csv

# 3. Merge data
python merge_data.py

# 4. Commit data
git add data/
git commit -m "Add memberX data and update master dataset"
git push

# 5. Training
python train_with_args.py --data data/master_dataset.csv --register-model

# 6. Commit results
git add model_registry/registry.json
git commit -m "Training results: LR 5e-5 (F1: 0.80)"
git push
```

---

**Created:** 2026-04-20  
**Version:** 1.0  
**Repo:** https://github.com/Escanor292/PhanLoaiCamXuc.git
