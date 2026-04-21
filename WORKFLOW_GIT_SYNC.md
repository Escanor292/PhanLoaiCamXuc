# Git Sync Workflow - Đồng Bộ Dữ Liệu Team

## 🎯 Mục đích

Hướng dẫn **workflow git sync** để đảm bảo bạn luôn có:
- ✅ **Data mới nhất** từ team
- ✅ **Model tốt nhất** đã được deploy
- ✅ **Code mới nhất** với bug fixes
- ✅ **Tránh conflicts** khi push

---

## 🔄 Tại sao phải Pull thường xuyên?

### **Vấn đề nếu KHÔNG pull:**

```
Tình huống: Bạn offline 2 ngày
├─ Member A đã thêm 100 câu tiếng Việt
├─ Member B đã training model mới (F1: 0.85)  
├─ Member C đã fix bug trong merge_data.py
└─ Model production đã được update

→ Máy bạn vẫn có data cũ, model cũ!
```

### **Hậu quả cụ thể:**

| Không Pull | Hậu quả | Ví dụ |
|------------|---------|-------|
| Training với data cũ | Model kém hơn | F1: 0.65 thay vì 0.80 |
| Push không được | Conflict error | `! [rejected] main -> main` |
| Dùng model cũ | Prediction kém | Confidence thấp |
| Miss bug fixes | Lỗi đã được fix | Script bị crash |

---

## 📋 Workflow Hàng Ngày

### **🌅 Mỗi sáng khi mở dự án:**

```bash
# 1. BẮT BUỘC: Pull trước khi làm gì
git pull

# 2. Kiểm tra có gì mới (optional)
git log --oneline -5

# 3. Kiểm tra data mới nhất
ls data/
wc -l data/master_dataset_vi.csv

# 4. Kiểm tra model registry
python model_registry.py list --top 3

# 5. Bắt đầu làm việc
```

### **🌆 Mỗi tối trước khi tắt máy:**

```bash
# 1. Add công việc của bạn
git add .

# 2. Commit với message rõ ràng
git commit -m "Add member_TenBan data (50 comments) + training results"

# 3. Pull trước khi push (tránh conflict)
git pull

# 4. Push lên GitHub
git push
```

---

## 🎯 Khi nào BẮT BUỘC phải Pull?

### **1. Trước khi Training**

```bash
# ❌ SAI - Training với data cũ
python train_with_args.py --data data/master_dataset_vi.csv --register-model

# ✅ ĐÚNG - Pull data mới nhất trước
git pull
python train_with_args.py --data data/master_dataset_vi.csv --register-model
```

**Tại sao:**
- `master_dataset_vi.csv` có thể đã được update với data mới
- Training với data cũ → Model kém hơn
- Lãng phí thời gian training

### **2. Trước khi Merge Data**

```bash
# ❌ SAI - Merge mà không có data mới nhất
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv

# ✅ ĐÚNG - Pull trước khi merge
git pull
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv
```

**Tại sao:**
- Có thể có `member_*.csv` mới từ người khác
- Tránh overwrite data của team

### **3. Trước khi Commit & Push**

```bash
# ❌ SAI - Push mà không pull
git add .
git commit -m "My changes"
git push  # ← Có thể bị reject!

# ✅ ĐÚNG - Pull trước khi push
git add .
git commit -m "My changes"
git pull  # ← Tránh conflict
git push
```

**Tại sao:**
- Tránh conflict với changes của người khác
- Git sẽ reject push nếu branch outdated

### **4. Trước khi Test Model**

```bash
# ❌ SAI - Test với model cũ
python my_test.py

# ✅ ĐÚNG - Pull model mới nhất trước
git pull
python my_test.py
```

**Tại sao:**
- `saved_model/` có thể đã được update với model tốt hơn
- Test với model cũ → Kết quả không chính xác

---

## 🚨 Xử lý Conflicts

### **Khi pull bị conflict:**

```bash
git pull
# CONFLICT (content): Merge conflict in data/master_dataset_vi.csv
# Automatic merge failed; fix conflicts and then commit the result.
```

### **Bước xử lý:**

```bash
# 1. Mở file có conflict
notepad data/master_dataset_vi.csv

# 2. Tìm conflict markers:
# <<<<<<< HEAD
# Your changes (local)
# =======
# Their changes (remote)
# >>>>>>> origin/main

# 3. Quyết định giữ phần nào:
# - Giữ phần của bạn: Xóa markers + their changes
# - Giữ phần của họ: Xóa markers + your changes  
# - Giữ cả hai: Merge thủ công

# 4. Save file và resolve conflict
git add data/master_dataset_vi.csv
git commit -m "Resolve merge conflict in master dataset"

# 5. Push
git push
```

### **Tips xử lý conflict:**

**Với data files (CSV):**
- Thường giữ phần **remote** (của team)
- Vì master dataset được merge từ nhiều sources

**Với code files:**
- Đọc kỹ cả 2 phần
- Merge thủ công nếu cần
- Test sau khi merge

---

## 📊 Ví dụ Thực Tế

### **Scenario: Sáng thứ 2 mở dự án**

```bash
# Kiểm tra status
git status
# On branch main
# Your branch is behind 'origin/main' by 15 commits ← CẦN PULL!

# Pull để cập nhật
git pull
# Updating abc1234..def5678
# Fast-forward
#  data/member_alice.csv        | 100 ++++++++++++++++
#  data/member_bob.csv          | 150 ++++++++++++++++
#  data/master_dataset_vi.csv   | 250 ++++++++++++++++
#  model_registry/registry.json |  50 ++++++++
#  4 files changed, 550 insertions(+)

# Kiểm tra data mới
wc -l data/master_dataset_vi.csv
# 501 data/master_dataset_vi.csv  ← Từ 200 → 500 câu!

# Kiểm tra model mới
python model_registry.py list --top 1
# ⭐ 1. model_20260421_090000 [BEST]
#    Macro F1: 0.85  ← Tốt hơn model cũ (0.65)

# Bây giờ training sẽ tốt hơn!
python train_with_args.py --data data/master_dataset_vi.csv --register-model
```

---

## 🤖 Tự Động Hóa

### **Script tự động: `update_project.ps1`**

```powershell
# update_project.ps1
Write-Host "🔄 Pulling latest changes..." -ForegroundColor Yellow
git pull

Write-Host "`n📊 Checking data status..." -ForegroundColor Cyan
$csvFiles = Get-ChildItem data/*.csv
Write-Host "CSV files found: $($csvFiles.Count)"
foreach ($file in $csvFiles) {
    $lines = (Get-Content $file.FullName | Measure-Object -Line).Lines
    Write-Host "  $($file.Name): $lines lines"
}

Write-Host "`n🏆 Checking best model..." -ForegroundColor Green
python model_registry.py list --top 1

Write-Host "`n✅ Project updated! Ready to work!" -ForegroundColor Green
```

**Sử dụng:**
```bash
.\update_project.ps1
```

### **Git alias (Advanced):**

```bash
# Thêm vào ~/.gitconfig
[alias]
    sync = !git pull && echo "📊 Data status:" && ls -la data/*.csv && echo "🏆 Best model:" && python model_registry.py list --top 1

# Sử dụng:
git sync
```

---

## 📅 Frequency Khuyến Nghị

### **Hàng ngày:**
- ✅ **Sáng**: Pull để cập nhật
- ✅ **Tối**: Push công việc của bạn

### **Trước mỗi task:**
- ✅ **Trước training**: `git pull`
- ✅ **Trước testing**: `git pull`
- ✅ **Trước deploy**: `git pull`

### **Khi có notification:**
- ✅ **Slack/Discord**: "Model mới đã deploy"
- ✅ **Email**: "Data mới đã được thêm"

---

## 💡 Best Practices

### **✅ NÊN:**

1. **Pull trước khi làm việc**
   ```bash
   git pull  # Luôn luôn đầu tiên
   ```

2. **Commit thường xuyên**
   ```bash
   # Mỗi ngày hoặc sau mỗi task
   git add .
   git commit -m "Clear message"
   git push
   ```

3. **Message commit rõ ràng**
   ```bash
   # ✅ Tốt
   git commit -m "Add member_nam Vietnamese data (100 comments)"
   git commit -m "Training: LR 5e-5 on 500 samples (Macro F1: 0.82)"
   
   # ❌ Kém
   git commit -m "update"
   git commit -m "fix"
   ```

4. **Kiểm tra trước khi push**
   ```bash
   git status    # Xem files nào sẽ được commit
   git diff      # Xem changes cụ thể
   git pull      # Pull trước khi push
   git push
   ```

### **❌ KHÔNG NÊN:**

1. **Làm việc nhiều ngày không pull**
   - Dẫn đến conflicts lớn
   - Miss updates quan trọng

2. **Push mà không pull trước**
   - Git sẽ reject
   - Phải fix conflicts sau

3. **Training mà không pull data mới**
   - Lãng phí thời gian
   - Model kém hơn

4. **Ignore conflicts**
   - Có thể mất data
   - Gây lỗi cho team

---

## 🔍 Troubleshooting

### **Problem 1: "Your branch is behind"**

```bash
git status
# Your branch is behind 'origin/main' by 5 commits

# Giải pháp:
git pull
```

### **Problem 2: "Push rejected"**

```bash
git push
# ! [rejected] main -> main (fetch first)

# Giải pháp:
git pull
git push
```

### **Problem 3: "Merge conflict"**

```bash
git pull
# CONFLICT (content): Merge conflict in file.csv

# Giải pháp:
# 1. Mở file, sửa conflict
# 2. git add file.csv
# 3. git commit -m "Resolve conflict"
# 4. git push
```

### **Problem 4: "Detached HEAD"**

```bash
git status
# HEAD detached at abc1234

# Giải pháp:
git checkout main
git pull
```

---

## 📋 Checklist Hàng Ngày

### **Sáng:**
- [ ] `git pull`
- [ ] Kiểm tra `data/master_dataset_vi.csv` có update không
- [ ] Kiểm tra model registry có model mới không
- [ ] Bắt đầu làm việc

### **Trong ngày:**
- [ ] Commit progress thường xuyên
- [ ] Pull trước khi training
- [ ] Pull trước khi merge data

### **Tối:**
- [ ] `git add .`
- [ ] `git commit -m "Clear message"`
- [ ] `git pull`
- [ ] `git push`

---

## 🎯 Tóm Tắt

### **Quy tắc vàng:**
1. **LUÔN pull trước khi làm việc**
2. **LUÔN pull trước khi push**
3. **Commit thường xuyên với message rõ ràng**
4. **Xử lý conflicts ngay lập tức**

### **Commands cần nhớ:**
```bash
git pull          # Lấy updates mới nhất
git status        # Kiểm tra trạng thái
git add .         # Stage tất cả changes
git commit -m ""  # Commit với message
git push          # Push lên GitHub
```

### **Workflow tóm gọn:**
```bash
# Mỗi ngày:
git pull → work → git add . → git commit -m "" → git pull → git push
```

**Nhớ:** Git sync giống như đồng bộ Google Drive - luôn cần update để có version mới nhất! 📁🔄✨

---

**Created:** 2026-04-21  
**Version:** 1.0  
**Repo:** https://github.com/Escanor292/PhanLoaiCamXuc.git