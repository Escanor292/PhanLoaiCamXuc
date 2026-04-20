# Hướng Dẫn Chi Tiết Cho Team Members

## 🎯 Mục Đích

Hướng dẫn này giúp bạn (team member) hiểu rõ:
- **Làm gì**: Mục đích của từng bước
- **Làm như nào**: Commands cụ thể
- **Tại sao**: Lý do cần làm bước đó

---

## 📋 Tổng Quan Workflow

```
1. Setup lần đầu (1 lần duy nhất)
   ↓
2. Góp data của bạn
   ↓
3. Training với config của bạn
   ↓
4. So sánh kết quả
   ↓
5. Deploy model tốt nhất
```

---

## 🚀 PHẦN 1: SETUP LẦN ĐẦU (1 LẦN DUY NHẤT)

### Bước 1.1: Cài Đặt Git

**Làm gì:** Cài Git để có thể clone code từ GitHub

**Làm như nào:**
1. Download Git từ: https://git-scm.com/download/win
2. Chạy installer
3. Chọn "Next" hết (dùng default settings)
4. Verify: Mở PowerShell và gõ:
   ```bash
   git --version
   ```
   Nếu thấy `git version 2.x.x` là OK

**Tại sao:** Git là tool để download code và sync với team

---

### Bước 1.2: Clone Repository

**Làm gì:** Download toàn bộ code về máy bạn

**Làm như nào:**
```bash
# 1. Mở PowerShell
# 2. Vào folder bạn muốn lưu code (ví dụ: D:\)
cd D:\

# 3. Clone repo
git clone https://github.com/Escanor292/PhanLoaiCamXuc.git

# 4. Vào folder project
cd PhanLoaiCamXuc
```

**Kết quả:**
```
D:\PhanLoaiCamXuc\
├── config.py
├── model.py
├── train.py
├── data/
├── ...
```

**Tại sao:** Bạn cần có code trên máy để làm việc

---

### Bước 1.3: Cài Đặt Python Packages

**Làm gì:** Cài các thư viện Python cần thiết

**Làm như nào:**
```bash
# Trong folder D:\PhanLoaiCamXuc
pip install -r requirements.txt
```

**Kết quả:**
```
Installing collected packages: torch, transformers, pandas, numpy, ...
Successfully installed ...
```

**Tại sao:** Code cần các thư viện này để chạy (PyTorch, Transformers, ...)

---

### Bước 1.4: Verify Setup

**Làm gì:** Kiểm tra xem setup đã OK chưa

**Làm như nào:**
```bash
# Test import
python -c "import torch; import transformers; print('OK')"
```

**Kết quả:** Nếu thấy `OK` là thành công

**Tại sao:** Đảm bảo mọi thứ hoạt động trước khi bắt đầu

---

## 📊 PHẦN 2: GÓP DATA TIẾNG VIỆT (MỖI TUẦN)

> **⚠️ QUAN TRỌNG:** Model hiện tại được train trên tiếng Anh, nên độ chính xác với tiếng Việt rất thấp (40-50%). Để cải thiện, team cần đóng góp dữ liệu tiếng Việt!

### Tại sao cần data tiếng Việt?

| Tình huống | Độ chính xác | Lý do |
|------------|--------------|-------|
| Train trên tiếng Anh, test tiếng Việt | 40-50% | Model không hiểu tiếng Việt |
| Train trên tiếng Việt, test tiếng Việt | 75-85% | Model học được tiếng Việt |

**Mục tiêu:** Mỗi thành viên đóng góp 50-100 câu tiếng Việt

---

### Bước 2.1: Pull Code Mới Nhất

**Làm gì:** Lấy code và data mới nhất từ GitHub

**Làm như nào:**
```bash
# Trong folder D:\PhanLoaiCamXuc
git pull
```

**Kết quả:**
```
Updating abc1234..def5678
Fast-forward
 data/master_dataset_vi.csv | 100 ++++++++++++++++++
 1 file changed, 100 insertions(+)
```

**Tại sao:** 
- Người khác có thể đã thêm data mới
- Bạn cần data mới nhất để merge

---

### Bước 2.2: Tạo Data Tiếng Việt Của Bạn

**Làm gì:** Tạo file CSV chứa bình luận tiếng Việt với nhãn cảm xúc

**Làm như nào:**

**Option 1: Copy từ template (KHUYẾN NGHỊ)**
```bash
# Copy template
copy data\TEMPLATE_DONG_GOP_DATA.csv data\member_TenBan.csv

# Mở và edit file
notepad data\member_TenBan.csv
```

**Option 2: Dùng Excel**
1. Mở file `data/TEMPLATE_DONG_GOP_DATA.csv` trong Excel
2. Xóa các dòng mẫu (giữ lại header)
3. Thêm dữ liệu của bạn
4. Save as: `data/member_TenBan.csv`

**Format yêu cầu:**

| comment | joy | trust | fear | surprise | sadness | disgust | anger | anticipation | love | worried | disappointed | proud | embarrassed | jealous | calm | excited |
|---------|-----|-------|------|----------|---------|---------|-------|--------------|------|---------|--------------|-------|-------------|---------|------|---------|
| Tôi rất vui vì được thăng chức! | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| Tôi lo lắng về kỳ thi ngày mai | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Sản phẩm này thật tuyệt vời! | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

**16 cảm xúc (Emotion Labels):**

| Tiếng Việt | Tiếng Anh | Cột CSV |
|------------|-----------|---------|
| vui vẻ | joy | joy |
| tin tưởng | trust | trust |
| sợ hãi | fear | fear |
| ngạc nhiên | surprise | surprise |
| buồn bã | sadness | sadness |
| ghê tởm | disgust | disgust |
| tức giận | anger | anger |
| mong đợi | anticipation | anticipation |
| yêu thương | love | love |
| lo lắng | worried | worried |
| thất vọng | disappointed | disappointed |
| tự hào | proud | proud |
| xấu hổ | embarrassed | embarrassed |
| ghen tị | jealous | jealous |
| bình tĩnh | calm | calm |
| phấn khích | excited | excited |

**Cách đánh nhãn:**
- **1** = Câu này CÓ cảm xúc đó
- **0** = Câu này KHÔNG CÓ cảm xúc đó
- Một câu có thể có NHIỀU cảm xúc (multi-label)

**Ví dụ chi tiết:**

```csv
"Tôi rất vui vì được thăng chức!",1,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1
```
- ✅ vui vẻ (joy) = 1
- ✅ tin tưởng (trust) = 1
- ✅ ngạc nhiên (surprise) = 1
- ✅ tự hào (proud) = 1
- ✅ phấn khích (excited) = 1
- ❌ Các cảm xúc khác = 0

**Tips để tạo data tốt:**

✅ **NÊN:**
- Viết câu tự nhiên, như bình luận thật
- Đa dạng chủ đề: sản phẩm, dịch vụ, sự kiện, cảm xúc cá nhân
- Đa dạng độ dài: ngắn (5-10 từ) và dài (20-30 từ)
- Một câu có thể có nhiều cảm xúc
- Đánh nhãn chính xác

❌ **KHÔNG NÊN:**
- Copy paste từ internet (có thể vi phạm bản quyền)
- Câu quá ngắn: "Tốt", "Hay" (không đủ ngữ cảnh)
- Câu quá dài: > 50 từ (khó phân tích)
- Đánh nhãn sai hoặc thiếu

**Tại sao:** 
- Mỗi người góp data tiếng Việt để model học được tiếng Việt
- Dataset lớn → Model tốt hơn
- 500 câu tiếng Việt → Độ chính xác 75-85%

> 📖 **Xem thêm:** File `HUONG_DAN_DONG_GOP_DATA.md` có hướng dẫn chi tiết hơn

---

### Bước 2.3: Merge Data Vào Master Dataset

**Làm gì:** Gộp data của bạn vào dataset chung

**Làm như nào:**
```bash
# Merge tất cả file member_*.csv thành master dataset
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv
```

**Kết quả:**
```
======================================================================
MERGING DATASETS
======================================================================
✓ Loaded data/member_nam.csv: 100 samples
✓ Loaded data/member_linh.csv: 150 samples
✓ Loaded data/member_hoa.csv: 200 samples

Merging 3 files...
✓ Removed 5 duplicate texts

======================================================================
MERGE COMPLETE
======================================================================
Total samples loaded: 450
Duplicates removed: 5
Final dataset size: 445
Output: data/master_dataset_vi.csv
======================================================================
```

**Script làm gì:**
1. Tìm tất cả file `member_*.csv` trong `data/`
2. Gộp tất cả lại
3. Xóa duplicates (câu trùng lặp)
4. Validate format (kiểm tra đúng format)
5. Tạo `data/master_dataset_vi.csv`

**Giải thích tham số:**
- `data/member_*.csv`: Tất cả file bắt đầu với `member_` trong folder `data/`
- `--output data/master_dataset_vi.csv`: Lưu kết quả vào file này

**Tại sao:** 
- Dataset chung có data của tất cả mọi người
- Training trên dataset chung → Model không bị bias
- **QUAN TRỌNG:** Phải train trên master dataset, không phải file riêng của bạn!

---

### Bước 2.4: Commit và Push Data

**Làm gì:** Upload data của bạn lên GitHub để team khác dùng

**Làm như nào:**
```bash
# 1. Add files
git add data/member_TenBan.csv
git add data/master_dataset_vi.csv

# 2. Commit với message rõ ràng
git commit -m "Add member_TenBan Vietnamese data (100 comments)"

# 3. Push lên GitHub
git push
```

**Kết quả:**
```
[main abc1234] Add member_TenBan Vietnamese data (100 comments)
 2 files changed, 100 insertions(+)
Enumerating objects: 5, done.
...
To https://github.com/Escanor292/PhanLoaiCamXuc.git
   abc1234..def5678  main -> main
```

**Giải thích commands:**
- `git add`: Đánh dấu files cần upload
- `git commit -m "..."`: Lưu thay đổi với message
- `git push`: Upload lên GitHub

**Tại sao:** 
- Team khác cần data của bạn
- Master dataset được update cho mọi người
- Tất cả cùng train trên dataset chung

---

## 🎓 PHẦN 3: TRAINING (MỖI TUẦN)

> **⚠️ LƯU Ý QUAN TRỌNG:** Luôn train với file `master_dataset_vi.csv` (dataset chung), KHÔNG train với file riêng của bạn!

### Tại sao phải train trên master dataset?

| Cách train | Kết quả | Vấn đề |
|------------|---------|--------|
| Train trên file riêng (`member_nam.csv`) | Model chỉ tốt với data của bạn | ❌ Bias, overfit, quên data cũ |
| Train trên master dataset (`master_dataset_vi.csv`) | Model tốt với tất cả data | ✅ Không bias, tổng quát hóa tốt |

**Quy tắc vàng:** Mỗi người train trên CÙNG master dataset, nhưng với KHÁC config (lr, batch_size, epochs, ...)

---

### Bước 3.1: Pull Master Dataset Mới Nhất

**Làm gì:** Lấy master dataset có data của tất cả mọi người

**Làm như nào:**
```bash
git pull
```

**Kết quả:**
```
Updating def5678..ghi9012
Fast-forward
 data/master_dataset_vi.csv | 150 ++++++++++++++++++
 1 file changed, 150 insertions(+)
```

**Tại sao:** 
- Người khác có thể đã thêm data
- Bạn cần train trên dataset mới nhất (có data của tất cả)

---

### Bước 3.2: Chọn Hyperparameter Của Bạn

**Làm gì:** Chọn config khác với người khác để thử nghiệm

**Các hyperparameters có thể thử:**

| Hyperparameter | Giá trị mặc định | Giá trị có thể thử | Ý nghĩa |
|----------------|------------------|-------------------|---------|
| `--lr` | 2e-5 | 1e-5, 2e-5, 5e-5 | Learning rate (tốc độ học) |
| `--batch-size` | 16 | 8, 16, 32 | Số samples mỗi batch |
| `--epochs` | 5 | 3, 5, 10 | Số vòng training |
| `--dropout` | 0.3 | 0.1, 0.3, 0.5 | Dropout rate (tránh overfit) |

**Ví dụ phân công:**
- Member 1: Thử `--lr 5e-5`
- Member 2: Thử `--lr 1e-5`
- Member 3: Thử `--batch-size 32`
- Member 4: Thử `--epochs 10`
- Member 5: Thử `--dropout 0.5`

**Tại sao:** 
- Mỗi người thử config khác nhau
- Tìm config tốt nhất cho model

---

### Bước 3.3: Training Model

**Làm gì:** Train model với config của bạn trên master dataset

**Làm như nào:**
```bash
python train_with_args.py \
    --data data/master_dataset_vi.csv \
    --output experiments/member_TenBan_lr5e5/ \
    --experiment-name "Member TenBan - LR 5e-5" \
    --lr 5e-5 \
    --register-model
```

**⚠️ QUAN TRỌNG:**
- **PHẢI dùng** `--data data/master_dataset_vi.csv` (dataset chung)
- **KHÔNG dùng** `--data data/member_TenBan.csv` (file riêng của bạn)

**Giải thích từng tham số:**
- `--data data/master_dataset_vi.csv`: **QUAN TRỌNG** - Dùng master dataset (có data của tất cả)
- `--output experiments/member_TenBan_lr5e5/`: Lưu model vào folder này
- `--experiment-name "..."`: Tên experiment (để nhận biết)
- `--lr 5e-5`: Learning rate = 5e-5
- `--register-model`: **QUAN TRỌNG** - Đăng ký model vào registry

**Các script training:**

| Script | File data | Khi nào dùng |
|--------|-----------|--------------|
| `train.py` | `data/sample_comments.csv` (cố định) | ❌ Không dùng (file cũ, tiếng Anh) |
| `train_with_args.py` | Tùy chọn với `--data` | ✅ Luôn dùng (linh hoạt) |

**Kết quả:**
```
======================================================================
MULTI-LABEL EMOTION CLASSIFICATION - TRAINING
======================================================================

Experiment: Member TenBan - LR 5e-5

Configuration:
  Data: data/master_dataset_vi.csv
  Learning Rate: 5e-5
  ...

======================================================================
LOADING DATA
======================================================================
✓ Loaded 445 samples

======================================================================
TRAINING
======================================================================

Epoch 1/5
Training: 100%|████████████| 20/20 [00:30<00:00]
  Train Loss: 0.3456
  Val Loss: 0.2987
  Micro F1: 0.7823
  Macro F1: 0.7456

...

Epoch 5/5
Training: 100%|████████████| 20/20 [00:30<00:00]
  Train Loss: 0.1234
  Val Loss: 0.1567
  Micro F1: 0.8567
  Macro F1: 0.8234

======================================================================
FINAL EVALUATION ON TEST SET
======================================================================

Test Results:
  Test Loss: 0.1678
  Micro F1: 0.8445
  Macro F1: 0.8123
  Hamming Loss: 0.0756

======================================================================
REGISTERING MODEL
======================================================================
Model ID: model_20260420_143022
Macro F1: 0.8123
Micro F1: 0.8445
Person: TenBan
======================================================================

🎉 NEW BEST MODEL FOUND!
======================================================================
Previous Best: model_20260420_120000
New Best: model_20260420_143022
Macro F1: 0.8123
Improvement: +0.0123
======================================================================

✓ Model registered with ID: model_20260420_143022
✓ Check registry: python model_registry.py list

======================================================================
TRAINING COMPLETE!
======================================================================
```

**Quá trình training:**
1. Load data từ `master_dataset_vi.csv` (445 samples)
2. Split: 70% train, 15% val, 15% test
3. Train 5 epochs (~2-3 phút/epoch trên CPU, ~30s/epoch trên GPU)
4. Evaluate trên test set
5. Register model vào registry
6. Check xem có phải best model không

**Tại sao:** 
- Train model với config của bạn
- Registry tự động track metrics
- Tự động so sánh với models khác
- **Train trên master dataset** → Model không bị bias

---

### Bước 3.4: Commit Kết Quả

**Làm gì:** Upload kết quả training lên GitHub

**Làm như nào:**
```bash
# Commit registry (chứa metrics)
git add model_registry/registry.json
git commit -m "Training: Member TenBan - LR 5e-5 (Macro F1: 0.8123)"
git push
```

**Lưu ý:** 
- **KHÔNG commit** folder `experiments/` (quá lớn)
- **CHỈ commit** `model_registry/registry.json` (chứa metrics)

**Tại sao:** 
- Team khác biết kết quả của bạn
- So sánh được với kết quả của họ

---

## 📊 PHẦN 4: SO SÁNH KẾT QUẢ

### Bước 4.1: Pull Kết Quả Mới Nhất

**Làm gì:** Lấy kết quả training của tất cả mọi người

**Làm như nào:**
```bash
git pull
```

**Tại sao:** Người khác có thể đã training xong

---

### Bước 4.2: Xem Tất Cả Models

**Làm gì:** Xem danh sách tất cả models và metrics

**Làm như nào:**
```bash
python model_registry.py list
```

**Kết quả:**
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
     • Person:        alice
     • Experiment:    Member Alice - LR 1e-5
     • Learning Rate: 1e-05
     • Batch Size:    16
     • Epochs:        5

📦 2. model_20260420_143022 [REGISTERED]
   ────────────────────────────────────────────────────────────────────────────
   Metrics:
     • Macro F1:      0.8123
     • Micro F1:      0.8445
     • Test Loss:     0.2234
     • Hamming Loss:  0.0756
   Metadata:
     • Person:        john
     • Experiment:    Member John - LR 5e-5
     • Learning Rate: 5e-05
     • Batch Size:    16
     • Epochs:        5

📦 3. model_20260420_140000 [REGISTERED]
   ────────────────────────────────────────────────────────────────────────────
   Metrics:
     • Macro F1:      0.8012
     • Micro F1:      0.8334
     • Test Loss:     0.2345
     • Hamming Loss:  0.0789
   Metadata:
     • Person:        bob
     • Experiment:    Member Bob - Batch Size 32
     • Learning Rate: 2e-05
     • Batch Size:    32
     • Epochs:        5

================================================================================
Legend: 🚀 = Production | ⭐ = Best | 📦 = Registered
================================================================================

Summary:
  Total models: 3
  Production model: None
  Best model: model_20260420_150000
```

**Giải thích:**
- **⭐ BEST**: Model tốt nhất (Macro F1 cao nhất)
- **🚀 PRODUCTION**: Model đang dùng trong production
- **📦 REGISTERED**: Model đã training nhưng chưa deploy

**Tại sao:** 
- Biết model nào tốt nhất
- So sánh config của mình với người khác

---

### Bước 4.3: So Sánh Chi Tiết (Optional)

**Làm gì:** So sánh chi tiết nhiều experiments

**Làm như nào:**
```bash
python compare_experiments.py \
    experiments/member_john_lr5e5/ \
    experiments/member_alice_lr1e5/ \
    experiments/member_bob_bs32/
```

**Kết quả:**
```
================================================================================
EXPERIMENT COMPARISON
================================================================================

Experiment                    | Macro F1 | Micro F1 | Test Loss | Config
------------------------------|----------|----------|-----------|------------------
Member Alice - LR 1e-5       | 0.8234   | 0.8567   | 0.2145    | lr=1e-5, bs=16
Member John - LR 5e-5        | 0.8123   | 0.8445   | 0.2234    | lr=5e-5, bs=16
Member Bob - Batch Size 32   | 0.8012   | 0.8334   | 0.2345    | lr=2e-5, bs=32

================================================================================
BEST MODEL: Member Alice - LR 1e-5
Macro F1: 0.8234
Improvement over baseline: +8.2%
================================================================================
```

**Tại sao:** 
- Hiểu rõ hơn về performance
- Học được config nào tốt

---

## 🚀 PHẦN 5: DEPLOY MODEL TỐT NHẤT

### Bước 5.1: Deploy Model

**Làm gì:** Deploy model tốt nhất vào production

**Làm như nào:**

**Option 1: Manual Deploy**
```bash
# Deploy model cụ thể
python model_registry.py deploy --model-id model_20260420_150000
```

**Option 2: Auto Deploy (Khuyến nghị)**
```bash
# Enable auto-deploy (1 lần)
$env:AUTO_DEPLOY = "true"

# Sau đó training sẽ tự động deploy nếu tốt hơn
python train_with_args.py --data data/master_dataset.csv --register-model
```

**Kết quả:**
```
======================================================================
DEPLOYING MODEL TO PRODUCTION
======================================================================
✓ Backed up current model to: model_registry/backups/backup_20260420_143022

✓ Model deployed successfully!
  Model ID: model_20260420_150000
  Macro F1: 0.8234
  Micro F1: 0.8567
  Deployed at: 2026-04-20 15:00:22
======================================================================
```

**Quá trình deploy:**
1. Backup model cũ (nếu có)
2. Copy model mới vào `saved_model/`
3. Update registry
4. Model trong `saved_model/` sẵn sàng dùng

**Tại sao:** 
- Model tốt nhất được dùng trong production
- Backup model cũ để rollback nếu cần

---

### Bước 5.2: Test Model Production

**Làm gì:** Test model vừa deploy

**Làm như nào:**
```bash
python predict.py
```

**Kết quả:**
```
======================================================================
EMOTION PREDICTION - INTERACTIVE MODE
======================================================================

Model loaded from: saved_model/
Ready for prediction!

Enter text (or 'quit' to exit): I love this product!

Predicted Emotions: joy, love, excited
Confidence Scores:
  joy         : 0.9234
  love        : 0.8567
  excited     : 0.7823
  trust       : 0.4567
  ...

Enter text (or 'quit' to exit):
```

**Tại sao:** 
- Verify model hoạt động đúng
- Test với real examples

---

## 📋 CHECKLIST HÀNG TUẦN

### Đầu Tuần (Monday)

- [ ] Pull code mới nhất: `git pull`
- [ ] Tạo data tiếng Việt của bạn: `data/member_TenBan.csv` (50-100 câu)
- [ ] Merge data: `python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv`
- [ ] Commit và push: `git add data/ && git commit -m "Add Vietnamese data" && git push`

### Giữa Tuần (Wednesday - Friday)

- [ ] Pull master dataset mới: `git pull`
- [ ] Training với master dataset: `python train_with_args.py --data data/master_dataset_vi.csv --register-model`
- [ ] Commit kết quả: `git add model_registry/ && git commit -m "Training results" && git push`

### Cuối Tuần (Friday)

- [ ] Pull kết quả của team: `git pull`
- [ ] Xem tất cả models: `python model_registry.py list`
- [ ] Deploy best model: `python model_registry.py deploy --model-id <best>`
- [ ] Test model: `python my_test.py` hoặc `python demo_prediction.py`

---

## 🆘 TROUBLESHOOTING

### Problem 1: Git pull bị conflict

**Lỗi:**
```
CONFLICT (content): Merge conflict in data/master_dataset.csv
```

**Giải pháp:**
```bash
# 1. Mở file có conflict
notepad data/master_dataset.csv

# 2. Tìm và xóa conflict markers
# <<<<<<< HEAD
# Your changes
# =======
# Their changes
# >>>>>>> branch-name

# 3. Giữ lại phần đúng, xóa phần sai

# 4. Add và commit
git add data/master_dataset.csv
git commit -m "Resolve merge conflict"
git push
```

---

### Problem 2: Training bị lỗi "CUDA out of memory"

**Lỗi:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
```bash
# Giảm batch size
python train_with_args.py --batch-size 8 --register-model

# Hoặc train trên CPU (chậm hơn)
python train_with_args.py --device cpu --register-model
```

---

### Problem 3: Merge data bị lỗi format

**Lỗi:**
```
✗ Error in data/member_john_data.csv: Missing 'text' column
```

**Giải pháp:**
```bash
# Check format của file
python -c "import pandas as pd; df = pd.read_csv('data/member_john_data.csv'); print(df.columns)"

# Đảm bảo có đủ columns:
# ['text', 'joy', 'trust', 'fear', ...]
```

---

### Problem 4: Push bị reject

**Lỗi:**
```
! [rejected]        main -> main (fetch first)
```

**Giải pháp:**
```bash
# Pull trước khi push
git pull
git push
```

---

## 💡 TIPS & BEST PRACTICES

### Tip 1: Commit Message Rõ Ràng

**❌ Bad:**
```bash
git commit -m "update"
git commit -m "fix"
```

**✅ Good:**
```bash
git commit -m "Add member_nam Vietnamese data (100 comments)"
git commit -m "Training: LR 5e-5 on Vietnamese data (Macro F1: 0.8123)"
git commit -m "Deploy best model (Macro F1: 0.8234)"
```

---

### Tip 2: Pull Thường Xuyên

```bash
# Pull trước khi bắt đầu làm việc
git pull

# Pull trước khi commit
git pull

# Pull trước khi push
git pull
```

---

### Tip 3: Backup Data Của Bạn

```bash
# Backup trước khi merge
copy data\member_TenBan.csv data\member_TenBan_backup.csv
```

---

### Tip 4: Luôn Train Trên Master Dataset

**❌ SAI:**
```bash
# Train trên file riêng của bạn
python train_with_args.py --data data/member_nam.csv --register-model
```

**✅ ĐÚNG:**
```bash
# Train trên master dataset (có data của tất cả)
python train_with_args.py --data data/master_dataset_vi.csv --register-model
```

**Tại sao:**
- Train trên file riêng → Model bị bias, overfit, quên data cũ
- Train trên master dataset → Model tổng quát, không bias

---

### Tip 5: Document Config Của Bạn

Tạo file `experiments/member_TenBan_notes.txt`:
```
Experiment 1: LR 5e-5
- Macro F1: 0.8123
- Notes: Tốt nhưng hơi overfit

Experiment 2: LR 1e-5
- Macro F1: 0.8234
- Notes: Tốt nhất! Không overfit

Experiment 3: Batch Size 32
- Macro F1: 0.8012
- Notes: Training nhanh hơn nhưng F1 thấp hơn
```

---

## 📚 TÀI LIỆU THAM KHẢO

### Quick Reference
- **HUONG_DAN_DONG_GOP_DATA.md** - Hướng dẫn chi tiết đóng góp data tiếng Việt
- **QUICK_START.md** - Quick start guide
- **CACH_SU_DUNG_PREDICTION.md** - Cách test prediction
- **README.md** - Project overview

### Detailed Guides
- **COLLABORATIVE_TRAINING_WORKFLOW.md** - Full workflow (nếu có)
- **README.md** - Project overview

### Tools Documentation
- **merge_data.py --help** - Merge data tool
- **model_registry.py --help** - Model registry tool
- **compare_experiments.py --help** - Compare tool
- **train_with_args.py --help** - Training tool với tham số

---

## ✅ TÓM TẮT

### Workflow Tổng Quát

```
1. Setup lần đầu (1 lần)
   git clone → pip install

2. Hàng tuần - Góp data tiếng Việt
   git pull → Tạo data (50-100 câu) → merge_data.py → git push

3. Training trên master dataset
   git pull → train_with_args.py --data master_dataset_vi.csv --register-model → git push

4. Deploy
   model_registry.py list → deploy best model → test với my_test.py
```

### Commands Quan Trọng

```bash
# Pull code mới
git pull

# Merge data tiếng Việt
python merge_data.py data/member_*.csv --output data/master_dataset_vi.csv

# Training trên master dataset (QUAN TRỌNG!)
python train_with_args.py --data data/master_dataset_vi.csv --register-model

# Xem models
python model_registry.py list

# Deploy
python model_registry.py deploy --model-id <model_id>

# Test prediction
python my_test.py
python demo_prediction.py

# Commit và push
git add .
git commit -m "Message"
git push
```

### Quy Tắc Vàng

1. **Luôn train trên `master_dataset_vi.csv`** (dataset chung), không train trên file riêng
2. **Mỗi người đóng góp 50-100 câu tiếng Việt** để model học được tiếng Việt
3. **Mỗi người thử config khác nhau** (lr, batch_size, epochs) để tìm config tốt nhất
4. **Pull thường xuyên** trước khi làm việc, commit, và push
5. **Commit message rõ ràng** để team hiểu bạn làm gì

---

**Created:** 2026-04-20  
**Version:** 1.0  
**Repo:** https://github.com/Escanor292/PhanLoaiCamXuc.git
