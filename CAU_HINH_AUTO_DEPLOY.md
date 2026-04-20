# CẤU HÌNH AUTO-DEPLOY

## Vấn đề

Hiện tại, `AUTO_DEPLOY` là biến môi trường trên máy từng người:

```powershell
# Member A phải set
$env:AUTO_DEPLOY = "true"

# Member B phải set
$env:AUTO_DEPLOY = "true"

# Member C phải set
$env:AUTO_DEPLOY = "true"
```

**Vấn đề:** Mỗi người phải set thủ công, dễ quên!

---

## Giải pháp: File .env (Đã cài đặt)

### File `.env` đã được tạo sẵn:

```bash
# .env
AUTO_DEPLOY=true
REGISTRY_DIR=model_registry
DEFAULT_EPOCHS=5
DEFAULT_BATCH_SIZE=16
DEFAULT_LEARNING_RATE=2e-5
```

### Cách hoạt động:

1. **File `.env` được commit lên GitHub**
2. **Tất cả members pull về → Có cùng config**
3. **Script tự động đọc từ `.env`**

```
Member A: git pull → Có .env → AUTO_DEPLOY=true ✓
Member B: git pull → Có .env → AUTO_DEPLOY=true ✓
Member C: git pull → Có .env → AUTO_DEPLOY=true ✓
```

---

## Cách sử dụng

### Bước 1: Cài đặt python-dotenv (1 lần duy nhất)

```bash
pip install python-dotenv
```

Hoặc:

```bash
pip install -r requirements.txt
```

### Bước 2: Training (tự động đọc .env)

```bash
python train_with_args.py --data data/master_dataset_vi.csv --register-model
```

**Kết quả:**
- Script tự động đọc `AUTO_DEPLOY=true` từ file `.env`
- Nếu model tốt hơn → Tự động deploy
- Không cần set biến môi trường thủ công!

---

## Thay đổi cấu hình

### Option 1: Sửa file .env (Cho tất cả team)

```bash
# Mở file .env
notepad .env

# Sửa
AUTO_DEPLOY=false  # Tắt auto-deploy

# Commit và push
git add .env
git commit -m "Disable auto-deploy"
git push
```

**Kết quả:** Tất cả members pull về → Có config mới

### Option 2: Override tạm thời (Chỉ cho 1 lần)

```powershell
# Override cho 1 lần training
$env:AUTO_DEPLOY = "false"
python train_with_args.py --data data/master_dataset_vi.csv --register-model
```

**Kết quả:** Chỉ lần này không auto-deploy, lần sau vẫn đọc từ .env

---

## So sánh

| Cách | Ưu điểm | Nhược điểm |
|------|---------|------------|
| **Biến môi trường** (cũ) | Đơn giản | Mỗi người phải set thủ công |
| **File .env** (mới) ✅ | Tự động, share cho team | Cần cài python-dotenv |

---

## Các biến trong .env

| Biến | Mô tả | Giá trị mặc định |
|------|-------|------------------|
| `AUTO_DEPLOY` | Tự động deploy model tốt nhất | `true` |
| `REGISTRY_DIR` | Thư mục lưu registry | `model_registry` |
| `DEFAULT_EPOCHS` | Số epochs mặc định | `5` |
| `DEFAULT_BATCH_SIZE` | Batch size mặc định | `16` |
| `DEFAULT_LEARNING_RATE` | Learning rate mặc định | `2e-5` |

---

## Workflow với .env

### Workflow cũ (Thủ công):

```bash
# Mỗi người phải làm
$env:AUTO_DEPLOY = "true"
python train_with_args.py --data data/master_dataset_vi.csv --register-model
```

### Workflow mới (Tự động) ✅:

```bash
# Chỉ cần pull 1 lần
git pull

# Training (tự động đọc .env)
python train_with_args.py --data data/master_dataset_vi.csv --register-model
```

---

## Khi nào nên bật/tắt AUTO_DEPLOY?

### Nên BẬT (AUTO_DEPLOY=true):

- ✅ Team nhỏ (3-5 người)
- ✅ Tin tưởng metrics (Macro F1)
- ✅ Muốn model tự động cải thiện
- ✅ Có backup model cũ

### Nên TẮT (AUTO_DEPLOY=false):

- ❌ Team lớn (10+ người)
- ❌ Cần review thủ công trước khi deploy
- ❌ Đang thử nghiệm (không chắc chắn)
- ❌ Production quan trọng (cần kiểm tra kỹ)

---

## Khuyến nghị

### Cho team nhỏ (3-5 người):

```bash
# File .env
AUTO_DEPLOY=true
```

**Lý do:**
- Ít người → Ít conflict
- Metrics đáng tin → Tự động deploy OK
- Tiết kiệm thời gian

### Cho team lớn (10+ người):

```bash
# File .env
AUTO_DEPLOY=false
```

**Workflow:**
1. Training → Register model
2. Review metrics: `python model_registry.py list`
3. Deploy thủ công: `python model_registry.py deploy --model-id <id>`

**Lý do:**
- Nhiều người → Cần review
- Tránh deploy nhầm
- Kiểm soát tốt hơn

---

## Troubleshooting

### Lỗi: ModuleNotFoundError: No module named 'dotenv'

**Giải pháp:**
```bash
pip install python-dotenv
```

### Lỗi: .env file not found

**Giải pháp:**
```bash
# Copy từ example
copy .env.example .env

# Hoặc pull từ GitHub
git pull
```

### AUTO_DEPLOY không hoạt động

**Kiểm tra:**
```bash
# 1. Xem file .env
cat .env

# 2. Kiểm tra giá trị
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('AUTO_DEPLOY'))"

# Kết quả mong đợi: true
```

---

## Tóm tắt

**Trước (Thủ công):**
```
Mỗi người: $env:AUTO_DEPLOY = "true"  ← Phải nhớ set
```

**Sau (Tự động) ✅:**
```
File .env: AUTO_DEPLOY=true  ← Tất cả đều có
```

**Lợi ích:**
- ✅ Không cần set thủ công
- ✅ Tất cả team có cùng config
- ✅ Dễ thay đổi (sửa 1 file, commit, push)
- ✅ Không quên bật/tắt

**Cài đặt:**
```bash
pip install python-dotenv
python train_with_args.py --data data/master_dataset_vi.csv --register-model
```

Done! 🎉
