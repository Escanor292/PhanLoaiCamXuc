# 🔄 Model Sharing Setup - Giải Quyết Vấn Đề Model Quá Lớn

## 🚨 Vấn Đề:
- **Model size:** 419 MB
- **GitHub limit:** 100 MB/file
- **Kết quả:** ❌ Không thể push model lên GitHub!

## 💡 Giải Pháp: Cloud Storage

### Option 1: Hugging Face Model Hub (KHUYẾN NGHỊ)

#### Bước 1: Cài Đặt
```bash
pip install -U huggingface_hub
```

#### Bước 2: Tạo Account và Token
1. **Đăng ký tài khoản** tại: https://huggingface.co/
2. **Tạo fine-grained token:**
   - Vào: https://huggingface.co/settings/tokens
   - Chọn "New token" → "Fine-grained"
   - **Cấp quyền write repository** (quan trọng!)
   - Không cần bật webhooks, billing, jobs hay discussions
3. **Login:**
   ```bash
   hf auth login
   # Nhập token vừa tạo
   ```

#### Bước 3: Cấu Hình Repo Dùng Chung Cho Team
```bash
python model_sharing.py config --method huggingface --repo "your-org/emotion-classification"
```

**Lưu ý:** Dùng **organization repo** thay vì personal repo để team cùng truy cập.

#### Bước 4: Upload Model Tốt Nhất
```bash
python model_sharing.py sync
```

### Option 2: Google Drive (Backup)

#### Bước 1: Cài Đặt
```bash
pip install pydrive2
```

#### Bước 2: Setup Google Drive API
1. Tạo project tại: https://console.developers.google.com/
2. Enable Google Drive API
3. Tạo credentials
4. Download `credentials.json`

#### Bước 3: Cấu Hình
```bash
python model_sharing.py config --method gdrive
```

---

## 🔄 Workflow Mới Cho Team:

### Người Training (Upload):
```bash
# 1. Training như bình thường
python train_simple.py

# 2. Upload model tốt nhất lên cloud
python model_sharing.py sync

# 3. Commit chỉ metadata (không có model files)
git add model_registry/registry.json && git commit -m "Training results" && git push
```

### Người Khác (Download):
```bash
# 1. Pull metadata mới
git pull

# 2. Training - script tự động download model nếu cần
python train_simple.py
# → Script tự động download model tốt nhất từ cloud
```

---

## 🎯 Lợi Ích:

### ✅ Giải Quyết Được:
- Model sharing không giới hạn kích thước
- GitHub repository nhẹ (chỉ code + metadata)
- Transfer learning vẫn hoạt động bình thường
- Team collaboration không bị gián đoạn

### ✅ Workflow Đơn Giản:
- Người training: `python train_simple.py` → `python model_sharing.py sync` → `git push`
- Người khác: `git pull` → `python train_simple.py` (tự động download)

---

## 🔧 Commands Hữu Ích:

### Upload Model Cụ Thể:
```bash
python model_sharing.py upload --model-id model_20260422_124631
```

### Download Model Cụ Thể:
```bash
python model_sharing.py download --model-id model_20260422_124631
```

### Sync Model Tốt Nhất:
```bash
python model_sharing.py sync
```

### Xem Cấu Hình:
```bash
cat model_sharing_config.json
```

---

## 🆘 Troubleshooting:

### Lỗi: "huggingface_hub not installed"
```bash
pip install huggingface_hub
```

### Lỗi: "Authentication failed"
```bash
hf auth login
# Nhập token từ https://huggingface.co/settings/tokens
# Đảm bảo token có quyền write repository
```

### Lỗi: "Model not found on cloud"
```bash
# Upload model trước
python model_sharing.py upload --model-id <model_id>
```

### Fallback: Manual Download
Nếu auto-download thất bại, có thể download thủ công:
1. Vào Hugging Face repo: https://huggingface.co/your-username/emotion-classification
2. Download files cần thiết
3. Đặt vào `model_registry/models/<model_id>/`

---

## 📋 Migration Plan:

### Bước 1: Setup (1 lần)
```bash
# Cài đặt dependencies
pip install -U huggingface_hub

# Login với token có quyền write
hf auth login

# Cấu hình organization repo
python model_sharing.py config --method huggingface --repo "your-org/emotion-classification"

# Upload model hiện tại
python model_sharing.py sync
```

### Bước 2: Update .gitignore
```bash
# Đã tự động thêm vào .gitignore:
model_registry/models/*/pytorch_model.bin
model_registry/models/*/model.bin
```

### Bước 3: Clean Repository
```bash
# Remove model files từ git history (optional)
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch model_registry/models/*/pytorch_model.bin' --prune-empty --tag-name-filter cat -- --all
```

---

## ✅ Kết Quả:

**Sau khi setup:**
- ✅ GitHub repository nhẹ (~50 MB thay vì 3 GB)
- ✅ Model sharing không giới hạn
- ✅ Transfer learning hoạt động bình thường
- ✅ Team collaboration mượt mà
- ✅ Backup model trên cloud (an toàn hơn)

**Team chỉ cần chạy `python train_simple.py` như bình thường!**