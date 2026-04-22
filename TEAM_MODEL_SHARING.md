# 🚀 Team Model Sharing - Hướng Dẫn Nhanh

## 🎯 Mục Tiêu
Chia sẻ model 419MB giữa các thành viên team để Transfer Learning hoạt động.

## ⚡ Setup 1 Lần (Chỉ 1 Người Làm)

### Bước 1: Cài Đặt
```bash
pip install -U huggingface_hub
```

### Bước 2: Tạo Organization Repo
1. **Đăng ký Hugging Face:** https://huggingface.co/
2. **Tạo Organization:** https://huggingface.co/organizations/new
3. **Tạo Model Repo:** https://huggingface.co/new (chọn owner = organization)
4. **Tạo Token:**
   - Vào: https://huggingface.co/settings/tokens
   - Chọn "Fine-grained" → **Cấp quyền write repository**
   - Copy token: `hf_xxxxxxxxxxxxxxxxxxxxx`

### Bước 3: Login và Cấu Hình
```bash
# Login
hf auth login
# Paste token: hf_xxxxxxxxxxxxxxxxxxxxx

# Cấu hình repo (thay your-org bằng tên organization thực)
python model_sharing.py config --method huggingface --repo "your-org/emotion-classification"

# Upload model hiện tại
python model_sharing.py sync
```

## 🔄 Workflow Hàng Ngày

### Người Training:
```bash
# 1. Training
python train_simple.py

# 2. Upload model mới (nếu tốt hơn)
python model_sharing.py sync

# 3. Share metadata
git add model_registry/ && git commit -m "Training results" && git push
```

### Thành Viên Khác:
```bash
# 1. Pull metadata
git pull

# 2. Training (tự động download model nếu cần)
python train_simple.py
```

## ✅ Lợi Ích

- **Model sharing không giới hạn:** 419MB → ∞
- **GitHub nhẹ:** 3GB → 50MB
- **Transfer Learning hoạt động:** Model tự động download
- **Workflow đơn giản:** 1 lệnh training

## 🆘 Troubleshooting

### Lỗi: Authentication failed
```bash
hf auth login
# Đảm bảo token có quyền write repository
```

### Lỗi: Model not found
```bash
# Upload model trước
python model_sharing.py sync
```

### Lỗi: huggingface_hub not installed
```bash
pip install -U huggingface_hub
```

## 📋 Commands Hữu Ích

```bash
# Xem cấu hình
cat model_sharing_config.json

# Upload model cụ thể
python model_sharing.py upload --model-id model_20260422_124631

# Download model cụ thể  
python model_sharing.py download --model-id model_20260422_124631

# Sync model tốt nhất
python model_sharing.py sync
```

## 🎯 Kết Quả

**Trước:**
- ❌ Model 419MB không thể push GitHub
- ❌ Team không share được model
- ❌ Transfer Learning bị gián đoạn

**Sau:**
- ✅ Model share qua Hugging Face Hub
- ✅ GitHub chỉ lưu metadata (~1KB)
- ✅ Transfer Learning hoạt động tự động
- ✅ Team training mượt mà

**Chỉ cần chạy `python train_simple.py` - mọi thứ khác tự động!** 🚀