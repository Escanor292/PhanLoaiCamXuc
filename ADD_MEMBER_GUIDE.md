# 👥 Hướng Dẫn Thêm Member Vào Organization

## 🎯 Mục Tiêu
Thêm `Escanor292` vào organization `emotion-classification-vn` để có thể upload model.

## 📋 Các Bước (Bạn làm trên Hugging Face)

### Bước 1: Vào Organization Settings
1. **Truy cập:** https://huggingface.co/emotion-classification-vn
2. **Click:** "Settings" (ở góc phải)
3. **Chọn:** "Members" tab

### Bước 2: Invite Member
1. **Click:** "Invite member" button
2. **Username:** `Escanor292`
3. **Role:** Chọn "Admin" hoặc "Member" (khuyến nghị: Admin)
4. **Click:** "Send invitation"

### Bước 3: Xác Nhận Quyền
Đảm bảo member có quyền:
- ✅ **Read repositories**
- ✅ **Write repositories** 
- ✅ **Create repositories**

## 🔄 Alternative: Tạo Token Mới

Nếu không muốn thêm member, bạn có thể:

### Bước 1: Tạo Token Với Quyền Organization
1. **Vào:** https://huggingface.co/settings/tokens
2. **New token** → **Fine-grained**
3. **Select organizations:** `emotion-classification-vn`
4. **Permissions:** Write access to contents of selected repos
5. **Copy token**

### Bước 2: Share Token
```bash
# Bạn share token này cho tôi để setup
hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

## ✅ Test Sau Khi Setup

Sau khi thêm member hoặc có token mới:

```bash
# Test upload
python switch_repo.py
# Chọn option 2: emotion-classification-vn/emotion-classification

# Hoặc thủ công
python model_sharing.py config --method huggingface --repo "emotion-classification-vn/emotion-classification"
python model_sharing.py sync
```

## 🎯 Kết Quả Mong Đợi

```
✅ Repository ready: emotion-classification-vn/emotion-classification
✅ Model uploaded to: https://huggingface.co/emotion-classification-vn/emotion-classification
🎉 SWITCH SUCCESSFUL!
```

---

**Chọn 1 trong 2 cách trên để tôi có thể upload model lên organization repo!** 🚀